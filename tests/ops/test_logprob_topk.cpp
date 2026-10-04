#include "ninfer/ops/logprob_topk.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int K = ops::kLogprobTopK;

// Full-vocabulary logprob values span roughly [-log(V), 0]; the dominant term is the logsumexp
// reduction, so a tight absolute bound with a small relative slack is the natural criterion.
const PointwiseCriterion kLogprobCriterion{/*absolute=*/3.0e-5, /*relative=*/1.0e-5};
const PointwiseCriterion kLseCriterion{/*absolute=*/3.0e-5, /*relative=*/1.0e-5};

struct CaseSpec {
    float temperature = 1.0f;
    float presence    = 0.0f;
    float frequency   = 0.0f;
    bool masked       = false; // contiguous 10% suffix set to -inf
    bool holes        = false; // scattered -inf inside the full token domain
    bool counts       = false;
    bool ties         = false;
};

std::vector<double> f32_as_double(const void* device, std::size_t count) {
    const auto values = from_device<float>(device, count);
    return {values.begin(), values.end()};
}

std::vector<float> make_logits(std::int32_t physical_rows, std::int32_t rows, const CaseSpec& spec,
                               std::uint32_t seed) {
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * rows);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    const std::int32_t valid_start = physical_rows / 10;
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t v = 0; v < physical_rows; ++v) {
            float value = dist(rng);
            if (spec.masked && v >= valid_start) { value = -std::numeric_limits<float>::infinity(); }
            // Scattered masked holes inside the full domain: the top-K must skip every one.
            if (spec.holes && v >= valid_start && v < 2 * valid_start && (v % 3) == 0) {
                value = -std::numeric_limits<float>::infinity();
            }
            if (spec.ties) { value = static_cast<float>((v + r) % 5); }
            // ninfer layout: ne[0]-contiguous, so element (v, r) is at r*physical_rows + v.
            logits[static_cast<std::size_t>(r) * physical_rows + v] =
                bf16_to_f32(f32_to_bf16(value));
        }
    }
    return logits;
}

// Row-major [rows, physical_rows]: row r's token_counts array starts at r*physical_rows.
std::vector<std::int32_t> make_counts(std::int32_t physical_rows, std::int32_t rows,
                                      std::uint32_t seed) {
    std::vector<std::int32_t> counts(static_cast<std::size_t>(physical_rows) * rows);
    std::mt19937 rng(seed);
    for (auto& value : counts) { value = static_cast<std::int32_t>(rng() % 5u); }
    return counts;
}

struct OracleOutput {
    std::vector<std::int32_t> top_ids;   // [K, rows]
    std::vector<double> top_values;      // [K, rows]
    std::vector<double> lse;             // [rows]
};

// Independent naive FP64 oracle over exactly the BF16 values the kernel reads.
OracleOutput logprob_oracle(const std::vector<float>& logits,
                            const std::vector<std::int32_t>& counts, std::int32_t physical_rows,
                            std::int32_t rows, std::int32_t token_domain, const CaseSpec& spec,
                            bool counts_present) {
    OracleOutput out;
    out.top_ids.assign(static_cast<std::size_t>(K) * rows, 0);
    out.top_values.assign(static_cast<std::size_t>(K) * rows, 0.0);
    out.lse.assign(rows, 0.0);
    const double temperature = spec.temperature > 0.0f ? spec.temperature : 1.0;

    for (std::int32_t r = 0; r < rows; ++r) {
        std::vector<double> scaled(token_domain);
        for (std::int32_t v = 0; v < token_domain; ++v) {
            const double raw = logits[static_cast<std::size_t>(r) * physical_rows + v];
            double adjusted = raw;
            if (counts_present) {
                const double count = counts[static_cast<std::size_t>(r) * physical_rows + v];
                if (count > 0.0) { adjusted -= spec.presence; }
                adjusted -= spec.frequency * count;
            }
            scaled[v] = adjusted / temperature;
        }
        // Masked (-inf) tokens are not candidates, and do not contribute to the logsumexp.
        std::vector<std::int32_t> order;
        for (std::int32_t v = 0; v < token_domain; ++v) {
            if (std::isfinite(scaled[v])) { order.push_back(v); }
        }
        const bool any = !order.empty();
        const double maximum = any ? scaled[order[0]] : 0.0;
        double sum = 0.0;
        for (const std::int32_t v : order) { sum += std::exp(scaled[v] - maximum); }
        const double lse = maximum + std::log(sum);
        out.lse[r]      = lse;

        std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
            return scaled[a] > scaled[b];
        });
        for (int k = 0; k < K; ++k) {
            // ninfer layout: [K, rows] is ne[0]-contiguous, so element (k, r) is at k + r*K.
            const std::size_t slot =
                static_cast<std::size_t>(k) + static_cast<std::size_t>(r) * static_cast<std::size_t>(K);
            if (static_cast<std::size_t>(k) < order.size()) {
                const std::int32_t id = order[static_cast<std::size_t>(k)];
                out.top_ids[slot]      = id;
                out.top_values[slot]   = scaled[id] - lse;
            } else {
                out.top_ids[slot]    = -1;
                out.top_values[slot] = -std::numeric_limits<double>::infinity();
            }
        }
    }
    return out;
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t rows,
             const CaseSpec& spec) {
    const std::int32_t token_domain = spec.masked ? physical_rows / 10 : physical_rows;
    const auto logits               = make_logits(physical_rows, rows, spec, 0xC0FFEEu + rows);
    const auto counts_host          = make_counts(physical_rows, rows, 0xBEEFu + rows);
    const bool counts_present       = spec.counts;

    const auto oracle = logprob_oracle(logits, counts_host, physical_rows, rows, token_domain, spec,
                                       counts_present);

    DeviceBuffer count_buffer(counts_present ? counts_host.size() * sizeof(std::int32_t) : 0);
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(rows));
    for (std::int32_t r = 0; r < rows; ++r) {
        configs[static_cast<std::size_t>(r)].temperature        = spec.temperature;
        configs[static_cast<std::size_t>(r)].top_k              = 20;
        configs[static_cast<std::size_t>(r)].presence_penalty   = spec.presence;
        configs[static_cast<std::size_t>(r)].frequency_penalty  = spec.frequency;
        configs[static_cast<std::size_t>(r)].token_counts =
            counts_present ? static_cast<std::int32_t*>(count_buffer.p) +
                                 static_cast<std::ptrdiff_t>(r) * physical_rows
                           : nullptr;
    }
    if (counts_present) { count_buffer.copy_from_host(counts_host.data(), count_buffer.bytes); }
    DeviceBuffer config_buffer(configs.size() * sizeof(ops::SamplingConfig));
    config_buffer.copy_from_host(configs.data(), config_buffer.bytes);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    {
        std::vector<std::uint16_t> packed(logits.size());
        for (std::size_t i = 0; i < logits.size(); ++i) {
            packed[i] = f32_to_bf16(logits[i]);
        }
        device_logits.copy_from_host(packed.data(), device_logits.bytes());
    }
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(K) * rows * sizeof(std::int32_t));
    GuardedDeviceBuffer device_values(static_cast<std::size_t>(K) * rows * sizeof(float));
    GuardedDeviceBuffer device_lse(static_cast<std::size_t>(rows) * sizeof(float));
    GuardedDeviceBuffer device_active(sizeof(std::int32_t));
    device_ids.fill(0xcd);
    device_values.fill(0xcd);
    device_lse.fill(0xcd);
    {
        std::int32_t active = 1;
        device_active.copy_from_host(&active, sizeof(active));
    }

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, rows});
    Tensor ids_tensor(device_ids.data(), DType::I32, {K, rows});
    Tensor values_tensor(device_values.data(), DType::FP32, {K, rows});
    Tensor lse_tensor(device_lse.data(), DType::FP32, {rows});
    Tensor active_tensor(device_active.data(), DType::I32, {1});
    auto* config_pointer =
        reinterpret_cast<const ops::SamplingConfig*>(config_buffer.p);

    const std::size_t capacity = ops::logprob_topk_workspace_capacity_bytes(token_domain, rows);
    WorkspaceArena workspace(capacity);
    ops::logprob_topk(logits_tensor, config_pointer, token_domain, ids_tensor, values_tensor,
                      lse_tensor, active_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = 0;
    failures += verify_exact((label + " top_ids").c_str(),
                             from_device<std::int32_t>(device_ids.data(),
                                                       static_cast<std::size_t>(K) * rows),
                             oracle.top_ids);
    failures += verify_pointwise(label + " top_values",
                                 f32_as_double(device_values.data(),
                                               static_cast<std::size_t>(K) * rows),
                                 oracle.top_values, kLogprobCriterion);
    failures += verify_pointwise(label + " lse", f32_as_double(device_lse.data(), rows),
                                 oracle.lse, kLseCriterion);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_ids.verify_guards(label + " top_ids guards");
    failures += device_values.verify_guards(label + " top_values guards");
    failures += device_lse.verify_guards(label + " lse guards");
    return failures;
}

int run_inactive_case() {
    // active==0 must not launch work; outputs stay at their prefilled sentinel.
    const std::int32_t physical_rows = 4096;
    const std::int32_t rows          = 2;
    const CaseSpec spec{};
    const auto logits = make_logits(physical_rows, rows, spec, 0x51EEDu);
    std::vector<ops::SamplingConfig> configs(rows);
    DeviceBuffer config_buffer(configs.size() * sizeof(ops::SamplingConfig));
    config_buffer.copy_from_host(configs.data(), config_buffer.bytes);
    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    {
        std::vector<std::uint16_t> packed(logits.size());
        for (std::size_t i = 0; i < logits.size(); ++i) { packed[i] = f32_to_bf16(logits[i]); }
        device_logits.copy_from_host(packed.data(), device_logits.bytes());
    }
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(K) * rows * sizeof(std::int32_t));
    GuardedDeviceBuffer device_values(static_cast<std::size_t>(K) * rows * sizeof(float));
    GuardedDeviceBuffer device_lse(static_cast<std::size_t>(rows) * sizeof(float));
    GuardedDeviceBuffer device_active(sizeof(std::int32_t));
    device_ids.fill(0xcd);
    device_values.fill(0xcd);
    device_lse.fill(0xcd);
    std::int32_t active = 0;
    device_active.copy_from_host(&active, sizeof(active));

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, rows});
    Tensor ids_tensor(device_ids.data(), DType::I32, {K, rows});
    Tensor values_tensor(device_values.data(), DType::FP32, {K, rows});
    Tensor lse_tensor(device_lse.data(), DType::FP32, {rows});
    Tensor active_tensor(device_active.data(), DType::I32, {1});
    auto* config_pointer = reinterpret_cast<const ops::SamplingConfig*>(config_buffer.p);

    WorkspaceArena workspace(ops::logprob_topk_workspace_capacity_bytes(physical_rows, rows));
    ops::logprob_topk(logits_tensor, config_pointer, physical_rows, ids_tensor, values_tensor,
                      lse_tensor, active_tensor, workspace, nullptr);
    cuda_synchronize();

    int failures = 0;
    const auto ids = from_device<std::int32_t>(device_ids.data(), static_cast<std::size_t>(K) * rows);
    const bool untouched = std::all_of(ids.begin(), ids.end(), [](std::int32_t value) {
        return (static_cast<std::uint32_t>(value) & 0xffu) == 0xcdu;
    });
    if (!untouched) {
        std::cerr << "inactive gate: top_ids was written while active==0\n";
        failures += 1;
    }
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(64 * 2 * sizeof(std::uint16_t));
    DeviceBuffer config_data(2 * sizeof(ops::SamplingConfig));
    DeviceBuffer ids_data(K * 2 * sizeof(std::int32_t));
    DeviceBuffer values_data(K * 2 * sizeof(float));
    DeviceBuffer lse_data(2 * sizeof(float));
    DeviceBuffer active_data(sizeof(std::int32_t));
    Tensor logits(logits_data.p, DType::BF16, {64, 2});
    Tensor ids(ids_data.p, DType::I32, {K, 2});
    Tensor values(values_data.p, DType::FP32, {K, 2});
    Tensor lse(lse_data.p, DType::FP32, {2});
    Tensor active(active_data.p, DType::I32, {1});
    auto* configs = reinterpret_cast<const ops::SamplingConfig*>(config_data.p);

    int failures = 0;
    failures += expect_invalid("logprob_topk rejects null configs", [&] {
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, nullptr, 64, ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects token_domain=0", [&] {
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, 0, ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects token_domain<kLogprobTopK", [&] {
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, K - 1, ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects token_domain>physical_rows", [&] {
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, 65, ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects top_ids shape mismatch", [&] {
        Tensor wrong_ids(ids_data.p, DType::I32, {K, 1});
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, 64, wrong_ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects lse dtype", [&] {
        Tensor wrong_lse(lse_data.p, DType::BF16, {2});
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, 64, ids, values, wrong_lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects non-contiguous logits", [&] {
        Tensor strided = logits;
        strided.nb[1] += 2;
        WorkspaceArena workspace(1024);
        ops::logprob_topk(strided, configs, 64, ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects active dtype", [&] {
        Tensor wrong_active(active_data.p, DType::FP32, {1});
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, 64, ids, values, lse, wrong_active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects output/input overlap", [&] {
        Tensor alias_ids(logits_data.p, DType::I32, {K, 2});
        WorkspaceArena workspace(1024);
        ops::logprob_topk(logits, configs, 64, alias_ids, values, lse, active, workspace, nullptr);
    });
    failures += expect_invalid("logprob_topk rejects zero-row profile", [&] {
        (void)ops::logprob_topk_workspace_capacity_bytes(64, 0);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("logprob_topk plain t=1 rows=1", 151936, 1, CaseSpec{});
    failures += run_case("logprob_topk plain t=1 rows=8", 151936, 8, CaseSpec{});
    failures += run_case("logprob_topk plain t=0.7 rows=8", 151936, 8,
                         CaseSpec{.temperature = 0.7f});
    failures += run_case("logprob_topk greedy rows=4", 151936, 4,
                         CaseSpec{.temperature = 0.0f});
    failures += run_case("logprob_topk penalties rows=8", 151936, 8,
                         CaseSpec{.presence = 0.5f, .frequency = 0.3f, .counts = true});
    failures += run_case("logprob_topk masked rows=8", 151936, 8, CaseSpec{.masked = true});
    failures += run_case("logprob_topk masked holes rows=8", 151936, 8, CaseSpec{.holes = true});
    failures += run_case("logprob_topk masked penalties rows=4", 151936, 4,
                         CaseSpec{.presence = 0.5f, .frequency = 0.3f, .masked = true,
                                  .counts = true});
    failures += run_case("logprob_topk ties rows=1", 4096, 1, CaseSpec{.ties = true});
    failures += run_case("logprob_topk minimum vocabulary V=20", 20, 3, CaseSpec{});
    failures += run_case("logprob_topk split boundary V=128", 128, 2, CaseSpec{});
    failures += run_case("logprob_topk split boundary V=129", 129, 2, CaseSpec{});
    failures += run_inactive_case();
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " logprob_topk\n";
    return failures ? 1 : 0;
}
