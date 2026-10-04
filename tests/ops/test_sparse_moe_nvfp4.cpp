// Host-only coverage for the nvfp4 routed-bank binding and validation path in
// prepare_sparse_moe_weights. No artifact and no CUDA device are required: the function only
// interprets WeightView geometry, so this exercises the codec accept/reject logic that the
// real-artifact loading test cannot cover without --artifact.

#include "core/weight.h"
#include "core/weight_view.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;

// Exact SparseMoe geometry.
constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kExperts      = 256;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kGateRows     = 512;

int g_failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++g_failures;
    }
}

template <class Fn>
void expect_invalid(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong error: " << error.what() << ")\n";
        ++g_failures;
        return;
    }
    std::cerr << "FAIL: " << message << " (no error)\n";
    ++g_failures;
}

// Stable parent/view storage: deques keep element references valid as the bank is filled.
struct Storage {
    std::deque<std::vector<std::byte>> buffers;
    std::deque<WeightParent> parents;
    std::deque<WeightView> views;

    std::vector<std::byte>& buffer(std::size_t bytes) { return buffers.emplace_back(bytes); }

    WeightParent& parent(const WeightGeometry& geometry, std::vector<std::byte>& bytes,
                         float divisor) {
        return parents.emplace_back(WeightParent{geometry, bytes.data(), divisor});
    }

    WeightView& view(std::vector<std::uint64_t> shape, const WeightParent& parent,
                     std::uint64_t begin, std::uint64_t end) {
        return views.emplace_back(WeightView{std::move(shape), {{&parent, begin, end}}});
    }
};

struct BankOptions {
    bool first_gate_q8       = false; // gate/up parent format differs from the down nvfp4 bank
    bool up_divisor_differs  = false; // A4 gate/up activation divisors disagree
    bool short_gate_geometry = false; // routed gate/up logical shape is wrong
};

struct Nvfp4Bank {
    Storage storage;
    WeightView* router_view       = nullptr;
    WeightView* shared_score_view = nullptr;
    WeightView* shared_gate_view  = nullptr;
    WeightView* shared_up_view    = nullptr;
    WeightView* shared_down_view  = nullptr;
    std::vector<WeightInput> gate_up_inputs;
    std::vector<WeightInput> down_inputs;
    std::vector<float> gate_up_divisors;
    std::vector<float> down_divisors;

    // The single-projection inputs are constructed here so WeightInput's reference member is
    // never default-constructed or assigned.
    SparseMoeWeights bind() const {
        const WeightInput router{*router_view, LinearPolicy::A16Only, std::nullopt};
        const WeightInput score{*shared_score_view, LinearPolicy::A16Only, std::nullopt};
        const WeightInput shared_gate{*shared_gate_view, LinearPolicy::A16Only, std::nullopt};
        const WeightInput shared_up{*shared_up_view, LinearPolicy::A16Only, std::nullopt};
        const WeightInput shared_down{*shared_down_view, LinearPolicy::A16Only, std::nullopt};
        return prepare_sparse_moe_weights(router, score, gate_up_inputs, down_inputs, shared_gate,
                                          shared_up, shared_down);
    }
};

void fill_bank(Nvfp4Bank& bank, const BankOptions& options) {
    const auto geometry2 = [](QType format, QuantLayout layout, std::uint64_t rows,
                              std::uint64_t columns) {
        const std::array<std::uint64_t, 2> shape{rows, columns};
        return weight_geometry(format, layout, shape);
    };
    const auto nvfp4_gate =
        geometry2(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, kGateRows, kHidden);
    const auto nvfp4_down =
        geometry2(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, kHidden, kIntermediate);
    const auto q8_gate = geometry2(QType::Q8_G32_FP16, QuantLayout::RowSplit, kGateRows, kHidden);
    const auto router  = geometry2(QType::BF16, QuantLayout::Contiguous, kExperts + 1, kHidden);
    const auto shared_gate =
        geometry2(QType::Q8_G32_FP16, QuantLayout::RowSplit, 2 * kIntermediate, kHidden);
    const auto shared_down =
        geometry2(QType::Q8_G32_FP16, QuantLayout::RowSplit, kHidden, kIntermediate);

    auto& router_bytes      = bank.storage.buffer(router.bytes);
    auto& shared_gate_bytes = bank.storage.buffer(shared_gate.bytes);
    auto& shared_down_bytes = bank.storage.buffer(shared_down.bytes);
    std::vector<std::vector<std::byte>*> gate_bytes;
    std::vector<std::vector<std::byte>*> down_bytes;
    for (int index = 0; index < 2 * kExperts; ++index) {
        gate_bytes.push_back(&bank.storage.buffer(nvfp4_gate.bytes));
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        down_bytes.push_back(&bank.storage.buffer(nvfp4_down.bytes));
    }

    auto& router_parent      = bank.storage.parent(router, router_bytes, 0.0F);
    auto& shared_gate_parent = bank.storage.parent(shared_gate, shared_gate_bytes, 0.0F);
    auto& shared_down_parent = bank.storage.parent(shared_down, shared_down_bytes, 0.0F);

    std::vector<const WeightParent*> gate_parents;
    for (int index = 0; index < 2 * kExperts; ++index) {
        const float divisor = 8.0F + static_cast<float>(index) * 0.25F;
        bank.gate_up_divisors.push_back(divisor);
        if (options.first_gate_q8 && index == 0) {
            auto& q8_bytes = bank.storage.buffer(q8_gate.bytes);
            gate_parents.push_back(&bank.storage.parent(q8_gate, q8_bytes, 0.0F));
        } else {
            gate_parents.push_back(&bank.storage.parent(nvfp4_gate, *gate_bytes[index], divisor));
        }
    }
    std::vector<const WeightParent*> down_parents;
    for (int expert = 0; expert < kExperts; ++expert) {
        const float divisor = 4.0F + static_cast<float>(expert) * 0.5F;
        bank.down_divisors.push_back(divisor);
        down_parents.push_back(&bank.storage.parent(nvfp4_down, *down_bytes[expert], divisor));
    }

    const auto gate_elements = static_cast<std::uint64_t>(kGateRows) * kHidden;
    const auto down_elements = static_cast<std::uint64_t>(kHidden) * kIntermediate;

    bank.router_view = &bank.storage.view({kExperts, kHidden}, router_parent, 0,
                                          static_cast<std::uint64_t>(kExperts) * kHidden);
    bank.shared_score_view =
        &bank.storage.view({1, kHidden}, router_parent,
                           static_cast<std::uint64_t>(kExperts) * kHidden,
                           static_cast<std::uint64_t>(kExperts + 1) * kHidden);
    bank.shared_gate_view =
        &bank.storage.view({kIntermediate, kHidden}, shared_gate_parent, 0,
                           static_cast<std::uint64_t>(kIntermediate) * kHidden);
    bank.shared_up_view =
        &bank.storage.view({kIntermediate, kHidden}, shared_gate_parent,
                           static_cast<std::uint64_t>(kIntermediate) * kHidden,
                           static_cast<std::uint64_t>(2 * kIntermediate) * kHidden);
    bank.shared_down_view =
        &bank.storage.view({kHidden, kIntermediate}, shared_down_parent, 0, down_elements);

    for (int index = 0; index < 2 * kExperts; ++index) {
        const auto shape = options.short_gate_geometry && index == 0
                               ? std::vector<std::uint64_t>{kGateRows - 1, kHidden}
                               : std::vector<std::uint64_t>{kGateRows, kHidden};
        const auto end = options.short_gate_geometry && index == 0 ? gate_elements - kHidden
                                                                   : gate_elements;
        float divisor = 1.5F;
        if (options.up_divisor_differs && index == 1) { divisor = 1.75F; }
        bank.gate_up_inputs.push_back(WeightInput{
            bank.storage.view(shape, *gate_parents[index], 0, end), LinearPolicy::AllowA4,
            divisor});
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        bank.down_inputs.push_back(WeightInput{
            bank.storage.view({kHidden, kIntermediate}, *down_parents[expert], 0, down_elements),
            LinearPolicy::AllowA4, 2.5F});
    }
}

void accepts_nvfp4_bank() {
    Nvfp4Bank bank;
    fill_bank(bank, {});
    const auto weights = bank.bind();
    check(weights.routed_gate_up.qtype == QType::NVFP4 &&
              weights.routed_down.qtype == QType::NVFP4,
          "nvfp4 routed bank was not reported as nvfp4");
    check(weights.routed_gate_up_experts.size() == 2 * kExperts &&
              weights.routed_down_experts.size() == kExperts,
          "nvfp4 routed bank lost expert projections");
    for (std::size_t index = 0; index < weights.routed_gate_up_experts.size(); ++index) {
        const auto& expert = weights.routed_gate_up_experts[index];
        check(expert.qtype == QType::NVFP4 &&
                  expert.weight_scale_divisor == bank.gate_up_divisors[index] &&
                  expert.input_scale_divisor == 1.5F,
              "nvfp4 gate/up projection was not bound with its own divisor");
    }
    for (std::size_t expert = 0; expert < weights.routed_down_experts.size(); ++expert) {
        const auto& weight = weights.routed_down_experts[expert];
        check(weight.qtype == QType::NVFP4 &&
                  weight.weight_scale_divisor == bank.down_divisors[expert] &&
                  weight.input_scale_divisor == 2.5F,
              "nvfp4 down projection was not bound with its own divisor");
    }
}

void rejects_non_nvfp4_and_bad_geometry() {
    {
        Nvfp4Bank bank;
        fill_bank(bank, {.first_gate_q8 = true});
        expect_invalid([&] { (void)bank.bind(); },
                       "a routed bank mixing q8 gate/up with nvfp4 down was accepted");
    }
    {
        Nvfp4Bank bank;
        fill_bank(bank, {.short_gate_geometry = true});
        expect_invalid([&] { (void)bank.bind(); },
                       "a routed gate/up parent with the wrong geometry was accepted");
    }
}

void rejects_mismatched_gate_up_divisor() {
    Nvfp4Bank bank;
    fill_bank(bank, {.up_divisor_differs = true});
    expect_invalid([&] { (void)bank.bind(); },
                   "nvfp4 gate/up activation divisors disagreeing was accepted");
}

} // namespace

int main() {
    try {
        accepts_nvfp4_bank();
        rejects_non_nvfp4_and_bad_geometry();
        rejects_mismatched_gate_up_divisor();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (g_failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 binding\n";
    return g_failures == 0 ? 0 : 1;
}
