// Full fused nvfp4 grouped-route oracle (ticket #212 lineage, map #175). Drives the fork route's
// grouped W4A4 pass through the merged moe_cuda::fork_nvfp4 launch (packed-route grouping ->
// nvfp4 routed gate/up -> nvfp4 routed down -> contribution reduce -> shared expert -> final
// merge) and compares the result against an independent FP64 oracle.
//
// The oracle decodes the production quantised activation planes and the stored BF16 shared
// activation, and decodes every weight from its packed representation with the fixture's
// independent logical decode; it never re-runs a kernel. The routed/shared activation
// quantisation is a private execution choice, so it is read back rather than re-derived, per
// op-development.md ("packed inputs are independently decoded with their stored scales"). The
// routed gate/up GEMM itself is qualified by its own oracle in
// ninfer_sparse_moe_nvfp4_gate_up_test (#211), and the A16 route below the grouped boundary by
// ninfer_sparse_moe_nvfp4_decode_test (#213).

#include "ops/fork_nvfp4_bank.h"

#include "ops/moe/kernels/cuda/groupwise_int/launch.h"
#include "ops/moe/kernels/cuda/groupwise_int/workspace.h"
#include "ops/moe/nvfp4/sparse_moe_nvfp4_down.h"
#include "ops/moe/workspace.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace fn = ninfer::test::fork_nvfp4;
namespace qw = ninfer::test::quantized_weight;
namespace kn = ninfer::ops::detail::moe_cuda::fork_nvfp4;
namespace gi = ninfer::ops::detail::moe_cuda::groupwise_int;

namespace {

constexpr std::int32_t kHidden       = fn::kHidden;
constexpr std::int32_t kExperts      = fn::kExperts;
constexpr std::int32_t kTopK         = fn::kTopK;
constexpr std::int32_t kIntermediate = fn::kIntermediate;
constexpr std::int32_t kRouterRows   = kExperts + 1;
constexpr std::int32_t kSharedRows   = 2 * kIntermediate;

// The complete fused chain mixes two FP4 quantisation steps plus the Q8 shared path, so the
// bound is looser than a single Op's. Verified cases land well inside it.
constexpr ReductionCriterion kFusedTolerance{
    /*relative_l2*/ 2.0e-2,
    /*gross_absolute*/ 6.0e-3,
    /*gross_relative_to_max_reference*/ 0.0,
};

int device_multiprocessor_count() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    int count = 0;
    cuda_check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device),
               "cudaDeviceGetAttribute(MultiProcessorCount)");
    return count;
}

std::uint32_t lcg(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

std::vector<float> random_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    const double span = static_cast<double>(high) - static_cast<double>(low);
    for (std::size_t i = 0; i < count; ++i) {
        const double unit = (lcg(seed) >> 8) * (1.0 / 16777216.0);
        values[i]         = static_cast<float>(static_cast<double>(low) + unit * span);
    }
    return values;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// Independent decode of a NVFP4 parent into its logical FP32 matrix (E2M1 code x E4M3 scale /
// weight divisor), keyed only by the stored representation.
std::vector<float> decode_nvfp4_matrix(const qw::PackedWeight& packed) {
    const std::int32_t n       = packed.weight.n;
    const std::int32_t k       = packed.weight.k;
    const std::uint8_t* codes  = packed.payload.data();
    const std::uint8_t* scales = packed.payload.data() + packed.scale_plane_offset;
    const double divisor       = packed.weight.weight_scale_divisor;
    const std::int32_t k_tiles = k / 64;
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        const std::int32_t row_tile  = row / 128;
        const std::int32_t row_inner = row % 128;
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint8_t byte = codes[static_cast<std::size_t>(row) * (k / 2) + column / 2];
            const std::uint8_t code = (column & 1) == 0 ? (byte & 0x0fU) : (byte >> 4);
            const std::int32_t group      = column / 16;
            const std::int32_t scale_tile = group / 4;
            const std::int32_t scale_lane = group % 4;
            const std::size_t scale_offset =
                static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512U +
                static_cast<std::size_t>(row_inner % 32) * 16U +
                static_cast<std::size_t>(row_inner / 32) * 4U + scale_lane;
            const double decoded = static_cast<double>(qw::detail::decode_e2m1(code)) *
                                   static_cast<double>(qw::detail::decode_e4m3fn(
                                       scales[scale_offset])) /
                                   divisor;
            out[static_cast<std::size_t>(row) * k + column] = static_cast<float>(decoded);
        }
    }
    return out;
}

// Independent decode of a row-split Q8 parent via the fixture's FP64 logical weight.
std::vector<float> decode_q8_matrix(const qw::PackedWeight& packed) {
    const std::int32_t n = packed.weight.n;
    const std::int32_t k = packed.weight.k;
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            out[static_cast<std::size_t>(row) * k + column] =
                static_cast<float>(qw::logical_weight_fp64(packed, row, column));
        }
    }
    return out;
}

// Independent decode of the production down-stage FP4 plane: one compact row per assignment.
std::vector<float> decode_down_plane(const std::vector<std::uint8_t>& codes,
                                     const std::vector<std::uint8_t>& scales, int assignments) {
    std::vector<float> out(static_cast<std::size_t>(assignments) * kIntermediate);
    for (int column = 0; column < assignments; ++column) {
        for (std::int32_t k = 0; k < kIntermediate; ++k) {
            const auto byte = codes[static_cast<std::size_t>(column) * (kIntermediate / 2) +
                                    k / 2];
            const auto code = static_cast<std::uint8_t>((k & 1) == 0 ? (byte & 0x0fU) : (byte >> 4));
            const auto scale = scales[static_cast<std::size_t>(column) * (kIntermediate / 16) +
                                      k / 16];
            out[static_cast<std::size_t>(column) * kIntermediate + k] =
                static_cast<float>(static_cast<double>(qw::detail::decode_e2m1(code)) *
                                   static_cast<double>(qw::detail::decode_e4m3fn(scale)));
        }
    }
    return out;
}

struct SharedExpert {
    qw::PackedWeight gate_up_packed, down_packed;
    ops::SwiGLUWeights weights;
    DeviceBuffer gate_up_storage, down_storage;
    std::vector<float> coefficients; // [tokens], consumed as the term's shared_scale
};

SharedExpert make_shared(std::int32_t tokens) {
    SharedExpert shared;
    shared.gate_up_packed =
        qw::make_patterned_weight(QType::Q8_G32_FP16, kSharedRows, kHidden, 7u);
    shared.down_packed =
        qw::make_patterned_weight(QType::Q8_G32_FP16, kHidden, kIntermediate, 11u);
    // The shared gate/up codes are zeroed so the fused output isolates the routed path and the
    // packed-column reduce; the shared path's numerics are covered by the merged suite.
    std::fill(shared.gate_up_packed.payload.begin(),
              shared.gate_up_packed.payload.begin() +
                  static_cast<std::ptrdiff_t>(shared.gate_up_packed.code_plane_bytes),
              static_cast<std::uint8_t>(0));
    shared.gate_up_storage = to_device(shared.gate_up_packed.payload);
    shared.down_storage    = to_device(shared.down_packed.payload);
    shared.weights.gate_up.weight =
        shared.gate_up_packed.device_weight(shared.gate_up_storage.span());
    shared.weights.down.weight =
        shared.down_packed.device_weight(shared.down_storage.span());
    shared.coefficients = random_values(static_cast<std::size_t>(tokens), 0x77u, 0.25F, 1.0F);
    return shared;
}

struct Run {
    std::vector<std::uint16_t> destination;
    std::vector<std::uint16_t> residual;
    std::vector<std::uint16_t> grouped_output;
    std::vector<std::uint8_t> down_codes;
    std::vector<std::uint8_t> down_scales;
    std::vector<std::uint16_t> shared_activation;
    std::vector<int> packed_index; // inverse: assignment -> packed column
    std::vector<int> expert_offsets;
    std::vector<int> ids;
    std::vector<float> alpha;
    std::vector<float> shared_scale;
    std::int32_t assignments = 0;
    std::int32_t tokens      = 0;
};

Run run_fused(fn::Bank& bank, const SharedExpert& shared, std::int32_t tokens) {
    const std::int32_t assignments = tokens * kTopK;
    const std::vector<float> x_values =
        random_values(static_cast<std::size_t>(kHidden) * tokens, 0xabcdefu, -1.0F, 1.0F);
    const std::vector<float> residual_values =
        random_values(static_cast<std::size_t>(kHidden) * tokens, 0x5555u, -0.5F, 0.5F);
    DeviceBuffer x_storage        = to_device_bf16(x_values);
    DeviceBuffer residual_storage = to_device_bf16(residual_values);
    DeviceBuffer destination_storage(cuda_environment().context.device(),
                                     static_cast<std::size_t>(kHidden) * tokens *
                                         sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(destination_storage.data(), residual_storage.data(),
                          destination_storage.size(), cudaMemcpyDeviceToDevice),
               "seed fused destination");

    Run run;
    run.tokens      = tokens;
    run.assignments = assignments;
    run.residual    = bf16_bits(residual_values);
    fn::fill_routes(run.ids, run.alpha, tokens, 3);
    run.shared_scale = shared.coefficients;

    DeviceBuffer ids    = to_device<int>(run.ids);
    DeviceBuffer coefs  = to_device<float>(run.alpha);
    DeviceBuffer shared_coefs = to_device<float>(shared.coefficients);

    const auto device = cuda_environment().context.device();
    const Tensor x(x_storage.data(), device, DType::BF16, {kHidden, tokens});
    Tensor destination(destination_storage.data(), device, DType::BF16, {kHidden, tokens});
    const ops::ExpertRoutes routes{
        Tensor(ids.data(), device, DType::I32, {kTopK, tokens}),
        Tensor(coefs.data(), device, DType::FP32, {kTopK, tokens})};
    const ops::ScaledSwiGLUTerm term{
        shared.weights,
        Tensor(shared_coefs.data(), device, DType::FP32, {tokens})};

    const kn::Plan plan{
        .grouped = true, .queued_ctas_per_sm = 32,
        .shared  = {.token   = gi::TokenPlan{},
                    .columns = 64}};
    WorkspaceLayoutBuilder probe;
    (void)kn::allocate(probe, plan, tokens, true, false);
    (void)gi::allocate_shared(probe, tokens);
    GuardedDeviceBuffer scratch(probe.peak_bytes(1));
    MemoryArena arena(scratch.span());
    const auto storage        = kn::allocate(arena, plan, tokens, true, false);
    const auto shared_storage = gi::allocate_shared(arena, tokens);

    const auto execution = cuda_environment().context.execution_view(nullptr);
    kn::launch(x, bank.experts(), routes, ops::ExpertSubset{ops::ExpertSubsetKind::All, {}},
               &term, ops::MoeEpilogue::AddResidual, destination, plan, storage, shared_storage,
               execution);
    cuda_synchronize();

    run.destination = from_device<std::uint16_t>(destination_storage,
                                                static_cast<std::size_t>(kHidden) * tokens);
    run.grouped_output =
        from_device<std::uint16_t>(storage.routed_io,
                                   static_cast<std::size_t>(kHidden) * assignments);
    run.down_codes  = from_device<std::uint8_t>(storage.product_codes,
                                               static_cast<std::size_t>(assignments) * (512 / 2));
    run.down_scales = from_device<std::uint8_t>(storage.product_scales,
                                               static_cast<std::size_t>(assignments) * (512 / 16));
    run.shared_activation =
        from_device<std::uint16_t>(shared_storage.shared_activation,
                                   static_cast<std::size_t>(kIntermediate) * tokens);
    run.packed_index   = from_device<int>(storage.inverse, assignments);
    run.expert_offsets = from_device<int>(storage.expert_offsets, kExperts + 1);
    return run;
}

// Shared oracle state: the column->expert map and a lazily decoded per-expert down matrix.
struct Oracle {
    const fn::Bank& bank;
    const std::vector<int>& expert_offsets;
    const qw::PackedWeight* shared_down = nullptr;
    std::vector<int> column_expert;
    std::vector<std::unique_ptr<std::vector<float>>> down_matrices;
    std::vector<float> down_activation;

    Oracle(const fn::Bank& bank_ref, const Run& run)
        : bank(bank_ref), expert_offsets(run.expert_offsets),
          down_matrices(static_cast<std::size_t>(kExperts)) {
        column_expert.assign(run.assignments, -1);
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            for (int column = run.expert_offsets[expert]; column < run.expert_offsets[expert + 1];
                 ++column) {
                column_expert[column] = expert;
            }
        }
        down_activation = decode_down_plane(run.down_codes, run.down_scales, run.assignments);
    }

    const std::vector<float>& down_weights(int expert) {
        auto& slot = down_matrices[expert];
        if (slot == nullptr) { slot = std::make_unique<std::vector<float>>(
                                   decode_nvfp4_matrix(bank.down_packed(expert))); }
        return *slot;
    }
};

int check_down_stage(Oracle& oracle, const Run& run, const char* label, int column_stride) {
    std::vector<double> actual;
    std::vector<double> reference;
    for (int column = 0; column < run.assignments; column += column_stride) {
        const std::vector<float>& weights = oracle.down_weights(oracle.column_expert[column]);
        const float* a = &oracle.down_activation[static_cast<std::size_t>(column) * kIntermediate];
        for (std::int32_t n = 0; n < kHidden; ++n) {
            const float* w = &weights[static_cast<std::size_t>(n) * kIntermediate];
            double accumulator = 0.0;
            for (std::int32_t k = 0; k < kIntermediate; ++k) {
                accumulator += static_cast<double>(a[k]) * static_cast<double>(w[k]);
            }
            reference.push_back(bf16_to_f32(f32_to_bf16(static_cast<float>(accumulator))));
            actual.push_back(
                bf16_to_f32(run.grouped_output[static_cast<std::size_t>(column) * kHidden + n]));
        }
    }
    return verify_reduction(std::string(label) + " routed down output", actual, reference,
                            kFusedTolerance);
}

int check_fused(Oracle& oracle, const Run& run, const char* label) {
    const std::int32_t tokens      = run.tokens;
    const std::int32_t assignments = run.assignments;

    std::vector<double> down_output(static_cast<std::size_t>(assignments) * kHidden);
    for (int column = 0; column < assignments; ++column) {
        const std::vector<float>& weights = oracle.down_weights(oracle.column_expert[column]);
        const float* a = &oracle.down_activation[static_cast<std::size_t>(column) * kIntermediate];
        double* out    = &down_output[static_cast<std::size_t>(column) * kHidden];
        for (std::int32_t n = 0; n < kHidden; ++n) {
            const float* w = &weights[static_cast<std::size_t>(n) * kIntermediate];
            double accumulator = 0.0;
            for (std::int32_t k = 0; k < kIntermediate; ++k) {
                accumulator += static_cast<double>(a[k]) * static_cast<double>(w[k]);
            }
            out[n] = bf16_to_f32(f32_to_bf16(static_cast<float>(accumulator)));
        }
    }

    // The shared down path consumes the stored BF16 shared activation, [T, I] row-major.
    const std::vector<float> shared_down_weights = decode_q8_matrix(*oracle.shared_down);
    std::vector<double> shared_out(static_cast<std::size_t>(tokens) * kHidden);
    std::vector<float> shared_row(kIntermediate);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t k = 0; k < kIntermediate; ++k) {
            shared_row[k] =
                bf16_to_f32(run.shared_activation[static_cast<std::size_t>(token) * kIntermediate +
                                                 k]);
        }
        for (std::int32_t n = 0; n < kHidden; ++n) {
            const float* w = &shared_down_weights[static_cast<std::size_t>(n) * kIntermediate];
            double accumulator = 0.0;
            for (std::int32_t k = 0; k < kIntermediate; ++k) {
                accumulator += static_cast<double>(shared_row[k]) * static_cast<double>(w[k]);
            }
            shared_out[static_cast<std::size_t>(token) * kHidden + n] = accumulator;
        }
    }

    std::vector<double> actual;
    std::vector<double> reference;
    actual.reserve(static_cast<std::size_t>(tokens) * kHidden);
    reference.reserve(static_cast<std::size_t>(tokens) * kHidden);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t n = 0; n < kHidden; ++n) {
            double routed_sum = 0.0;
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const int column = run.packed_index[token * kTopK + rank];
                routed_sum += static_cast<double>(run.alpha[token * kTopK + rank]) *
                              down_output[static_cast<std::size_t>(column) * kHidden + n];
            }
            const double merged =
                static_cast<double>(
                    bf16_to_f32(run.residual[static_cast<std::size_t>(token) * kHidden + n])) +
                routed_sum +
                static_cast<double>(run.shared_scale[token]) *
                    shared_out[static_cast<std::size_t>(token) * kHidden + n];
            reference.push_back(bf16_to_f32(f32_to_bf16(static_cast<float>(merged))));
            actual.push_back(
                bf16_to_f32(run.destination[static_cast<std::size_t>(token) * kHidden + n]));
        }
    }
    return verify_reduction(std::string(label) + " fused prefill output", actual, reference,
                            kFusedTolerance);
}

// Exercises an expert with more than one 64-column route job, covering the `column_base > 0`
// route-job mapping directly (the fill_routes spread never concentrates that hard).
int check_down_multi_job(fn::Bank& bank) {
    constexpr int M = 128;
    std::vector<int> offsets(kExperts + 1, M);
    offsets[0] = 0;
    const std::vector<int> job_experts{0, 0};
    const std::vector<int> job_columns{0, 64};
    const std::vector<int> job_count{2, 0};
    const std::vector<float> act =
        random_values(static_cast<std::size_t>(M) * 512, 0x888u, -4.0F, 4.0F);
    DeviceBuffer act_dev = to_device_bf16(act);
    const auto device    = cuda_environment().context.device();
    DeviceBuffer codes(device, M * 256);
    DeviceBuffer scales(device, M * 32);
    DeviceBuffer out(device, static_cast<std::size_t>(M) * kHidden * 2);
    DeviceBuffer off_dev = to_device(offsets);
    DeviceBuffer je_dev  = to_device(job_experts);
    DeviceBuffer jc_dev  = to_device(job_columns);
    DeviceBuffer jn_dev  = to_device(job_count);
    const Tensor act_t(act_dev.data(), device, DType::BF16, {512, M});
    ninfer::ops::detail::sparse_moe_nvfp4_down_launch(
        act_t, ninfer::ops::detail::make_nvfp4_down_bases(bank.view()),
        static_cast<const int*>(off_dev.data()),
        static_cast<const int*>(je_dev.data()), static_cast<const int*>(jc_dev.data()),
        static_cast<const int*>(jn_dev.data()), static_cast<std::uint8_t*>(codes.data()),
        static_cast<std::uint8_t*>(scales.data()), static_cast<__nv_bfloat16*>(out.data()),
        device_multiprocessor_count(), nullptr);
    cuda_synchronize();

    const auto plane = decode_down_plane(from_device<std::uint8_t>(codes.data(), M * 256),
                                         from_device<std::uint8_t>(scales.data(), M * 32), M);
    const auto produced = from_device<std::uint16_t>(out.data(),
                                                    static_cast<std::size_t>(M) * kHidden);
    const auto weights  = decode_nvfp4_matrix(bank.down_packed(0));
    std::vector<double> actual;
    std::vector<double> reference;
    actual.reserve(static_cast<std::size_t>(M) * kHidden);
    reference.reserve(static_cast<std::size_t>(M) * kHidden);
    for (int column = 0; column < M; ++column) {
        for (std::int32_t n = 0; n < kHidden; ++n) {
            const float* w = &weights[static_cast<std::size_t>(n) * 512];
            const float* a = &plane[static_cast<std::size_t>(column) * 512];
            double accumulator = 0.0;
            for (std::int32_t k = 0; k < 512; ++k) {
                accumulator += static_cast<double>(a[k]) * static_cast<double>(w[k]);
            }
            reference.push_back(bf16_to_f32(f32_to_bf16(static_cast<float>(accumulator))));
            actual.push_back(
                bf16_to_f32(produced[static_cast<std::size_t>(column) * kHidden + n]));
        }
    }
    return verify_reduction("nvfp4 down multi-job", actual, reference, kFusedTolerance);
}

int run_case(fn::Bank& bank, const SharedExpert& shared, std::int32_t tokens, int column_stride) {
    const std::string label = "nvfp4 fused T=" + std::to_string(tokens);
    const Run run           = run_fused(bank, shared, tokens);
    Oracle oracle(bank, run);
    oracle.shared_down = &shared.down_packed;
    int failures = 0;
    failures += check_down_stage(oracle, run, label.c_str(), column_stride);
    failures += check_fused(oracle, run, label.c_str());
    if (failures == 0) {
        std::cout << label << ": " << run.assignments << " packed columns\n";
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    try {
        fn::Bank bank;
        SharedExpert shared = make_shared(256);
        // T=20 verifies the down stage on every packed column; the larger interior extent samples
        // columns (the fused output is still checked for every token).
        failures += check_down_multi_job(bank);
        // The gate/up host payloads are not part of the oracle; release them after the runs that
        // decode them.
        bank.release_gate_up();
        failures += run_case(bank, shared, 20, 1);
        failures += run_case(bank, shared, 200, 8);
    } catch (const std::exception& error) {
        std::cerr << "nvfp4 fused prefill test: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 fused prefill\n";
    return failures == 0 ? 0 : 1;
}
