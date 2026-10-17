// Correctness for the NVFP4 routed decode / small-T route (ticket #213 lineage, map #175).
// Drives the fork route's A16 SIMT pair through the merged moe_cuda::fork_nvfp4 launch below the
// grouped boundary and compares the result against an independent FP64 oracle. The oracle decodes
// every weight from its packed E2M1/E4M3 representation with the stored per-parent FP32 divisor,
// and reads the production FP32 routed activation, which the T<20 route keeps in FP32 (so it is
// compared directly, unlike the grouped route's private FP4 activation planes). Routes are inputs
// here: the top-k router is qualified by the merged routing suite.
// No CUDA device is optional: the route is device execution.

#include "ops/fork_nvfp4_bank.h"

#include "ops/moe/workspace.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace fn = ninfer::test::fork_nvfp4;

namespace {

constexpr std::int32_t kHidden       = fn::kHidden;
constexpr std::int32_t kExperts      = fn::kExperts;
constexpr std::int32_t kTopK         = fn::kTopK;
constexpr std::int32_t kIntermediate = fn::kIntermediate;

// The routed activation of the T<20 route is FP32 (SwiGLU of FP32 gates), so the only error
// against the FP64 oracle is accumulation rounding; a structural error (wrong codec, scale, or
// divisor) is O(1) relative and cannot hide under this bound.
constexpr ReductionCriterion kActivationTolerance{
    /*relative_l2*/ 1.0e-3,
    /*gross_absolute*/ 1.0e-2,
    /*gross_relative_to_max_reference*/ 1.0e-3,
};
// The routed-only output rounds to BF16 once, so the bound is dominated by that ~2^-8 step.
constexpr ReductionCriterion kFusedTolerance{
    /*relative_l2*/ 2.0e-2,
    /*gross_absolute*/ 6.0e-3,
    /*gross_relative_to_max_reference*/ 0.0,
};

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

struct Route {
    std::vector<float> input;      // BF16-rounded, [tokens, kHidden]
    std::vector<int> ids;          // [tokens * kTopK]
    std::vector<float> alpha;      // [tokens * kTopK]
    std::vector<float> activation; // [tokens * kTopK, kIntermediate] FP32
    std::vector<std::uint16_t> destination;
    std::vector<std::uint16_t> residual;
    std::int32_t tokens = 0;
};

Route run_route(fn::Bank& bank, std::int32_t tokens) {
    std::vector<float> x_values =
        random_values(static_cast<std::size_t>(kHidden) * tokens, 0xabcdefu, -1.0F, 1.0F);
    round_to_bf16(x_values);
    std::vector<float> residual_values =
        random_values(static_cast<std::size_t>(kHidden) * tokens, 0x5555u, -0.5F, 0.5F);
    round_to_bf16(residual_values);
    DeviceBuffer x_storage        = to_device_bf16(x_values);
    DeviceBuffer residual_storage = to_device_bf16(residual_values);
    DeviceBuffer destination_storage(cuda_environment().context.device(),
                                     static_cast<std::size_t>(kHidden) * tokens *
                                         sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(destination_storage.data(), residual_storage.data(),
                          destination_storage.size(), cudaMemcpyDeviceToDevice),
               "seed destination");

    Route route;
    route.tokens = tokens;
    route.input  = x_values;
    fn::fill_routes(route.ids, route.alpha, tokens, tokens <= 8 ? 1 : 3);
    DeviceBuffer ids   = to_device<int>(route.ids);
    DeviceBuffer coefs = to_device<float>(route.alpha);

    const auto device = cuda_environment().context.device();
    const Tensor x(x_storage.data(), device, DType::BF16, {kHidden, tokens});
    Tensor destination(destination_storage.data(), device, DType::BF16, {kHidden, tokens});
    const ops::ExpertRoutes routes{
        Tensor(ids.data(), device, DType::I32, {kTopK, tokens}),
        Tensor(coefs.data(), device, DType::FP32, {kTopK, tokens})};

    const fn::route::Plan plan{.grouped = false, .queued_ctas_per_sm = 32, .shared = {}};
    ninfer::WorkspaceLayoutBuilder layout;
    (void)fn::route::allocate(layout, plan, tokens, true, false);
    GuardedDeviceBuffer scratch(layout.peak_bytes(1));
    MemoryArena arena(scratch.span());
    const auto storage = fn::route::allocate(arena, plan, tokens, true, false);

    const auto execution = cuda_environment().context.execution_view(nullptr);
    fn::route::launch(x, bank.experts(), routes, ops::ExpertSubset{ops::ExpertSubsetKind::All, {}},
                      nullptr, ops::MoeEpilogue::AddResidual, destination, plan, storage, {},
                      execution);
    cuda_synchronize();

    route.destination = from_device<std::uint16_t>(destination_storage,
                                                  static_cast<std::size_t>(kHidden) * tokens);
    route.residual    = bf16_bits(residual_values);
    route.activation  = from_device<float>(storage.vector_activation,
                                          static_cast<std::size_t>(tokens) * kTopK * kIntermediate);
    return route;
}

// silu(gate) * up for every routed path, from the BF16 hidden state and the decoded gate/up
// parents. Row sampling keeps the FP64 reference bounded; the kernel's row work is
// data-independent, and the last row is always included.
int check_activation(const fn::Bank& bank, const Route& route, const char* label, int row_stride) {
    std::vector<double> actual;
    std::vector<double> reference;
    for (std::int32_t token = 0; token < route.tokens; ++token) {
        const float* x = &route.input[static_cast<std::size_t>(token) * kHidden];
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const int expert = route.ids[static_cast<std::size_t>(token) * kTopK + rank];
            const float* activation =
                &route.activation[(static_cast<std::size_t>(token) * kTopK + rank) * kIntermediate];
            const auto check_row = [&](std::int32_t row) {
                double g = 0.0;
                double u = 0.0;
                for (std::int32_t k = 0; k < kHidden; ++k) {
                    const double value = x[k];
                    g += value * bank.gate_weight(expert, row, k);
                    u += value * bank.up_weight(expert, row, k);
                }
                reference.push_back((g / (1.0 + std::exp(-g))) * u);
                actual.push_back(activation[row]);
            };
            for (std::int32_t row = 0; row < kIntermediate; row += row_stride) {
                check_row(row);
            }
            if (row_stride > 1) { check_row(kIntermediate - 1); }
        }
    }
    return verify_reduction(std::string(label) + " routed activation", actual, reference,
                            kActivationTolerance);
}

// The fused output: residual + sum_rank alpha * down(FP32 routed activation), rounded to BF16.
int check_fused(const fn::Bank& bank, const Route& route, const char* label) {
    const std::int32_t tokens = route.tokens;
    std::vector<double> actual;
    std::vector<double> reference;
    actual.reserve(static_cast<std::size_t>(tokens) * kHidden);
    reference.reserve(static_cast<std::size_t>(tokens) * kHidden);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t n = 0; n < kHidden; ++n) {
            double routed = 0.0;
            for (std::int32_t rank = 0; rank < kTopK; ++rank) {
                const int expert = route.ids[static_cast<std::size_t>(token) * kTopK + rank];
                const float* activation =
                    &route.activation[(static_cast<std::size_t>(token) * kTopK + rank) *
                                      kIntermediate];
                double dot = 0.0;
                for (std::int32_t k = 0; k < kIntermediate; ++k) {
                    dot += static_cast<double>(activation[k]) *
                           bank.down_weight(expert, n, k);
                }
                routed += static_cast<double>(
                              route.alpha[static_cast<std::size_t>(token) * kTopK + rank]) *
                          dot;
            }
            const double merged =
                static_cast<double>(bf16_to_f32(route.residual[static_cast<std::size_t>(token) *
                                                                   kHidden +
                                                               n])) +
                routed;
            reference.push_back(bf16_to_f32(f32_to_bf16(static_cast<float>(merged))));
            actual.push_back(
                bf16_to_f32(route.destination[static_cast<std::size_t>(token) * kHidden + n]));
        }
    }
    return verify_reduction(std::string(label) + " fused output", actual, reference,
                            kFusedTolerance);
}

int run_case(fn::Bank& bank, std::int32_t tokens, int row_stride) {
    const std::string label = "nvfp4 a16 T=" + std::to_string(tokens);
    const Route route       = run_route(bank, tokens);
    int failures            = 0;
    failures += check_activation(bank, route, label.c_str(), row_stride);
    failures += check_fused(bank, route, label.c_str());
    if (failures == 0) {
        std::cout << label << ": " << route.tokens << " tokens, " << (tokens * kTopK)
                  << " routed assignments\n";
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
        // T=1 exercises the nine-warp-shaped single-token A16 pass on every routed row; the
        // small-T cases cover the [2,19] interval boundary with row sampling (the fused output
        // is still checked for every token).
        failures += run_case(bank, 1, 1);
        failures += run_case(bank, 2, 16);
        failures += run_case(bank, 8, 16);
        failures += run_case(bank, 19, 16);
    } catch (const std::exception& error) {
        std::cerr << "nvfp4 a16 test: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 decode/small-T\n";
    return failures == 0 ? 0 : 1;
}
