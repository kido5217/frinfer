// Correctness for the NVFP4 routed decode + small-T route (ticket #213, map #175). Drives the
// complete public SparseMoe Op on the T<20 nvfp4 route -- T=1 takes the decode D1/D2/D3/D4 kernels,
// T in [2,19] the small-T S1/S2 plus the same D3/D4 kernels -- and compares the result against an
// independent FP64 oracle. The oracle decodes every weight from its packed E2M1/E4M3 representation
// with the stored per-parent FP32 divisor, and reads the production FP32 routed activation, which
// the T<20 route keeps in FP32 (so it is compared directly, unlike the prefill route's private FP4
// activation plane). No CUDA device is optional here: the route is device execution.

#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

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
namespace qw = ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kExperts      = 256;
constexpr std::int32_t kTopK         = 8;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kGateRows     = kIntermediate;
constexpr std::int32_t kRouterRows   = kExperts + 1;
constexpr std::int32_t kSharedRows   = 2 * kIntermediate;

// The routed activation of the T<20 route is FP32 (SwiGLU of FP32 gates), so the only error against
// the FP64 oracle is accumulation rounding; a structural error (wrong codec, scale, or divisor)
// is O(1) relative and cannot hide under this bound.
constexpr ReductionCriterion kActivationTolerance{
    /*relative_l2*/ 1.0e-3,
    /*gross_absolute*/ 1.0e-2,
    /*gross_relative_to_max_reference*/ 1.0e-3,
};
// The fused output rounds to BF16 once, so the bound is dominated by that ~2^-8 step.
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

// block_scale_k16_m128x4 scale-plane offset within a parent's scale plane.
std::int64_t scale_plane_offset(std::int32_t row, std::int32_t group, std::int32_t k) {
    return (static_cast<std::int64_t>(row / 128) * (k / 64) + group / 4) * 512 +
           (row & 31) * 16 + ((row & 127) >> 5) * 4 + (group & 3);
}

// Host copy of one encoded nvfp4 parent plus its device twin. The host payload is kept for the
// oracle's independent element decode; only the device twin is bound into the Op.
struct Parent {
    qw::PackedWeight host;
    DeviceBuffer storage;
    Weight device{};

    double weight(std::int32_t row, std::int32_t column) const {
        const std::int32_t k       = host.weight.k;
        const std::uint8_t* codes  = host.payload.data();
        const std::uint8_t* scales = host.payload.data() + host.scale_plane_offset;
        const std::uint8_t byte    = codes[static_cast<std::size_t>(row) * (k / 2) + column / 2];
        const std::uint8_t code    = (column & 1) == 0 ? (byte & 0x0fU) : (byte >> 4);
        const std::uint8_t scale   = scales[scale_plane_offset(row, column / 16, k)];
        return qw::detail::decode_e2m1(code) * qw::detail::decode_e4m3fn(scale) /
               static_cast<double>(host.weight.weight_scale_divisor);
    }
};

Parent make_parent(std::int32_t n, std::int32_t k, std::uint32_t seed, float divisor) {
    Parent parent;
    parent.host = qw::make_patterned_weight(
        QType::NVFP4, n, k, seed,
        qw::PatternedWeightOptions{qw::RowSplitScalePattern::Unit,
                                   qw::RowSplitCodePattern::Coordinate, divisor, divisor});
    parent.storage = to_device(parent.host.payload);
    parent.device  = parent.host.device_weight(parent.storage.p);
    return parent;
}

struct Bank {
    std::vector<Parent> gate_up; // 2 * kExperts
    std::vector<Parent> down;    // kExperts
    qw::PackedWeight shared_gate_up;
    qw::PackedWeight shared_down;
    DeviceBuffer shared_gate_up_storage;
    DeviceBuffer shared_down_storage;
    DeviceBuffer router_storage;
    Weight router{};
    ops::SparseMoeWeights weights{};
};

Bank build_bank() {
    Bank bank;
    bank.gate_up.reserve(2 * kExperts);
    bank.down.reserve(kExperts);
    for (std::int32_t expert = 0; expert < kExperts; ++expert) {
        // gate/up share one per-expert divisor (the source's rule); down carries its own.
        const float gate_divisor = 2.0F + 0.25F * static_cast<float>(expert);
        const float down_divisor = 3.0F + 0.5F * static_cast<float>(expert);
        bank.gate_up.push_back(make_parent(
            kGateRows, kHidden, 0x9e37u + static_cast<std::uint32_t>(expert) * 2u, gate_divisor));
        bank.gate_up.push_back(make_parent(
            kGateRows, kHidden, 0x85ebu + static_cast<std::uint32_t>(expert) * 2u + 1u, gate_divisor));
        bank.down.push_back(make_parent(
            kHidden, kIntermediate, 0xc2b2u + static_cast<std::uint32_t>(expert) * 3u, down_divisor));
    }

    bank.shared_gate_up =
        qw::make_patterned_weight(QType::Q8_G32_FP16, kSharedRows, kHidden, 7u);
    bank.shared_down = qw::make_patterned_weight(QType::Q8_G32_FP16, kHidden, kIntermediate, 11u);
    // The shared expert is zeroed so the fused output isolates the routed path; the shared path's
    // numerics are covered by the packed-profile suite.
    std::fill(bank.shared_gate_up.payload.begin(),
              bank.shared_gate_up.payload.begin() +
                  static_cast<std::ptrdiff_t>(bank.shared_gate_up.code_plane_bytes),
              static_cast<std::uint8_t>(0));
    std::fill(bank.shared_down.payload.begin(),
              bank.shared_down.payload.begin() +
                  static_cast<std::ptrdiff_t>(bank.shared_down.code_plane_bytes),
              static_cast<std::uint8_t>(0));
    bank.shared_gate_up_storage = to_device(bank.shared_gate_up.payload);
    bank.shared_down_storage    = to_device(bank.shared_down.payload);

    const std::vector<float> router_values =
        random_values(static_cast<std::size_t>(kRouterRows) * kHidden, 0x1234u, -0.5F, 0.5F);
    bank.router_storage         = to_device_bf16(router_values);
    bank.router                 = Weight{};
    bank.router.qtype           = QType::BF16;
    bank.router.layout          = QuantLayout::Contiguous;
    bank.router.ndim            = 2;
    bank.router.shape[0]        = kRouterRows;
    bank.router.shape[1]        = kHidden;
    bank.router.padded_shape[0] = kRouterRows;
    bank.router.padded_shape[1] = kHidden;
    bank.router.n               = kRouterRows;
    bank.router.k               = kHidden;
    bank.router.payload         = bank.router_storage.p;
    bank.router.payload_bytes   = bank.router_storage.bytes;
    bank.router.qdata           = bank.router_storage.p;

    bank.weights.router_shared_gate = bank.router;
    bank.weights.routed_gate_up     = bank.gate_up.front().device;
    bank.weights.routed_down        = bank.down.front().device;
    bank.weights.shared_gate_up =
        bank.shared_gate_up.device_weight(bank.shared_gate_up_storage.p);
    bank.weights.shared_down = bank.shared_down.device_weight(bank.shared_down_storage.p);
    bank.weights.routed_gate_up_experts.reserve(2 * kExperts);
    for (const auto& parent : bank.gate_up) {
        bank.weights.routed_gate_up_experts.push_back(parent.device);
    }
    bank.weights.routed_down_experts.reserve(kExperts);
    for (const auto& parent : bank.down) {
        bank.weights.routed_down_experts.push_back(parent.device);
    }
    return bank;
}

struct Route {
    std::vector<float> input;         // BF16-rounded, [tokens, kHidden]
    std::vector<int> ids;             // [tokens * kTopK]
    std::vector<float> alpha;         // [tokens * kTopK]
    std::vector<float> activation;    // [tokens * (kTopK + 1), kIntermediate] FP32
    std::vector<std::uint16_t> destination;
    std::vector<std::uint16_t> residual;
    std::int32_t tokens = 0;
};

Route run_route(Bank& bank, std::int32_t tokens) {
    std::vector<float> x_values = random_values(
        static_cast<std::size_t>(kHidden) * tokens, 0xabcdefu, -1.0F, 1.0F);
    round_to_bf16(x_values);
    std::vector<float> residual_values = random_values(
        static_cast<std::size_t>(kHidden) * tokens, 0x5555u, -0.5F, 0.5F);
    round_to_bf16(residual_values);
    DeviceBuffer x_storage        = to_device_bf16(x_values);
    DeviceBuffer residual_storage = to_device_bf16(residual_values);
    DeviceBuffer destination_storage(static_cast<std::size_t>(kHidden) * tokens *
                                     sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(destination_storage.p, residual_storage.p, destination_storage.bytes,
                          cudaMemcpyDeviceToDevice),
               "seed destination");

    const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(
        QType::NVFP4, QType::NVFP4, tokens, tokens);
    WorkspaceArena workspace(workspace_bytes);

    Tensor x(x_storage.p, DType::BF16, {kHidden, tokens});
    Tensor destination(destination_storage.p, DType::BF16, {kHidden, tokens});
    ops::sparse_moe(x, bank.weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                    nullptr);
    cuda_synchronize();

    // The Op rewinds its arena on return, so re-allocating reproduces the identical layout and
    // exposes what the run wrote.
    Route route;
    route.tokens      = tokens;
    route.input       = x_values;
    route.destination = from_device<std::uint16_t>(destination_storage,
                                                  static_cast<std::size_t>(kHidden) * tokens);
    route.residual    = bf16_bits(residual_values);
    if (tokens == 1) {
        const ops::detail::SparseMoeDecodeWorkspace views =
            ops::detail::allocate_sparse_moe_decode_workspace(workspace);
        route.ids          = from_device<int>(views.ids.data, kTopK);
        route.alpha        = from_device<float>(views.alpha.data, kTopK);
        route.activation   = from_device<float>(views.scratch.data,
                                                static_cast<std::size_t>(kTopK + 1) * kIntermediate);
    } else {
        const ops::detail::SparseMoeSmallTWorkspace views =
            ops::detail::allocate_sparse_moe_small_t_workspace(workspace, tokens);
        route.ids        = from_device<int>(views.token_ids.data,
                                            static_cast<std::size_t>(tokens) * kTopK);
        route.alpha      = from_device<float>(views.token_alpha.data,
                                              static_cast<std::size_t>(tokens) * kTopK);
        route.activation = from_device<float>(
            views.scratch.data,
            static_cast<std::size_t>(tokens) * (kTopK + 1) * kIntermediate);
    }
    return route;
}

// The D3 stage: silu(gate) * up for every routed path, from the BF16 hidden state and the decoded
// gate/up parents. Row sampling keeps the FP64 reference bounded; the kernel's row work is
// data-independent, and the last row is always included.
int check_activation(const Bank& bank, const Route& route, const char* label, int row_stride) {
    std::vector<double> actual;
    std::vector<double> reference;
    for (std::int32_t token = 0; token < route.tokens; ++token) {
        const float* x = &route.input[static_cast<std::size_t>(token) * kHidden];
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const int expert = route.ids[static_cast<std::size_t>(token) * kTopK + rank];
            const Parent& gate = bank.gate_up[static_cast<std::size_t>(2 * expert)];
            const Parent& up   = bank.gate_up[static_cast<std::size_t>(2 * expert + 1)];
            const float* activation =
                &route.activation[(static_cast<std::size_t>(token) * (kTopK + 1) + rank) *
                                  kIntermediate];
            for (std::int32_t row = 0; row < kIntermediate; row += row_stride) {
                double g = 0.0;
                double u = 0.0;
                for (std::int32_t k = 0; k < kHidden; ++k) {
                    const double value = x[k];
                    g += value * gate.weight(row, k);
                    u += value * up.weight(row, k);
                }
                reference.push_back((g / (1.0 + std::exp(-g))) * u);
                actual.push_back(activation[row]);
            }
            if (row_stride > 1) {
                const std::int32_t row = kIntermediate - 1;
                double g = 0.0;
                double u = 0.0;
                for (std::int32_t k = 0; k < kHidden; ++k) {
                    const double value = x[k];
                    g += value * gate.weight(row, k);
                    u += value * up.weight(row, k);
                }
                reference.push_back((g / (1.0 + std::exp(-g))) * u);
                actual.push_back(activation[row]);
            }
        }
    }
    return verify_reduction(std::string(label) + " routed activation", actual, reference,
                            kActivationTolerance);
}

// The fused output: residual + sum_rank alpha * down(FP32 routed activation), rounded to BF16.
int check_fused(const Bank& bank, const Route& route, const char* label) {
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
                    &route.activation[(static_cast<std::size_t>(token) * (kTopK + 1) + rank) *
                                      kIntermediate];
                const Parent& down = bank.down[static_cast<std::size_t>(expert)];
                double dot = 0.0;
                for (std::int32_t k = 0; k < kIntermediate; ++k) {
                    dot += static_cast<double>(activation[k]) * down.weight(n, k);
                }
                routed += static_cast<double>(route.alpha[static_cast<std::size_t>(token) * kTopK +
                                                          rank]) *
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

int run_case(Bank& bank, std::int32_t tokens, int row_stride) {
    const std::string label = "nvfp4 decode T=" + std::to_string(tokens);
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
        Bank bank = build_bank();
        // T=1 exercises the decode nine-warp D3/D4 kernels on every routed row; the small-T cases
        // cover the [2,19] interval boundary with row sampling (the fused output is still checked
        // for every token).
        failures += run_case(bank, 1, 1);
        failures += run_case(bank, 2, 16);
        failures += run_case(bank, 8, 16);
        failures += run_case(bank, 19, 16);
    } catch (const std::exception& error) {
        std::cerr << "nvfp4 decode test: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 decode/small-T\n";
    return failures == 0 ? 0 : 1;
}
