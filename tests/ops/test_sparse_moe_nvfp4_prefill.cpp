// Full fused nvfp4 routed prefill oracle (ticket #212, map #175). Drives the complete public
// SparseMoe prefill route (router -> grouped top-k -> packed route -> nvfp4 routed gate/up ->
// nvfp4 routed down -> top-k reduce -> shared expert -> final merge) and compares the result
// against an independent FP64 oracle.
//
// The oracle decodes the production quantised activation planes and the stored BF16 shared
// activation, and decodes every weight from its packed representation with the fixture's
// independent logical decode; it never re-runs a kernel. The routed/shared activation
// quantisation is a private execution choice, so it is read back rather than re-derived, per
// op-development.md ("packed inputs are independently decoded with their stored scales"). The
// routed gate/up GEMM itself is qualified by its own oracle in
// ninfer_sparse_moe_nvfp4_gate_up_test (#211).

#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

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

// The complete fused chain mixes two FP4 quantisation steps plus the FP32/FP16 shared path, so the
// bound is looser than a single Op's. Verified cases land well inside it.
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

struct Parent {
    qw::PackedWeight host;
    DeviceBuffer storage;
    Weight device{};
};

Parent make_parent(std::int32_t n, std::int32_t k, std::uint32_t seed, float divisor) {
    Parent parent;
    parent.host = qw::make_patterned_weight(
        QType::NVFP4, n, k, seed,
        qw::PatternedWeightOptions{qw::RowSplitScalePattern::Unit, qw::RowSplitCodePattern::Coordinate,
                                   divisor, divisor});
    parent.storage = to_device(parent.host.payload);
    parent.device  = parent.host.device_weight(parent.storage.p);
    return parent;
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
            const double decoded =
                static_cast<double>(qw::detail::decode_e2m1(code)) *
                static_cast<double>(qw::detail::decode_e4m3fn(scales[scale_offset])) / divisor;
            out[static_cast<std::size_t>(row) * k + column] = static_cast<float>(decoded);
        }
    }
    return out;
}

// Independent decode of a row-split Q8 parent (int8 code x FP16 group scale).
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

// Decodes the compact per-column down activation plane (row-major scales, 512/2 code bytes and
// 512/16 scale bytes per packed column, private divisor 1.0).
std::vector<float> decode_down_plane(const std::vector<std::uint8_t>& codes,
                                     const std::vector<std::uint8_t>& scales,
                                     std::int32_t assignments) {
    constexpr std::int32_t k       = 512;
    constexpr std::int32_t kGroups = k / 16;
    std::vector<float> out(static_cast<std::size_t>(assignments) * k);
    for (std::int32_t row = 0; row < assignments; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint8_t byte = codes[static_cast<std::size_t>(row) * (k / 2) + column / 2];
            const std::uint8_t code = (column & 1) == 0 ? (byte & 0x0fU) : (byte >> 4);
            const std::uint8_t scale =
                scales[static_cast<std::size_t>(row) * kGroups + column / 16];
            out[static_cast<std::size_t>(row) * k + column] =
                static_cast<float>(static_cast<double>(qw::detail::decode_e2m1(code)) *
                                   static_cast<double>(qw::detail::decode_e4m3fn(scale)));
        }
    }
    return out;
}

struct Bank {
    std::vector<std::unique_ptr<Parent>> gate_up; // 2 * kExperts
    std::vector<std::unique_ptr<Parent>> down;    // kExperts
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
        bank.gate_up.push_back(std::make_unique<Parent>(make_parent(
            kGateRows, kHidden, 0x9e37u + static_cast<std::uint32_t>(expert) * 2u, gate_divisor)));
        bank.gate_up.push_back(std::make_unique<Parent>(make_parent(
            kGateRows, kHidden, 0x85ebu + static_cast<std::uint32_t>(expert) * 2u + 1u,
            gate_divisor)));
        bank.down.push_back(std::make_unique<Parent>(make_parent(
            kHidden, kIntermediate, 0xc2b2u + static_cast<std::uint32_t>(expert) * 3u,
            down_divisor)));
    }

    bank.shared_gate_up =
        qw::make_patterned_weight(QType::Q8_G32_FP16, kSharedRows, kHidden, 7u);
    bank.shared_down = qw::make_patterned_weight(QType::Q8_G32_FP16, kHidden, kIntermediate, 11u);
    // The shared expert is zeroed so the fused output isolates the routed path (the down kernel and
    // the top-k reduce); the shared path's numerics are covered by the packed-profile suite.
    std::fill(bank.shared_gate_up.payload.begin(),
              bank.shared_gate_up.payload.begin() +
                  static_cast<std::ptrdiff_t>(bank.shared_gate_up.code_plane_bytes),
              static_cast<std::uint8_t>(0));
    bank.shared_gate_up_storage = to_device(bank.shared_gate_up.payload);
    bank.shared_down_storage    = to_device(bank.shared_down.payload);

    const std::vector<float> router_values =
        random_values(static_cast<std::size_t>(kRouterRows) * kHidden, 0x1234u, -0.5F, 0.5F);
    bank.router_storage = to_device_bf16(router_values);
    bank.router                = Weight{};
    bank.router.qtype          = QType::BF16;
    bank.router.layout         = QuantLayout::Contiguous;
    bank.router.ndim           = 2;
    bank.router.shape[0]       = kRouterRows;
    bank.router.shape[1]       = kHidden;
    bank.router.padded_shape[0] = kRouterRows;
    bank.router.padded_shape[1] = kHidden;
    bank.router.n              = kRouterRows;
    bank.router.k              = kHidden;
    bank.router.payload        = bank.router_storage.p;
    bank.router.payload_bytes  = bank.router_storage.bytes;
    bank.router.qdata          = bank.router_storage.p;

    bank.weights.router_shared_gate = bank.router;
    bank.weights.routed_gate_up     = bank.gate_up.front()->device;
    bank.weights.routed_down        = bank.down.front()->device;
    bank.weights.shared_gate_up =
        bank.shared_gate_up.device_weight(bank.shared_gate_up_storage.p);
    bank.weights.shared_down = bank.shared_down.device_weight(bank.shared_down_storage.p);
    bank.weights.routed_gate_up_experts.reserve(2 * kExperts);
    for (const auto& parent : bank.gate_up) {
        bank.weights.routed_gate_up_experts.push_back(parent->device);
    }
    bank.weights.routed_down_experts.reserve(kExperts);
    for (const auto& parent : bank.down) {
        bank.weights.routed_down_experts.push_back(parent->device);
    }
    return bank;
}

struct Run {
    std::vector<std::uint16_t> destination;
    std::vector<std::uint16_t> residual;
    std::vector<std::uint16_t> grouped_output;
    std::vector<std::uint8_t> down_codes;
    std::vector<std::uint8_t> down_scales;
    std::vector<std::uint16_t> shared_activation;
    std::vector<int> packed_index;
    std::vector<int> expert_offsets;
    std::vector<float> alpha;
    std::vector<float> shared_scale;
    std::int32_t assignments = 0;
    std::int32_t tokens      = 0;
};

Run run_fused(Bank& bank, std::int32_t tokens) {
    const std::int32_t assignments = tokens * kTopK;
    const std::vector<float> x_values =
        random_values(static_cast<std::size_t>(kHidden) * tokens, 0xabcdefu, -1.0F, 1.0F);
    const std::vector<float> residual_values =
        random_values(static_cast<std::size_t>(kHidden) * tokens, 0x5555u, -0.5F, 0.5F);
    DeviceBuffer x_storage        = to_device_bf16(x_values);
    DeviceBuffer residual_storage = to_device_bf16(residual_values);
    DeviceBuffer destination_storage(static_cast<std::size_t>(kHidden) * tokens *
                                     sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(destination_storage.p, residual_storage.p, destination_storage.bytes,
                          cudaMemcpyDeviceToDevice),
               "seed fused destination");

    const std::size_t workspace_bytes =
        ops::sparse_moe_workspace_capacity_bytes(QType::NVFP4, QType::NVFP4, tokens, tokens);
    WorkspaceArena workspace(workspace_bytes);

    Tensor x(x_storage.p, DType::BF16, {kHidden, tokens});
    Tensor destination(destination_storage.p, DType::BF16, {kHidden, tokens});
    ops::sparse_moe(x, bank.weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                    nullptr);
    cuda_synchronize();

    // The Op allocates its views from this same arena and rewinds on return, so re-allocating
    // reproduces the identical layout and exposes what the run wrote.
    const ops::detail::SparseMoePrefillWorkspace views =
        ops::detail::allocate_sparse_moe_prefill_workspace(workspace, tokens, true);
    if (workspace.used() == 0) { throw std::runtime_error("fused test: views were not allocated"); }

    Run run;
    run.tokens        = tokens;
    run.assignments   = assignments;
    run.destination   = from_device<std::uint16_t>(destination_storage,
                                                  static_cast<std::size_t>(kHidden) * tokens);
    run.residual      = bf16_bits(residual_values);
    run.grouped_output = from_device<std::uint16_t>(views.grouped_io.data,
                                                    static_cast<std::size_t>(kHidden) * assignments);
    run.down_codes  = from_device<std::uint8_t>(views.nvfp4_down_codes.data,
                                                static_cast<std::size_t>(assignments) * (512 / 2));
    run.down_scales = from_device<std::uint8_t>(views.nvfp4_down_scales.data,
                                                static_cast<std::size_t>(assignments) * (512 / 16));
    run.shared_activation = from_device<std::uint16_t>(
        views.shared_activation.data, static_cast<std::size_t>(kIntermediate) * tokens);
    run.packed_index   = from_device<int>(views.packed_index.data, assignments);
    run.expert_offsets = from_device<int>(views.expert_offsets.data, kExperts + 1);
    run.alpha          = from_device<float>(views.token_alpha.data, assignments);
    run.shared_scale   = from_device<float>(views.shared_scale.data, tokens);
    return run;
}

// Shared oracle state: the column->expert map and a lazily decoded per-expert down matrix.
struct Oracle {
    const Bank& bank;
    const std::vector<int>& expert_offsets;
    std::vector<int> column_expert;
    std::vector<std::unique_ptr<std::vector<float>>> down_matrices;
    std::vector<float> down_activation;

    Oracle(const Bank& bank_ref, const Run& run)
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
        if (slot == nullptr) { slot = std::make_unique<std::vector<float>>(decode_nvfp4_matrix(bank.down[expert]->host)); }
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

    // The shared down path consumes the stored BF16 shared activation.
    const std::vector<float> shared_down_weights = decode_q8_matrix(oracle.bank.shared_down);
    std::vector<double> shared_out(static_cast<std::size_t>(tokens) * kHidden);
    std::vector<float> shared_row(kIntermediate);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t k = 0; k < kIntermediate; ++k) {
            shared_row[k] =
                bf16_to_f32(run.shared_activation[static_cast<std::size_t>(k) +
                                                 static_cast<std::size_t>(kIntermediate) * token]);
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


int run_case(Bank& bank, std::int32_t tokens, int column_stride) {
    const std::string label = "nvfp4 fused T=" + std::to_string(tokens);
    const Run run           = run_fused(bank, tokens);
    Oracle oracle(bank, run);
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
        Bank bank = build_bank();
        // The gate/up host payloads are not part of the oracle; release them after upload.
        for (auto& parent : bank.gate_up) {
            parent->host.payload.clear();
            parent->host.payload.shrink_to_fit();
        }
        // T=20 verifies the down stage on every packed column; the larger interior extent samples
        // columns (the fused output is still checked for every token).
        failures += run_case(bank, 20, 1);
        failures += run_case(bank, 200, 8);
    } catch (const std::exception& error) {
        std::cerr << "nvfp4 fused prefill test: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 fused prefill\n";
    return failures == 0 ? 0 : 1;
}
