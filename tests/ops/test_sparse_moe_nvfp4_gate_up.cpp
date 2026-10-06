// Correctness for the NVFP4 routed gate/up prefill stage (ticket #211, map #175). The production
// launch is qualified against an independent FP64 oracle: the packed weight parents and the
// production-quantised activation plane are decoded with their stored E2M1/E4M3 scales, and the
// logical SwiGLU formula is evaluated naively. No CUDA device is optional here: the stage is a
// device kernel.

#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/sparse_moe/nvfp4/sparse_moe_nvfp4_gate_up.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kExperts      = 256;
constexpr std::int32_t kTopK         = 8;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kGateRows     = kIntermediate;

constexpr std::size_t kCodeBytes   = std::size_t(kGateRows) * kHidden / 2;   // 524288
constexpr std::size_t kScaleOffset = kCodeBytes;                             // 256-aligned
constexpr std::size_t kScaleBytes  = std::size_t(kGateRows) * (kHidden / 16); // 65536
constexpr std::size_t kParentBytes = kScaleOffset + kScaleBytes + 4;         // 589828

// The oracle accepts the production route's arithmetic; the bound is dominated by the BF16
// destination rounding (~2^-8) of a prefill-scale routed activation.
constexpr ReductionCriterion kNvfp4GateUpTolerance{
    /*relative_l2*/ 2.0e-2,
    /*gross_absolute*/ 1.0e-2,
    /*gross_relative_to_max_reference*/ 1.0e-2,
};

double e2m1_decode(std::uint8_t code) {
    constexpr double kTable[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double value         = kTable[code & 7u];
    return (code & 8u) != 0 ? -value : value;
}

double e4m3_decode(std::uint8_t byte) {
    const int sign = byte >> 7;
    const int exp  = (byte >> 3) & 0xF;
    const int mant = byte & 0x7;
    double value   = 0.0;
    if (exp == 0) {
        value = std::ldexp(static_cast<double>(mant), -9);
    } else {
        value = std::ldexp(1.0 + static_cast<double>(mant) / 8.0, exp - 7);
    }
    return sign != 0 ? -value : value;
}

// block_scale_k16_m128x4 scale-plane offset within a parent's scale plane.
std::int64_t scale_plane_offset(std::int32_t row, std::int32_t group) {
    return (static_cast<std::int64_t>(row / 128) * (kHidden / 64) + group / 4) * 512 +
           (row & 31) * 16 + ((row & 127) >> 5) * 4 + (group & 3);
}

std::uint32_t lcg(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

struct PackedParent {
    std::vector<std::uint8_t> bytes;
    float divisor = 1.0f;

    std::vector<std::uint8_t> codes() const { return {bytes.begin(), bytes.begin() + kCodeBytes}; }
    std::vector<std::uint8_t> scales() const {
        return {bytes.begin() + kScaleOffset, bytes.begin() + kScaleOffset + kScaleBytes};
    }
};

// A syntactically valid nvfp4 parent: arbitrary E2M1 codes, finite E4M3 scales, one FP32 divisor.
PackedParent make_parent(std::uint32_t seed, float divisor) {
    PackedParent parent;
    parent.bytes.assign(kParentBytes, 0);
    parent.divisor = divisor;
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < kCodeBytes; ++i) {
        parent.bytes[i] = static_cast<std::uint8_t>(lcg(state) & 0xFFFF);
    }
    for (std::int32_t row = 0; row < kGateRows; ++row) {
        for (std::int32_t group = 0; group < kHidden / 16; ++group) {
            const int exp  = 6 + static_cast<int>(lcg(state) % 5); // 6..10
            const int mant = static_cast<int>(lcg(state) % 8);
            // Block scales are unsigned E4M3 (the MMA's scale type); NVFP4 derives them from an
            // absmax, so the sign bit is clear. A random sign bit would be read as part of an
            // unsigned value and is not a legal NVFP4 scale.
            const std::uint8_t byte = static_cast<std::uint8_t>((exp << 3) | mant);
            parent.bytes[kScaleOffset + scale_plane_offset(row, group)] = byte;
        }
    }
    const std::uint32_t word = [] (float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }(divisor);
    std::memcpy(&parent.bytes[kScaleOffset + kScaleBytes], &word, sizeof(word));
    return parent;
}

// Decoded weight element W[row, k] of a parent already resident in host bytes.
double parent_weight(const PackedParent& parent, std::int32_t row, std::int32_t k) {
    const std::uint8_t code = parent.bytes[std::size_t(row) * (kHidden / 2) + k / 2];
    const std::uint8_t nibble = (k & 1) != 0 ? (code >> 4) : (code & 0xF);
    const std::uint8_t scale = parent.bytes[kScaleOffset + scale_plane_offset(row, k / 16)];
    return e2m1_decode(nibble) * e4m3_decode(scale) / parent.divisor;
}

double activation_value(const std::vector<std::uint8_t>& codes,
                        const std::vector<std::uint8_t>& scales, std::int32_t token,
                        std::int32_t k) {
    const std::uint8_t code = codes[std::size_t(token) * (kHidden / 2) + k / 2];
    const std::uint8_t nibble = (k & 1) != 0 ? (code >> 4) : (code & 0xF);
    const std::uint8_t scale = scales[std::size_t(token) * (kHidden / 16) + k / 16];
    // The kernel's activation divisor is 1.0, so the E2M1 * E4M3 product is the represented value.
    return e2m1_decode(nibble) * e4m3_decode(scale);
}

struct Route {
    std::vector<int> packed_token;
    std::vector<int> expert_offsets;
    std::vector<int> job_experts;
    std::vector<int> job_columns;
    int assignments = 0;
};

// A grouped packed route in the prefill scan's layout: each expert's columns are contiguous, each
// route job is one 64-column tile of one expert.
Route build_route(std::int32_t tokens, std::int32_t experts_used) {
    Route route;
    std::vector<std::vector<int>> tokens_by_expert(kExperts);
    std::vector<int> assignment_expert(kTopK * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const int expert = (token * 5 + rank * 29 + 7) % experts_used;
            assignment_expert[token * kTopK + rank] = expert;
            tokens_by_expert[expert].push_back(token);
        }
    }
    route.assignments = tokens * kTopK;
    route.packed_token.resize(route.assignments);
    route.expert_offsets.assign(kExperts + 1, 0);
    for (int expert = 0; expert < kExperts; ++expert) {
        route.expert_offsets[expert + 1] =
            route.expert_offsets[expert] + static_cast<int>(tokens_by_expert[expert].size());
        for (int index = 0; index < static_cast<int>(tokens_by_expert[expert].size()); ++index) {
            route.packed_token[route.expert_offsets[expert] + index] = tokens_by_expert[expert][index];
        }
        const int count = static_cast<int>(tokens_by_expert[expert].size());
        for (int column = 0; column < count; column += 64) {
            route.job_experts.push_back(expert);
            route.job_columns.push_back(column);
        }
    }
    return route;
}

struct ExpertHost {
    PackedParent gate;
    PackedParent up;
};

int run_case(const char* label, std::int32_t tokens, std::int32_t experts_used,
             std::int32_t column_stride) {
    const Route route = build_route(tokens, experts_used);

    // One parent per used expert projection; unused experts keep a null table entry.
    std::vector<ExpertHost> host(kExperts);
    std::vector<DeviceBuffer> device_parents;
    std::vector<Weight> gate_weights(kExperts);
    std::vector<Weight> up_weights(kExperts);
    for (int expert = 0; expert < experts_used; ++expert) {
        host[expert].gate = make_parent(0x9E3779B9u ^ (expert * 2u), 2.0f + 0.25f * expert);
        host[expert].up   = make_parent(0x85EBCA6Bu ^ (expert * 2u + 1u), 3.0f + 0.5f * expert);
        for (int projection = 0; projection < 2; ++projection) {
            PackedParent& parent = projection == 0 ? host[expert].gate : host[expert].up;
            DeviceBuffer device  = to_device(parent.bytes);
            Weight weight;
            weight.payload_bytes    = kParentBytes;
            weight.qtype            = QType::NVFP4;
            weight.layout           = QuantLayout::BlockScaleK16M128x4;
            weight.group_size       = 16;
            weight.group            = 16;
            weight.n                = kGateRows;
            weight.k                = kHidden;
            weight.ndim             = 2;
            weight.shape[0]         = kGateRows;
            weight.shape[1]         = kHidden;
            weight.padded_shape[0]  = kGateRows;
            weight.padded_shape[1]  = kHidden;
            weight.scale_dtype      = DType::FP8_E4M3FN;
            weight.weight_scale_divisor = parent.divisor;
            weight.input_scale_divisor  = 1.0f;
            weight.payload          = device.p;
            weight.qdata            = device.p;
            weight.scales           = static_cast<const std::uint8_t*>(device.p) + kScaleOffset;
            if (projection == 0) {
                gate_weights[expert] = weight;
            } else {
                up_weights[expert] = weight;
            }
            device_parents.push_back(std::move(device));
        }
    }

    std::vector<Weight> routed_gate_up(2 * kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        routed_gate_up[2 * expert]     = gate_weights[expert];
        routed_gate_up[2 * expert + 1] = up_weights[expert];
    }
    ops::SparseMoeWeights weights{};
    weights.routed_gate_up_experts = std::move(routed_gate_up);
    weights.routed_gate_up.qtype   = QType::NVFP4;
    weights.routed_down.qtype      = QType::NVFP4;

    std::vector<float> input(static_cast<std::size_t>(tokens) * kHidden);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.75f * std::sin(0.013f * static_cast<float>(i)) +
                   0.25f * std::cos(0.041f * static_cast<float>(i));
    }
    DeviceBuffer device_x = to_device_bf16(input);

    const int jobs = static_cast<int>(route.job_experts.size());
    DeviceBuffer device_token    = to_device<int>(route.packed_token);
    DeviceBuffer device_offsets  = to_device<int>(route.expert_offsets);
    DeviceBuffer device_jobs     = to_device<int>(route.job_experts);
    DeviceBuffer device_columns  = to_device<int>(route.job_columns);
    DeviceBuffer device_jobcount = to_device<int>(std::vector<int>{jobs, experts_used});
    DeviceBuffer device_codes(std::size_t(tokens) * (kHidden / 2));
    DeviceBuffer device_scales(std::size_t(tokens) * (kHidden / 16));
    DeviceBuffer device_output(std::size_t(kIntermediate) * route.assignments * sizeof(std::uint16_t));

    Tensor x(device_x.p, DType::BF16, {kHidden, tokens});
    ops::detail::sparse_moe_nvfp4_gate_up_launch(
        x, weights, static_cast<const int*>(device_token.p),
        static_cast<const int*>(device_offsets.p), static_cast<const int*>(device_jobs.p),
        static_cast<const int*>(device_columns.p), static_cast<const int*>(device_jobcount.p),
        static_cast<std::uint8_t*>(device_codes.p),
        static_cast<std::uint8_t*>(device_scales.p),
        static_cast<__nv_bfloat16*>(device_output.p), nullptr);
    cuda_synchronize();

    const std::vector<std::uint8_t> codes =
        from_device<std::uint8_t>(device_codes, device_codes.bytes);
    const std::vector<std::uint8_t> scales =
        from_device<std::uint8_t>(device_scales, device_scales.bytes);
    const std::vector<double> got =
        from_device_bf16(device_output, std::size_t(kIntermediate) * route.assignments);

    // The oracle is O(columns * rows * K) in FP64; verify every column at T=20 and a subset that
    // still crosses every expert tile boundary at the larger extent.
    std::vector<char> verify(route.assignments, column_stride <= 1 ? 1 : 0);
    if (column_stride > 1) {
        for (int column = 0; column < route.assignments; column += column_stride) {
            verify[column] = 1;
        }
        for (int expert = 0; expert < kExperts; ++expert) {
            if (route.expert_offsets[expert + 1] > route.expert_offsets[expert]) {
                verify[route.expert_offsets[expert + 1] - 1] = 1;
            }
        }
    }

    std::vector<double> reference;
    reference.reserve(got.size());
    std::vector<double> actual;
    actual.reserve(got.size());
    for (int expert = 0; expert < kExperts; ++expert) {
        const int begin = route.expert_offsets[expert];
        const int end   = route.expert_offsets[expert + 1];
        for (int column = begin; column < end; ++column) {
            if (!verify[column]) { continue; }
            const int token = route.packed_token[column];
            std::vector<double> activation(kHidden);
            for (std::int32_t k = 0; k < kHidden; ++k) {
                activation[k] = activation_value(codes, scales, token, k);
            }
            for (std::int32_t row = 0; row < kIntermediate; ++row) {
                double gate = 0.0;
                double up   = 0.0;
                for (std::int32_t k = 0; k < kHidden; ++k) {
                    const double a = activation[k];
                    gate += a * parent_weight(host[expert].gate, row, k);
                    up += a * parent_weight(host[expert].up, row, k);
                }
                const double silu = gate / (1.0 + std::exp(-gate));
                reference.push_back(silu * up);
                actual.push_back(got[std::size_t(column) * kIntermediate + row]);
            }
        }
    }

    int failures =
        verify_reduction(std::string(label) + " routed gate/up", actual, reference,
                         kNvfp4GateUpTolerance);
    if (failures == 0) {
        std::cout << label << ": " << route.assignments << " packed columns, " << jobs
                  << " route jobs\n";
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
    // T=20 concentrates every assignment on one expert, so the three route jobs cover every
    // 64-column A-tile position (two full tiles and a partial tail) and the oracle checks all of
    // them. T=200 spreads overlap across 128 experts with a sampled oracle to keep the FP64
    // reference bounded while still crossing every expert and partial-tile boundary.
    failures += run_case("nvfp4 gate/up T=20", 20, 1, 1);
    failures += run_case("nvfp4 gate/up T=200", 200, 128, 8);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 gate/up correctness\n";
    return failures == 0 ? 0 : 1;
}
