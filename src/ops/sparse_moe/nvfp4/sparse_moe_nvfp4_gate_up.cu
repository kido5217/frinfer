// NVFP4 routed gate/up prefill stage (ticket #211, map #175). The routed gate/up GEMM runs on the
// Blackwell W4A4 tensor-core instruction `mma_nvfp4_e4m3` over the packed route, decoding one
// complete per-expert nvfp4 parent per expert projection (#206). The routed activation x is
// quantised once per token into a compact FP4 plane and shared by every expert a token routes to;
// the activation divisor is one private execution choice, not the source's per-expert
// activation_input_divisor (activation quantisation is a private execution choice per the Op
// contract, and one shared divisor lets one quantised copy serve all eight experts).

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/sparse_moe/nvfp4/sparse_moe_nvfp4_gate_up.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kNvfp4Hidden       = 2048;
constexpr int kNvfp4Intermediate = 512;
// One CTA tile: 64 packed columns and 64 weight rows. The 64 rows interleave gate and up so that a
// processed pair (logical intermediate row r) is row 2r (gate) and 2r+1 (up); the nvfp4 MMA keeps
// adjacent b-rows in one lane's accumulator pair, so the SwiGLU combine stays warp-local.
constexpr int kNvfp4BlockColumns = 64;
constexpr int kNvfp4BlockRows    = 64;
constexpr int kNvfp4LogicalRows  = kNvfp4BlockRows / 2;
constexpr int kNvfp4RowBlocks    = kNvfp4Intermediate / kNvfp4LogicalRows; // 16
constexpr int kNvfp4BlockK       = 256;
constexpr int kNvfp4KTiles       = kNvfp4Hidden / kNvfp4BlockK; // 8
// Activation quantisation divisor. E2M1 codes are normalised by the per-group E4M3 scale, so the
// divisor cancels in reconstruction; 1.0 keeps the E4M3 scale inside its normal range for the
// post-norm hidden state.
constexpr float kNvfp4ActivationDivisor = 1.0f;
// Upper bound on the persistent grid (matches the packed prefill cap: 32 * 170 SMs).
constexpr int kNvfp4MaxBlocks = 32 * 170;

using Nvfp4GateUpSchedule =
    Nvfp4A4MmaSchedule<kNvfp4BlockColumns, kNvfp4BlockRows, kNvfp4BlockK, 2, 2, 2, 2>;

template <class Schedule>
__device__ __forceinline__ void stage_nvfp4_gate_up_activation(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const int* __restrict__ packed_token, int packed_base, int cols,
    Nvfp4A4SharedStorage<Schedule>& shared, int stage, int k_tile) {
    constexpr int k              = kNvfp4Hidden;
    constexpr int kSegmentsPerRow = Schedule::kSegmentsPerRow;
    constexpr int kCodeTasks     = Schedule::kBlockTokens * kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row         = task / kSegmentsPerRow;
        const int segment     = task - row * kSegmentsPerRow;
        const bool valid      = row < cols;
        const int token       = valid ? packed_token[packed_base + row] : 0;
        const int physical    = nvfp4_a4_swizzled_byte<Schedule>(row, segment * 16);
        auto* destination     = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical;
        const auto* source    = codes + static_cast<std::int64_t>(token) * (k / 2) +
                                k_tile * Schedule::kCodeRowBytes + segment * 16;
        cp_async_zfill<16, Schedule::kActivationCache>(destination, source, valid ? 16 : 0);
    }

    constexpr int kScaleBytes = Schedule::kK64PerStage * 4;
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockTokens;
         row += Schedule::kThreads) {
        const bool valid  = row < cols;
        const int token   = valid ? packed_token[packed_base + row] : 0;
        auto* destination = &shared.a_scale4[stage][row * Schedule::kK64PerStage];
        const auto* source = scales + (static_cast<std::int64_t>(token) * (k / 64) +
                                       k_tile * Schedule::kK64PerStage) * 4;
        cp_async_zfill<kScaleBytes>(destination, source, valid ? kScaleBytes : 0);
    }
}

template <class Schedule>
__device__ __forceinline__ void stage_nvfp4_gate_up_weight(
    const std::uint8_t* __restrict__ gate_codes, const std::uint8_t* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_codes, const std::uint8_t* __restrict__ up_scales,
    int logical_begin, Nvfp4A4SharedStorage<Schedule>& shared, int stage, int k_tile) {
    constexpr int k               = kNvfp4Hidden;
    constexpr int kSegmentsPerRow = Schedule::kSegmentsPerRow;
    constexpr int kCodeTasks      = Schedule::kBlockRows * kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row         = task / kSegmentsPerRow;
        const int segment     = task - row * kSegmentsPerRow;
        const bool is_up      = (row & 1) != 0;
        const int weight_row  = logical_begin + (row >> 1);
        const std::uint8_t* codes = is_up ? up_codes : gate_codes;
        const int physical    = nvfp4_a4_swizzled_byte<Schedule>(row, segment * 16);
        auto* destination     = shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical;
        const auto* source    = codes + static_cast<std::int64_t>(weight_row) * (k / 2) +
                                k_tile * Schedule::kCodeRowBytes + segment * 16;
        cp_async<16, Schedule::kWeightCache>(destination, source);
    }

    for (int task = static_cast<int>(threadIdx.x);
         task < Schedule::kBlockRows * Schedule::kK64PerStage; task += Schedule::kThreads) {
        const int row        = task / Schedule::kK64PerStage;
        const int local_k64  = task - row * Schedule::kK64PerStage;
        const bool is_up     = (row & 1) != 0;
        const int weight_row = logical_begin + (row >> 1);
        const std::uint8_t* scales = is_up ? up_scales : gate_scales;
        const int global_k64 = k_tile * Schedule::kK64PerStage + local_k64;
        auto* destination =
            shared.b_scales[stage] + (row * Schedule::kK64PerStage + local_k64) * 4;
        const auto* source = scales + nvfp4_scale_byte_offset(weight_row, global_k64 * 4, k);
        cp_async<4>(destination, source);
    }
}

template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void sparse_moe_nvfp4_gate_up_kernel(const std::uint8_t* __restrict__ x_codes,
                                     const std::uint8_t* __restrict__ x_scales,
                                     const int* __restrict__ packed_token,
                                     const int* __restrict__ expert_offsets,
                                     const int* __restrict__ route_job_experts,
                                     const int* __restrict__ route_job_columns,
                                     const int* __restrict__ route_job_count,
                                     __nv_bfloat16* __restrict__ activation, Nvfp4GateUpBases bases,
                                     float activation_divisor) {
    constexpr int kWaitGroups = Schedule::kStages - 1;
    auto* storage             = nvfp4_shared_storage<Schedule::kSharedBytes>();
    auto& shared              = *reinterpret_cast<Nvfp4A4SharedStorage<Schedule>*>(storage);

    const int jobs       = route_job_count[0];
    const int row_blocks = kNvfp4RowBlocks;
    const int total      = jobs * row_blocks;

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / Schedule::kWarpsRows;
    const int warp_n = warp - warp_m * Schedule::kWarpsRows;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;
    const int ar            = lane >> 2;

    for (int work = static_cast<int>(blockIdx.x); work < total;
         work += static_cast<int>(gridDim.x)) {
        const int job         = work / row_blocks;
        const int row_block   = work - job * row_blocks;
        const int expert      = route_job_experts[job];
        const int column_base = route_job_columns[job];
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int cols        = min(kNvfp4BlockColumns, count - column_base);
        const int packed_base = begin + column_base;
        const int logical_begin = row_block * kNvfp4LogicalRows;

        const auto* gate_codes  = static_cast<const std::uint8_t*>(bases.gate_codes[expert]);
        const auto* gate_scales = static_cast<const std::uint8_t*>(bases.gate_scales[expert]);
        const auto* up_codes    = static_cast<const std::uint8_t*>(bases.up_codes[expert]);
        const auto* up_scales   = static_cast<const std::uint8_t*>(bases.up_scales[expert]);
        const float alpha_gate  = 1.0f / (activation_divisor * bases.gate_divisor[expert]);
        const float alpha_up    = 1.0f / (activation_divisor * bases.up_divisor[expert]);

#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            if (stage < kNvfp4KTiles) {
                stage_nvfp4_gate_up_activation<Schedule>(x_codes, x_scales, packed_token,
                                                         packed_base, cols, shared, stage, stage);
                stage_nvfp4_gate_up_weight<Schedule>(gate_codes, gate_scales, up_codes, up_scales,
                                                     logical_begin, shared, stage, stage);
                cp_commit();
            }
        }

        float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};

        for (int k_tile = 0; k_tile < kNvfp4KTiles; ++k_tile) {
            const int stage = k_tile % Schedule::kStages;
            cp_wait<kWaitGroups>();
            __syncthreads();

#pragma unroll
            for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
                unsigned a_fragments[Schedule::kMmaTokens][4];
                unsigned b_fragments[Schedule::kMmaRows][2];
                unsigned a_scales[Schedule::kMmaTokens];
                unsigned b_scales[Schedule::kMmaRows];

#pragma unroll
                for (int mma_m = 0; mma_m < Schedule::kMmaTokens; ++mma_m) {
                    const int row = warp_m * Schedule::kWarpTokens + mma_m * 16 + a_row_offset;
                    const int logical_byte  = local_k64 * 32 + a_column_byte;
                    const int physical_byte = nvfp4_a4_swizzled_byte<Schedule>(row, logical_byte);
                    const auto* address =
                        shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                    ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1],
                                a_fragments[mma_m][2], a_fragments[mma_m][3], smem_addr(address));
                    const int scale_row =
                        warp_m * Schedule::kWarpTokens + mma_m * 16 + sfa_row;
                    a_scales[mma_m] =
                        shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
                }

#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaRows; ++mma_n) {
                    const int row = warp_n * Schedule::kWarpRows + mma_n * 8 + b_row_offset;
                    const int logical_byte  = local_k64 * 32 + b_column_byte;
                    const int physical_byte = nvfp4_a4_swizzled_byte<Schedule>(row, logical_byte);
                    const auto* address =
                        shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                    ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                    const int scale_row = warp_n * Schedule::kWarpRows + mma_n * 8 + sfb_row;
                    b_scales[mma_n]     = load_vec<unsigned>(
                        shared.b_scales[stage] + (scale_row * Schedule::kK64PerStage + local_k64) * 4);
                }

#pragma unroll
                for (int mma_m = 0; mma_m < Schedule::kMmaTokens; ++mma_m) {
#pragma unroll
                    for (int mma_n = 0; mma_n < Schedule::kMmaRows; ++mma_n) {
                        mma_nvfp4_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                       accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                       a_fragments[mma_m][0], a_fragments[mma_m][1],
                                       a_fragments[mma_m][2], a_fragments[mma_m][3],
                                       b_fragments[mma_n][0], b_fragments[mma_n][1],
                                       a_scales[mma_m], b_scales[mma_n]);
                    }
                }
            }

            __syncthreads();
            const int next_k_tile = k_tile + Schedule::kStages;
            if (next_k_tile < kNvfp4KTiles) {
                stage_nvfp4_gate_up_activation<Schedule>(x_codes, x_scales, packed_token,
                                                         packed_base, cols, shared, stage,
                                                         next_k_tile);
                stage_nvfp4_gate_up_weight<Schedule>(gate_codes, gate_scales, up_codes, up_scales,
                                                     logical_begin, shared, stage, next_k_tile);
            }
            cp_commit();
        }

        cp_wait<0>();
        __syncthreads();

        // Row 2r is gate and row 2r+1 is up for the same logical intermediate row, and the MMA
        // keeps an adjacent b-row pair in accumulator [0]/[1] (and [2]/[3] for the +8 token), so
        // the SwiGLU combine never crosses a warp.
#pragma unroll
        for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
            const int local_token0 = warp_m * Schedule::kWarpTokens + mt * 16 + ar;
            const int local_token1 = local_token0 + 8;
            const int col0         = packed_base + local_token0;
            const int col1         = packed_base + local_token1;
#pragma unroll
            for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
                const int logical =
                    logical_begin + warp_n * (Schedule::kWarpRows / 2) + mr * 4 + (lane & 3);
                const float gate0 = accumulators[mt][mr][0] * alpha_gate;
                const float up1   = accumulators[mt][mr][1] * alpha_up;
                const float gate2 = accumulators[mt][mr][2] * alpha_gate;
                const float up3   = accumulators[mt][mr][3] * alpha_up;
                if (local_token0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kNvfp4Intermediate + logical] =
                        __float2bfloat16_rn(silu(gate0) * up1);
                }
                if (local_token1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kNvfp4Intermediate + logical] =
                        __float2bfloat16_rn(silu(gate2) * up3);
                }
            }
        }
        __syncthreads();
    }
}

} // namespace

void sparse_moe_nvfp4_gate_up_launch(const Tensor& x, const SparseMoeWeights& weights,
                                     const int* packed_token, const int* expert_offsets,
                                     const int* route_job_experts,
                                     const int* route_job_columns, const int* route_job_count,
                                     std::uint8_t* activation_codes,
                                     std::uint8_t* activation_scales,
                                     __nv_bfloat16* routed_activation, cudaStream_t stream) {
    const int tokens    = x.ne[1];
    const auto* input   = static_cast<const __nv_bfloat16*>(x.data);

    using ActivationGeometry = Nvfp4ActivationGeometry<kNvfp4Hidden>;
    constexpr int kQuantizeThreads = 256;
    const int quantize_tasks       = tokens * ActivationGeometry::kGroupsPerRow;
    const int quantize_blocks      = (quantize_tasks + kQuantizeThreads - 1) / kQuantizeThreads;
    nvfp4_a4_quantize_kernel<ActivationGeometry, kQuantizeThreads, Nvfp4ScaleLayout::RowMajor>
        <<<quantize_blocks, kQuantizeThreads, 0, stream>>>(input, activation_codes,
                                                           activation_scales, tokens, tokens,
                                                           kNvfp4ActivationDivisor);
    CUDA_CHECK(cudaGetLastError());

    Nvfp4GateUpBases bases{};
    for (int expert = 0; expert < kNvfp4RoutedExperts; ++expert) {
        const Weight& gate = weights.routed_gate_up_experts[2 * expert];
        const Weight& up   = weights.routed_gate_up_experts[2 * expert + 1];
        bases.gate_codes[expert]    = gate.qdata;
        bases.gate_scales[expert]   = gate.scales;
        bases.up_codes[expert]      = up.qdata;
        bases.up_scales[expert]     = up.scales;
        bases.gate_divisor[expert]  = gate.weight_scale_divisor;
        bases.up_divisor[expert]    = up.weight_scale_divisor;
    }

    // The persistent kernel strides its work list by gridDim.x, so a fixed cap is correct for any
    // job count and the job count is only ever read on the device; never dereference the device
    // route map on the host.
    sparse_moe_nvfp4_gate_up_kernel<Nvfp4GateUpSchedule>
        <<<kNvfp4MaxBlocks, Nvfp4GateUpSchedule::kThreads, 0, stream>>>(
            activation_codes, activation_scales, const_cast<int*>(packed_token),
            const_cast<int*>(expert_offsets), const_cast<int*>(route_job_experts),
            const_cast<int*>(route_job_columns), const_cast<int*>(route_job_count),
            routed_activation, bases, kNvfp4ActivationDivisor);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
