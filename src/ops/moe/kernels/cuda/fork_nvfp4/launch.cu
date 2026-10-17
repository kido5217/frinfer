// Host-side pipeline for the fork 35B-A3B NVFP4 route: packed-route grouping, the ported W4A4
// grouped expert stages (T >= 20) or the A16 SIMT expert pair (T < 20), the contribution reduce,
// and the shared-expert composition through the groupwise_int family. Upstream's own route for
// this geometry is groupwise_int; the fork route replaces only the expert GEMMs, so the shared
// term and the epilogue ride the same Q8 shared kernels and output policies.

#include "ops/moe/kernels/cuda/fork_nvfp4/launch.h"

#include "core/device.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/moe/kernels/cuda/output.cuh"
#include "ops/moe/nvfp4/sparse_moe_nvfp4_down.h"
#include "ops/moe/nvfp4/sparse_moe_nvfp4_gate_up.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail::moe_cuda::fork_nvfp4 {
namespace {
using common::bf16x2_bits_to_float2;
using common::load_vec;
using common::nvfp4_dequant_f16x16;
using common::nvfp4_scale_byte_offset;
using common::silu;
using common::warp_reduce_sum;

__device__ __forceinline__ bool selected(ExpertSubsetKind kind, const int* mask, int expert) {
    return kind == ExpertSubsetKind::All || mask[expert] == (kind == ExpertSubsetKind::MaskOnes);
}

// --- packed-route grouping (the fork column tile is 64 assignments wide) ----------------------

__global__ void count_experts(const int* ids, ExpertSubsetKind kind, const int* mask, int* counts,
                              int assignments) {
    __shared__ int histogram[kExperts];
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { histogram[e] = 0; }
    __syncthreads();
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < assignments) {
        const int e = ids[a];
        if (selected(kind, mask, e)) { atomicAdd(histogram + e, 1); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) {
        if (histogram[e]) { atomicAdd(counts + e, histogram[e]); }
    }
}

__device__ __forceinline__ void expert_scan(int* scan, int value, int e) {
    scan[e] = value;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < kExperts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
}

__global__ void scan_experts(Storage s, int columns) {
    __shared__ int scan[kExperts], touched[kExperts];
    const int e     = threadIdx.x;
    const int count = s.counts[e];
    expert_scan(scan, count, e);
    s.expert_offsets[e] = e ? scan[e - 1] : 0;
    s.cursors[e]        = s.expert_offsets[e];
    if (e == kExperts - 1) { s.expert_offsets[kExperts] = scan[e]; }
    __syncthreads();
    const int jobs = (count + columns - 1) / columns;
    expert_scan(scan, jobs, e);
    const int first = e ? scan[e - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        s.job_experts[first + j] = e;
        s.job_columns[first + j] = j * columns;
    }
    __syncthreads();
    expert_scan(touched, count > 0 ? 1 : 0, e);
    if (e == kExperts - 1) {
        s.job_count[0] = scan[e];
        s.job_count[1] = touched[e];
    }
}

__global__ void index_experts(const int* ids, ExpertSubsetKind kind, const int* mask, Storage s,
                              int assignments) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) { return; }
    const int expert = ids[a];
    if (!selected(kind, mask, expert)) {
        s.inverse[a] = -1;
        return;
    }
    const int packed = atomicAdd(s.cursors + expert, 1);
    s.packed_token[packed] = a / kTopK;
    s.inverse[a]           = packed;
}

// One-block grouping for the short end of the grouped route: counts, scan, jobs and the packed
// maps all live in shared memory, so the pass replaces the three launches above.
__global__ void group_single(const int* ids, ExpertSubsetKind kind, const int* mask, Storage s,
                             int assignments, int columns) {
    __shared__ int counts[kExperts], ranks[kTopK * 64], scan[kExperts];
    const int e = threadIdx.x;
    counts[e]   = 0;
    __syncthreads();
    for (int a = e; a < assignments; a += kExperts) {
        const int expert = ids[a];
        ranks[a] = selected(kind, mask, expert) ? atomicAdd(counts + expert, 1) : -1;
    }
    __syncthreads();
    const int count = counts[e];
    expert_scan(scan, count, e);
    const int begin = e ? scan[e - 1] : 0;
    s.expert_offsets[e] = begin;
    if (e == kExperts - 1) { s.expert_offsets[kExperts] = scan[e]; }
    const int jobs = (count + columns - 1) / columns;
    __syncthreads();
    expert_scan(scan, jobs, e);
    const int first = e ? scan[e - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        s.job_experts[first + j] = e;
        s.job_columns[first + j] = j * columns;
    }
    __syncthreads();
    expert_scan(counts, s.expert_offsets[e], e);  // reuse counts as cursors
    __syncthreads();
    if (e == kExperts - 1) {
        s.job_count[0] = scan[e];
        int touched    = 0;
        for (int x = 0; x < kExperts; ++x) {
            touched += s.expert_offsets[x + 1] > s.expert_offsets[x] ? 1 : 0;
        }
        s.job_count[1] = touched;
    }
    for (int a = e; a < assignments; a += kExperts) {
        const int expert  = ids[a];
        const int packed  = ranks[a] >= 0 ? counts[expert] + ranks[a] : -1;
        s.inverse[a]      = packed;
        if (packed >= 0) { s.packed_token[packed] = a / kTopK; }
    }
}

// --- grouped epilogue: BF16 column contributions -> destination or routed_sum -----------------

template <class Output>
__global__ void reduce_grouped(const __nv_bfloat16* __restrict__ grouped, const int* __restrict__ inverse,
                               const float* __restrict__ coefficients, Output output,
                               int elements) {
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const int token = i / kHidden, row = i % kHidden;
    float value     = 0.0F;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
        const int assignment = token * kTopK + j;
        const int column     = inverse[assignment];
        if (column >= 0) {
            value = fmaf(coefficients[assignment],
                         __bfloat162float(grouped[std::int64_t(column) * kHidden + row]), value);
        }
    }
    output.store(i, output.initial(i) + value);
}

// --- A16 SIMT expert pair (T < 20) -------------------------------------------------------------

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float divisor;
};

__device__ __forceinline__ Matrix bank_matrix(const ExpertMatrixDeviceView& matrix, int expert) {
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const std::uint8_t*>(matrix.scales) +
                (matrix.offsets ? offset : expert * matrix.scale_stride),
            matrix.divisors[expert * matrix.divisor_stride]};
}

__device__ __forceinline__ void expand_group(Matrix weight, int row, int group, int k,
                                             float* values) {
    const auto decoded =
        nvfp4_dequant_f16x16(weight.codes + std::int64_t(row) * (k / 2) + group * 8,
                             weight.scales[nvfp4_scale_byte_offset(row, group, k)]);
    const int words[8] = {decoded.lo.x, decoded.lo.y, decoded.lo.z, decoded.lo.w,
                          decoded.hi.x, decoded.hi.y, decoded.hi.z, decoded.hi.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto pair   = __half22float2(*reinterpret_cast<const __half2*>(&words[i]));
        values[2 * i]     = pair.x;
        values[2 * i + 1] = pair.y;
    }
}

__global__ void vector_gate(const __nv_bfloat16* __restrict__ x, ExpertMatrixDeviceView bank,
                            const int* __restrict__ ids, ExpertSubsetKind kind, const int* mask,
                            float* activation) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!selected(kind, mask, expert)) { return; }
    const auto weight = bank_matrix(bank, expert);
    const int row     = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const auto* input = x + std::int64_t(assignment / kTopK) * kHidden;
    float gate = 0, up = 0;
#pragma unroll
    for (int group = lane; group < kHidden / 16; group += 32) {
        float g[16], u[16];
        expand_group(weight, row, group, kHidden, g);
        expand_group(weight, row + kIntermediate, group, kHidden, u);
        const auto x0           = load_vec<uint4>(input + group * 16);
        const auto x1           = load_vec<uint4>(input + group * 16 + 8);
        const unsigned words[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 v = bf16x2_bits_to_float2(words[pair]);
            gate           = fmaf(g[pair * 2], v.x, gate);
            gate           = fmaf(g[pair * 2 + 1], v.y, gate);
            up             = fmaf(u[pair * 2], v.x, up);
            up             = fmaf(u[pair * 2 + 1], v.y, up);
        }
    }
    gate = warp_reduce_sum(gate) / weight.divisor;
    up   = warp_reduce_sum(up) / weight.divisor;
    if (!lane) { activation[std::int64_t(assignment) * kIntermediate + row] = silu(gate) * up; }
}

template <int Rows, class Output>
__global__ void vector_down(const float* __restrict__ activation, ExpertMatrixDeviceView bank,
                            const int* __restrict__ ids, const float* __restrict__ coefficients,
                            ExpertSubsetKind kind, const int* mask, Output output) {
    __shared__ float values[kTopK][Rows];
    const int token = blockIdx.y, route = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int assignment = token * kTopK + route, expert = ids[assignment];
    float sum[Rows] = {};
    if (selected(kind, mask, expert)) {
        const auto weight = bank_matrix(bank, expert);
        for (int group = lane; group < kIntermediate / 16; group += 32) {
            float x[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                x[k] = activation[std::int64_t(assignment) * kIntermediate + group * 16 + k];
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                float w[16];
                expand_group(weight, blockIdx.x * Rows + r, group, kIntermediate, w);
#pragma unroll
                for (int k = 0; k < 16; ++k) { sum[r] = fmaf(x[k], w[k], sum[r]); }
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            sum[r] = warp_reduce_sum(sum[r]) * (coefficients[assignment] / weight.divisor);
        }
    }
    if (!lane) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) { values[route][r] = sum[r]; }
    }
    __syncthreads();
    if (threadIdx.x < Rows) {
        float result = 0;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) { result += values[j][threadIdx.x]; }
        const int i = token * kHidden + blockIdx.x * Rows + threadIdx.x;
        output.store(i, output.initial(i) + result);
    }
}

} // namespace

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
            Tensor& destination, const Plan& plan, const Storage& storage,
            groupwise_int::ExpertStorage shared_storage, CudaExecutionView execution) {
    const auto stream  = execution.stream();
    const int  tokens  = x.ne[1];
    if (subset.kind != ExpertSubsetKind::None) {
        const int         assignments = tokens * kTopK;
        const auto*       ids         = static_cast<const int*>(routes.ids.data);
        const auto*       mask        = static_cast<const int*>(subset.mask.data);
        const auto*       coefficients = static_cast<const float*>(routes.coefficients.data);
        auto* const grouped = static_cast<__nv_bfloat16*>(storage.routed_io);
        auto* const product = static_cast<std::uint8_t*>(storage.product_scales);
        if (plan.grouped) {
            if (tokens <= 64) {
                group_single<<<1, kExperts, 0, stream>>>(ids, subset.kind, mask, storage,
                                                         assignments, 64);
            } else {
                CUDA_CHECK(cudaMemsetAsync(storage.counts, 0, kExperts * sizeof(int), stream));
                count_experts<<<(assignments + 255) / 256, 256, 0, stream>>>(
                    ids, subset.kind, mask, storage.counts, assignments);
                scan_experts<<<1, kExperts, 0, stream>>>(storage, 64);
                index_experts<<<(assignments + 255) / 256, 256, 0, stream>>>(
                    ids, subset.kind, mask, storage, assignments);
            }
            CUDA_CHECK(cudaGetLastError());
            sparse_moe_nvfp4_gate_up_launch(
                x, make_nvfp4_gate_up_bases(experts.bank), storage.packed_token,
                storage.expert_offsets, storage.job_experts, storage.job_columns,
                storage.job_count, storage.input_codes,
                static_cast<std::uint8_t*>(storage.input_scales), grouped,
                execution.multiprocessor_count(), stream);
            const Tensor activation(static_cast<__nv_bfloat16*>(storage.routed_io),
                                    experts.bank.device, DType::BF16,
                                    {kIntermediate, assignments});
            sparse_moe_nvfp4_down_launch(
                activation, make_nvfp4_down_bases(experts.bank), storage.expert_offsets,
                storage.job_experts, storage.job_columns, storage.job_count,
                storage.product_codes, product, grouped,
                execution.multiprocessor_count(), stream);
        } else {
            vector_gate<<<dim3(kIntermediate / 4, assignments), 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind,
                mask, storage.vector_activation);
            CUDA_CHECK(cudaGetLastError());
            const auto down = [&](auto output) {
                vector_down<4><<<dim3(kHidden / 4, tokens), kTopK * 32, 0, stream>>>(
                    storage.vector_activation, experts.bank.down, ids, coefficients, subset.kind,
                    mask, output);
                CUDA_CHECK(cudaGetLastError());
            };
            if (shared) {
                down(FloatContribution{storage.routed_sum});
            } else {
                with_output(destination, mode, down);
            }
        }
        if (plan.grouped) {
            const int elements = tokens * kHidden;
            if (shared) {
                // The shared-down pass adds routed_sum and publishes the destination once, so
                // the epilogue mode applies to the composed sum, not to the routed part alone.
                reduce_grouped<<<(elements + 255) / 256, 256, 0, stream>>>(
                    grouped, storage.inverse, coefficients,
                    FloatContribution{storage.routed_sum}, elements);
                CUDA_CHECK(cudaGetLastError());
            } else {
                with_output(destination, mode, [&](auto output) {
                    reduce_grouped<<<(elements + 255) / 256, 256, 0, stream>>>(
                        grouped, storage.inverse, coefficients, output, elements);
                });
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    if (shared) {
        // The routed pass finished into routed_sum; hand it to the groupwise shared-down pass,
        // which adds it and publishes the destination once with the requested mode.
        auto shared_experts          = shared_storage;
        shared_experts.routed_sum    = storage.routed_sum;
        groupwise_int::launch_shared_swiglu(x, *shared, mode, destination, shared_experts,
                                            plan.shared, execution);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::fork_nvfp4
