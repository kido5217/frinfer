#pragma once

// Implements: include/ninfer/ops/logprob_topk.h
// Match: contiguous BF16 [physical_rows,rows], a device SamplingConfig[rows] array, and
// contiguous I32/FP32 outputs. Only vocabulary rows v in [0,token_domain) participate.
// Algorithm assumptions: a split-CTA partial pass (one CTA per vocabulary chunk of each row)
// feeding one combine CTA per row; the bounded top-K uses an insertion-sorted per-thread list and
// a shared tree merge, so no full sort or global staging is needed.

#include "ninfer/ops/logprob_topk.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kLogprobTopkBlock = 128;
// Maximum partial CTAs per row. The launcher uses fewer when several rows share the grid, so the
// per-CTA merge overhead is amortized without starving the vocabulary scan of CTAs.
inline constexpr int kLogprobTopkSplit = 128;

struct LogprobScratch {
    float* pmax;         // [rows*split]
    float* psum;         // [rows*split]
    float* ptop_values;  // [rows*split*K]
    std::int32_t* ptop_ids; // [rows*split*K]
};

struct LogprobCandidate {
    float value;
    std::int32_t id;
};

__device__ __forceinline__ bool logprob_better(float av, int aid, float bv, int bid) {
    return sampling_better(av, aid, bv, bid);
}

__device__ __forceinline__ float logprob_block_max(float value, float* scratch, int tid) {
    scratch[tid] = value;
    __syncthreads();
    for (int step = kLogprobTopkBlock / 2; step > 0; step >>= 1) {
        if (tid < step) { scratch[tid] = fmaxf(scratch[tid], scratch[tid + step]); }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float logprob_block_sum(float value, float* scratch, int tid) {
    scratch[tid] = value;
    __syncthreads();
    for (int step = kLogprobTopkBlock / 2; step > 0; step >>= 1) {
        if (tid < step) { scratch[tid] += scratch[tid + step]; }
        __syncthreads();
    }
    return scratch[0];
}

// Merges the kLogprobTopkBlock per-thread sorted candidate lists held in `cand` into slots
// [0,K) of thread 0's list. All threads must call; `cand` is [kLogprobTopkBlock*K].
__device__ __forceinline__ void logprob_block_merge_topk(LogprobCandidate* cand, int tid) {
    for (int step = kLogprobTopkBlock / 2; step > 0; step >>= 1) {
        if (tid < step) {
            int i = 0;
            int j = 0;
            LogprobCandidate out[kLogprobTopK];
#pragma unroll
            for (int k = 0; k < kLogprobTopK; ++k) {
                bool take_left;
                if (i >= kLogprobTopK) {
                    take_left = false;
                } else if (j >= kLogprobTopK) {
                    take_left = true;
                } else {
                    const LogprobCandidate& left  = cand[tid * kLogprobTopK + i];
                    const LogprobCandidate& right = cand[(tid + step) * kLogprobTopK + j];
                    take_left = logprob_better(left.value, left.id, right.value, right.id);
                }
                out[k] = take_left ? cand[tid * kLogprobTopK + i]
                                   : cand[(tid + step) * kLogprobTopK + j];
                if (take_left) {
                    ++i;
                } else {
                    ++j;
                }
            }
#pragma unroll
            for (int k = 0; k < kLogprobTopK; ++k) { cand[tid * kLogprobTopK + k] = out[k]; }
        }
        __syncthreads();
    }
}

// Per-chunk partial: max scaled logit and sum(exp(scaled-max)) for the logsumexp, plus a bounded
// top-K list over the chunk. `*active==0` makes the whole Op a no-op.
__global__ void logprob_topk_partials_kernel(
    const __nv_bfloat16* __restrict__ logits, const SamplingConfig* __restrict__ sampling,
    std::int32_t token_domain, std::int32_t rows, std::int32_t split,
    const std::int32_t* __restrict__ active, float* __restrict__ pmax, float* __restrict__ psum,
    float* __restrict__ ptop_values, std::int32_t* __restrict__ ptop_ids) {
    if (*active == 0) { return; }

    __shared__ LogprobCandidate sCand[kLogprobTopkBlock * kLogprobTopK];
    __shared__ float sRed[kLogprobTopkBlock];

    const int split_index = static_cast<int>(blockIdx.x);
    const int row         = static_cast<int>(blockIdx.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int chunk       = (token_domain + split - 1) / split;
    const int begin       = split_index * chunk;
    const int end         = min(begin + chunk, token_domain);
    const int partial     = row * split + split_index;

    const SamplingConfig cfg = sampling[row];
    const float temperature  = cfg.temperature > 0.0f ? cfg.temperature : 1.0f;
    const float inv_temp     = 1.0f / temperature;
    const std::int64_t row_stride = rows;

    float local_max = -CUDART_INF_F;
    for (int v = begin + tid; v < end; v += kLogprobTopkBlock) {
        const float raw = __bfloat162float(logits[static_cast<std::int64_t>(v) * row_stride + row]);
        local_max       = fmaxf(local_max, sampling_adjusted_logit(raw, v, cfg) * inv_temp);
    }
    const float chunk_max = logprob_block_max(local_max, sRed, tid);

    LogprobCandidate local_top[kLogprobTopK];
#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) {
        local_top[k].value = -CUDART_INF_F;
        local_top[k].id    = INT_MAX;
    }
    float local_sum = 0.0f;
    if (chunk_max > -CUDART_INF_F) {
        for (int v = begin + tid; v < end; v += kLogprobTopkBlock) {
            const float raw = __bfloat162float(logits[static_cast<std::int64_t>(v) * row_stride + row]);
            const float scaled = sampling_adjusted_logit(raw, v, cfg) * inv_temp;
            local_sum += expf(scaled - chunk_max);
            if (logprob_better(scaled, v, local_top[kLogprobTopK - 1].value,
                               local_top[kLogprobTopK - 1].id)) {
                int position = kLogprobTopK - 1;
                while (position > 0 && logprob_better(scaled, v, local_top[position - 1].value,
                                                      local_top[position - 1].id)) {
                    local_top[position] = local_top[position - 1];
                    --position;
                }
                local_top[position].value = scaled;
                local_top[position].id    = v;
            }
        }
    }
    const float chunk_sum = logprob_block_sum(local_sum, sRed, tid);

#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) { sCand[tid * kLogprobTopK + k] = local_top[k]; }
    __syncthreads();
    logprob_block_merge_topk(sCand, tid);

    if (tid == 0) {
        pmax[partial] = chunk_max;
        psum[partial] = chunk_sum;
#pragma unroll
        for (int k = 0; k < kLogprobTopK; ++k) {
            ptop_values[partial * kLogprobTopK + k] = sCand[k].value;
            ptop_ids[partial * kLogprobTopK + k]    = sCand[k].id;
        }
    }
}

// Combine the per-row partials into lse and the final top-K logprob list. One CTA per row.
__global__ void logprob_topk_combine_kernel(
    const float* __restrict__ pmax, const float* __restrict__ psum,
    const float* __restrict__ ptop_values, const std::int32_t* __restrict__ ptop_ids,
    std::int32_t rows, std::int32_t split, const std::int32_t* __restrict__ active,
    float* __restrict__ top_values, std::int32_t* __restrict__ top_ids, float* __restrict__ lse) {
    if (*active == 0) { return; }

    __shared__ LogprobCandidate sCand[kLogprobTopkBlock * kLogprobTopK];
    __shared__ float sRed[kLogprobTopkBlock];

    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int base = row * split;

    float row_max = -CUDART_INF_F;
    for (int s = tid; s < split; s += kLogprobTopkBlock) { row_max = fmaxf(row_max, pmax[base + s]); }
    const float maximum = logprob_block_max(row_max, sRed, tid);

    float accumulator = 0.0f;
    for (int s = tid; s < split; s += kLogprobTopkBlock) {
        const float partial_max = pmax[base + s];
        if (partial_max > -CUDART_INF_F) {
            accumulator += psum[base + s] * expf(partial_max - maximum);
        }
    }
    const float total = logprob_block_sum(accumulator, sRed, tid);
    const float logsumexp = maximum + logf(total);

    LogprobCandidate local_top[kLogprobTopK];
#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) {
        local_top[k].value = -CUDART_INF_F;
        local_top[k].id    = INT_MAX;
    }
    const int candidate_count = split * kLogprobTopK;
    const std::int64_t candidate_base = static_cast<std::int64_t>(row) * split * kLogprobTopK;
    for (int i = tid; i < candidate_count; i += kLogprobTopkBlock) {
        const float value = ptop_values[candidate_base + i];
        const std::int32_t id = ptop_ids[candidate_base + i];
        if (logprob_better(value, id, local_top[kLogprobTopK - 1].value,
                           local_top[kLogprobTopK - 1].id)) {
            int position = kLogprobTopK - 1;
            while (position > 0 && logprob_better(value, id, local_top[position - 1].value,
                                                  local_top[position - 1].id)) {
                local_top[position] = local_top[position - 1];
                --position;
            }
            local_top[position].value = value;
            local_top[position].id    = id;
        }
    }
#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) { sCand[tid * kLogprobTopK + k] = local_top[k]; }
    __syncthreads();
    logprob_block_merge_topk(sCand, tid);

    if (tid == 0) {
#pragma unroll
        for (int k = 0; k < kLogprobTopK; ++k) {
            top_values[k * rows + row] = sCand[k].value - logsumexp;
            top_ids[k * rows + row]    = sCand[k].id;
        }
        lse[row] = logsumexp;
    }
}

} // namespace ninfer::ops
