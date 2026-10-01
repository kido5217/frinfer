#pragma once

// Implements: include/ninfer/ops/apply_mask.h
// Match: BF16 logits [vocab, batch] (columns == 1) or [vocab, columns, batch] with arbitrary
// column/batch strides, a contiguous I32 bitmask row table and an I32[1] active flag. One thread
// per (lane, 32-bit mask word); fully-set words are skipped, cleared bits write the BF16 -inf bit
// pattern at the vocabulary offset. A zero active flag returns before any write.

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kApplyMaskMinusInfBf16 = 0xFF80u;

// 32-bit mask words per row, lanes = columns * batch, grid.y = lanes, grid.x covers the words.
template <int Block>
__launch_bounds__(Block) __global__ void apply_mask_kernel(
    __nv_bfloat16* __restrict__ logits, const std::int32_t* __restrict__ masks,
    const std::int32_t* __restrict__ active, std::int32_t words, std::int32_t token_domain,
    std::int32_t columns, std::int32_t batch, std::int32_t mask_lane_stride,
    std::int64_t column_stride_bytes, std::int64_t batch_stride_bytes) {
    if (active[0] == 0) { return; }

    const std::int32_t lane = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t column = columns == 1 ? 0 : lane / batch;
    const std::int32_t row    = columns == 1 ? lane : lane % batch;
    const std::int64_t lane_offset = column * column_stride_bytes + row * batch_stride_bytes;
    auto* lane_logits =
        reinterpret_cast<unsigned short*>(reinterpret_cast<unsigned char*>(logits) + lane_offset);
    const std::uint32_t* mask_row =
        reinterpret_cast<const std::uint32_t*>(masks) +
        static_cast<std::int64_t>(column * mask_lane_stride + row) * words;

    for (std::int32_t word_index = static_cast<std::int32_t>(blockIdx.x) * Block +
                                   static_cast<std::int32_t>(threadIdx.x);
         word_index < words;
         word_index += static_cast<std::int32_t>(gridDim.x) * Block) {
        const std::uint32_t word = mask_row[word_index];
        if (word == 0xFFFFFFFFu) { continue; }
        const std::int32_t first = word_index * 32;
        const std::int32_t limit = min(32, token_domain - first);
        for (std::int32_t bit = 0; bit < limit; ++bit) {
            if ((word & (1u << bit)) == 0) { lane_logits[first + bit] = kApplyMaskMinusInfBf16; }
        }
    }
}

} // namespace ninfer::ops
