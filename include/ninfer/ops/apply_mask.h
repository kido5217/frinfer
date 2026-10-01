#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Applies a dense grammar mask to target logits in place: lane (column c, batch b) keeps the logit
// of token v exactly when bit v of mask row (c * mask_lane_stride + b) is set, and receives -inf
// (BF16) otherwise. Bits at or beyond `token_domain` are padding and are never inspected; the
// vocabulary rows beyond `token_domain` keep their logits untouched.
//
// `logits` must be well-formed BF16 with ne[0] >= token_domain:
//   - columns == 1: shape [vocab, batch] with the batch stride in nb[1] (a slice of the round
//     logits region is allowed to be non-contiguous);
//   - columns > 1: shape [vocab, columns, batch] with the column stride in nb[1] and the batch
//     stride in nb[2].
// `masks` is a contiguous I32 [words, rows] bitmask table: ne[0] = words = ceil(token_domain/32)
// is the 32-bit word count of one row, ne[1] = rows is the row table length, and each row's words
// are little-endian (bit v lives in word v>>5, position v&31). rows must be at least
// (columns - 1) * mask_lane_stride + batch. `active` is a device-resident I32 scalar:
// when it is zero the Op writes nothing, so an unconstrained round is byte-identical.
//
// Exact Op: a kept logit is never rewritten (a cleared bit writes the -inf bit pattern), so the
// result is a pure bit selection over the input. The Op is stream-ordered with its producer and
// consumers and has no host-side synchronization.
void apply_mask(Tensor& logits, std::int32_t columns, std::int32_t batch, const Tensor& masks,
                const Tensor& active, std::int32_t token_domain, std::int32_t mask_lane_stride,
                cudaStream_t stream);

} // namespace ninfer::ops
