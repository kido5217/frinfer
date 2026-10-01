#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Preconditions are the validated ones of ops::apply_mask (see include/ninfer/ops/apply_mask.h).
void apply_mask_launch(Tensor& logits, std::int32_t columns, std::int32_t batch,
                       const Tensor& masks, const Tensor& active, std::int32_t token_domain,
                       std::int32_t mask_lane_stride, cudaStream_t stream);

} // namespace ninfer::ops::detail
