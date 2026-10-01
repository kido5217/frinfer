#include "ops/launcher/apply_mask.h"

#include "core/device.h"
#include "ops/kernel/apply_mask.cuh"

#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

void apply_mask_launch(Tensor& logits, std::int32_t columns, std::int32_t batch, const Tensor& masks,
                       const Tensor& active, std::int32_t token_domain, std::int32_t mask_lane_stride,
                       cudaStream_t stream) {
    constexpr int block = 256;
    const std::int32_t words = (token_domain + 31) / 32;
    const std::int32_t lanes = columns * batch;
    const std::int64_t column_stride_bytes = columns == 1 ? 0 : logits.nb[1];
    const std::int64_t batch_stride_bytes  = columns == 1 ? logits.nb[1] : logits.nb[2];
    const unsigned grid_x = static_cast<unsigned>((words + block - 1) / block);
    const unsigned grid_y = static_cast<unsigned>(lanes);
    apply_mask_kernel<block><<<dim3(grid_x, grid_y), block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(masks.data),
        static_cast<const std::int32_t*>(active.data), words, token_domain, columns, batch,
        mask_lane_stride, column_stride_bytes, batch_stride_bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
