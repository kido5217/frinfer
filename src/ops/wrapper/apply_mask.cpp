#include "ninfer/ops/apply_mask.h"

#include "ops/launcher/apply_mask.h"

#include <stdexcept>

namespace ninfer::ops {

void apply_mask(Tensor& logits, std::int32_t columns, std::int32_t batch, const Tensor& masks,
                const Tensor& active, std::int32_t token_domain, std::int32_t mask_lane_stride,
                cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("apply_mask: logits must be BF16");
    }
    if (masks.dtype != DType::I32) { throw std::invalid_argument("apply_mask: masks must be I32"); }
    if (active.dtype != DType::I32) {
        throw std::invalid_argument("apply_mask: active must be I32");
    }
    if (logits.data == nullptr || masks.data == nullptr || active.data == nullptr) {
        throw std::invalid_argument("apply_mask: tensors must have non-null data");
    }
    if (columns <= 0 || batch <= 0 || mask_lane_stride <= 0 || mask_lane_stride < batch) {
        throw std::invalid_argument("apply_mask: lane geometry is invalid");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("apply_mask: token domain is outside the logits rows");
    }
    if (columns == 1) {
        if (logits.ne[1] != batch || logits.ne[2] != 1 || logits.ne[3] != 1) {
            throw std::invalid_argument("apply_mask: single-column logits must be [vocab, batch]");
        }
    } else if (logits.ne[1] != columns || logits.ne[2] != batch || logits.ne[3] != 1) {
        throw std::invalid_argument(
            "apply_mask: multi-column logits must be [vocab, columns, batch]");
    }
    if (logits.nb[0] != static_cast<std::int64_t>(dtype_size(DType::BF16))) {
        throw std::invalid_argument("apply_mask: the vocabulary dimension must be contiguous");
    }
    if (logits.nb[1] <= 0 || (columns > 1 && logits.nb[2] <= 0)) {
        throw std::invalid_argument("apply_mask: lane strides must be positive");
    }
    const std::int32_t words = (token_domain + 31) / 32;
    if (masks.ne[0] != words) {
        throw std::invalid_argument("apply_mask: mask rows must carry one word per 32 tokens");
    }
    if (masks.ne[1] < (columns - 1) * mask_lane_stride + batch) {
        throw std::invalid_argument("apply_mask: mask row table is too short");
    }
    if (masks.nb[1] !=
        static_cast<std::int64_t>(words) * static_cast<std::int64_t>(sizeof(std::int32_t))) {
        throw std::invalid_argument("apply_mask: mask rows must be contiguous");
    }
    detail::apply_mask_launch(logits, columns, batch, masks, active, token_domain, mask_lane_stride,
                              stream);
}

} // namespace ninfer::ops
