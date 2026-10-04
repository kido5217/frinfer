// ninfer::ops - logprob_topk wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/logprob_topk.h"

#include "ops/launcher/logprob_topk.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("logprob_topk: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_vector(const Tensor& tensor, std::int32_t length, const char* label) {
    if (tensor.ne[0] != length || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("logprob_topk: ") + label + " must have shape [" +
                                    std::to_string(length) + "]");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("logprob_topk: ") + label + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("logprob_topk: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("logprob_topk: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

std::size_t logprob_topk_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t rows) {
    if (token_domain <= 0 || rows <= 0) {
        throw std::invalid_argument("logprob_topk workspace: invalid profile");
    }
    return detail::logprob_topk_workspace_exact_bytes(rows);
}

void logprob_topk(const Tensor& logits, const SamplingConfig* sampling,
                  std::int32_t token_domain, Tensor& top_ids, Tensor& top_values, Tensor& lse,
                  const Tensor& active, WorkspaceArena& workspace, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("logprob_topk: logits must be BF16");
    }
    if (top_ids.dtype != DType::I32) {
        throw std::invalid_argument("logprob_topk: top_ids must be I32");
    }
    if (top_values.dtype != DType::FP32) {
        throw std::invalid_argument("logprob_topk: top_values must be FP32");
    }
    if (lse.dtype != DType::FP32) {
        throw std::invalid_argument("logprob_topk: lse must be FP32");
    }
    if (active.dtype != DType::I32) {
        throw std::invalid_argument("logprob_topk: active must be I32");
    }
    if (sampling == nullptr) {
        throw std::invalid_argument("logprob_topk: sampling configs must be non-null");
    }

    require_rank_two(logits, "logits");
    const std::int32_t rows = logits.ne[1];
    if (top_ids.ne[0] != kLogprobTopK || top_ids.ne[1] != rows || top_ids.ne[2] != 1 ||
        top_ids.ne[3] != 1) {
        throw std::invalid_argument("logprob_topk: top_ids must have shape [kLogprobTopK, rows]");
    }
    if (top_values.ne[0] != kLogprobTopK || top_values.ne[1] != rows || top_values.ne[2] != 1 ||
        top_values.ne[3] != 1) {
        throw std::invalid_argument(
            "logprob_topk: top_values must have shape [kLogprobTopK, rows]");
    }
    require_vector(lse, rows, "lse");
    require_vector(active, 1, "active");
    if (token_domain < kLogprobTopK || token_domain > logits.ne[0]) {
        throw std::invalid_argument(
            "logprob_topk: token_domain must be in [kLogprobTopK, physical_rows]");
    }

    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(top_ids, alignof(std::int32_t), "top_ids");
    require_accessible(top_values, alignof(float), "top_values");
    require_accessible(lse, alignof(float), "lse");
    require_accessible(active, alignof(std::int32_t), "active");
    if (overlaps(top_ids, logits) || overlaps(top_values, logits) || overlaps(lse, logits) ||
        overlaps(top_ids, active) || overlaps(top_values, active) || overlaps(lse, active) ||
        overlaps(top_ids, top_values) || overlaps(top_ids, lse) || overlaps(top_values, lse)) {
        throw std::invalid_argument("logprob_topk: outputs must not overlap inputs or each other");
    }

    auto scratch_scope = workspace.scope();
    const std::size_t bytes = detail::logprob_topk_workspace_exact_bytes(rows);
    const DeviceSpan scratch = workspace.alloc_bytes(bytes);
    detail::logprob_topk_launch(logits, sampling, token_domain, top_ids, top_values, lse, active,
                                scratch, stream);
}

} // namespace ninfer::ops
