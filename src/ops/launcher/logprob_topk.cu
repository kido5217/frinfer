// Implements: include/ninfer/ops/logprob_topk.h
// Match: wrapper-validated contiguous tensors, a device SamplingConfig[rows] array, and a
// caller-owned scratch span sized by logprob_topk_workspace_exact_bytes().
// Algorithm assumptions: a split-CTA partial pass then one combine CTA per row (kernel header).
#include "ops/launcher/logprob_topk.h"

#include "core/device.h"
#include "ops/kernel/logprob_topk.cuh"

namespace ninfer::ops::detail {

std::size_t logprob_topk_workspace_exact_bytes(std::int32_t rows) {
    const std::size_t partials = static_cast<std::size_t>(rows) * kLogprobTopkSplit;
    return partials * (2 * sizeof(float) +
                       kLogprobTopK * (sizeof(float) + sizeof(std::int32_t)));
}

void logprob_topk_launch(const Tensor& logits, const SamplingConfig* sampling,
                         std::int32_t token_domain, Tensor& top_ids, Tensor& top_values,
                         Tensor& lse, const Tensor& active, DeviceSpan workspace,
                         cudaStream_t stream) {
    const std::int32_t rows   = logits.ne[1];
    const std::int32_t split  = rows == 1 ? kLogprobTopkSplit : kLogprobTopkSplit / 2;
    const std::size_t partials = static_cast<std::size_t>(rows) * kLogprobTopkSplit;

    auto* base               = static_cast<unsigned char*>(workspace.data);
    LogprobScratch scratch{};
    scratch.pmax             = reinterpret_cast<float*>(base);
    scratch.psum             = scratch.pmax + partials;
    scratch.ptop_values      = scratch.psum + partials;
    scratch.ptop_ids =
        reinterpret_cast<std::int32_t*>(scratch.ptop_values + partials * kLogprobTopK);

    const auto* active_flag = static_cast<const std::int32_t*>(active.data);
    const dim3 partial_grid(static_cast<unsigned int>(split), static_cast<unsigned int>(rows));
    logprob_topk_partials_kernel<<<partial_grid, kLogprobTopkBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), sampling, token_domain, rows, split,
        active_flag, scratch.pmax, scratch.psum, scratch.ptop_values, scratch.ptop_ids);
    CUDA_CHECK(cudaGetLastError());

    logprob_topk_combine_kernel<<<static_cast<unsigned int>(rows), kLogprobTopkBlock, 0, stream>>>(
        scratch.pmax, scratch.psum, scratch.ptop_values, scratch.ptop_ids, rows, split, active_flag,
        static_cast<float*>(top_values.data), static_cast<std::int32_t*>(top_ids.data),
        static_cast<float*>(lse.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
