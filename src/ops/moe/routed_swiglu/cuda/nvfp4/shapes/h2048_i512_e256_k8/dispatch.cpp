#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
// Fork 35B-A3B NVFP4 route. The fork kernels select their implementation on the token count:
// a SIMT A16 path below 20 tokens and the grouped W4A4 persistent path at or above 20; the
// boundaries mirror the former sparse_moe prefill/small_t split. Expert parents are consumed
// through ops/moe/nvfp4/sparse_moe_nvfp4_gate_up.h (kNvfp4RoutedExperts = 256).
Plan select_plan(int tokens, bool) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    plan.block_scaled   = {.format                = QType::NVFP4,
                           .gate_columns          = tokens < 20 ? 0 : 32,
                           .down_columns          = tokens < 20 ? 0 : 32,
                           .vector                = false,
                           .atomic_output         = tokens >= 128,
                           .single_block_grouping = tokens <= 64,
                           .shared_algorithm      = tokens < 8
                                                        ? kernel::block_scaled::SharedAlgorithm::Vector
                                                        : kernel::block_scaled::SharedAlgorithm::Mma,
                           .queued_ctas_per_sm    = 32};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2048_i512_e256_k8() {
    static constexpr int boundaries[]{2, 20, 128, 129};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed
