#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

namespace ninfer::ops::detail::moe_complete {
namespace {
// Fork 35B-A3B NVFP4 route. The router follows the groupwise h2048 line except that the fork
// grouped route runs its own counts pass (no router count fusion), so the shared pass stays
// on the MMA branch that consumes the fork route's routed_sum.
Plan select_plan(int tokens) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::ForkNvfp4;
    plan.router         = {.algorithm         = tokens == 1
                              ? RouterAlgorithm::Vector
                              : tokens < kernel::fork_nvfp4::kGroupedFirstTokens
                                  ? RouterAlgorithm::TokenTile
                                  : RouterAlgorithm::Mma,
                           .programmatic      = false,
                           .count_assignments = false};
    plan.fork           = {.grouped          = tokens >= kernel::fork_nvfp4::kGroupedFirstTokens,
                           .queued_ctas_per_sm = 32,
                           .shared           = {.token   = {.paths_per_block    = 1,
                                                            .output_rows        = tokens == 1 ? 1 : 4,
                                                            .single_token       = tokens == 1,
                                                            .programmatic_input = false},
                                                .columns = 64}};
    return plan;
}
} // namespace

const ShapePolicy& nvfp4_h2048_i512_e256_k8() {
    static constexpr int boundaries[]{kernel::fork_nvfp4::kGroupedFirstTokens,
                                      kernel::fork_nvfp4::kGroupedFirstTokens + 1};
    static const ShapePolicy policy{.select = select_plan, .allocation_boundaries = boundaries};
    return policy;
}
} // namespace ninfer::ops::detail::moe_complete
