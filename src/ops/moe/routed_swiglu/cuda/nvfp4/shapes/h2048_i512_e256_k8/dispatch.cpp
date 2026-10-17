#include "ops/moe/routed_swiglu/cuda/dispatch.h"

namespace ninfer::ops::detail::moe_routed {
namespace {
// Fork 35B-A3B NVFP4 route. The fork kernels select their implementation on the token count:
// a SIMT A16 path below 20 tokens and the grouped W4A4 persistent path at or above 20; the
// boundaries mirror the former sparse_moe prefill/small_t split. Expert parents are consumed
// through ops/moe/nvfp4/sparse_moe_nvfp4_gate_up.h (kNvfp4RoutedExperts = 256) as one affine
// resident bank; the shared Q8 expert rides the groupwise_int shared kernels.
Plan select_plan(int tokens, bool shared) {
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::ForkNvfp4;
    plan.fork           = {.grouped          = tokens >= kernel::fork_nvfp4::kGroupedFirstTokens,
                           .queued_ctas_per_sm = 32,
                           .shared           = {.token   = {.paths_per_block    = 1,
                                                            .output_rows        = tokens == 1 ? 1 : 4,
                                                            .single_token       = tokens == 1,
                                                            .programmatic_input = false},
                                                // The fork composition hands routed_sum to the
                                                // shared-down pass, which exists only in the MMA
                                                // shared branch.
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
} // namespace ninfer::ops::detail::moe_routed
