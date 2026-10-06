#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
    // nvfp4 routed profile only: one complete encoded parent per expert projection. Gate and up
    // are separate source tensors and the source selects an FP32 divisor per expert (one divisor
    // per parent is the format's granularity); gate/up share that divisor within an expert.
    // Gate/up are expert-major (expert e owns entries 2e and 2e+1); down is expert-major.
    // Empty for the packed Q4/Q5/Q6/Q8 profiles, which keep their single-parent banks above.
    std::vector<Weight> routed_gate_up_experts;
    std::vector<Weight> routed_down_experts;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

/**
 * Optional per-call execution hints. Every field is a pure cache hint with no numeric effect:
 * the same call with a default-constructed SparseMoeHints produces bit-identical output.
 *
 * next_weight_prefetch names a weight span the next decode-step consumer will stream; the decode
 * D4 epilogue issues fire-and-forget L2 prefetches over its first bytes. The span is caller-owned
 * and read once, inside the call: the Op keeps no state between calls, and no hidden channel
 * carries it.
 */
struct SparseMoeHints {
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op for the exact future 35B-A3B geometry.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and Q8+Q8 for the two routed banks; both
 * shared banks are Q8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs. The nvfp4 routed profile is admitted as per-expert parents
 * (routed_gate_up_experts/routed_down_experts, one complete [512,2048] or [2048,512] parent per
 * expert projection) because gate/up are separate tensors and the source selects an FP32 divisor
 * per expert; gate/up share that divisor within an expert. At T >= 20 the nvfp4 routed prefill
 * path executes end to end on the Blackwell W4A4 tensor-core path (`mma_nvfp4_e4m3`): the routed
 * activation is quantised once per token into a compact FP4 plane shared by every expert the token
 * routes to for the grouped 64-column/64-row gate/up GEMM, the SwiGLU result is quantised per
 * packed column for the grouped down GEMM, and the routed output is merged with the shared expert.
 * Each quantisation uses one private activation divisor (activation quantisation is a private
 * execution choice, not the source's per-expert activation_input_divisor).
 *
 * Every positive T is supported by the packed Q4/Q5/Q6/Q8 profiles. The nvfp4 routed profile is
 * admitted at T >= 20; its routed gate/up, routed down, and merge execute on the Blackwell W4A4
 * tensor-core path.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * The same Op with caller-supplied execution hints. Semantics, workspace requirement and output
 * are exactly those of the overload above; hints only steer cache warming.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, WorkspaceArena& workspace,
                cudaStream_t stream);

} // namespace ninfer::ops
