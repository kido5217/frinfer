#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Fixed OpenAI top-logprob width. The response layer trims to the requested
// top_logprobs; the sentinel rule is defined against this top-20 set.
inline constexpr std::int32_t kLogprobTopK = 20;

/**
 * Op: Bounded top-K log-probability gather over sampler logits.
 *
 * Math / indexing:
 *   Let l[v,r] be the exact real value represented by logits[v,r] and c[v] the token-count entry
 *   the sampler reads for row r. With
 *
 *     T_r        = sampling[r].temperature > 0 ? sampling[r].temperature : 1
 *     adjusted   = l[v,r] - sampling[r].presence_penalty * (c[v] > 0)
 *                           - sampling[r].frequency_penalty * c[v]
 *     scaled     = adjusted / T_r
 *     lse_r      = logsumexp_{v in [0,token_domain)} scaled
 *
 *   the Op reports top_ids[k,r] = the k-th token id ordered by scaled descending (lower id
 *   breaks ties), top_values[k,r] = scaled - lse_r (the log-probability under the full,
 *   pre-truncation distribution), and lse[r] = lse_r, for k in [0,kLogprobTopK).
 *
 *   A masked token (scaled == -inf, e.g. a grammar/logit mask) is not a candidate: it is excluded
 *   from both the top-K list and the logsumexp. When a row has fewer than kLogprobTopK finite
 *   candidates, the remaining slots are empty, written as top_ids == -1 and top_values == -inf.
 *
 * Logical shapes:
 *   logits is BF16 [physical_rows,rows] with kLogprobTopK<=token_domain<=physical_rows and rows>0.
 *   sampling is a non-null device pointer to `rows` contiguous SamplingConfig records. top_ids is I32
 *   [kLogprobTopK,rows], top_values is FP32 [kLogprobTopK,rows], lse is FP32 [rows]. active is a
 *   device I32 [1]: when *active==0 the Op performs no device work and outputs are unspecified.
 *   Tensors are ninfer-layout (ne[0]-contiguous), so element (v,r) of logits is at r*ne[0]+v and
 *   element (k,r) of top_ids/top_values is at k + r*kLogprobTopK.
 *
 * Supported domain:
 *   logits is contiguous BF16, sampling is a non-null device pointer to `rows` contiguous
 *   SamplingConfig records, and every output is contiguous FP32/I32. Every row has at least one
 *   finite value in [0,token_domain); otherwise the row's output is unspecified. SamplingConfig
 *   token-count arrays, when non-null, are device I32 [token_domain] and are only read.
 *
 * Numeric:
 *   Output values are the FP32 numerical approximation of the ideal; reduction association and
 *   private accumulator precision are implementation choices. The independent oracle evaluates
 *   the formula in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes top_ids, top_values and lse and preserves every input. Outputs must not overlap
 *   logits or active.
 *
 * Workspace:
 *   Uses caller-owned transient storage reported by logprob_topk_workspace_capacity_bytes() and
 *   owns no persistent state.
 *
 * Execution:
 *   Enqueues work on stream. `active` is read on device so the Op is a cheap no-op when logprobs
 *   are not requested for any row of the round; this lets one captured CUDA graph carry the Op
 *   unconditionally.
 */
void logprob_topk(const Tensor& logits, const SamplingConfig* sampling,
                  std::int32_t token_domain, Tensor& top_ids, Tensor& top_values, Tensor& lse,
                  const Tensor& active, WorkspaceArena& workspace, cudaStream_t stream);

// Transient device bytes logprob_topk() needs for the given profile. Zero when the Op needs no
// scratch (it always needs scratch for rows>=1, so this is always positive for a valid profile).
[[nodiscard]] std::size_t logprob_topk_workspace_capacity_bytes(std::int32_t token_domain,
                                                                std::int32_t rows);

} // namespace ninfer::ops
