#pragma once

// Host-side assembly of the logprob gather results for one committed token. Shared by the ordinary
// prefill path and the speculative (MTP/DFlash) decode bodies so the gather layout convention lives
// in one place.

#include "ninfer/types.h"
#include "runtime/contract/execution.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// The gather Op writes ninfer [kMaximumTokenLogprobs, rows] (ne[0]-contiguous), so element (k, row)
// is at k + row*kMaximumTokenLogprobs. The drawn token is normally the finite top-1, so its own
// value comes from the matching entry; kLogprobSentinel is the OpenAI "outside the reported top
// set" marker for the case it does not appear.
[[nodiscard]] inline runtime::RawTokenLogprob
assemble_logprob(const std::int32_t* ids, const float* values, std::int32_t row, TokenId token) {
    runtime::RawTokenLogprob record;
    record.id      = token;
    record.logprob = kLogprobSentinel;
    for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
        const std::size_t slot = k + static_cast<std::size_t>(row) * kMaximumTokenLogprobs;
        record.top_ids[k]      = ids[slot];
        record.top_values[k]   = values[slot];
        if (record.top_ids[k] == token && std::isfinite(record.top_values[k])) {
            record.logprob = record.top_values[k];
        }
    }
    return record;
}

// The speculative gather view is ninfer [V, width, batch] (ne[0]-contiguous), so vocabulary element
// (v, col, lane) is at v + (col + lane*width)*V. The flattened gather row for verify column `col`
// and batch lane `lane` is therefore col + lane*width: the column axis varies fastest, the batch
// axis slowest (not col*batch + lane).
[[nodiscard]] inline std::int32_t speculation_logprob_column(std::int32_t col, std::int32_t lane,
                                                             std::int32_t width) {
    return col + lane * width;
}

} // namespace ninfer::models::qwen3_5::detail
