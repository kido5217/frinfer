#pragma once

// The per-round logprob gather protocol, shared by the ordinary and speculative (MTP/DFlash) decode
// bodies. A round owns the request flag OR, the ingress gate, the optional per-verify-column sampling
// seed, and the post-round host assembly into the pending records. The gather row arithmetic for both
// layouts lives here so the [kMaximumTokenLogprobs, rows] ordinary gather and the [V, width, batch]
// speculative gather are never re-derived at the three call sites.

#include "models/qwen3_5/program/logprob_assembly.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::detail {

// The optional speculative sampling seed: the round's ingress logprob_sampling array. The ordinary
// body leaves this empty and seeds nothing.
struct LogprobSamplingView {
    std::span<ops::SamplingConfig> configs;
};

class LogprobRound {
  public:
    // `gate` is the round's ingress logprob_active field, `gather_ids`/`gather_values` are that
    // round's egress top-k arrays, and `pending` is the stable per-round record storage.
    // `speculative_width` is the verify width of a speculative round: it sets both the sampling
    // gather row (col + lane*width) and the pending-record stride. It is 0 for the ordinary round.
    LogprobRound(std::int32_t& gate, const std::int32_t* gather_ids, const float* gather_values,
                 std::span<runtime::RawTokenLogprob> pending, LogprobSamplingView sampling = {},
                 std::int32_t speculative_width = 0)
        : gate_(gate), gather_ids_(gather_ids), gather_values_(gather_values), pending_(pending),
          sampling_(sampling), speculative_width_(speculative_width) {}

    // OR one application request's flag into the round. Call once for every active row.
    void enable(bool requested) { enabled_ = enabled_ || requested; }

    // Publish the accumulated flag to the device gate. Call once per round after every enable().
    void publish() { gate_ = enabled_ ? 1 : 0; }

    [[nodiscard]] bool enabled() const { return enabled_; }

    // Seed the speculative sampling view before the row loop so an idle lane can never hand the
    // gather a stale config. No-op for the ordinary round.
    void seed_sampling() {
        std::fill(sampling_.configs.begin(), sampling_.configs.end(), ops::SamplingConfig{});
    }

    // Bind one lane's sampling config to every verify column of the speculative view, at the row
    // index the gather reads (col + lane*width). No-op for the ordinary round.
    void bind_sampling(std::size_t lane, const ops::SamplingConfig& config) {
        for (std::int32_t column = 0; column < speculative_width_; ++column) {
            sampling_.configs[static_cast<std::size_t>(speculation_logprob_column(
                column, static_cast<std::int32_t>(lane), speculative_width_))] = config;
        }
    }

    // Assemble one committed token from the ordinary [kMaximumTokenLogprobs, rows] gather into the
    // pending record for batch `row`.
    void assemble(std::size_t row, TokenId token) {
        pending_[row] =
            assemble_logprob(gather_ids_, gather_values_, static_cast<std::int32_t>(row), token);
    }

    // Assemble the committed token of verify column `index` for speculative batch `lane` from the
    // [V, width, batch] gather into the lane's pending record.
    void assemble(std::size_t lane, std::size_t index, TokenId token) {
        const std::int32_t row = speculation_logprob_column(
            static_cast<std::int32_t>(index), static_cast<std::int32_t>(lane), speculative_width_);
        pending_[lane * static_cast<std::size_t>(speculative_width_) + index] =
            assemble_logprob(gather_ids_, gather_values_, row, token);
    }

  private:
    std::int32_t& gate_;
    const std::int32_t* gather_ids_;
    const float* gather_values_;
    std::span<runtime::RawTokenLogprob> pending_;
    LogprobSamplingView sampling_;
    std::int32_t speculative_width_ = 0;
    bool enabled_ = false;
};

} // namespace ninfer::models::qwen3_5::detail
