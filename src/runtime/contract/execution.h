#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <compare>
#include <span>

namespace ninfer::runtime {

// Borrowed for one synchronous Program call. Engine maps compact rows to request-owned matchers.
class TokenMaskProvider {
public:
    virtual ~TokenMaskProvider()                                           = default;
    [[nodiscard]] virtual bool constrained(std::size_t row) const noexcept = 0;
    // Returns dead-end position bits. Even dead/unreachable positions receive a safe nonempty
    // device mask. A dead end fails a row only if verification reaches that position.
    [[nodiscard]] virtual std::uint32_t fill(std::size_t row, std::span<const TokenId> drafts,
                                             std::span<std::uint32_t> words) = 0;
};

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
    FailedReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    bool failed                   = false;
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

// Device-gather result for one committed token before tokenizer bytes are attached. Row-major over
// the [rows,row_stride] token extent; empty when logprobs were not requested for the round.
struct RawTokenLogprob {
    TokenId id        = 0;
    float logprob     = 0.0f;
    std::array<TokenId, kMaximumTokenLogprobs> top_ids{};
    std::array<float, kMaximumTokenLogprobs> top_values{};
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
    // Optional per-token logprob records aligned with `tokens`.
    std::span<const RawTokenLogprob> logprobs;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::span<const RawTokenLogprob> logprobs;
    std::uint32_t row_stride = 1;
    ExecutionTiming timing;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
