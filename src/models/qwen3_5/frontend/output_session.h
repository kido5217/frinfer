#pragma once
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {
class GrammarSession;
} // namespace ninfer::text

namespace ninfer::models::qwen3_5 {
namespace frontend {
class Tokenizer;
struct ToolCallOutputContract;
} // namespace frontend
class Frontend;

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }

    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }

    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }

    void clear() noexcept;
    void push_back(OutputDelta value);

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    // `logprobs` is optional per-token log-probability data aligned with `tokens` (empty when the
    // request did not opt in). Records for content tokens accumulate on the session and are exposed
    // through content_logprobs().
    [[nodiscard]] runtime::OutputDecision
    preview_model(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining,
                  FinishReason limit_reason,
                  std::span<const runtime::RawTokenLogprob> logprobs = {});
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    // Arms an early close of the model-origin reasoning block for the next model round. The close is
    // committed exactly as a thinking-budget boundary commits it, under the same capacity rule: a
    // remaining output budget that cannot fit the complete control suffix plus one post-close model
    // token ends the request instead. A session that never opened reasoning, or that already applied
    // control, ignores the request. Idempotent.
    void request_reasoning_close() noexcept;

    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    // Releases the current preview when a round is rolled back; the matcher returns to its last
    // committed state.
    void discard_preview() noexcept;
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

    // Content-token logprob records committed so far, in generation order. Empty unless the request
    // opted in. Records carry the tokenizer's decoded bytes for the token.
    [[nodiscard]] std::span<const TokenLogprob> content_logprobs() const noexcept;

    // Request-owned constrained-decoding matcher (ticket #247). `grammar_masks` fills the legal
    // next-token bitmask for the given draft prefix; the session advances the matcher as tokens
    // commit and rolls back a discarded preview.
    [[nodiscard]] bool constrained() const noexcept;
    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId> drafts,
                                              std::span<std::uint32_t> words);

private:
    class Impl;
    OutputSession(std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy,
                  OutputOptions output, bool starts_in_reasoning, ThinkingControlOptions thinking,
                  std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                  std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output,
                  std::unique_ptr<text::GrammarSession> grammar = {});
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

} // namespace ninfer::models::qwen3_5
