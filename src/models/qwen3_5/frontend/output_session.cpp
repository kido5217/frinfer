#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/chat_parse_core.h"
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/unicode.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {
namespace fi                                = frontend;
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // Replace one maximal subpart. The first byte that cannot continue this sequence
                // is deliberately left for the next iteration, so valid following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

template <std::size_t Size>
consteval std::array<std::size_t, Size> make_prefix_failure_table(std::string_view pattern) {
    std::array<std::size_t, Size> failure{};
    for (std::size_t index = 1; index < Size; ++index) {
        std::size_t matched = failure[index - 1U];
        while (matched != 0 && pattern[index] != pattern[matched]) {
            matched = failure[matched - 1U];
        }
        if (pattern[index] == pattern[matched]) { ++matched; }
        failure[index] = matched;
    }
    return failure;
}

struct PrefixExecutionTracker {
    static constexpr std::string_view kBoundary = fi::kCanonicalReasoningCloseSerialization;
    static constexpr auto kFailure = make_prefix_failure_table<kBoundary.size()>(kBoundary);

    // Returns the byte offset immediately after the first completed boundary in this token.
    [[nodiscard]] std::optional<std::size_t> feed(std::string_view bytes) noexcept {
        if (!tracking) { return std::nullopt; }
        for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
            const char byte = bytes[offset];
            while (matched != 0 && byte != kBoundary[matched]) { matched = kFailure[matched - 1U]; }
            if (byte == kBoundary[matched]) { ++matched; }
            if (matched != kBoundary.size()) { continue; }
            tracking = false;
            matched  = 0;
            return offset + 1U;
        }
        return std::nullopt;
    }

    std::size_t matched = 0;
    bool tracking       = false;
};

// Presentation state that is not owned by the parsing core: UTF-8 reassembly, stop-string
// holding, observations. Channel membership, marker boundaries and the terminal tool-call
// resolution belong to ChatParseCore.
struct DecoderState {
    std::string utf8_pending;
    std::array<std::string, 2> stop_pending;
    std::string content_whitespace_pending;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
};

struct ThinkingSessionState {
    std::optional<std::uint32_t> budget;
    std::uint32_t budget_thinking_tokens = 0;
    std::uint32_t injected_tokens        = 0;
    bool control_pending                 = false;
    bool applied                         = false;
};

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

// The content channel withholds its trailing format-whitespace run (B1: the whitespace framing in
// front of a tool-call region is trimmed). The run is released unchanged when a non-whitespace
// byte follows, dropped when the region is accepted, and re-emitted verbatim when the region is
// demoted or the turn ends without one.
std::size_t visible_content_end(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return end;
}

void feed_content_channel(DecoderState& state, std::string_view text, const StopPolicy& policy,
                          PublishedOutput& emitted, std::uint32_t committed_tokens,
                          StopMatch* match) {
    if (text.empty() && state.content_whitespace_pending.empty()) { return; }
    std::string combined = std::move(state.content_whitespace_pending);
    state.content_whitespace_pending.clear();
    combined.append(text);
    const std::size_t end = visible_content_end(combined);
    feed_channel(state, OutputChannel::Content, std::string_view(combined).substr(0, end), policy,
                 emitted, committed_tokens, match);
    state.content_whitespace_pending.assign(combined, end, combined.size() - end);
}

void publish_content_whitespace(DecoderState& state, PublishedOutput& emitted) {
    append_delta(emitted, OutputChannel::Content, std::move(state.content_whitespace_pending));
    state.content_whitespace_pending.clear();
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning, ThinkingControlOptions thinking_,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::shared_ptr<const fi::ToolCallOutputContract> tool_call_output_)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          thinking_control_tokens(std::move(thinking_control_tokens_)),
          preserve_special(output.raw || output.preserve_special_tokens),
          raw_presentation(output.raw), split_reasoning(starts_in_reasoning && !output.raw),
          core(output.raw ? nullptr : std::move(tool_call_output_),
               fi::ChatParseOptions{.thinking_enabled     = starts_in_reasoning,
                                    .tool_name_max_length = output.tool_name_max_length}) {
        if (thinking_.budget && *thinking_.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        thinking.budget = thinking_.budget;
        // The prefix execution tracker observes the canonical close serialization only while the
        // prompt actually opened the reasoning block.
        prefix_execution.tracking = starts_in_reasoning;
    }

    // Feeds one decoded token's bytes through the parsing core and publishes the core's channel
    // deltas through the stop machinery. `match` is the round's shared best stop candidate; it is
    // null for control tokens, which never end the turn. Raw presentation publishes every byte as
    // content; the core still tracks the close boundary so the thinking budget follows R8.
    // Returns true when these bytes contributed to the content channel (raw mode counts all bytes).
    bool feed_token(std::string_view bytes, std::uint32_t committed_tokens, StopMatch* match) {
        const fi::ChatParseResult parsed = core.preview_feed(bytes);
        if (raw_presentation) {
            feed_channel(preview_state, OutputChannel::Content, bytes, policy, preview_output,
                         committed_tokens, match);
            return !bytes.empty();
        }
        feed_channel(preview_state, OutputChannel::Reasoning, parsed.reasoning_delta, policy,
                     preview_output, committed_tokens, match);
        feed_content_delta(parsed.content_delta, committed_tokens, match);
        // A token whose content bytes are entirely trailing format whitespace (for example the
        // "\n\n" before a tool-call region) is withheld, not content: it must not contribute a
        // logprob entry. Only a token with visible content of its own counts.
        return visible_content_end(parsed.content_delta) > 0;
    }

    void feed_content_delta(std::string_view text, std::uint32_t committed_tokens,
                            StopMatch* match) {
        if (raw_presentation) {
            feed_channel(preview_state, OutputChannel::Content, text, policy, preview_output,
                         committed_tokens, match);
            return;
        }
        feed_content_channel(preview_state, text, policy, preview_output, committed_tokens, match);
    }

    // Reassembles complete UTF-8 text from the token's decoded bytes (the decode authority) and
    // feeds it to the core.
    bool feed_token_bytes(std::string_view bytes, std::uint32_t committed_tokens,
                          StopMatch* match) {
        preview_state.utf8_pending.append(bytes);
        const std::string text = consume_generated_utf8(preview_state.utf8_pending);
        if (text.empty()) { return false; }
        const bool content = feed_token(text, committed_tokens, match);
        round_fed.append(text);
        return content;
    }

    // Restarts the active preview over exactly the first `bytes` of this round's fed bytes, so a
    // discarded token or a stop cut cannot leak withheld bytes into the terminal state.
    void rewind_core(std::size_t bytes) {
        core.begin_preview();
        (void)core.preview_feed(std::string_view(round_fed).substr(0, bytes));
    }

    // A token budget can end between byte-level tokens of one code point: publish the standard
    // replacement character rather than an invalid UTF-8 suffix.
    void flush_partial_utf8() {
        if (preview_state.utf8_pending.empty()) { return; }
        preview_state.utf8_pending.clear();
        const fi::ChatParseResult parsed = core.preview_feed(kUtf8Replacement);
        if (raw_presentation) {
            feed_content_delta(kUtf8Replacement, 0, nullptr);
            return;
        }
        feed_channel(preview_state, OutputChannel::Reasoning, parsed.reasoning_delta, policy,
                     preview_output, 0, nullptr);
        feed_content_delta(parsed.content_delta, 0, nullptr);
    }

    // Resolves the core as a terminal flush. With `merge_held` the withheld bytes publish through
    // the session's channels (limit and terminal paths, mirroring the old terminalize()); the
    // stop-string path already owns the cut prefix and only needs the terminal state (tool calls,
    // diagnostics).
    void flush_terminal(bool merge_held) {
        const bool ended_in_reasoning      = core.preview_in_reasoning();
        const fi::ChatParseResult terminal = core.preview_finish();
        if (!merge_held) { return; }
        if (raw_presentation) {
            close_channel(preview_state, OutputChannel::Content, preview_output);
            return;
        }
        // ADR-0002: a demoted tool-call region is published byte-exact but marked, so a streaming
        // consumer can withhold `text.substr(text_offset)` and signal the class instead of showing
        // the markup. Everything appended before the region in this commit stays ordinary content.
        const auto publish_terminal_content = [&](std::string text) {
            if (text.empty()) { return; }
            const std::size_t offset =
                !preview_output.empty() && preview_output.back().channel == OutputChannel::Content
                    ? preview_output.back().text.size()
                    : 0;
            append_delta(preview_output, OutputChannel::Content, std::move(text));
            if (terminal.diagnostics.fallback_reason != ninfer::ToolCallParseFallbackReason::None) {
                preview_output.back().demotion = ninfer::ToolCallDemotion{
                    .fallback_reason = terminal.diagnostics.fallback_reason,
                    .call_attempted  = terminal.diagnostics.call_attempted,
                    .text_offset     = offset,
                };
            }
        };
        if (ended_in_reasoning) {
            append_delta(preview_output, OutputChannel::Reasoning, terminal.reasoning_delta);
            publish_content_whitespace(preview_state, preview_output);
            close_channel(preview_state, OutputChannel::Reasoning, preview_output);
            publish_terminal_content(terminal.content_delta);
            return;
        }
        append_delta(preview_output, OutputChannel::Reasoning, terminal.reasoning_delta);
        close_channel(preview_state, OutputChannel::Content, preview_output);
        if (terminal.content_delta.empty() && !terminal.tool_calls.empty()) {
            // The tool-call region was accepted: the withheld whitespace is its framing.
            preview_state.content_whitespace_pending.clear();
        } else {
            publish_content_whitespace(preview_state, preview_output);
        }
        publish_terminal_content(terminal.content_delta);
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool preserve_special = false;
    bool raw_presentation = false;
    bool split_reasoning  = false;
    fi::ChatParseCore core;
    DecoderState state;
    DecoderState preview_state;
    ThinkingSessionState thinking;
    ThinkingSessionState preview_thinking;
    PrefixExecutionTracker prefix_execution;
    PrefixExecutionTracker preview_prefix_execution;
    std::optional<std::uint32_t> preview_execution_split_after;
    PublishedOutput preview_output;
    std::vector<TokenLogprob> preview_logprobs;
    std::vector<TokenLogprob> content_logprobs;
    std::string round_fed;
    bool preview_ready = false;
};

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(
    std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy, OutputOptions output,
    bool starts_in_reasoning, ThinkingControlOptions thinking,
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
    std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output)
    : impl_(std::make_unique<Impl>(
          std::move(tokenizer), std::move(policy), output, starts_in_reasoning, thinking,
          std::move(thinking_control_tokens), std::move(tool_call_output))) {}

runtime::OutputDecision
OutputSession::preview_model(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining,
                             FinishReason limit_reason,
                             std::span<const runtime::RawTokenLogprob> logprobs) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!logprobs.empty() && logprobs.size() != tokens.size()) {
        throw std::invalid_argument("logprob preview does not align with the token round");
    }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (impl_->thinking.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_thinking         = impl_->thinking;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    impl_->preview_logprobs.clear();
    impl_->round_fed.clear();
    impl_->core.begin_preview();

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        if (reason != FinishReason::None) { impl_->preview_thinking.control_pending = false; }
        if (impl_->preview_execution_split_after && *impl_->preview_execution_split_after > count) {
            throw std::logic_error("prefix execution split exceeds the accepted token prefix");
        }
        impl_->preview_ready = true;
        return runtime::OutputDecision{
            .accepted_tokens              = count,
            .finish_reason                = reason,
            .continuation                 = continuation,
            .prefix_execution_split_after = impl_->preview_execution_split_after,
        };
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count          = static_cast<std::uint32_t>(index + 1);
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);

        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = count;
        }

        // R8: the reasoning phase of the parsing core is the single close authority, so a quoted
        // `</think>` neither ends the channel nor the thinking budget.
        const bool in_reasoning = impl_->core.preview_in_reasoning();
        if (in_reasoning && impl_->split_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        if (impl_->preview_thinking.budget && in_reasoning) {
            ++impl_->preview_thinking.budget_thinking_tokens;
            if (impl_->preview_thinking.budget_thinking_tokens > *impl_->preview_thinking.budget) {
                throw std::logic_error("model output exceeded the licensed thinking budget");
            }
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        DecoderState before_state;
        PublishedOutput before_output;
        const std::size_t before_fed = impl_->round_fed.size();
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        StopMatch match;
        const std::string_view bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        const std::size_t logprobs_before = impl_->preview_logprobs.size();
        const bool contributes_content = impl_->feed_token_bytes(bytes, count, &match);
        if (contributes_content && !logprobs.empty()) {
            const runtime::RawTokenLogprob& raw = logprobs[index];
            TokenLogprob record;
            record.id         = raw.id;
            record.logprob    = raw.logprob;
            record.bytes.assign(bytes.data(), bytes.size());
            record.top_ids    = raw.top_ids;
            record.top_values = raw.top_values;
            for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
                // Masked/short rows leave empty slots as id -1: they have no token to decode.
                if (raw.top_ids[k] < 0) {
                    record.top_ids[k] = -1;
                    continue;
                }
                record.top_bytes[k] =
                    std::string(impl_->tokenizer->decoded_token(raw.top_ids[k]).bytes);
            }
            impl_->preview_logprobs.push_back(std::move(record));
        }

        if (match.found) {
            // The stop string owns the published prefix; the terminal flush only records the
            // terminal state over the accepted bytes.
            impl_->preview_logprobs.resize(logprobs_before);
            impl_->rewind_core(before_fed);
            impl_->flush_terminal(/*merge_held=*/false);
            impl_->preview_state = terminal_state(std::move(impl_->preview_state));
            impl_->preview_state.matched_stop_order = match.declaration_order;
            impl_->preview_output                   = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
                impl_->preview_logprobs.resize(logprobs_before);
                impl_->rewind_core(before_fed);
            }
            impl_->flush_terminal(/*merge_held=*/true);
            impl_->preview_state.terminal = true;
            return complete(count, FinishReason::StopToken);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        impl_->flush_partial_utf8();
        impl_->flush_terminal(/*merge_held=*/true);
        impl_->preview_state.terminal = true;
        return complete(count, limit_reason);
    }
    if (impl_->core.preview_in_reasoning() && impl_->preview_thinking.budget &&
        impl_->preview_thinking.budget_thinking_tokens == *impl_->preview_thinking.budget) {
        impl_->preview_thinking.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr || !impl_->thinking.budget || !impl_->core.in_reasoning() ||
        impl_->thinking.applied) {
        return total_budget_remaining;
    }
    if (impl_->thinking.control_pending ||
        impl_->thinking.budget_thinking_tokens >= *impl_->thinking.budget) {
        return 0;
    }
    return std::min(total_budget_remaining,
                    *impl_->thinking.budget - impl_->thinking.budget_thinking_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->thinking.control_pending || !impl_->thinking_control_tokens) {
        return {};
    }
    return *impl_->thinking_control_tokens;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || tokens.size() != expected.size() ||
        !std::equal(tokens.begin(), tokens.end(), expected.begin())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_thinking         = impl_->thinking;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    impl_->round_fed.clear();
    impl_->core.begin_preview();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = static_cast<std::uint32_t>(index + 1U);
        }
        if (impl_->core.preview_in_reasoning() && impl_->split_reasoning) {
            ++impl_->preview_state.reasoning_tokens;
        }
        const std::string_view presentation_bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        impl_->feed_token_bytes(presentation_bytes, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    // The canonical control span must close the reasoning phase under the core's boundary rule.
    if (impl_->core.preview_in_reasoning()) {
        throw std::logic_error("canonical thinking control did not close the reasoning channel");
    }
    impl_->preview_thinking.control_pending = false;
    impl_->preview_thinking.applied         = true;
    impl_->preview_thinking.injected_tokens = static_cast<std::uint32_t>(tokens.size());
    impl_->preview_ready                    = true;
    return runtime::OutputDecision{
        .accepted_tokens              = static_cast<std::uint32_t>(tokens.size()),
        .prefix_execution_split_after = impl_->preview_execution_split_after,
    };
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!impl_->thinking.budget || !impl_->core.in_reasoning() ||
        effective_output_tokens <= *impl_->thinking.budget) {
        return;
    }
    const std::uint64_t remaining =
        static_cast<std::uint64_t>(effective_output_tokens) - *impl_->thinking.budget;
    const std::uint64_t required =
        static_cast<std::uint64_t>(impl_->thinking_control_tokens->size()) + 1U;
    if (remaining < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control "
            "suffix and one post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state            = impl_->state;
    impl_->preview_thinking         = impl_->thinking;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_thinking.control_pending = false;
    impl_->preview_output.clear();
    impl_->round_fed.clear();
    impl_->core.begin_preview();
    impl_->flush_partial_utf8();
    impl_->flush_terminal(/*merge_held=*/true);
    impl_->preview_state.terminal = true;
    impl_->preview_ready          = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    swap(impl_->thinking, impl_->preview_thinking);
    swap(impl_->prefix_execution, impl_->preview_prefix_execution);
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->content_logprobs.insert(
        impl_->content_logprobs.end(), std::make_move_iterator(impl_->preview_logprobs.begin()),
        std::make_move_iterator(impl_->preview_logprobs.end()));
    impl_->preview_logprobs.clear();
    impl_->preview_ready = false;
    impl_->core.commit();
    return output;
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? impl_->core.take_tool_calls() : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->core.diagnostics() : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{
        .configured_budget      = impl_->thinking.budget,
        .budget_thinking_tokens = impl_->thinking.budget_thinking_tokens,
        .injected_tokens        = impl_->thinking.injected_tokens,
        .applied                = impl_->thinking.applied,
    };
}

std::span<const TokenLogprob> OutputSession::content_logprobs() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return std::span<const TokenLogprob>(impl_->content_logprobs);
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->state.matched_stop_order;
    if (index >= impl_->policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->policy.strings[index].text;
}

} // namespace ninfer::models::qwen3_5
