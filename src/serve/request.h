#pragma once

#include "product/media_acquire/source.h"

#include <ninfer/types.h>

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens = 8192;
};

enum class ContentKind {
    Text,
    Image,
    Video,
};

struct CacheBoundary {
    enum class Ttl : std::uint8_t {
        Default,
        FiveMinutes,
        OneHour,
    };

    ninfer::PromptCacheMarkerKind kind       = ninfer::PromptCacheMarkerKind::SharedStablePrefix;
    ninfer::SharedCandidateEvidence evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary;
    Ttl ttl                                  = Ttl::Default;

    [[nodiscard]] friend constexpr bool operator==(CacheBoundary, CacheBoundary) noexcept = default;
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original wire "type" string for diagnostics
    ninfer::product::media_acquire::Source source;
    ninfer::ImageResizePolicy image_resize_policy = ninfer::ImageResizePolicy::Downsize;
    std::optional<CacheBoundary> cache_boundary_after;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string input_schema_json;
    bool strict = false;
    std::optional<std::string> input_examples_json;
    std::optional<CacheBoundary> cache_boundary_after;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Constrained tool calling (upstream 41e50d0d): the serve layer shares the Engine's
// ToolChoice rather than a parallel enum. Selection (required/named) and cardinality
// (parallel) travel to the frontend boundary, where select_tool_call_contract resolves
// them against the declared tools.
using ToolChoiceMode = ninfer::ToolChoiceMode;
using ToolChoice     = ninfer::ToolChoice;

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart> content; // ordered parts; may be empty when wire content is empty
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id; // populated for role=tool
    // Optional protocol assertion for a tool result. Call-graph normalization verifies it against
    // the function identified by tool_call_id before the Engine sees the history.
    std::optional<std::string> tool_result_name;
    bool tool_result_is_error = false;
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
    std::optional<CacheBoundary> cache_boundary_after;
};

// Sampling overrides that have an executable Engine meaning. Protocol-only
// fields are normalized or rejected before this value is constructed.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<double> min_p;
    std::optional<int> top_k;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Protocol-level effort vocabulary. Each wire adapter accepts the values from
// its external contract; translation passes explicit values to the selected template.
enum class RequestedReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::optional<RequestedReasoningEffort>
parse_requested_reasoning_effort(std::string_view value) noexcept {
    if (value == "none") { return RequestedReasoningEffort::None; }
    if (value == "minimal") { return RequestedReasoningEffort::Minimal; }
    if (value == "low") { return RequestedReasoningEffort::Low; }
    if (value == "medium") { return RequestedReasoningEffort::Medium; }
    if (value == "high") { return RequestedReasoningEffort::High; }
    if (value == "xhigh") { return RequestedReasoningEffort::XHigh; }
    if (value == "max") { return RequestedReasoningEffort::Max; }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view
requested_reasoning_effort_name(RequestedReasoningEffort effort) noexcept {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return "none";
    case RequestedReasoningEffort::Minimal:
        return "minimal";
    case RequestedReasoningEffort::Low:
        return "low";
    case RequestedReasoningEffort::Medium:
        return "medium";
    case RequestedReasoningEffort::High:
        return "high";
    case RequestedReasoningEffort::XHigh:
        return "xhigh";
    case RequestedReasoningEffort::Max:
        return "max";
    }
    return {};
}

// Which request field produced GenerationRequest::constraint; a constrained-decoding failure maps
// back to that field's contract codes.
enum class ConstraintSource : std::uint8_t { None, Grammar, Choice, Regex, JsonSchema };

// The request field a constraint source is attributed to in error params. `grammar` is the
// llama.cpp-compatible GBNF field; choice and regex are the NInfer `structured_outputs` members.
[[nodiscard]] constexpr std::string_view
constraint_source_param(ConstraintSource source) noexcept {
    switch (source) {
    case ConstraintSource::Grammar:
        return "grammar";
    case ConstraintSource::Choice:
        return "structured_outputs.choice";
    case ConstraintSource::Regex:
        return "structured_outputs.regex";
    case ConstraintSource::JsonSchema:
        return "response_format";
    case ConstraintSource::None:
        break;
    }
    return "messages";
}

struct GenerationRequest {
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    bool stop_strings_apply_to_reasoning = false;
    int max_tokens                       = 0; // resolved budget; zero means immediate output limit
    std::optional<bool> enable_thinking;      // unset => use the server default
    std::optional<std::uint32_t> thinking_budget;
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
    // Opt-in per-token log probabilities for the response builders; `top_logprobs` is the requested
    // alternative count in [0,20] (the Engine always gathers 20).
    bool logprobs   = false;
    int top_logprobs = 0;
    ninfer::PromptContinuationMode continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    bool allow_engine_automatic_shared_prefixes = true;
    SamplingParams sampling;
    // Resolved output constraint for the answer stream: the client's GBNF text, JSON object or
    // schema, choice, or regex. The Engine compiles it at submit; a rejection is mapped back to
    // the originating field through constraint_source.
    std::optional<ninfer::OutputConstraint> constraint;
    ConstraintSource constraint_source = ConstraintSource::None;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None &&
               (!tool_choice.allowed_names || !tool_choice.allowed_names->empty());
    }

    // A selection, cardinality, or strict-declaration policy the Engine must enforce with
    // a tool grammar rather than best-effort parsing. Basic is the default policy, so any
    // declared tool set constrains the turn unless the choice is explicitly automatic.
    [[nodiscard]] bool constrains_tools() const noexcept {
        if (tools.empty() || constraint) { return false; }
        if (!uses_tools()) { return true; }
        if (tool_choice.mode == ToolChoiceMode::Required || !tool_choice.parallel ||
            tool_choice.allowed_names ||
            tool_choice.constraints == ninfer::ToolConstraintMode::Basic) {
            return true;
        }
        return std::any_of(tools.begin(), tools.end(),
                           [](const auto& tool) { return tool.strict; });
    }

    [[nodiscard]] std::size_t media_item_count() const noexcept {
        std::size_t count = 0;
        for (const ChatTurn& message : messages) {
            for (const ContentPart& part : message.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
            }
        }
        return count;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }
};

} // namespace ninfer::serve
