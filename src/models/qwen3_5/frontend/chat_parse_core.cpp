#include "models/qwen3_5/frontend/chat_parse_core.h"

#include "chat-peg-parser.h"
#include "peg-parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

// Rule names of the ported region arena; the mapper is the AST walker below, not the vendored
// semantic mapper, because normalization follows the frontend contract (B6) rather than the
// vendored qwen3-coder handler.
constexpr const char* kRuleToolCall        = "qwen3-5-tool-call";
constexpr const char* kRuleFunction        = "qwen3-5-function";
constexpr const char* kRuleFunctionName    = "qwen3-5-function-name";
constexpr const char* kRuleParameter       = "qwen3-5-parameter";
constexpr const char* kRuleParameterName   = "qwen3-5-parameter-name";
constexpr const char* kRuleParameterValue  = "qwen3-5-parameter-value";
constexpr const char* kRuleNestedParameter = "qwen3-5-nested-parameter";

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

// Longest suffix of `text` that is a prefix of `marker`; with `allow_complete = false` the whole
// marker does not count (a complete marker at the end of the available bytes is withheld by the
// callers explicitly, because the next byte decides how it reads).
std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

void publish(std::string& channel, std::string_view text) {
    if (!text.empty()) { channel.append(text); }
}

// Structural region grammar over the wire format. It carries the wire's own strictness where the
// corpus pins it and tolerance where the corpus grants it:
//   - `<tool_call>`, `<function=NAME>`, `<parameter=NAME>` markers with format whitespace between
//     structural parts (B3);
//   - a missing `</tool_call>` before the next `<tool_call>` is tolerated (B3, row 15);
//   - a parameter value is raw bytes with balanced nested `<parameter=...>...</parameter>` markup;
//     any `<parameter=` occurrence that does not open a well-formed, balanced nested parameter
//     makes the value unrepresentable (B4, rows 21/22);
//   - nothing after the last call except format whitespace: the root requires the end of input
//     (B2), which is how a candidate "consumes the response to its end".
common_peg_arena build_region_arena(const ChatParseWireFormat& format) {
    const std::string parameter_open(format.parameter_open);
    const std::string parameter_close(format.parameter_close);
    const std::string function_open(format.function_open);
    const std::string function_close(format.function_close);
    const std::string tool_open(format.tool_call_open);
    const std::string tool_close(format.tool_call_close);
    return build_chat_peg_parser([&](common_chat_peg_builder& p) {
        auto format_ws      = p.chars("[ \t\r\n]", 0, -1);
        auto function_name  = p.rule(kRuleFunctionName, p.chars("[^<>]", 1, -1));
        auto parameter_name = p.rule(kRuleParameterName, p.chars("[^<>]", 1, -1));
        auto value          = p.rule(kRuleParameterValue, [&]() {
            auto nested = p.rule(kRuleNestedParameter, p.literal(parameter_open) + parameter_name +
                                                                    ">" + p.ref(kRuleParameterValue) +
                                                                    p.literal(parameter_close));
            auto text   = p.one_or_more(
                p.negate(p.literal(parameter_open) | p.literal(parameter_close)) + p.any());
            return p.zero_or_more(nested | text);
        });
        auto parameter = p.rule(kRuleParameter, p.literal(parameter_open) + parameter_name + ">" +
                                                    value + p.literal(parameter_close));
        auto function =
            p.rule(kRuleFunction, p.literal(function_open) + function_name + ">" + format_ws +
                                      p.zero_or_more(format_ws + parameter) + format_ws +
                                      p.literal(function_close));
        auto call = p.rule(kRuleToolCall, p.literal(tool_open) + format_ws + function + format_ws +
                                              p.optional(p.literal(tool_close)));
        return format_ws + p.optional(call + p.zero_or_more(format_ws + call)) + format_ws +
               p.end();
    });
}

const common_peg_arena& region_arena() {
    static const common_peg_arena arena = build_region_arena(ChatParseWireFormat::qwen3_5());
    return arena;
}

struct RawParameter {
    std::string name;
    std::string value;
};

struct RawToolCall {
    std::string name;
    std::vector<RawParameter> parameters;
};

// Walks the region AST: direct children of the root are calls, direct children of a call are its
// function, and direct children of a parameter are its name and raw value bytes. Nested parameter
// markup lives inside the value node and is deliberately left there.
bool collect_calls(const common_peg_ast_arena& ast, const common_peg_parse_result& result,
                   std::vector<RawToolCall>& calls) {
    calls.clear();
    for (const common_peg_ast_id call_id : result.nodes) {
        const common_peg_ast_node& call_node = ast.get(call_id);
        if (call_node.rule != kRuleToolCall) { continue; }

        RawToolCall call;
        for (const common_peg_ast_id function_id : call_node.children) {
            const common_peg_ast_node& function_node = ast.get(function_id);
            if (function_node.rule != kRuleFunction) { continue; }
            for (const common_peg_ast_id member_id : function_node.children) {
                const common_peg_ast_node& member = ast.get(member_id);
                if (member.rule == kRuleFunctionName) {
                    call.name.assign(member.text);
                } else if (member.rule == kRuleParameter) {
                    RawParameter parameter;
                    for (const common_peg_ast_id argument_id : member.children) {
                        const common_peg_ast_node& argument = ast.get(argument_id);
                        if (argument.rule == kRuleParameterName) {
                            parameter.name.assign(argument.text);
                        } else if (argument.rule == kRuleParameterValue) {
                            parameter.value.assign(argument.text);
                        }
                    }
                    call.parameters.push_back(std::move(parameter));
                }
            }
        }
        calls.push_back(std::move(call));
    }
    return !calls.empty();
}

// Parses one candidate region: it must parse and consume the region to its end.
FallbackReason parse_candidate_region(std::string_view region, std::vector<RawToolCall>& calls) {
    common_peg_parse_context context(std::string(region), COMMON_PEG_PARSE_FLAG_NONE);
    const common_peg_parse_result result = region_arena().parse(context);
    if (!result.success() || result.end != context.input.size()) {
        return FallbackReason::MalformedStructure;
    }
    if (!collect_calls(context.ast, result, calls)) { return FallbackReason::MalformedStructure; }
    return FallbackReason::None;
}

// ---------------------------------------------------------------- value normalization (B6)
//
// Ported from the bespoke frontend parser (tool_call_parser.cpp at the port baseline), which
// documents and implements these rules in docs/serving.md. The frontend integration replaces that
// parser with this core, so the rules move here unchanged.

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
               (byte >= '0' && byte <= '9') || byte == '_' || byte == '-';
    });
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && number[pos] >= '0' && number[pos] <= '9') { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && number[pos] >= '0' && number[pos] <= '9') { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && number[pos] >= '0' && number[pos] <= '9') {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || (value.front() >= '0' && value.front() <= '9')) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

std::string encode_json_string(std::string_view value) { return Json(std::string(value)).dump(); }

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
};

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(framed)};
    }

    const std::string_view value = trim_format_whitespace(framed);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (classify_json_value(value, kind)) {
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::string(value)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(framed)};
}

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(value)};
}

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);
    if (tool != nullptr && !tool->unambiguous) { tool = nullptr; }

    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized = normalize_parameter(raw_parameter.value, parameter);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = raw.name, .arguments_json = std::move(arguments)};
}

// ---------------------------------------------------------------- channel state machine (R1-R8)
//
// The reasoning boundary and hold rules need bytes the PEG does not track, so they live here as a
// small byte-level machine over the ported region parser. Only bytes whose interpretation is
// decided are published; everything else stays in `held` until the next round or the terminal
// flush resolves it.

struct ParseContext {
    const ChatParseWireFormat* format = nullptr;
    bool tools_enabled                = false;
};

struct ParseState {
    enum class Phase : std::uint8_t { Reasoning, Content, ToolRegion };

    Phase phase                = Phase::Content;
    bool at_turn_start         = false;
    bool strip_content_leading = false;
    bool finished              = false;
    std::string held;
    std::string reasoning;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

void resolve_tool_region(ParseState& state, const ParseContext& context, const Contract& contract,
                         std::size_t max_name_length) {
    const std::string_view region(state.held);
    const std::string_view tool_open = context.format->tool_call_open;

    FallbackReason first_failure = FallbackReason::MalformedStructure;
    bool failure_recorded        = false;
    std::size_t accepted         = std::string_view::npos;
    std::vector<RawToolCall> accepted_calls;

    for (std::size_t candidate = region.find(tool_open); candidate != std::string_view::npos;
         candidate             = region.find(tool_open, candidate + 1)) {
        std::vector<RawToolCall> calls;
        FallbackReason failure = parse_candidate_region(region.substr(candidate), calls);
        if (failure == FallbackReason::None) {
            // Structure first, then the declared-name policy, then the parameters (B5).
            for (const RawToolCall& call : calls) {
                if (!valid_function_name(call.name, max_name_length)) {
                    failure = FallbackReason::InvalidToolName;
                    break;
                }
            }
            if (failure == FallbackReason::None && contract.enforce_declared_names) {
                for (const RawToolCall& call : calls) {
                    if (find_tool_contract(contract, call.name) == nullptr) {
                        failure = FallbackReason::UndeclaredTool;
                        break;
                    }
                }
            }
            if (failure == FallbackReason::None) {
                for (const RawToolCall& call : calls) {
                    for (std::size_t i = 0; i < call.parameters.size(); ++i) {
                        for (std::size_t j = i + 1; j < call.parameters.size(); ++j) {
                            if (call.parameters[i].name == call.parameters[j].name) {
                                failure = FallbackReason::DuplicateParameter;
                                break;
                            }
                        }
                        if (failure != FallbackReason::None) { break; }
                    }
                    if (failure != FallbackReason::None) { break; }
                }
            }
        }
        if (failure == FallbackReason::None) {
            accepted       = candidate;
            accepted_calls = std::move(calls);
            break;
        }
        if (!failure_recorded) {
            first_failure    = failure;
            failure_recorded = true;
        }
    }

    state.diagnostics.marker_seen = true;
    if (accepted == std::string_view::npos) {
        // No candidate qualifies: the whole region is ordinary content (B2).
        publish(state.content, region);
        state.diagnostics.fallback_reason = first_failure;
        return;
    }

    publish(state.content, rtrim_format_whitespace(region.substr(0, accepted)));
    state.diagnostics.structured_call_count = static_cast<std::uint32_t>(accepted_calls.size());
    state.tool_calls.reserve(state.tool_calls.size() + accepted_calls.size());
    for (const RawToolCall& raw : accepted_calls) {
        state.tool_calls.push_back(normalize_raw_tool_call(raw, contract, state.diagnostics));
    }
}

void feed_content(ParseState& state, const ParseContext& context, std::string_view text) {
    state.held.append(text);
    const std::string_view tool_open = context.format->tool_call_open;

    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < state.held.size() && is_format_whitespace(state.held[begin])) { ++begin; }
        state.held.erase(0, begin);
        if (!state.held.empty()) { state.strip_content_leading = false; }
    }
    if (state.held.empty()) { return; }

    if (!context.tools_enabled) {
        publish(state.content, state.held);
        state.held.clear();
        return;
    }

    const std::size_t marker = state.held.find(tool_open);
    if (marker != std::string::npos) {
        // The format-whitespace run before the marker is withheld with the region and re-emitted
        // only if the region is demoted (B1).
        std::size_t start = marker;
        while (start != 0 && is_format_whitespace(state.held[start - 1])) { --start; }
        publish(state.content, std::string_view(state.held).substr(0, start));
        state.held.erase(0, start);
        state.phase                   = ParseState::Phase::ToolRegion;
        state.diagnostics.marker_seen = true;
        return;
    }

    const std::size_t hold = longest_suffix_prefix(state.held, tool_open, false);
    const std::size_t safe = state.held.size() - hold;
    publish(state.content, std::string_view(state.held).substr(0, safe));
    state.held.erase(0, safe);
}

void feed_reasoning(ParseState& state, const ParseContext& context, std::string_view text) {
    state.held.append(text);
    const ChatParseWireFormat& format     = *context.format;
    const std::string_view thinking_close = format.thinking_close;
    const std::string_view tool_open      = format.tool_call_open;
    std::size_t begin                     = 0;

    if (state.at_turn_start) {
        const std::string_view buffer(state.held);
        if (buffer.size() < format.thinking_open.size()) {
            if (format.thinking_open.substr(0, buffer.size()) != buffer) {
                state.at_turn_start = false;
            } else {
                return; // a partial leading opener stays held
            }
        } else if (buffer.substr(0, format.thinking_open.size()) == format.thinking_open) {
            begin               = format.thinking_open.size();
            state.at_turn_start = false;
        } else {
            state.at_turn_start = false;
        }
    }

    const std::string_view buffer(state.held);
    // R2: close at the first `</think>` followed by format whitespace. A marker followed by any
    // other byte is quoted protocol text and stays reasoning (R3).
    std::size_t close = std::string_view::npos;
    for (std::size_t marker = buffer.find(thinking_close, begin); marker != std::string_view::npos;
         marker             = buffer.find(thinking_close, marker + thinking_close.size())) {
        const std::size_t after = marker + thinking_close.size();
        if (after >= buffer.size()) { break; } // held until the next byte decides (R4)
        if (is_format_whitespace(buffer[after])) {
            close = marker;
            break;
        }
    }
    // B1: a tool-call opener closes the reasoning channel and starts the structured region.
    const std::size_t tool =
        context.tools_enabled ? buffer.find(tool_open, begin) : std::string_view::npos;

    if (tool != std::string_view::npos && (close == std::string_view::npos || tool < close)) {
        publish(state.reasoning, buffer.substr(begin, tool - begin));
        state.held.erase(0, tool);
        state.phase                   = ParseState::Phase::ToolRegion;
        state.diagnostics.marker_seen = true;
        return;
    }
    if (close != std::string_view::npos) {
        publish(state.reasoning, buffer.substr(begin, close - begin));
        state.held.erase(0, close + thinking_close.size());
        state.phase                 = ParseState::Phase::Content;
        state.strip_content_leading = true;
        feed_content(state, context, {});
        return;
    }

    std::size_t hold = longest_suffix_prefix(buffer.substr(begin), thinking_close, false);
    if (buffer.size() >= thinking_close.size() &&
        buffer.compare(buffer.size() - thinking_close.size(), thinking_close.size(),
                       thinking_close) == 0) {
        hold = std::max(hold, thinking_close.size());
    }
    if (context.tools_enabled) {
        hold = std::max(hold, longest_suffix_prefix(buffer.substr(begin), tool_open, false));
    }
    const std::size_t safe = buffer.size() - hold;
    publish(state.reasoning, buffer.substr(begin, safe - begin));
    state.held.erase(0, safe);
}

void feed(ParseState& state, const ParseContext& context, std::string_view text) {
    if (text.empty() && state.held.empty()) { return; }
    switch (state.phase) {
    case ParseState::Phase::Reasoning:
        feed_reasoning(state, context, text);
        break;
    case ParseState::Phase::Content:
        feed_content(state, context, text);
        break;
    case ParseState::Phase::ToolRegion:
        state.held.append(text);
        break;
    }
}

void terminalize(ParseState& state, const ParseContext& context, const Contract* contract,
                 std::size_t max_name_length) {
    switch (state.phase) {
    case ParseState::Phase::Reasoning: {
        // R5: a close marker still held at the end of the turn is the model's implicit close; it
        // is consumed, not published, and the bytes before it are reasoning.
        const std::string_view held         = state.held;
        const std::string_view close_marker = context.format->thinking_close;
        std::size_t close                   = std::string_view::npos;
        if (held.size() >= close_marker.size() &&
            held.compare(held.size() - close_marker.size(), close_marker.size(), close_marker) ==
                0) {
            close = held.size() - close_marker.size();
        }
        publish(state.reasoning, held.substr(0, close));
        break;
    }
    case ParseState::Phase::Content:
        // R2: whitespace directly after the close is never published as content.
        if (!state.strip_content_leading) { publish(state.content, state.held); }
        break;
    case ParseState::Phase::ToolRegion:
        if (contract != nullptr) {
            resolve_tool_region(state, context, *contract, max_name_length);
        }
        break;
    }
    state.held.clear();
    state.finished = true;
}

} // namespace

const ChatParseWireFormat& ChatParseWireFormat::qwen3_5() noexcept {
    static const ChatParseWireFormat format{
        .thinking_open   = "<think>",
        .thinking_close  = "</think>",
        .tool_call_open  = "<tool_call>",
        .tool_call_close = "</tool_call>",
        .function_open   = "<function=",
        .function_close  = "</function>",
        .parameter_open  = "<parameter=",
        .parameter_close = "</parameter>",
    };
    return format;
}

class ChatParseCore::Impl {
public:
    Impl(std::shared_ptr<const ToolCallOutputContract> contract, ChatParseOptions options)
        : contract_(std::move(contract)), options_(options) {
        committed_.phase =
            options.thinking_enabled ? ParseState::Phase::Reasoning : ParseState::Phase::Content;
        committed_.at_turn_start = options.thinking_enabled;
    }

    [[nodiscard]] ParseContext context() const {
        return ParseContext{.format        = &ChatParseWireFormat::qwen3_5(),
                            .tools_enabled = contract_ != nullptr};
    }

    std::shared_ptr<const ToolCallOutputContract> contract_;
    ChatParseOptions options_;
    ParseState committed_;
    ParseState preview_;
    bool preview_ready_ = false;
};

ChatParseCore::ChatParseCore(std::shared_ptr<const ToolCallOutputContract> contract,
                             ChatParseOptions options)
    : impl_(std::make_unique<Impl>(std::move(contract), options)) {}

ChatParseCore::~ChatParseCore()                                   = default;
ChatParseCore::ChatParseCore(ChatParseCore&&) noexcept            = default;
ChatParseCore& ChatParseCore::operator=(ChatParseCore&&) noexcept = default;

void ChatParseCore::warm_up() { (void)region_arena(); }

void ChatParseCore::begin_preview() {
    Impl& impl = *impl_;
    if (impl.committed_.finished) { throw std::logic_error("chat parse core is already terminal"); }
    impl.preview_       = impl.committed_;
    impl.preview_ready_ = true;
}

ChatParseResult ChatParseCore::preview_feed(std::string_view text) {
    Impl& impl = *impl_;
    if (!impl.preview_ready_) { throw std::logic_error("chat parse core has no active preview"); }
    const std::size_t reasoning_before = impl.preview_.reasoning.size();
    const std::size_t content_before   = impl.preview_.content.size();
    feed(impl.preview_, impl.context(), text);

    ChatParseResult result;
    result.reasoning_delta = impl.preview_.reasoning.substr(reasoning_before);
    result.content_delta   = impl.preview_.content.substr(content_before);
    result.diagnostics     = impl.preview_.diagnostics;
    return result;
}

ChatParseResult ChatParseCore::preview_finish() {
    Impl& impl = *impl_;
    if (!impl.preview_ready_) { throw std::logic_error("chat parse core has no active preview"); }
    if (impl.preview_.finished) { return {}; }
    const std::size_t reasoning_before = impl.preview_.reasoning.size();
    const std::size_t content_before   = impl.preview_.content.size();
    terminalize(impl.preview_, impl.context(), impl.contract_.get(),
                impl.options_.tool_name_max_length);

    ChatParseResult result;
    result.reasoning_delta = impl.preview_.reasoning.substr(reasoning_before);
    result.content_delta   = impl.preview_.content.substr(content_before);
    result.terminal        = true;
    result.tool_calls      = impl.preview_.tool_calls;
    result.diagnostics     = impl.preview_.diagnostics;
    return result;
}

bool ChatParseCore::preview_in_reasoning() const noexcept {
    return impl_->preview_ready_ && impl_->preview_.phase == ParseState::Phase::Reasoning;
}

bool ChatParseCore::in_reasoning() const noexcept {
    return impl_->committed_.phase == ParseState::Phase::Reasoning;
}

ChatParseResult ChatParseCore::preview(std::string_view round_text) {
    begin_preview();
    return preview_feed(round_text);
}

ChatParseResult ChatParseCore::preview_terminal() {
    begin_preview();
    return preview_finish();
}

void ChatParseCore::commit() {
    Impl& impl = *impl_;
    if (!impl.preview_ready_) {
        throw std::logic_error("chat parse core has no preview to commit");
    }
    impl.committed_     = std::move(impl.preview_);
    impl.preview_       = {};
    impl.preview_ready_ = false;
}

std::string_view ChatParseCore::reasoning() const noexcept { return impl_->committed_.reasoning; }

std::string_view ChatParseCore::content() const noexcept { return impl_->committed_.content; }

std::string_view ChatParseCore::held() const noexcept { return impl_->committed_.held; }

bool ChatParseCore::finished() const noexcept { return impl_->committed_.finished; }

const std::vector<GeneratedToolCall>& ChatParseCore::tool_calls() const noexcept {
    return impl_->committed_.tool_calls;
}

std::vector<GeneratedToolCall> ChatParseCore::take_tool_calls() noexcept {
    return std::move(impl_->committed_.tool_calls);
}

ToolCallParseDiagnostics ChatParseCore::diagnostics() const noexcept {
    return impl_->committed_.diagnostics;
}

} // namespace ninfer::models::qwen3_5::frontend
