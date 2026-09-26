// Driver for the chat-parsing oracle corpus (tests/fixtures/chat_parsing/corpus.json).
//
// The corpus is the acceptance contract for the ported Qwen3.5/froggeric parsing core
// (models/qwen3_5/frontend/chat_parse_core.h): every vector drives the core round by round and
// the driver asserts the cumulative channels, the held bytes, the terminal tool calls and the
// terminal diagnostics. Expectations are never re-derived from the core.

#include "models/qwen3_5/frontend/chat_parse_core.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nlohmann::ordered_json;
using namespace ninfer::models::qwen3_5::frontend;
using ninfer::GeneratedToolCall;
using ninfer::ToolCallParseDiagnostics;
using ninfer::ToolCallParseFallbackReason;
using ninfer::tool_call_parse_fallback_reason_name;

constexpr std::size_t kExpectedVectorCount = 40;

int g_failures = 0;

std::string quote(std::string_view text) {
    std::string out = "\"";
    for (const char byte : text) {
        switch (byte) {
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        default:
            out.push_back(byte);
            break;
        }
    }
    out += "\"";
    return out;
}

void check(bool condition, const std::string& context, const std::string& detail = {}) {
    if (condition) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s%s%s\n", context.c_str(), detail.empty() ? "" : ": ",
                 detail.c_str());
}

bool check_channel(std::string_view actual, const ordered_json& expected,
                   const std::string& context, std::string_view channel,
                   const std::string& vector_id) {
    const std::string expected_text = expected.get<std::string>();
    if (actual == expected_text) { return true; }
    check(false, vector_id + ": " + std::string(channel),
          "expected " + quote(expected_text) + ", got " + quote(actual));
    return false;
}

// The fixture's diagnostics vocabulary: every listed field is asserted; unlisted fields are not
// pinned by the corpus.
void check_diagnostics(const ToolCallParseDiagnostics& actual, const ordered_json& expected,
                       const std::string& vector_id) {
    for (const auto& [key, value] : expected.items()) {
        const std::string context = vector_id + ": diagnostics." + key;
        if (key == "marker_seen") {
            check(actual.marker_seen == value.get<bool>(), context);
        } else if (key == "structured_call_count") {
            check(actual.structured_call_count == value.get<std::uint32_t>(), context);
        } else if (key == "empty_arguments_omitted") {
            check(actual.empty_arguments_omitted == value.get<std::uint32_t>(), context);
        } else if (key == "schema_mismatch_arguments") {
            check(actual.schema_mismatch_arguments == value.get<std::uint32_t>(), context);
        } else if (key == "fallback_reason") {
            const std::string name             = value.get<std::string>();
            ToolCallParseFallbackReason reason = ToolCallParseFallbackReason::None;
            if (name == "None") {
                reason = ToolCallParseFallbackReason::None;
            } else if (name == "MalformedStructure") {
                reason = ToolCallParseFallbackReason::MalformedStructure;
            } else if (name == "DuplicateParameter") {
                reason = ToolCallParseFallbackReason::DuplicateParameter;
            } else if (name == "InvalidToolName") {
                reason = ToolCallParseFallbackReason::InvalidToolName;
            } else if (name == "UndeclaredTool") {
                reason = ToolCallParseFallbackReason::UndeclaredTool;
            } else if (name == "TrailingContent") {
                reason = ToolCallParseFallbackReason::TrailingContent;
            } else {
                check(false, context, "unknown fallback reason " + quote(name));
                continue;
            }
            check(actual.fallback_reason == reason, context,
                  std::string("expected ") + tool_call_parse_fallback_reason_name(reason) +
                      ", got " + tool_call_parse_fallback_reason_name(actual.fallback_reason));
        } else {
            check(false, context, "unasserted diagnostics field in the fixture");
        }
    }
}

bool check_tool_calls(const std::vector<GeneratedToolCall>& actual, const ordered_json& expected,
                      const std::string& vector_id) {
    if (actual.size() != expected.size()) {
        check(false, vector_id + ": tool call count",
              "expected " + std::to_string(expected.size()) + ", got " +
                  std::to_string(actual.size()));
        return false;
    }
    bool ok = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const std::string call_context = vector_id + ": tool call " + std::to_string(index);
        const ordered_json& want       = expected[index];
        if (actual[index].name != want.at("name").get<std::string>()) {
            check(false, call_context + " name",
                  "expected " + quote(want.at("name").get<std::string>()) + ", got " +
                      quote(actual[index].name));
            ok = false;
        }
        const ordered_json& want_arguments = want.at("arguments");
        const ordered_json actual_arguments =
            ordered_json::parse(actual[index].arguments_json, nullptr, false);
        if (actual_arguments.is_discarded() || actual_arguments.dump() != want_arguments.dump()) {
            check(false, call_context + " arguments",
                  "expected " + want_arguments.dump() + ", got " + actual[index].arguments_json);
            ok = false;
        }
    }
    return ok;
}

std::shared_ptr<const ToolCallOutputContract>
contract_for(const ordered_json& vector, const std::map<std::string, ordered_json>& tools,
             const std::string& vector_id) {
    std::vector<std::string> definitions;
    for (const auto& name : vector.at("tools")) {
        const auto tool = tools.find(name.get<std::string>());
        if (tool == tools.end()) {
            check(false, vector_id + ": undeclared fixture tool", name.get<std::string>());
            return nullptr;
        }
        definitions.push_back(tool->second.dump());
    }
    return build_tool_call_output_contract(definitions, true);
}

void run_vector(const ordered_json& vector, const std::map<std::string, ordered_json>& tools) {
    const std::string vector_id = vector.at("id").get<std::string>();
    auto contract               = contract_for(vector, tools, vector_id);
    if (contract == nullptr) { return; }

    ChatParseOptions options;
    options.thinking_enabled     = vector.value("thinking", false);
    options.tool_name_max_length = 64;
    ChatParseCore core(std::move(contract), options);

    const ordered_json& rounds = vector.at("rounds");
    if (rounds.empty()) {
        check(false, vector_id + ": empty rounds");
        return;
    }

    std::string reasoning;
    std::string content;
    for (std::size_t index = 0; index < rounds.size(); ++index) {
        const ordered_json& round = rounds[index];
        const bool last           = index + 1 == rounds.size();
        const std::string feed    = round.at("feed").get<std::string>();

        if (round.contains("control")) {
            std::fprintf(stderr,
                         "note: %s round %zu carries a frontend control expectation (deferred "
                         "to the frontend integration)\n",
                         vector_id.c_str(), index);
        }

        // Preview purity: a preview that is never committed leaves the committed state intact.
        const std::string committed_reasoning(core.reasoning());
        const std::string committed_content(core.content());
        const std::string committed_held(core.held());
        (void)core.preview(feed);
        check(core.reasoning() == committed_reasoning && core.content() == committed_content &&
                  core.held() == committed_held,
              vector_id + ": discarded preview changed the committed state",
              "at round " + std::to_string(index));

        const ChatParseResult fed = core.preview(feed);
        core.commit();
        reasoning += fed.reasoning_delta;
        content += fed.content_delta;

        if (last) {
            // A terminal round reports its held bytes before the terminal flush resolves them,
            // then the resolved channels, tool calls and diagnostics.
            if (round.contains("held")) {
                check_channel(core.held(), round.at("held"), vector_id, "held", vector_id);
            }
            const std::string pre_terminal_reasoning(core.reasoning());
            const std::string pre_terminal_content(core.content());
            (void)core.preview_terminal();
            check(core.reasoning() == pre_terminal_reasoning &&
                      core.content() == pre_terminal_content && !core.finished(),
                  vector_id + ": discarded terminal preview changed the committed state");
            const ChatParseResult terminal = core.preview_terminal();
            core.commit();
            reasoning += terminal.reasoning_delta;
            content += terminal.content_delta;
            if (round.contains("tool_calls")) {
                check_tool_calls(core.tool_calls(), round.at("tool_calls"), vector_id);
            }
        } else if (round.contains("held")) {
            check_channel(core.held(), round.at("held"), vector_id, "held", vector_id);
        }

        if (round.contains("reasoning")) {
            check_channel(reasoning, round.at("reasoning"), vector_id, "reasoning", vector_id);
        }
        if (round.contains("content")) {
            check_channel(content, round.at("content"), vector_id, "content", vector_id);
        }
    }

    const ordered_json& final = vector.at("final");
    check_channel(reasoning, final.at("reasoning"), vector_id, "final reasoning", vector_id);
    check_channel(content, final.at("content"), vector_id, "final content", vector_id);
    check_tool_calls(core.tool_calls(), final.at("tool_calls"), vector_id);
    if (final.contains("diagnostics")) {
        check_diagnostics(core.diagnostics(), final.at("diagnostics"), vector_id);
    }
    check(core.finished(), vector_id + ": terminal state");
}

} // namespace

int main() {
    const std::string path =
        std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/chat_parsing/corpus.json";
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot open corpus fixture %s\n", path.c_str());
        return 1;
    }
    const ordered_json corpus = ordered_json::parse(input);

    std::map<std::string, ordered_json> tools;
    for (const auto& tool : corpus.at("tools")) {
        tools.emplace(tool.at("function").at("name").get<std::string>(), tool);
    }

    const ordered_json& vectors = corpus.at("vectors");
    check(vectors.size() == kExpectedVectorCount, "fixture vector count",
          "expected " + std::to_string(kExpectedVectorCount) + ", got " +
              std::to_string(vectors.size()));

    for (const auto& vector : vectors) { run_vector(vector, tools); }

    if (g_failures == 0) {
        std::printf("chat parsing corpus: %zu vectors OK\n", vectors.size());
        return 0;
    }
    std::fprintf(stderr, "chat parsing corpus: %d failure(s)\n", g_failures);
    return 1;
}
