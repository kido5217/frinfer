#include "models/qwen3_5/frontend/tool_grammar.h"
#include "models/qwen3_5/frontend/chat_parse_core.h"
#include "text/json_schema.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;
using Builder  = text::ModelGrammarBuilder;
using Rule     = Builder::Rule;

// The non-strict tool arguments surface: any object of string values.
constexpr std::string_view kNonStrictArgumentsSchema =
    R"({"type":"object","additionalProperties":{"type":"string"}})";

// Adopts one tool's argument schema, attributing any construction failure to the tool declaration.
Rule arguments(Builder& builder, const Contract::Tool& tool) {
    try {
        const std::string source = tool.strict
                                       ? text::prepare_json_schema(tool.schema_json)
                                       : std::string(kNonStrictArgumentsSchema);
        return builder.json_schema_rule(source);
    } catch (const text::ModelGrammarError& error) {
        RequestErrorKind kind = RequestErrorKind::InvalidJsonSchema;
        switch (error.kind()) {
        case text::ModelGrammarError::Kind::UnsupportedSchema:
            kind = RequestErrorKind::UnsupportedJsonSchema;
            break;
        case text::ModelGrammarError::Kind::UnsatisfiableSchema:
            kind = RequestErrorKind::UnsatisfiableJsonSchema;
            break;
        case text::ModelGrammarError::Kind::Fatal:
            kind = RequestErrorKind::InvalidToolConstraint;
            break;
        case text::ModelGrammarError::Kind::InvalidSchema:
            break;
        }
        throw RequestError(kind, "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           "/" + std::to_string(tool.declaration_index) + "/parameters" +
                               error.pointer(),
                           RequestErrorSource::Tools);
    }
}

text::ModelGrammar build(const Contract& contract) {
    const ChatParseWireFormat& wire = ChatParseWireFormat::qwen3_5();
    const std::string tool_call_open(wire.tool_call_open);
    const std::string function_open(wire.function_open);
    const std::string function_close(wire.function_close);
    const std::string tool_call_close(wire.tool_call_close);
    Builder builder;
    if (contract.tools.empty()) {
        return builder.get(
            builder.rule("root", builder.tag_dispatch({}, false, {tool_call_open})));
    }
    std::vector<Rule> choices;
    std::optional<Rule> basic_arguments;
    for (const auto& tool : contract.tools) {
        const Rule body =
            !tool.strict && basic_arguments ? *basic_arguments : arguments(builder, tool);
        if (!tool.strict) { basic_arguments = body; }
        choices.push_back(builder.sequence(
            {builder.literal("\n" + function_open + tool.name + ">\n"), builder.reference(body),
             builder.literal(function_close + "\n" + tool_call_close)}));
    }
    const Rule call_body = builder.rule("call_body", builder.choice(choices));
    const Rule next_call =
        builder.sequence({builder.literal("\n" + tool_call_open), builder.reference(call_body)});
    const Rule tail =
        contract.parallel ? builder.repeat("calls", next_call, 0, -1) : builder.empty();
    const Rule first =
        builder.rule("calls", builder.sequence({builder.reference(call_body), tail}));
    const Rule root =
        contract.required
            ? builder.sequence({builder.literal(tool_call_open), builder.reference(first)})
            : builder.tag_dispatch({{tool_call_open, first}}, false, {});
    return builder.normalize(builder.rule("root", root));
}
} // namespace

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const Contract& contract,
                     std::string_view reasoning_close, std::string_view continuation) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel}};
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"] = Json::parse(tool.schema_json);
            // Failed compilations are cached too, including their declaration error location.
            entry["source_index"] = tool.declaration_index;
        }
        identity["qwen_tools"].push_back(std::move(entry));
    }
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract); }, reasoning_close, continuation);
}
} // namespace ninfer::models::qwen3_5::frontend
