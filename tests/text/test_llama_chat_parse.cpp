// Smoke test for the vendored llama.cpp chat-parsing stack (third_party/llama-chat).
//
// Builds a Qwen3-Coder-shaped PEG parser for a froggeric-like XML tool contract and
// parses one canonical froggeric turn into a structured tool call; a malformed turn
// (undeclared function) must fail. Links only the vendored library — no FrInfer core,
// no CUDA. The upstream model for this test is tests/test-chat-peg-parser.cpp
// (hand-built qwen3-coder-shaped parsers) plus common/parsers/qwen3-coder.cpp, whose
// builder chain the parser below mirrors.

#include "chat-peg-parser.h"
#include "chat.h"
#include "json.h"
#include "parsers/parsers.h"
#include "peg-parser.h"

#include <cstdio>
#include <string>

using json = common_json;

static int g_failures = 0;

static void check(bool ok, const char* what, const std::string& detail = {}) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s%s%s\n", what, detail.empty() ? "" : ": ", detail.c_str());
    }
}

static std::string fmt_quoted(const std::string& s) { return "\"" + s + "\""; }

// The generation prefix froggeric appends after the assistant header; the parser is
// fed the text the model produces after it, prefixed by it (mirrors qwen3-coder.cpp).
static const char* GEN_PREFIX = "<|im_start|>assistant\n";

// Qwen3-Coder / Qwen3.5 / froggeric XML tool calls:
//   <tool_call>\n<function=NAME>\n<parameter=K>\nVALUE\n</parameter>\n</function>\n</tool_call>
// Required arguments are accepted in any order; optional arguments follow, in any order.
static common_peg_arena build_qwen3_coder_parser(const json& tools) {
    return build_chat_peg_parser([&](common_chat_peg_builder& p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        // <think>\n...\n</think>\n\n, closed either by </think> or by the tool-call opener
        auto reasoning = p.eps();
        reasoning      = p.optional("<think>" + p.space() +
                                    p.reasoning(p.until_one_of({"</think>", "<tool_call>"})) +
                                    (p.literal("</think>") | p.peek(p.literal("<tool_call>"))));

        auto arg_close  = p.tool_arg_close(p.literal("\n</parameter>\n"));
        auto arg_string = p.rule(
            "xml-arg-string", p.ac(p.tool_arg_string_value(p.until("\n</parameter>\n")) + arg_close,
                                   "\n</parameter>\n"));

        auto tool_choice = p.choice();
        foreach_function(tools, [&](const json& tool) {
            const auto& function = tool.at("function");
            std::string name     = function.at("name");

            std::vector<common_peg_parser> required_args;
            std::vector<common_peg_parser> optional_args;

            foreach_parameter(function, [&](const common_chat_schema_property& param,
                                            const common_chat_schema_document_ptr& doc) {
                auto rule_name = "tool-" + name + "-arg-" + param.name;
                auto arg_open =
                    p.tool_arg_open("<parameter=" + p.tool_arg_name(p.literal(param.name)) + ">\n");

                auto types     = param.schema->value_types();
                auto arg_value = p.eps();
                if (!types.has(common_chat_schema::TYPE_STRING)) {
                    arg_value = p.tool_arg_json_value(
                                    p.schema(p.json(), rule_name + "-schema", doc, *param.schema)) +
                                arg_close;
                } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
                    arg_value = arg_string;
                } else {
                    arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(arg_string)) | arg_string,
                                       "xml-arg-string");
                }

                auto arg_rule = p.rule(rule_name, p.tool_arg(arg_open + arg_value));
                (param.required ? required_args : optional_args).push_back(arg_rule);
            });

            auto args = p.permute("tool-" + name + "-args", required_args);
            if (!optional_args.empty()) { args = args + p.zero_or_more(p.choice(optional_args)); }

            auto func = p.tool(p.tool_open("<function=" + p.tool_name(p.literal(name)) + ">\n") +
                               p.tool_args(args) + p.tool_close(p.literal("</function>\n")));

            tool_choice |= p.rule("tool-" + name, func);
        });

        auto tool_call_body = tool_choice + "</tool_call>" + p.space();
        auto tool_call      = p.rule("tool-call", "<tool_call>\n" + tool_call_body);
        auto calls          = tool_call + p.zero_or_more(tool_call);
        auto tool_calls     = p.trigger_rule("tool-call-root", p.repeat(calls, 0, 1));

        return generation_prompt + (reasoning << p.content(p.until("<tool_call>")) << tool_calls) +
               p.end();
    });
}

static json make_tools() {
    return json::parse(R"([
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a location",
            "parameters": {"type": "object",
                "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["location"]}}},
        {"type": "function", "function": {
            "name": "search_docs",
            "description": "Search the project documentation",
            "parameters": {"type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]}}}
    ])");
}

// Canonical froggeric turn: reasoning, content, one XML tool call with a required and
// an optional argument.
static const char* CANONICAL_TURN = "<|im_start|>assistant\n"
                                    "<think>\n"
                                    "The user asks about the weather; I should call the tool.\n"
                                    "</think>\n"
                                    "\n"
                                    "Let me check the forecast."
                                    "<tool_call>\n"
                                    "<function=get_weather>\n"
                                    "<parameter=location>\n"
                                    "Paris\n"
                                    "</parameter>\n"
                                    "<parameter=unit>\n"
                                    "celsius\n"
                                    "</parameter>\n"
                                    "</function>\n"
                                    "</tool_call>";

// Same turn with an undeclared function name (only that differs): must not parse.
static const char* MALFORMED_TURN = "<|im_start|>assistant\n"
                                    "<think>\n"
                                    "The user asks about the weather; I should call the tool.\n"
                                    "</think>\n"
                                    "\n"
                                    "Let me check the forecast."
                                    "<tool_call>\n"
                                    "<function=delete_everything>\n"
                                    "<parameter=location>\n"
                                    "Paris\n"
                                    "</parameter>\n"
                                    "</function>\n"
                                    "</tool_call>";

static void check_tool_call(const common_chat_msg& msg) {
    check(msg.tool_calls.size() == 1, "one tool call",
          "got " + std::to_string(msg.tool_calls.size()));
    if (msg.tool_calls.size() != 1) { return; }
    const auto& call = msg.tool_calls[0];
    check(call.name == "get_weather", "tool name", fmt_quoted(call.name));
    check(call.arguments.find("Paris") != std::string::npos, "arguments carry location",
          fmt_quoted(call.arguments));
    check(call.arguments.find("celsius") != std::string::npos, "arguments carry unit",
          fmt_quoted(call.arguments));

    // Arguments are assembled as a JSON object; verify they are structured, not raw text.
    try {
        auto args = json::parse(call.arguments);
        check(args.is_object(), "arguments are a JSON object", fmt_quoted(call.arguments));
        check(args.value("location", std::string()) == "Paris", "location value",
              fmt_quoted(args.value("location", std::string())));
        check(args.value("unit", std::string()) == "celsius", "unit value",
              fmt_quoted(args.value("unit", std::string())));
    } catch (const std::exception& e) {
        check(false, "arguments parse as JSON",
              std::string(e.what()) + " in " + fmt_quoted(call.arguments));
    }
}

int main() {
    const json tools = make_tools();
    const auto arena = build_qwen3_coder_parser(tools);

    // 1. Canonical turn through the raw engine (strict flags): full consumption required.
    {
        common_peg_parse_context ctx(CANONICAL_TURN, COMMON_PEG_PARSE_FLAG_NONE);
        const auto result = arena.parse(ctx);
        check(!result.fail(), "canonical turn parses");
        if (!result.fail()) {
            common_chat_msg msg;
            common_chat_peg_mapper mapper(msg);
            mapper.from_ast(ctx.ast, result);
            check(msg.reasoning_content ==
                      "The user asks about the weather; I should call the tool.\n",
                  "reasoning content", fmt_quoted(msg.reasoning_content));
            check(msg.content == "Let me check the forecast.", "content", fmt_quoted(msg.content));
            check_tool_call(msg);
        }
    }

    // 2. Same turn through the vendored chat entry point (common_chat_parse route).
    {
        common_chat_parser_params params;
        params.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
        auto msg      = common_chat_peg_parse(arena, CANONICAL_TURN, /*is_partial=*/false, params);
        check_tool_call(msg);
    }

    // 3. The vendored Qwen3-Coder handler (common/parsers/qwen3-coder.cpp) builds a
    //    working arena through the template route, including argument schemas and
    //    grammar emission.
    {
        common_chat_template tmpl(
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}", "", "");

        autoparser::generation_params inputs;
        inputs.messages    = json::parse(R"([{"role": "user", "content": "weather in Paris?"}])");
        inputs.tools       = tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

        const auto params = common_chat_params_init_qwen3_coder(tmpl, inputs);
        check(params.format == COMMON_CHAT_FORMAT_PEG_NATIVE, "handler format");
        check(!params.parser.empty(), "handler serialized a parser");
        check(!params.grammar.empty(), "handler emitted a grammar");

        auto handler_arena = common_peg_arena::from_json(json::parse(params.parser));
        const auto msg =
            common_chat_peg_parse(handler_arena, CANONICAL_TURN, /*is_partial=*/false, params);
        check_tool_call(msg);
    }

    // 4. Malformed turn (undeclared function): the strict parse must fail.
    {
        common_peg_parse_context ctx(MALFORMED_TURN, COMMON_PEG_PARSE_FLAG_NONE);
        const auto result = arena.parse(ctx);
        check(result.fail(), "malformed turn fails to parse");
    }

    // 5. Control: the same turn with a declared function name parses, so the failure
    //    above is the name and not an unrelated malformation.
    {
        std::string declared(MALFORMED_TURN);
        const auto name_pos = declared.find("delete_everything");
        declared.replace(name_pos, std::string("delete_everything").size(), "get_weather");
        common_peg_parse_context ctx(declared, COMMON_PEG_PARSE_FLAG_NONE);
        const auto result = arena.parse(ctx);
        check(!result.fail(), "declared-name control parses");
        if (!result.fail()) {
            common_chat_msg msg;
            common_chat_peg_mapper mapper(msg);
            mapper.from_ast(ctx.ast, result);
            check(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "get_weather",
                  "control yielded the declared tool call");
        }
    }

    if (g_failures == 0) {
        std::printf("llama-chat parse smoke: OK\n");
        return 0;
    }
    std::fprintf(stderr, "llama-chat parse smoke: %d failure(s)\n", g_failures);
    return 1;
}
