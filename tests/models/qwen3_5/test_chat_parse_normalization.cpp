// Focused declared-schema normalization cases for the ported chat parsing core. The wire-format
// oracle corpus (tests/models/qwen3_5/test_chat_parsing_corpus.cpp) is the semantic authority for
// the Qwen3.5/froggeric markup; these cases carry the typing rules over from the replaced bespoke
// parser test: declared primitive types, case-insensitive booleans with CRLF framing, exact
// integer lexemes, composed anyOf/oneOf unions, legacy fallback for unsupported or ambiguous
// schemas, and the function-name length bound.
#include "models/qwen3_5/frontend/chat_parse_core.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer::models::qwen3_5::frontend;
using ninfer::GeneratedToolCall;
using ninfer::ToolCallParseDiagnostics;
using ninfer::ToolCallParseFallbackReason;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

Json prop(const char* type) { return Json{{"type", type}}; }

Json prop_types(std::initializer_list<const char*> types) {
    Json list = Json::array();
    for (const char* type : types) { list.push_back(type); }
    return Json{{"type", list}};
}

Json any_of(std::initializer_list<Json> members) {
    Json list = Json::array();
    for (const Json& member : members) { list.push_back(member); }
    return Json{{"anyOf", list}};
}

Json one_of(std::initializer_list<Json> members) {
    Json list = Json::array();
    for (const Json& member : members) { list.push_back(member); }
    return Json{{"oneOf", list}};
}

std::string tool_definition(const std::string& name, const Json& properties) {
    return Json{{"type", "function"},
                {"function", Json{{"name", name},
                                  {"parameters", Json{{"type", "object"},
                                                      {"properties", properties},
                                                      {"required", Json::array()}}}}}}
        .dump();
}

std::string tool_call(const std::string& name,
                      const std::vector<std::pair<std::string, std::string>>& arguments = {}) {
    std::string text = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& [parameter, value] : arguments) {
        text += "<parameter=" + parameter + ">\n" + value + "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

struct Parsed {
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostics;
};

Parsed parse(const std::vector<std::string>& definitions, const std::string& text,
             std::size_t max_name_length = 64, bool declared = true) {
    std::shared_ptr<const ToolCallOutputContract> contract;
    if (declared) {
        contract = build_tool_call_output_contract(definitions, true);
    } else {
        auto open_contract                    = std::make_shared<ToolCallOutputContract>();
        open_contract->enforce_declared_names = false;
        contract                              = std::move(open_contract);
    }
    ChatParseCore core(contract, ChatParseOptions{.thinking_enabled     = false,
                                                  .tool_name_max_length = max_name_length});
    (void)core.preview(text);
    core.commit();
    (void)core.preview_terminal();
    core.commit();
    return Parsed{.calls = core.tool_calls(), .diagnostics = core.diagnostics()};
}

Json arguments(const GeneratedToolCall& call) { return Json::parse(call.arguments_json); }

int declared_json_types() {
    const Json properties =
        Json{{"count", prop("integer")},  {"total", prop("number")},
             {"ratio", prop("number")},   {"enabled", prop("boolean")},
             {"payload", prop("object")}, {"items", prop("array")},
             {"unset", prop("null")},     {"optional", prop_types({"integer", "null"})}};
    const auto parsed = parse({tool_definition("configure", properties)},
                              tool_call("configure", {{"count", "7"},
                                                      {"total", "8"},
                                                      {"ratio", "1.5"},
                                                      {"enabled", "true"},
                                                      {"payload", "{\"x\":1}"},
                                                      {"items", "[\"a\",2]"},
                                                      {"unset", "null"},
                                                      {"optional", "null"}}));
    int failures      = check(parsed.calls.size() == 1, "declared JSON values were rejected");
    if (parsed.calls.size() != 1) { return failures; }
    const Json args = arguments(parsed.calls.front());
    failures += check(args.at("count") == 7 && args.at("total") == 8 && args.at("ratio") == 1.5,
                      "declared numeric types were not decoded");
    failures += check(args.at("enabled") == true, "declared boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "declared object was not decoded");
    failures += check(args.at("items").is_array() && args.at("items").at(1) == 2,
                      "declared array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null (direct and unioned) was not decoded");
    return failures;
}

int boolean_boundary() {
    const Json properties = Json{{"lower", prop("boolean")},   {"title", prop("boolean")},
                                 {"upper", prop("boolean")},   {"mixed", prop("boolean")},
                                 {"spaced", prop("boolean")},  {"windows", prop("boolean")},
                                 {"rejected", prop("boolean")}};
    const auto parsed     = parse({tool_definition("configure", properties)},
                                  "<tool_call>\n<function=configure>\n"
                                      "<parameter=lower>\ntrue\n</parameter>\n"
                                      "<parameter=title>\nTrue\n</parameter>\n"
                                      "<parameter=upper>\nTRUE\n</parameter>\n"
                                      "<parameter=mixed>\nfAlSe\n</parameter>\n"
                                      "<parameter=spaced>\n \tTrUe \n</parameter>\n"
                                      "<parameter=windows>\r\nFaLsE\r\n</parameter>\n"
                                      "<parameter=rejected>\nyes\n</parameter>\n"
                                      "</function>\n</tool_call>");
    int failures = check(parsed.calls.size() == 1, "case-insensitive booleans were rejected");
    if (parsed.calls.size() != 1) { return failures; }
    const Json args = arguments(parsed.calls.front());
    failures += check(args.at("lower") == true && args.at("title") == true &&
                          args.at("upper") == true && args.at("spaced") == true,
                      "boolean true variants were not canonicalized");
    failures += check(args.at("mixed") == false && args.at("windows") == false,
                      "boolean false variants were not canonicalized");
    failures +=
        check(args.at("rejected") == "yes" && parsed.diagnostics.schema_mismatch_arguments == 1,
              "non-boolean text did not stay a mismatching string");
    return failures;
}

int exact_integer_boundary() {
    const Json properties = Json{{"decimal", prop("integer")},
                                 {"exponent", prop("integer")},
                                 {"scaled", prop("integer")},
                                 {"negative_zero", prop("integer")},
                                 {"large", prop("integer")}};
    const auto integers   = parse({tool_definition("configure", properties)},
                                  tool_call("configure", {{"decimal", "7.0"},
                                                          {"exponent", "1e2"},
                                                          {"scaled", "100e-2"},
                                                          {"negative_zero", "-0.0"},
                                                          {"large", "9007199254740992.0"}}));
    int failures =
        check(integers.calls.size() == 1, "mathematically integral JSON numbers were rejected");
    if (integers.calls.size() == 1) {
        failures += check(integers.calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto fractional = parse({tool_definition("configure", Json{{"value", prop("integer")}})},
                                  tool_call("configure", {{"value", "9007199254740992.5"}}));
    failures +=
        check(fractional.calls.size() == 1 &&
                  fractional.calls.front().arguments_json == "{\"value\":9007199254740992.5}" &&
                  fractional.diagnostics.schema_mismatch_arguments == 1,
              "fractional integer mismatch lost its exact lexeme");

    const auto numbers = parse({tool_definition("configure", Json{{"value", prop("number")}})},
                               tool_call("configure", {{"value", "9007199254740992.5"}}));
    failures += check(numbers.calls.size() == 1 &&
                          numbers.calls.front().arguments_json == "{\"value\":9007199254740992.5}",
                      "declared number lost its original precision");
    return failures;
}

int composed_schema_types() {
    const Json properties =
        Json{{"flag", any_of({prop("boolean"), prop("null")})},
             {"unset", one_of({prop("null"), prop("boolean")})},
             {"count", any_of({prop("integer"), prop("null")})},
             {"nested", any_of({one_of({prop("boolean"), prop("null")}), prop("integer")})},
             {"string_or_number", one_of({prop("string"), prop("number")})}};
    const auto parsed = parse({tool_definition("configure", properties)},
                              tool_call("configure", {{"flag", "False"},
                                                      {"unset", "null"},
                                                      {"count", "7.0"},
                                                      {"nested", "TRUE"},
                                                      {"string_or_number", "7"}}));
    int failures      = check(parsed.calls.size() == 1, "explicit anyOf/oneOf union was rejected");
    if (parsed.calls.size() != 1) { return failures; }
    const Json args = arguments(parsed.calls.front());
    failures += check(args.at("flag") == false && args.at("unset").is_null(),
                      "nullable boolean composition was decoded incorrectly");
    failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                      "nested primitive composition was decoded incorrectly");
    failures += check(args.at("string_or_number") == "7",
                      "string-admitting composition did not preserve text");

    const auto fractional = parse(
        {tool_definition("configure", Json{{"count", any_of({prop("integer"), prop("null")})}})},
        tool_call("configure", {{"count", "7.5"}}));
    failures += check(fractional.calls.size() == 1 &&
                          fractional.calls.front().arguments_json == "{\"count\":7.5}" &&
                          fractional.diagnostics.schema_mismatch_arguments == 1,
                      "fractional integer union mismatch was not structured");
    return failures;
}

int legacy_fallbacks() {
    const Json unsupported_properties =
        Json{{"missing_type", Json::object()},
             {"alias", prop("int")},
             {"invalid_type_array", prop_types({"integer", "int"})},
             {"partial_anyof", any_of({prop("integer"), Json{{"enum", Json::array({1, 2})}}})},
             {"mixed_composition", Json{{"anyOf", Json::array({prop("boolean")})},
                                        {"oneOf", Json::array({prop("null")})}}}};
    const auto unsupported = parse({tool_definition("configure", unsupported_properties)},
                                   tool_call("configure", {{"missing_type", "7"},
                                                           {"alias", "8"},
                                                           {"invalid_type_array", "9"},
                                                           {"partial_anyof", "7.5"},
                                                           {"mixed_composition", "True"},
                                                           {"undeclared", "{\"x\":1}"}}));
    int failures           = check(unsupported.calls.size() == 1 &&
                                       unsupported.diagnostics.schema_mismatch_arguments == 1,
                                   "unsupported schema did not retain the legacy policy");
    if (unsupported.calls.size() == 1) {
        const Json args = arguments(unsupported.calls.front());
        failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                              args.at("invalid_type_array") == 9,
                          "legacy numeric inference changed");
        failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                          "unsupported composition was partially inferred");
        failures += check(args.at("undeclared").at("x") == 1,
                          "undeclared parameter legacy inference changed");
    }

    const std::string integer_definition =
        tool_definition("configure", Json{{"value", prop("integer")}});
    const std::string string_definition =
        tool_definition("configure", Json{{"value", prop("string")}});
    const auto identical =
        parse({integer_definition, integer_definition}, tool_call("configure", {{"value", "7"}}));
    failures +=
        check(identical.calls.size() == 1, "identical duplicate tool contracts became ambiguous");
    const auto conflicting =
        parse({integer_definition, string_definition}, tool_call("configure", {{"value", "7"}}));
    failures += check(conflicting.calls.size() == 1 &&
                          conflicting.calls.front().arguments_json == "{\"value\":7}" &&
                          conflicting.diagnostics.schema_mismatch_arguments == 0,
                      "conflicting duplicate tool contracts did not use legacy normalization");
    return failures;
}

int name_length_bound() {
    const std::string name(128, 'a');
    const auto wide   = parse({}, tool_call(name), 128, /*declared=*/false);
    const auto narrow = parse({}, tool_call(name), 64, /*declared=*/false);
    const auto over   = parse({}, tool_call(std::string(129, 'a')), 128, /*declared=*/false);
    int failures      = check(wide.calls.size() == 1, "128-character tool name was rejected");
    failures += check(narrow.calls.empty() && narrow.diagnostics.marker_seen &&
                          narrow.diagnostics.fallback_reason ==
                              ToolCallParseFallbackReason::InvalidToolName,
                      "128-character tool name passed the 64-character presentation bound");
    failures += check(over.calls.empty(), "129-character tool name was accepted");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += declared_json_types();
    failures += boolean_boundary();
    failures += exact_integer_boundary();
    failures += composed_schema_types();
    failures += legacy_fallbacks();
    failures += name_length_bound();
    if (failures == 0) { std::fprintf(stderr, "chat parse normalization: OK\n"); }
    return failures == 0 ? 0 : 1;
}
