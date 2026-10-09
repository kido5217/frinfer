// Focused test for the protocol-neutral constraint admission module: the shared
// `structured_outputs` extension, the top-level GBNF field, the JSON Schema contract, and the
// cross-language rejection factories.

#include "product/constraint/constraint_contract.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ninfer::constraint::ConstraintError;
using ninfer::constraint::ConstraintOrigin;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

struct Failure {
    bool thrown = false;
    std::string code;
    std::string param;
    std::string message;
};

template <typename Function>
Failure capture(Function&& function) {
    try {
        function();
    } catch (const ConstraintError& error) {
        return Failure{true, error.code(), error.param(), error.what()};
    }
    return {};
}

int test_structured_outputs() {
    int failures = 0;

    const auto choice = ninfer::constraint::admit_structured_outputs(
        Json{{"choice", Json::array({"positive", "negative"})}});
    failures += check(choice.origin == ConstraintOrigin::Choice &&
                          choice.field == "structured_outputs.choice" &&
                          choice.value.kind == ninfer::OutputConstraintKind::Choice &&
                          choice.value.choices ==
                              std::vector<std::string>({"positive", "negative"}),
                      "structured_outputs.choice is admitted as a choice constraint");

    const auto regex = ninfer::constraint::admit_structured_outputs(Json{{"regex", "[0-9]+"}});
    failures += check(regex.origin == ConstraintOrigin::Regex &&
                          regex.value.kind == ninfer::OutputConstraintKind::Regex,
                      "structured_outputs.regex is admitted as a regex constraint");

    const auto grammar = ninfer::constraint::admit_structured_outputs(
        Json{{"grammar", "root ::= \"a\""}});
    failures += check(grammar.origin == ConstraintOrigin::Grammar &&
                          grammar.value.kind == ninfer::OutputConstraintKind::Grammar,
                      "structured_outputs.grammar is admitted as a grammar constraint");

    failures += check(ninfer::constraint::structured_outputs_option(Json{{"regex", "x"}}) ==
                          "regex",
                      "structured_outputs_option returns the selected option");

    const std::vector<std::pair<Json, Failure>> malformed = {
        {Json{{"json", Json{{"type", "object"}}}},
         {true, "structured_outputs_invalid", "structured_outputs.json", {}}},
        {Json{{"grammar", "root ::= \"a\""}, {"regex", "a"}},
         {true, "structured_outputs_invalid", "structured_outputs", {}}},
        {Json{{"choice", Json::array()}}, {true, "invalid_choice", "structured_outputs.choice", {}}},
        {Json{{"choice", Json::array({"a", 5})}},
         {true, "invalid_choice", "structured_outputs.choice/1", {}}},
        {Json{{"regex", 5}}, {true, "invalid_regex", "structured_outputs.regex", {}}},
        {Json{{"grammar", ""}}, {true, "grammar_invalid", "structured_outputs.grammar", {}}},
    };
    for (const auto& [value, expected] : malformed) {
        const Failure failure =
            capture([&] { (void)ninfer::constraint::admit_structured_outputs(value); });
        failures += check(failure.thrown && failure.code == expected.code &&
                              failure.param == expected.param,
                          "structured_outputs malformed shape: " + value.dump() + " -> " +
                              failure.code);
    }

    const Failure not_object =
        capture([&] { (void)ninfer::constraint::admit_structured_outputs(Json(5)); });
    failures += check(not_object.thrown && not_object.code == "structured_outputs_invalid" &&
                          not_object.param == "structured_outputs",
                      "a non-object structured_outputs is rejected");

    const Json oversized{{"grammar",
                          std::string(ninfer::constraint::kConstraintPayloadLimit + 1, 'x')}};
    const Failure too_large =
        capture([&] { (void)ninfer::constraint::admit_structured_outputs(oversized); });
    failures += check(too_large.thrown && too_large.code == "constraint_too_large" &&
                          too_large.param == "structured_outputs.grammar",
                      "an oversized structured_outputs.grammar is rejected");
    return failures;
}

int test_grammar_and_object() {
    int failures = 0;

    failures += check(!ninfer::constraint::admit_grammar(Json(""), "grammar").has_value(),
                      "an empty top-level grammar carries no constraint");

    const auto grammar = ninfer::constraint::admit_grammar(Json("root ::= \"a\""), "grammar");
    failures += check(grammar.has_value() && grammar->origin == ConstraintOrigin::Grammar &&
                          grammar->value.source == "root ::= \"a\"",
                      "a top-level grammar is admitted as a grammar constraint");

    const Failure non_string =
        capture([&] { (void)ninfer::constraint::admit_grammar(Json(5), "grammar"); });
    failures += check(non_string.thrown && non_string.code == "grammar_invalid" &&
                          non_string.param == "grammar",
                      "a non-string top-level grammar is rejected");

    const Failure oversized = capture([&] {
        (void)ninfer::constraint::admit_grammar(
            Json(std::string(ninfer::constraint::kConstraintPayloadLimit + 1, 'x')), "grammar");
    });
    failures += check(oversized.thrown && oversized.code == "constraint_too_large" &&
                          oversized.param == "grammar",
                      "an oversized top-level grammar is rejected");

    const auto object = ninfer::constraint::admit_json_object("response_format");
    failures += check(object.origin == ConstraintOrigin::JsonObject &&
                          object.value.kind == ninfer::OutputConstraintKind::JsonObject &&
                          object.field == "response_format",
                      "a json_object format is admitted as an object constraint");
    return failures;
}

int test_json_schema() {
    int failures = 0;
    const Json schema = Json{{"type", "object"}, {"additionalProperties", false}};

    const auto admitted =
        ninfer::constraint::admit_json_schema(schema, "text.format", "text.format");
    failures += check(admitted.origin == ConstraintOrigin::JsonSchema &&
                          admitted.field == "text.format" &&
                          admitted.value.kind == ninfer::OutputConstraintKind::JsonSchema,
                      "a JSON Schema document is admitted as a schema constraint");

    const Json oversized =
        Json{{"type", "string"},
             {"const", std::string(ninfer::constraint::kConstraintPayloadLimit, 'x')}};

    const Failure remapped = capture([&] {
        (void)ninfer::constraint::admit_json_schema(oversized, "text.format", "text.format");
    });
    failures += check(remapped.thrown && remapped.code == "constraint_too_large" &&
                          remapped.param == "text.format",
                      "a contract error is reported on the route's error param");

    const Failure contract_param = capture([&] {
        (void)ninfer::constraint::admit_json_schema(oversized, "response_format", "");
    });
    failures += check(contract_param.thrown && contract_param.code == "constraint_too_large" &&
                          contract_param.param == "response_format",
                      "an empty error_param keeps the contract's own path");

    const Failure unsupported = capture([&] {
        (void)ninfer::constraint::admit_json_schema(Json{{"type", "string"}, {"format", "email"}},
                                                    "output_config.format",
                                                    "output_config.format");
    });
    failures += check(unsupported.thrown && unsupported.code == "json_schema_unsupported" &&
                          unsupported.param == "output_config.format",
                      "an unenforced schema keyword is rejected on the route's field");
    return failures;
}

int test_rejections() {
    int failures = 0;

    const Failure second = capture(
        [] { ninfer::constraint::reject_second_constraint("structured_outputs"); });
    failures += check(second.thrown && second.code == "constrained_decoding_conflict" &&
                          second.param == "structured_outputs",
                      "the single-constraint rejection carries the route's param");

    const Failure tools = capture([] {
        ninfer::constraint::reject_constraint_with_tools("structured_outputs.grammar");
    });
    failures += check(tools.thrown && tools.code == "constrained_decoding_not_supported" &&
                          tools.param == "structured_outputs.grammar",
                      "the tools-conflict rejection carries the route's param");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_structured_outputs();
    failures += test_grammar_and_object();
    failures += test_json_schema();
    failures += test_rejections();
    if (failures == 0) { std::cout << "Constraint admission tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
