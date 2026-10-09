// Focused test for the vendor-neutral model-grammar builder facade: composition primitives and the
// classified construction errors the model frontend maps onto its own taxonomy.

#include "text/model_grammar.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {

using ninfer::text::ModelGrammarBuilder;
using ninfer::text::ModelGrammarError;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

// Runs json_schema_rule and returns the classified error kind, or `built` when it succeeded.
struct SchemaOutcome {
    bool built = false;
    ModelGrammarError::Kind kind{};
};

SchemaOutcome build_schema(std::string_view schema) {
    ModelGrammarBuilder builder;
    try {
        (void)builder.json_schema_rule(schema);
    } catch (const ModelGrammarError& error) {
        return {false, error.kind()};
    }
    return {true, {}};
}

} // namespace

int main() {
    int failures = 0;

    failures += check(
        build_schema(R"({"type":"object","properties":{"a":{"type":"string"}}})").built,
        "a supported schema builds a model grammar rule");

    const SchemaOutcome invalid = build_schema(R"({"type":5})");
    failures += check(!invalid.built && invalid.kind == ModelGrammarError::Kind::InvalidSchema,
                      "a malformed schema is classified InvalidSchema");

    // The builder vocabulary composes without the vendored API.
    {
        ModelGrammarBuilder builder;
        const auto mid  = builder.rule("mid", builder.choice({builder.literal("b"), builder.empty()}));
        const auto root = builder.rule(
            "root", builder.sequence({builder.literal("a"), builder.reference(mid)}));
        (void)builder.get(root);
    }

    {
        ModelGrammarBuilder builder;
        const auto calls = builder.repeat("calls", builder.literal("x"), 0, -1);
        const auto root =
            builder.rule("root", builder.tag_dispatch({{"<tool_call>", calls}}, false, {}));
        (void)builder.normalize(root);
    }

    if (failures == 0) { std::cout << "Model grammar facade tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
