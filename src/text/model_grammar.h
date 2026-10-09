#pragma once

// Vendor-neutral model-output grammar construction. A model expresses its own output framing
// through ModelGrammarBuilder and hands the finished grammar to GrammarCompiler::compile_model;
// the vendored builder API stays behind this seam (src/text/model_grammar.cpp).

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::text {

// A classified schema/grammar-construction failure from the vendored engine, before any protocol
// attribution. The caller maps `kind` onto its own taxonomy and prefixes `pointer`.
class ModelGrammarError : public std::runtime_error {
public:
    enum class Kind : std::uint8_t {
        InvalidSchema,
        UnsupportedSchema,
        UnsatisfiableSchema,
        Fatal,
    };

    ModelGrammarError(Kind kind, std::string message, std::string pointer = {})
        : std::runtime_error(std::move(message)), kind_(kind), pointer_(std::move(pointer)) {}

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] const std::string& pointer() const noexcept { return pointer_; }

private:
    Kind kind_;
    std::string pointer_;
};

// An opaque model-output grammar produced by ModelGrammarBuilder and consumed by
// GrammarCompiler::compile_model.
class ModelGrammar {
public:
    ModelGrammar();
    ~ModelGrammar();
    ModelGrammar(const ModelGrammar&)            = delete;
    ModelGrammar& operator=(const ModelGrammar&) = delete;
    ModelGrammar(ModelGrammar&&) noexcept;
    ModelGrammar& operator=(ModelGrammar&&) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class ModelGrammarBuilder;
    friend class GrammarCompiler;
};

// A small builder vocabulary over the vendored grammar builder, so a model can express its own
// output framing without naming the vendored API. `Rule` handles are opaque ids the builder hands
// out and accepts.
class ModelGrammarBuilder {
public:
    using Rule = std::int32_t;

    ModelGrammarBuilder();
    ~ModelGrammarBuilder();
    ModelGrammarBuilder(const ModelGrammarBuilder&)            = delete;
    ModelGrammarBuilder& operator=(const ModelGrammarBuilder&) = delete;
    ModelGrammarBuilder(ModelGrammarBuilder&&) noexcept;
    ModelGrammarBuilder& operator=(ModelGrammarBuilder&&) noexcept;

    // Converts a prepared JSON Schema document into a model arguments sub-grammar (the Qwen XML
    // value framing) and adopts it, returning its rule id. Throws ModelGrammarError.
    [[nodiscard]] Rule json_schema_rule(std::string_view schema_source);

    [[nodiscard]] Rule literal(std::string_view bytes);
    [[nodiscard]] Rule reference(Rule named_rule);
    [[nodiscard]] Rule sequence(std::initializer_list<Rule> parts);
    [[nodiscard]] Rule sequence(std::span<const Rule> parts);
    [[nodiscard]] Rule choice(std::initializer_list<Rule> alternatives);
    [[nodiscard]] Rule choice(std::span<const Rule> alternatives);
    [[nodiscard]] Rule repeat(std::string_view name_hint, Rule part, std::int32_t minimum,
                              std::int32_t maximum);
    [[nodiscard]] Rule empty();
    // Adds a named rule (a hint; the builder may suffix it) and returns its rule id.
    [[nodiscard]] Rule rule(std::string_view name_hint, Rule body);
    [[nodiscard]] Rule tag_dispatch(std::initializer_list<std::pair<std::string, Rule>> tags,
                                    bool loop, std::initializer_list<std::string> excludes);

    // Finishes the composition. `normalize` expands nested rules; `get` leaves the AST as built.
    [[nodiscard]] ModelGrammar normalize(Rule root_rule);
    [[nodiscard]] ModelGrammar get(Rule root_rule);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
