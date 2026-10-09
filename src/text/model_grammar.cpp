#include "text/model_grammar.h"
#include "text/model_grammar_internal.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "json_schema_converter.h"
#include "support/logging.h"

#include <optional>
#include <utility>

namespace ninfer::text {

ModelGrammar::ModelGrammar() : impl_(std::make_unique<Impl>()) {}

ModelGrammar::~ModelGrammar() = default;

ModelGrammar::ModelGrammar(ModelGrammar&&) noexcept            = default;

ModelGrammar& ModelGrammar::operator=(ModelGrammar&&) noexcept = default;

class ModelGrammarBuilder::Impl {
public:
    xgrammar::GrammarBuilder builder;
};

ModelGrammarBuilder::ModelGrammarBuilder() : impl_(std::make_unique<Impl>()) {}

ModelGrammarBuilder::~ModelGrammarBuilder() = default;

ModelGrammarBuilder::ModelGrammarBuilder(ModelGrammarBuilder&&) noexcept            = default;

ModelGrammarBuilder& ModelGrammarBuilder::operator=(ModelGrammarBuilder&&) noexcept = default;

ModelGrammarBuilder::Rule ModelGrammarBuilder::json_schema_rule(std::string_view schema_source) {
    try {
        xgrammar::Grammar arguments = xgrammar::JSONSchemaToGrammar(
            std::string(schema_source), false, std::nullopt,
            std::pair<std::string, std::string>{", ", ": "}, false, std::nullopt, false,
            xgrammar::JSONFormat::kQwenXML, {});
        return xgrammar::SubGrammarAdder::Apply(&impl_->builder, arguments);
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        auto kind = ModelGrammarError::Kind::InvalidSchema;
        if (error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema) {
            kind = ModelGrammarError::Kind::UnsupportedSchema;
        } else if (error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema) {
            kind = ModelGrammarError::Kind::UnsatisfiableSchema;
        }
        throw ModelGrammarError(kind, error.what(), error.pointer);
    } catch (const xgrammar::LogFatalError& error) {
        throw ModelGrammarError(ModelGrammarError::Kind::Fatal, error.what());
    }
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::literal(std::string_view bytes) {
    return impl_->builder.AddByteString(std::string(bytes));
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::reference(Rule named_rule) {
    return impl_->builder.AddRuleRef(named_rule);
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::sequence(std::initializer_list<Rule> parts) {
    return impl_->builder.AddSequence(std::vector<Rule>(parts.begin(), parts.end()));
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::sequence(std::span<const Rule> parts) {
    return impl_->builder.AddSequence(std::vector<Rule>(parts.begin(), parts.end()));
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::choice(std::initializer_list<Rule> alternatives) {
    return impl_->builder.AddChoices(
        std::vector<Rule>(alternatives.begin(), alternatives.end()));
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::choice(std::span<const Rule> alternatives) {
    return impl_->builder.AddChoices(std::vector<Rule>(alternatives.begin(), alternatives.end()));
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::repeat(std::string_view name_hint, Rule part,
                                                      std::int32_t minimum,
                                                      std::int32_t maximum) {
    return impl_->builder.AddRepeatFromExpr(std::string(name_hint), part, minimum, maximum);
}

ModelGrammarBuilder::Rule ModelGrammarBuilder::empty() { return impl_->builder.AddEmptyStr(); }

ModelGrammarBuilder::Rule ModelGrammarBuilder::rule(std::string_view name_hint, Rule body) {
    return impl_->builder.AddRuleWithHint(std::string(name_hint), body);
}

ModelGrammarBuilder::Rule
ModelGrammarBuilder::tag_dispatch(std::initializer_list<std::pair<std::string, Rule>> tags,
                                  bool loop, std::initializer_list<std::string> excludes) {
    xgrammar::Grammar::Impl::TagDispatch dispatch;
    dispatch.tag_rule_pairs.assign(tags.begin(), tags.end());
    dispatch.loop_after_dispatch = loop;
    dispatch.excludes.assign(excludes.begin(), excludes.end());
    return impl_->builder.AddTagDispatch(dispatch);
}

ModelGrammar ModelGrammarBuilder::normalize(Rule root_rule) {
    ModelGrammar grammar;
    grammar.impl_->grammar = xgrammar::GrammarNormalizer::Apply(impl_->builder.Get(root_rule));
    return grammar;
}

ModelGrammar ModelGrammarBuilder::get(Rule root_rule) {
    ModelGrammar grammar;
    grammar.impl_->grammar = impl_->builder.Get(root_rule);
    return grammar;
}

} // namespace ninfer::text
