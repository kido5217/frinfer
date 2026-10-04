#include "product/constraint/constraint_contract.h"
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/tool_call_signal.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

template <typename Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

Json base_request() {
    return Json{{"model", "qwen"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
}

OpenAIChatRequest parse(Json body) { return parse_chat_completion_request(body, limits()); }

ResolvedPromptSemantics semantics(const GenerationRequest& request) {
    ServeOptions server;
    return resolve_prompt_semantics(request, server);
}

ninfer::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), {});
}

ninfer::RequestOptions options(const GenerationRequest& request) {
    ServeOptions server;
    return to_request_options(request, server, semantics(request), true);
}

ninfer::RequestOptions options_with_backend(const GenerationRequest& request,
                                            ninfer::SpeculativeBackend backend) {
    ServeOptions server;
    server.speculative.backend = backend;
    return to_request_options(request, server, semantics(request), true);
}

Json parse_sse(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    if (!event.starts_with(prefix) || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_request_envelope_and_sampling() {
    int failures                  = 0;
    Json body                     = base_request();
    body["stream"]                = true;
    body["stream_options"]        = Json{{"include_usage", true}, {"include_obfuscation", false}};
    body["max_completion_tokens"] = 48;
    body["max_tokens"]            = 9;
    body["temperature"]           = 0.7;
    body["top_p"]                 = 0.8;
    body["presence_penalty"]      = 0.3;
    body["frequency_penalty"]     = -0.2;
    body["seed"]                  = -1;
    body["top_k"]                 = 17;
    body["min_p"]                 = 0.05;
    body["timings_per_token"]     = true;
    body["return_progress"]       = true;

    const OpenAIChatRequest request = parse(body);
    failures += check(request.model == "qwen", "model remains in OpenAI envelope");
    failures += check(request.stream && request.include_usage, "stream metadata parsed");
    failures += check(request.timings_per_token && request.return_progress,
                      "llama.cpp response observations remain in the protocol envelope");
    failures += check(request.output_tokens_explicit && request.generation.max_tokens == 48,
                      "max_completion_tokens wins and explicitness stays in envelope");
    failures += check(request.generation.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
                      "signed seed maps modulo 2^64");
    failures +=
        check(request.generation.sampling.top_k == 17 && request.generation.sampling.min_p == 0.05,
              "compatible sampler extensions parsed");
    const ninfer::RequestOptions translated = options(request.generation);
    failures +=
        check(translated.execution.sampling.top_k == 17, "top_k reaches Engine request options");
    failures +=
        check(translated.execution.sampling.min_p && *translated.execution.sampling.min_p == 0.05F,
              "min_p reaches Engine request options");
    failures +=
        check(translated.execution.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
              "signed seed reaches Engine request options");

    const OpenAIChatRequest defaults = parse(base_request());
    failures +=
        check(!defaults.stream && !defaults.include_usage && !defaults.output_tokens_explicit &&
                  !defaults.timings_per_token && !defaults.return_progress &&
                  defaults.generation.max_tokens == limits().default_max_tokens,
              "protocol defaults remain outside GenerationRequest");

    Json malformed              = base_request();
    malformed["stream_options"] = true;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "stream_options",
                      "malformed stream_options rejected");
    malformed                      = base_request();
    malformed["timings_per_token"] = "yes";
    failures += check(api_error([&] { (void)parse(malformed); }).param == "timings_per_token",
                      "non-boolean timings_per_token rejected");
    malformed                    = base_request();
    malformed["return_progress"] = 1;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "return_progress",
                      "non-boolean return_progress rejected");
    return failures;
}

int test_standard_field_policy() {
    int failures  = 0;
    auto rejected = [&](const char* key, Json value, const char* code) {
        Json body            = base_request();
        body[key]            = std::move(value);
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == key && error.code == code,
                          std::string(key) + " non-neutral value rejected");
    };

    rejected("n", 2, "n_not_supported");
    rejected("logit_bias", Json{{"12", 1}}, "logit_bias_not_supported");
    rejected("logprobs", true, "logprobs_not_supported");
    rejected("top_logprobs", 2, "logprobs_not_supported");
    rejected("response_format", Json{{"type", "yaml"}}, "response_format_not_supported");
    rejected("modalities", Json::array({"text", "audio"}), "modality_not_supported");
    rejected("web_search_options", Json::object(), "web_search_not_supported");
    rejected("moderation", Json::object(), "moderation_not_supported");
    rejected("verbosity", "high", "verbosity_not_supported");
    rejected("store", true, "store_not_supported");
    rejected("functions", Json::array({Json{{"name", "legacy"}}}), "legacy_tools_not_supported");

    Json neutral                      = base_request();
    neutral["n"]                      = 1;
    neutral["logit_bias"]             = Json{{"12", 0}, {"13", 0.0}};
    neutral["logprobs"]               = false;
    neutral["top_logprobs"]           = 0;
    neutral["response_format"]        = Json{{"type", "text"}};
    neutral["modalities"]             = Json::array({"text"});
    neutral["audio"]                  = Json{{"voice", "alloy"}};
    neutral["prediction"]             = Json{{"type", "content"}, {"content", "expected"}};
    neutral["verbosity"]              = "medium";
    neutral["store"]                  = false;
    neutral["functions"]              = Json::array();
    neutral["function_call"]          = "auto";
    neutral["metadata"]               = Json{{"trace", "client"}};
    neutral["user"]                   = "user-1";
    neutral["safety_identifier"]      = "safe-1";
    neutral["prompt_cache_key"]       = "cache-1";
    neutral["prompt_cache_options"]   = Json{{"retention", "24h"}};
    neutral["prompt_cache_retention"] = "24h";
    neutral["service_tier"]           = "priority";
    neutral["future_unknown_field"]   = Json{{"value", 1}};
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral controls and advisory hints are accepted");

    Json zero_limit                     = base_request();
    zero_limit["max_completion_tokens"] = 0;
    const OpenAIChatRequest zero        = parse(zero_limit);
    failures += check(zero.output_tokens_explicit && zero.generation.max_tokens == 0,
                      "an explicit zero output limit reaches Engine's no-generation path");
    return failures;
}

int test_constrained_decoding_extensions() {
    int failures = 0;

    // GBNF text is the supported constrained-decoding spelling and reaches the Engine contract.
    Json grammar_body       = base_request();
    grammar_body["grammar"] = "root ::= \"yes\" | \"no\"";
    const OpenAIChatRequest constrained = parse(grammar_body);
    failures += check(constrained.generation.grammar.has_value() &&
                          *constrained.generation.grammar == "root ::= \"yes\" | \"no\"" &&
                          constrained.generation.constraint_source == ConstraintSource::Grammar,
                      "grammar GBNF text is parsed into the request");
    const ninfer::RequestOptions constrained_options = options(constrained.generation);
    failures += check(constrained_options.constraint.has_value() &&
                          constrained_options.constraint->gbnf == "root ::= \"yes\" | \"no\"",
                      "grammar GBNF text reaches the Engine constraint contract");

    Json neutral                  = base_request();
    neutral["grammar"]            = "";
    neutral["structured_outputs"] = nullptr;
    neutral["guided_json"]        = nullptr;
    const OpenAIChatRequest plain = parse(neutral);
    failures += check(!plain.generation.grammar.has_value() &&
                          !options(plain.generation).constraint.has_value(),
                      "an empty grammar constrains nothing");

    Json typing_body       = base_request();
    typing_body["grammar"] = 5;
    const ApiError typing  = api_error([&] { (void)parse(typing_body); });
    failures += check(typing.param == "grammar" && typing.status == 400 &&
                          typing.code == "grammar_invalid",
                      "a non-string grammar is rejected");

    Json huge_grammar       = base_request();
    huge_grammar["grammar"] = std::string(ninfer::constraint::kConstraintPayloadLimit + 1, 'x');
    const ApiError huge     = api_error([&] { (void)parse(huge_grammar); });
    failures += check(huge.param == "grammar" && huge.code == "constraint_too_large",
                      "an oversized grammar is rejected");

    const std::vector<std::pair<const char*, Json>> unsupported = {
        {"structured_outputs", Json{{"json", Json{{"type", "object"}}}}},
        {"guided_json", Json{{"type", "object"}}},
        {"guided_regex", "[a-z]+"},
        {"guided_choice", Json::array({"yes", "no"})},
        {"guided_grammar", "root ::= \"yes\" | \"no\""},
    };
    for (const auto& [field, value] : unsupported) {
        Json body            = base_request();
        body[field]          = value;
        const ApiError error = api_error([&] { (void)parse(body); });
        failures +=
            check(error.param == field && error.code == "constrained_decoding_not_supported" &&
                      error.message.find(field) != std::string::npos,
                  std::string(field) + " remains an explicit rejection");
    }

    // response_format: text keeps no constraint; json_object and json_schema convert.
    Json text_format                  = base_request();
    text_format["response_format"]    = Json{{"type", "text"}};
    const OpenAIChatRequest text_only = parse(text_format);
    failures += check(!text_only.generation.grammar.has_value(),
                      "response_format text constrains nothing");

    Json object_format               = base_request();
    object_format["response_format"] = Json{{"type", "json_object"}};
    const OpenAIChatRequest object_json = parse(object_format);
    failures += check(object_json.generation.grammar.has_value() &&
                          object_json.generation.constraint_source == ConstraintSource::JsonSchema &&
                          object_json.generation.grammar->find("root") != std::string::npos,
                      "response_format json_object converts to a constraint");
    // The llama.cpp quirk of reading an extra `schema` member under json_object is not adopted.
    Json quirk_format               = base_request();
    quirk_format["response_format"] = Json{{"type", "json_object"},
                                           {"schema", Json{{"type", "array"}}}};
    failures += check(parse(quirk_format).generation.grammar == object_json.generation.grammar,
                      "json_object ignores an extra schema member");

    const Json diary_schema = Json{
        {"type", "object"},
        {"additionalProperties", false},
        {"required", Json::array({"title", "text"})},
        {"properties",
         Json{{"title", Json{{"type", "string"}, {"maxLength", 60}}},
              {"text", Json{{"type", "string"}, {"maxLength", 2400}}},
              {"facts", Json{{"type", "array"},
                             {"maxItems", 14},
                             {"items", Json{{"type", "string"}, {"maxLength", 120}}}}}}}};
    Json wrapper_body               = base_request();
    wrapper_body["response_format"] = Json{
        {"type", "json_schema"},
        {"json_schema", Json{{"name", "diary"}, {"strict", true}, {"schema", diary_schema}}}};
    const OpenAIChatRequest wrapper = parse(wrapper_body);
    failures += check(wrapper.generation.grammar.has_value() &&
                          wrapper.generation.constraint_source == ConstraintSource::JsonSchema,
                      "response_format json_schema wrapper converts");
    const ninfer::RequestOptions wrapper_options = options(wrapper.generation);
    failures += check(wrapper_options.constraint.has_value() &&
                          wrapper_options.constraint->gbnf == *wrapper.generation.grammar,
                      "the converted schema reaches the Engine constraint contract");

    Json bare_body               = base_request();
    bare_body["response_format"] = Json{{"type", "json_schema"}, {"json_schema", diary_schema}};
    failures += check(parse(bare_body).generation.grammar.has_value(),
                      "a bare schema under json_schema converts");

    const std::vector<Json> malformed_formats = {
        Json{{"type", "json_schema"}},
        Json{{"type", "json_schema"}, {"json_schema", 5}},
        Json{{"type", "json_schema"}, {"json_schema", Json{{"schema", 5}}}},
        Json{{"type", "json_schema"},
             {"json_schema", Json{{"schema", Json{{"type", "object"}}}, {"name", 5}}}},
        Json{{"type", "json_schema"},
             {"json_schema", Json{{"schema", Json{{"type", "object"}}}, {"strict", "yes"}}}},
        Json{{"type", "json_schema"}, {"json_schema", Json{{"name", "x"}}}},
        Json{{"type", 7}},
    };
    for (const Json& format : malformed_formats) {
        Json body               = base_request();
        body["response_format"] = format;
        const ApiError error    = api_error([&] { (void)parse(body); });
        failures += check(error.status == 400 && error.code == "json_schema_invalid",
                          "malformed response_format is rejected: " + format.dump());
    }

    Json yaml_body               = base_request();
    yaml_body["response_format"] = Json{{"type", "yaml"}};
    const ApiError yaml          = api_error([&] { (void)parse(yaml_body); });
    failures += check(yaml.param == "response_format" &&
                          yaml.code == "response_format_not_supported",
                      "an unknown response_format type is rejected");

    // Keywords the converter would not genuinely enforce are rejected, not silently loosened:
    // unknown keywords, patterns (degraded to "any string"), formats it ignores, combinators it
    // drops, and keywords used in a context where the typed model does not read them.
    const std::vector<Json> unenforced_schemas = {
        Json{{"type", "string"}, {"pattern", "^[a-z]+$"}},
        Json{{"type", "object"}, {"minProperties", 1}},
        Json{{"type", "string"}, {"format", "email"}},
        Json{{"type", "string"}, {"format", "uuid5"}},
        Json{{"type", "number"}, {"minimum", 5}},
        Json{{"minimum", 5}},
        Json{{"type", "integer"}, {"minimum", 0}, {"exclusiveMinimum", 5}},
        Json{{"type", "integer"}, {"maximum", 9}, {"exclusiveMaximum", 5}},
        Json{{"type", "string"}, {"maxItems", 3}},
        Json{{"type", "number"}, {"maxLength", 3}},
        Json{{"type", "string"}, {"properties", Json{{"a", Json{{"type", "string"}}}}}},
        Json{{"type", "object"},
             {"properties", Json{{"a", Json{{"type", "string"}}}}},
             {"allOf", Json::array({Json{{"required", Json::array({"a"})}}})}},
        Json{{"allOf", Json::array({Json{{"type", "string"}}, Json{{"maxLength", 3}}})}},
        Json{{"type", "object"},
             {"properties", Json{{"a", Json{{"type", "string"}}}}},
             {"oneOf", Json::array({Json{{"required", Json::array({"a"})}}})}},
        Json{{"type", "array"}, {"items", Json::array({Json{{"type", "string"}}})}, {"minItems", 2}},
        Json{{"type", "array"},
             {"prefixItems", Json::array({Json{{"type", "string"}}})},
             {"maxItems", 2}},
        Json{{"type", "array"},
             {"items", Json::array({Json{{"type", "number"}, {"minimum", 5}}})}},
        Json{{"anyOf", Json::array({Json{{"type", "string"}}})}, {"maxLength", 3}},
        Json{{"enum", Json::array({"a"})}, {"maxLength", 3}},
        Json{{"$ref", "#/$defs/x"}, {"minLength", 3}},
        Json{{"type", Json::array({"integer", "number"})}, {"minimum", 5}},
        Json{{"type", "array"},
             {"prefixItems", Json::array({Json{{"type", "string"}}})},
             {"items", Json{{"type", "integer"}}}},
    };
    for (const Json& schema : unenforced_schemas) {
        Json body               = base_request();
        body["response_format"] =
            Json{{"type", "json_schema"}, {"json_schema", Json{{"schema", schema}}}};
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.status == 400 && error.code == "json_schema_unsupported",
                          "an unenforced schema keyword is rejected: " + schema.dump());
    }

    // Malformed documents inside allowed keywords are rejected rather than ignored.
    for (const Json& schema :
         {Json{{"type", "object"}, {"properties", Json{{"a", Json{{"type", "string"}}}}},
               {"required", "a"}},
          Json{{"type", "object"}, {"properties", Json{{"a", Json{{"type", "string"}}}}},
               {"required", Json::array({5})}},
          Json{{"type", "string"}, {"enum", Json::array()}},
          Json{{"type", "string"}, {"enum", Json::array({1})}},
          Json{{"type", "string"}, {"const", 5}},
          Json{{"type", Json::array({"string", "null"})}, {"enum", Json::array({"a", 5})}}}) {
        Json body               = base_request();
        body["response_format"] =
            Json{{"type", "json_schema"}, {"json_schema", Json{{"schema", schema}}}};
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.status == 400 && error.code == "json_schema_invalid",
                          "a malformed schema document is rejected: " + schema.dump());
    }

    // The enforced subset still converts in every genuine context.
    const std::vector<Json> supported_schemas = {
        Json{{"type", "integer"}, {"minimum", 0}, {"maximum", 10}},
        Json{{"type", "array"},
             {"minItems", 1},
             {"maxItems", 3},
             {"items", Json{{"type", "string"}, {"maxLength", 4}}}},
        Json{{"type", "string"}, {"format", "date-time"}},
        Json{{"anyOf", Json::array({Json{{"type", "string"}},
                                    Json{{"type", "integer"}, {"minimum", 0}}})}},
        Json{{"type", "object"},
             {"$defs", Json{{"item", Json{{"type", "string"}}}}},
             {"properties", Json{{"a", Json{{"$ref", "#/$defs/item"}}}}}},
        Json{{"properties", Json{{"a", Json{{"type", "string"}, {"maxLength", 4}}}}},
             {"required", Json::array({"a"})},
             {"additionalProperties", false}},
        Json{{"items", Json{{"type", "string"}}}, {"maxItems", 2}},
        Json{{"type", Json::array({"string", "null"})}, {"maxLength", 3}},
        Json{{"type", Json::array({"integer", "null"})}, {"minimum", 5}},
        Json{{"type", Json::array({"object", "null"})},
             {"properties", Json{{"a", Json{{"type", "string"}}}}},
             {"required", Json::array({"a"})}},
        Json{{"type", "string"}, {"enum", Json::array({"a", "b"})}},
        Json{{"type", "integer"}, {"const", 7}},
        Json{{"type", Json::array({"string", "null"})}, {"enum", Json::array({"a", nullptr})}},
        Json{{"type", Json::array({"string", "null"})}, {"const", "a"}},
    };
    for (const Json& schema : supported_schemas) {
        Json body               = base_request();
        body["response_format"] =
            Json{{"type", "json_schema"}, {"json_schema", Json{{"schema", schema}}}};
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.status == 0,
                          "a supported schema converts: " + schema.dump() + " -> " + error.message);
    }

    Json big_schema_body               = base_request();
    big_schema_body["response_format"] = Json{
        {"type", "json_schema"},
        {"json_schema",
         Json{{"schema",
               Json{{"type", "string"}, {"const", std::string(ninfer::constraint::kConstraintPayloadLimit, 'x')}}}}}};
    const ApiError big_schema = api_error([&] { (void)parse(big_schema_body); });
    failures +=
        check(big_schema.code == "constraint_too_large", "an oversized JSON Schema is rejected");

    Json deep = Json{{"type", "string"}};
    for (int level = 0; level < ninfer::constraint::kConstraintNestingLimit + 2; ++level) {
        deep = Json{{"type", "array"}, {"items", deep}};
    }
    Json deep_body               = base_request();
    deep_body["response_format"] = Json{{"type", "json_schema"},
                                        {"json_schema", Json{{"schema", deep}}}};
    const ApiError deep_error    = api_error([&] { (void)parse(deep_body); });
    failures +=
        check(deep_error.code == "constraint_too_large", "an over-nested JSON Schema is rejected");

    // Exactly one constraint kind; text is not a constraint.
    Json conflict_body               = base_request();
    conflict_body["grammar"]         = "root ::= \"x\"";
    conflict_body["response_format"] = Json{{"type", "json_object"}};
    const ApiError conflict          = api_error([&] { (void)parse(conflict_body); });
    failures += check(conflict.param == "response_format" &&
                          conflict.code == "constrained_decoding_conflict",
                      "grammar with a constraining response_format is rejected");

    Json text_pair_body               = base_request();
    text_pair_body["grammar"]         = "root ::= \"x\"";
    text_pair_body["response_format"] = Json{{"type", "text"}};
    const GenerationRequest text_pair = parse(text_pair_body).generation;
    failures += check(text_pair.grammar.has_value() && *text_pair.grammar == "root ::= \"x\"",
                      "grammar with response_format text is allowed");

    // Tools with structured output stay fail-closed.
    const Json tool = Json{{"type", "function"},
                           {"function", Json{{"name", "weather"},
                                             {"description", "Get weather"},
                                             {"parameters", Json{{"type", "object"}}},
                                             {"strict", false}}}};
    Json tools_grammar         = base_request();
    tools_grammar["tools"]     = Json::array({tool});
    tools_grammar["grammar"]   = "root ::= \"x\"";
    const ApiError tools_grammar_error = api_error([&] { (void)parse(tools_grammar); });
    failures += check(tools_grammar_error.status == 400 &&
                          tools_grammar_error.code == "constrained_decoding_not_supported" &&
                          tools_grammar_error.param == "grammar",
                      "tools with a grammar constraint are rejected");
    Json empty_tools            = base_request();
    empty_tools["tools"]        = Json::array();
    empty_tools["grammar"]      = "root ::= \"x\"";
    const ApiError empty_tools_error = api_error([&] { (void)parse(empty_tools); });
    failures += check(empty_tools_error.status == 400 &&
                          empty_tools_error.code == "constrained_decoding_not_supported",
                      "an empty tools field with a constraint is rejected");
    Json tools_schema               = base_request();
    tools_schema["tools"]           = Json::array({tool});
    tools_schema["response_format"] = Json{{"type", "json_object"}};
    const ApiError tools_schema_error = api_error([&] { (void)parse(tools_schema); });
    failures += check(tools_schema_error.param == "response_format" &&
                          tools_schema_error.code == "constrained_decoding_not_supported",
                      "tools with a schema constraint are rejected");

    // The serve contract rejects a constrained request on a backend that cannot carry one,
    // naming the configured backend.
    for (const ninfer::SpeculativeBackend backend :
         {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::DFlash2}) {
        const ApiError backend_error =
            api_error([&] { (void)options_with_backend(constrained.generation, backend); });
        failures += check(backend_error.status == 400 &&
                              backend_error.code == "constrained_decoding_not_supported" &&
                              backend_error.message.find("constrained generation") !=
                                  std::string::npos,
                          "a constrained request on a DFlash backend is rejected");
    }
    const ninfer::RequestOptions mtp_options =
        options_with_backend(constrained.generation, ninfer::SpeculativeBackend::Mtp);
    failures += check(mtp_options.constraint.has_value(),
                      "a constrained request on MTP stays supported");

    return failures;
}

int test_constrained_thinking_default() {
    int failures = 0;

    // A constraining request without an explicit thinking field resolves thinking off: the
    // constrained default replaces both the server and the template defaults (issue #86).
    const std::vector<std::pair<const char*, Json>> constraints = {
        {"grammar", Json("root ::= \"yes\" | \"no\"")},
        {"response_format", Json{{"type", "json_object"}}},
        {"response_format",
         Json{{"type", "json_schema"}, {"json_schema", Json{{"type", "object"}}}}},
    };
    for (const auto& [field, value] : constraints) {
        Json body                   = base_request();
        body[field]                 = value;
        const OpenAIChatRequest parsed = parse(body);
        failures += check(parsed.generation.constraint_source != ConstraintSource::None,
                          std::string("the constraining request carries a constraint: ") + field);
        failures += check(semantics(parsed.generation).enable_thinking == false,
                          "a constraining request without a thinking field resolves thinking off: " +
                              std::string(field));
    }

    // An explicit enable overrides the constrained default: thinking runs and the constraint
    // stays on the request options (the v1 answer-region contract).
    Json enabled_body               = base_request();
    enabled_body["grammar"]         = "root ::= \"yes\" | \"no\"";
    enabled_body["enable_thinking"] = true;
    const OpenAIChatRequest enabled = parse(enabled_body);
    failures += check(semantics(enabled.generation).enable_thinking == true,
                      "an explicit enable_thinking overrides the constrained default");
    failures += check(options(enabled.generation).constraint.has_value(),
                      "the constraint stays on the request options when thinking is enabled");

    // Every non-none effort value counts as an explicit enable (the rule keys on presence,
    // not on a specific level).
    const std::vector<const char*> efforts = {"minimal", "low", "medium", "high", "xhigh", "max"};
    for (const char* effort : efforts) {
        Json effort_body                = base_request();
        effort_body["grammar"]          = "root ::= \"yes\" | \"no\"";
        effort_body["reasoning_effort"] = effort;
        failures += check(semantics(parse(effort_body).generation).enable_thinking == true,
                          "a non-none reasoning_effort overrides the constrained default: " +
                              std::string(effort));
    }
    Json effort_object_body               = base_request();
    effort_object_body["response_format"] = Json{{"type", "json_object"}};
    effort_object_body["reasoning_effort"] = "max";
    failures +=
        check(semantics(parse(effort_object_body).generation).enable_thinking == true,
              "a non-none reasoning_effort overrides the constrained default on json_object");

    // Conflict handling is unchanged for constraining requests: the throw precedes the rule,
    // so a conflicting explicit enable and none-effort is rejected, not defaulted.
    Json conflict_body                = base_request();
    conflict_body["grammar"]          = "root ::= \"yes\" | \"no\"";
    conflict_body["enable_thinking"]  = true;
    conflict_body["reasoning_effort"] = "none";
    const ApiError conflict = api_error([&] { (void)semantics(parse(conflict_body).generation); });
    failures += check(conflict.code == "conflicting_template_option" &&
                         conflict.param == "reasoning_effort",
                      "a constraining request with conflicting thinking fields still raises "
                      "conflicting_template_option");

    // Explicit disables stay off without error (same outcome as the constrained default).
    Json off_body               = base_request();
    off_body["grammar"]         = "root ::= \"yes\" | \"no\"";
    off_body["enable_thinking"] = false;
    failures += check(semantics(parse(off_body).generation).enable_thinking == false,
                      "an explicit enable_thinking false stays off");

    Json none_body                = base_request();
    none_body["grammar"]          = "root ::= \"yes\" | \"no\"";
    none_body["reasoning_effort"] = "none";
    failures += check(semantics(parse(none_body).generation).enable_thinking == false,
                      "an explicit reasoning_effort none stays off");

    // Unconstrained requests keep the existing resolution order: request field, then the
    // server default, then the template default (unspecified stays unspecified here).
    const OpenAIChatRequest plain = parse(base_request());
    failures += check(!semantics(plain.generation).enable_thinking.has_value(),
                      "an unconstrained request without a thinking field stays unspecified");
    Json plain_enabled_body = base_request();
    plain_enabled_body["enable_thinking"] = true;
    failures += check(semantics(parse(plain_enabled_body).generation).enable_thinking == true,
                      "an unconstrained request with an explicit enable stays on");

    ServeOptions no_thinking_server;
    no_thinking_server.enable_thinking = false;
    failures +=
        check(resolve_prompt_semantics(plain.generation, no_thinking_server).enable_thinking ==
                  false,
              "an unconstrained request honors the --no-thinking server default");

    // --no-thinking and the constrained default agree: off either way, no error.
    Json constrained_body         = base_request();
    constrained_body["grammar"]   = "root ::= \"yes\" | \"no\"";
    const OpenAIChatRequest constrained = parse(constrained_body);
    failures += check(
        resolve_prompt_semantics(constrained.generation, no_thinking_server).enable_thinking ==
            false,
        "--no-thinking and the constrained default agree");

    // The --default-thinking-budget cap follows the effective thinking state: a defaulted-off
    // constrained request receives no cap; an explicitly enabled one inherits the server cap.
    ServeOptions budget_server;
    budget_server.default_thinking_budget = 512;
    failures += check(
        !to_request_options(constrained.generation, budget_server,
                            resolve_prompt_semantics(constrained.generation, budget_server), true)
             .execution.thinking.budget.has_value(),
        "a defaulted-off constrained request receives no thinking cap");
    failures += check(
        to_request_options(enabled.generation, budget_server,
                           resolve_prompt_semantics(enabled.generation, budget_server), true)
            .execution.thinking.budget == 512,
        "an explicitly enabled constrained request inherits the server cap");

    return failures;
}

Json function_tool(std::string name = "weather", bool strict = false) {
    return Json{{"type", "function"},
                {"function", Json{{"name", std::move(name)},
                                  {"description", "Get weather"},
                                  {"parameters", Json{{"type", "object"}}},
                                  {"strict", strict}}}};
}

int test_tools() {
    int failures                      = 0;
    Json body                         = base_request();
    body["tools"]                     = Json::array({function_tool()});
    const OpenAIChatRequest automatic = parse(body);
    failures += check(automatic.generation.uses_tools(), "function tools default to auto");
    failures += check(prompt(automatic.generation).options.tool_jsons.size() == 1,
                      "auto tools reach PromptInput");

    body["tools"][0]["future_item_field"]                 = "ignored";
    body["tools"][0]["function"]["future_function_field"] = "ignored";
    const std::string normalized_definition = prompt(parse(body).generation).options.tool_jsons[0];
    failures += check(normalized_definition.find("future_item_field") == std::string::npos &&
                          normalized_definition.find("future_function_field") == std::string::npos,
                      "unknown tool fields do not silently alter the model prompt");

    body["tool_choice"]          = "none";
    body["parallel_tool_calls"]  = false;
    const OpenAIChatRequest none = parse(body);
    failures +=
        check(!none.generation.uses_tools() && prompt(none.generation).options.tool_jsons.empty(),
              "tool_choice none makes parallel_tool_calls neutral and removes executable tools");

    body["tool_choice"] = "required";
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_choice_not_supported",
                      "required tool choice rejected");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_choice_not_supported",
                      "named tool choice rejected");

    body          = base_request();
    body["tools"] = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "search"}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tools.size() == 1 && allowed.tools[0].name == "search" &&
                          prompt(allowed).options.tool_jsons.size() == 1,
                      "allowed_tools auto narrows the executable function set");

    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    const GenerationRequest direct_allowed = parse(body).generation;
    failures += check(direct_allowed.tools.size() == 1 && direct_allowed.tools[0].name == "weather",
                      "direct allowed_tools compatibility shape is accepted");
    body["tool_choice"]["mode"]     = "required";
    const ApiError required_allowed = api_error([&] { (void)parse(body); });
    failures +=
        check(required_allowed.code == "tool_choice_not_supported" &&
                  required_allowed.message.find("at least one tool call") != std::string::npos,
              "required allowed_tools reports the unenforceable guarantee");
    body["tool_choice"]["mode"]             = "auto";
    body["tool_choice"]["tools"][0]["name"] = "missing";
    failures += check(api_error([&] { (void)parse(body); }).param == "tool_choice",
                      "allowed_tools rejects names absent from the declared tool set");

    body          = base_request();
    body["tools"] = Json::array({function_tool("weather", true)});
    failures += check(api_error([&] { (void)parse(body); }).code == "strict_tools_not_supported",
                      "strict tools rejected");
    body["tools"] = Json::array({Json{{"type", "custom"}, {"name", "shell"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_type_not_supported",
                      "custom tools rejected");

    body                        = base_request();
    body["tools"]               = Json::array({function_tool()});
    body["parallel_tool_calls"] = false;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "parallel_tool_calls_not_supported",
              "parallel_tool_calls=false rejected when tools exist");
    body.erase("tools");
    failures += check(parse(body).generation.tools.empty(),
                      "parallel_tool_calls=false is neutral without tools");
    body["tool_choice"] = "auto";
    failures +=
        check(parse(body).generation.tools.empty(), "tool_choice auto is neutral without tools");

    Json history = base_request();
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                     Json{{"role", "assistant"},
                          {"content", nullptr},
                          {"tool_calls",
                           Json::array({Json{{"id", "call_1"},
                                             {"type", "function"},
                                             {"function", Json{{"name", "weather"},
                                                               {"arguments", "not-json-yet"}}}}})}},
                     Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}}});
    failures += check(parse(history).generation.has_tool_history(),
                      "tool-call history follows wire types without inventing JSON validation");

    Json mixed_assistant        = base_request();
    mixed_assistant["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "inspect"}},
         Json{{"role", "assistant"},
              {"content", "I will inspect it"},
              {"tool_calls",
               Json::array({Json{
                   {"id", "call_2"},
                   {"type", "function"},
                   {"function", Json{{"name", "inspect"}, {"arguments", R"({"path":"a"})"}}}}})}}});
    const GenerationRequest mixed_request  = parse(mixed_assistant).generation;
    const ninfer::PromptInput mixed_prompt = prompt(mixed_request);
    failures += check(mixed_request.messages[1].cache_boundary_after &&
                          !mixed_request.messages[1].content[0].cache_boundary_after &&
                          !mixed_prompt.context_cache.markers.empty() &&
                          mixed_prompt.context_cache.markers.back().location ==
                              ninfer::PromptCacheMarkerLocation::MessageBoundary &&
                          mixed_prompt.context_cache.markers.back().after_message_count == 2,
                      "automatic caching stops after a complete assistant text/tool-call turn");

    const Json ordered = Json::parse(
        R"({"model":"qwen","messages":[{"role":"user","content":"probe"}],"tools":[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]})");
    const ninfer::PromptInput ordered_prompt = prompt(parse(ordered).generation);
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}},"strict":false}})",
        "OpenAI Chat changed tool-schema member order before PromptInput");
    return failures;
}

int test_messages_and_media() {
    int failures                   = 0;
    Json body                      = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "alpha"}}, Json{{"type", "text"}, {"text", "beta"}}});
    const ninfer::PromptInput translated = prompt(parse(body).generation);
    failures += check(translated.messages[0].parts.size() == 2 &&
                          translated.messages[0].parts[0].text == "alpha" &&
                          translated.messages[0].parts[1].text == "beta",
                      "adjacent text parts preserve exact text without inserted newline");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"},
              {"image_url", Json{{"url", "https://example.test/a.png"}, {"detail", "auto"}}}},
         Json{{"type", "video_url"}, {"video_url", "https://example.test/a.mp4"}}});
    const GenerationRequest media = parse(body).generation;
    failures += check(media.media_item_count() == 2 &&
                          media.messages[0].content[0].kind == ContentKind::Image &&
                          media.messages[0].content[1].kind == ContentKind::Video,
                      "image and video compatibility inputs normalize to Engine media");

    body["messages"][0]["content"][0]["image_url"]["detail"] = "high";
    failures += check(api_error([&] { (void)parse(body); }).code == "image_detail_not_supported",
                      "explicit image preprocessing detail rejected");

    auto content_rejected = [&](const char* role, const char* type) {
        Json invalid                   = base_request();
        invalid["messages"][0]["role"] = role;
        invalid["messages"][0]["content"] =
            Json::array({Json{{"type", type}, {type, "https://example.test/x"}}});
        return api_error([&] { (void)parse(invalid); }).code == "modality_not_supported";
    };
    failures +=
        check(content_rejected("assistant", "image_url"), "assistant media history rejected");
    failures += check(content_rejected("system", "image_url"),
                      "system media rejected at protocol boundary");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "capture it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_capture"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "capture"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"},
              {"tool_call_id", "call_capture"},
              {"content",
               Json::array({Json{{"type", "text"}, {"text", "captured"}},
                            Json{{"type", "image_url"},
                                 {"image_url", Json{{"url", "https://example.test/capture.png"},
                                                    {"detail", "auto"}}}}})}}});
    const GenerationRequest tool_image = parse(body).generation;
    failures += check(tool_image.messages.back().role == ninfer::ChatRole::Tool &&
                          tool_image.messages.back().tool_call_id == "call_capture" &&
                          tool_image.messages.back().content.size() == 2 &&
                          tool_image.messages.back().content[0].kind == ContentKind::Text &&
                          tool_image.messages.back().content[1].kind == ContentKind::Image,
                      "tool result text and image parts normalize to one tool turn");

    body["messages"].back()["content"] = Json::array(
        {Json{{"type", "video_url"}, {"video_url", "https://example.test/capture.mp4"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "tool result video remains outside the Chat compatibility extension");

    body = base_request();
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "input_audio"}, {"input_audio", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "input audio rejected");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "file"}, {"file", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "file input rejected");

    body                        = base_request();
    body["messages"][0]["name"] = "speaker";
    failures += check(api_error([&] { (void)parse(body); }).code == "message_name_not_supported",
                      "message name rejected");

    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls",
         Json::array({Json{{"id", "call_1"},
                           {"type", "function"},
                           {"function", Json{{"name", "get_status"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(Json{
        {"role", "tool"}, {"name", "get_status"}, {"tool_call_id", "call_1"}, {"content", "ok"}});
    const GenerationRequest named_tool_history = parse(body).generation;
    const ChatTurn& named_tool                 = named_tool_history.messages.back();
    failures += check(named_tool.role == ninfer::ChatRole::Tool &&
                          named_tool.tool_call_id == "call_1" && !named_tool.tool_result_name &&
                          named_tool.content.size() == 1 && named_tool.content[0].text == "ok",
                      "tool message name is an ignored compatibility extension");

    body["messages"].back()["name"] = Json::array();
    failures +=
        check(api_error([&] { (void)parse(body); }).message == "message name must be a string",
              "tool message name remains type checked");

    body                           = base_request();
    body["messages"][0]["name"]    = "";
    body["messages"][0]["content"] = Json::array();
    failures += check(parse(body).generation.messages[0].content.empty(),
                      "empty names and empty content arrays remain neutral");

    body = base_request();
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "refusal"}, {"refusal", "part"}}})},
             {"refusal", "top-level"}});
    const GenerationRequest refusal_history = parse(body).generation;
    const ChatTurn& refusal                 = refusal_history.messages.back();
    failures += check(refusal.content.size() == 2 && refusal.content[0].text == "part" &&
                          refusal.content[1].text == "top-level",
                      "assistant refusal history is preserved as assistant text");

    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}});
    failures += check(parse(body).generation.messages.back().content.empty(),
                      "an empty assistant history turn is representable");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"function_call", Json{{"name", "legacy"}, {"arguments", R"({"value":1})"}}}},
         Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}}});
    const GenerationRequest legacy = parse(body).generation;
    failures += check(legacy.messages[1].tool_calls.size() == 1 &&
                          legacy.messages[1].tool_calls[0].name == "legacy" &&
                          legacy.messages[2].role == ninfer::ChatRole::Tool,
                      "legacy function-call history lowers to Engine tool history");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", ""},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", ""}, {"content", "done"}}});
    failures += check(parse(body).generation.has_tool_history(),
                      "string tool-call identifiers may be empty without changing history");
    return failures;
}

int test_reasoning_and_extensions() {
    int failures = 0;
    Json body    = base_request();
    body["messages"].push_back(Json{{"role", "assistant"},
                                    {"content", "answer"},
                                    {"reasoning_content", "thought"},
                                    {"reasoning", "thought"}});
    failures += check(parse(body).generation.messages.back().reasoning_content == "thought",
                      "assistant reasoning aliases normalize");
    body["messages"].back()["reasoning"] = "different";
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting assistant reasoning aliases rejected");
    body["messages"].back()["reasoning_content"] = "";
    failures += check(parse(body).generation.messages.back().reasoning_content == "different",
                      "an empty reasoning alias does not conflict with a meaningful alias");
    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "unfinished thought"}});
    failures +=
        check(parse(body).generation.messages.back().reasoning_content == "unfinished thought",
              "reasoning-only assistant history is preserved");

    body                         = base_request();
    body["enable_thinking"]      = true;
    body["preserve_thinking"]    = false;
    body["chat_template_kwargs"] = Json{{"enable_thinking", true}, {"preserve_thinking", false}};
    const GenerationRequest normalized = parse(body).generation;
    failures += check(normalized.enable_thinking == true && normalized.preserve_thinking == false,
                      "Qwen/vLLM template aliases normalize");
    body["chat_template_kwargs"]["enable_thinking"] = false;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking aliases rejected");
    body                         = base_request();
    body["chat_template_kwargs"] = Json{{"future", 1}};
    failures +=
        check(Json::parse(parse(body).generation.chat_template_kwargs_json).at("future") == 1,
              "custom template keyword did not survive protocol parsing");
    body["chat_template_kwargs"] = Json{{"future", nullptr}};
    failures += check(parse(body).generation.messages.size() == 1,
                      "null unknown template option is neutral");

    body                        = base_request();
    body["repetition_penalty"]  = 1.0;
    body["mm_processor_kwargs"] = Json{{"max_pixels", nullptr}};
    failures +=
        check(parse(body).generation.messages.size() == 1, "neutral ecosystem defaults accepted");
    body["repetition_penalty"] = 1.1;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "repetition_penalty_not_supported",
              "non-neutral repetition penalty rejected");
    body                        = base_request();
    body["mm_processor_kwargs"] = Json{{"max_pixels", 100}};
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "mm_processor_kwargs_not_supported",
              "non-empty media processor kwargs rejected");
    return failures;
}

int test_stops_and_ranges() {
    int failures                            = 0;
    Json body                               = base_request();
    body["stop"]                            = Json::array({"A", "B"});
    const ninfer::RequestOptions translated = options(parse(body).generation);
    failures += check(translated.stop.strings.size() == 4,
                      "each stop string applies to Content and Reasoning");
    failures += check(translated.stop.strings[0].channel == ninfer::OutputChannel::Content &&
                          translated.stop.strings[1].channel == ninfer::OutputChannel::Reasoning,
                      "stop channel ordering is explicit");

    body["stop"] = Json::array({"1", "2", "3", "4", "5"});
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "more than four stop strings rejected");
    body["stop"] = "";
    failures +=
        check(api_error([&] { (void)parse(body); }).param == "stop", "empty stop string rejected");

    body                                  = base_request();
    body["top_k"]                         = 21;
    const GenerationRequest invalid_top_k = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_top_k); }).param == "top_k",
                      "Engine translator owns sampler value range");
    body["top_k"]                         = 5;
    body["min_p"]                         = 1.1;
    const GenerationRequest invalid_min_p = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_min_p); }).param == "min_p",
                      "min_p range enforced by common Engine translator");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                                = "answer";
    outcome.reasoning                           = "thought";
    outcome.prompt_tokens                       = 20;
    outcome.completion_tokens                   = 7;
    outcome.reasoning_tokens                    = 3;
    outcome.finish_reason                       = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 12;
    outcome.metrics.prompt_wall_seconds         = 0.04;
    outcome.metrics.generation_wall_seconds     = 0.03;
    outcome.metrics.speculative_draft_tokens    = 9;
    outcome.metrics.speculative_accepted_tokens = 6;
    return outcome;
}

OpenAIChatResponseIdentity identity() {
    return OpenAIChatResponseIdentity{.id = "chatcmpl-test", .model = "qwen", .created = 42};
}

int test_aggregate_response() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    Json response             = Json::parse(make_chat_completion_response(identity(), outcome));
    failures += check(response["choices"][0]["message"]["content"] == "answer" &&
                          response["choices"][0]["message"]["reasoning_content"] == "thought" &&
                          response["choices"][0]["message"]["refusal"].is_null(),
                      "aggregate response separates reasoning and content");
    failures += check(response["choices"][0]["logprobs"].is_null(),
                      "aggregate choice carries nullable logprobs");
    failures += check(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3,
                      "aggregate usage exposes cache hits and reasoning tokens");
    failures += check(
        response["timings"]["cache_n"] == 12 && response["timings"]["prompt_n"] == 8 &&
            response["timings"]["prompt_ms"] == 40.0 &&
            response["timings"]["prompt_per_second"] == 200.0 &&
            response["timings"]["predicted_n"] == 7 &&
            response["timings"]["predicted_ms"] == 30.0 &&
            response["timings"]["predicted_per_second"] == 200.0 &&
            response["timings"]["draft_n"] == 9 && response["timings"]["draft_n_accepted"] == 6,
        "aggregate timings use exact cache and N-1 generation intervals");

    outcome.text.clear();
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    response         = Json::parse(make_chat_completion_response(identity(), outcome));
    const Json& call = response["choices"][0]["message"]["tool_calls"][0];
    failures += check(response["choices"][0]["finish_reason"] == "tool_calls" &&
                          response["choices"][0]["message"]["content"].is_null(),
                      "aggregate tool call has OpenAI terminal shape");
    failures += check(
        call["id"].get<std::string>().starts_with("call_") && call["function"]["name"] == "Edit" &&
            !Json::parse(call["function"]["arguments"].get<std::string>()).contains("replace_all"),
        "OpenAI adapter owns wire tool-call identifiers");
    return failures;
}

int test_stream_response() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true);
    Json role = parse_sse(stream.start());
    failures += check(role["choices"][0]["delta"]["role"] == "assistant" &&
                          role["choices"][0]["logprobs"].is_null() && role["usage"].is_null(),
                      "stream starts with role, nullable logprobs, and null usage");
    Json reasoning = parse_sse(stream.reasoning_delta("thought"));
    Json content   = parse_sse(stream.content_delta("ans"));
    failures += check(reasoning["choices"][0]["delta"]["reasoning_content"] == "thought" &&
                          content["choices"][0]["delta"]["content"] == "ans",
                      "stream separates reasoning and content deltas");

    GenerationOutcome outcome             = sample_outcome();
    const std::vector<std::string> events = stream.finish(outcome);
    failures +=
        check(events.size() == 4, "finish emits buffered suffix, terminal, usage, and done");
    failures += check(parse_sse(events[0])["choices"][0]["delta"]["content"] == "wer",
                      "terminal content suffix is emitted exactly once");
    failures += check(parse_sse(events[1])["choices"][0]["finish_reason"] == "stop",
                      "stream terminal finish reason emitted");
    const Json usage = parse_sse(events[2]);
    failures += check(usage["choices"].empty() &&
                          usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          usage["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3 &&
                          usage["timings"]["predicted_n"] == 7,
                      "dedicated stream usage carries token accounting and terminal timings");
    failures += check(events.back() == "data: [DONE]\n\n", "stream ends with DONE sentinel");

    OpenAIChatStream mismatch(identity(), false);
    (void)mismatch.start();
    (void)mismatch.content_delta("different");
    failures += check(throws_logic([&] { (void)mismatch.finish(outcome); }),
                      "stream encoder rejects terminal/content divergence");

    OpenAIChatStream tool_stream(identity(), false);
    (void)tool_stream.start();
    GenerationOutcome tool_outcome;
    tool_outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit", .arguments_json = R"({"file_path":"/tmp/probe.cpp"})"});
    tool_outcome.finish_reason                 = ninfer::FinishReason::StopToken;
    const std::vector<std::string> tool_events = tool_stream.finish(tool_outcome);
    const Json tool_delta                      = parse_sse(tool_events[0]);
    failures += check(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["id"].get<std::string>().starts_with(
            "call_") &&
            tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Edit" &&
            parse_sse(tool_events[1])["choices"][0]["finish_reason"] == "tool_calls",
        "stream encoder owns stable OpenAI tool-call shape");
    return failures;
}

int test_stream_observations() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true, true, true);
    const Json role = parse_sse(stream.start());
    failures += check(!role.contains("timings") && !role.contains("prompt_progress"),
                      "transport role chunk precedes Engine observations");

    stream.note_start(
        ninfer::GenerationStart{.prompt = {.prompt_tokens = 32}, .reused_prompt_tokens = 12});
    const Json initial = parse_sse(stream.initial_prompt_progress());
    failures +=
        check(initial["choices"][0]["delta"].empty() && initial["prompt_progress"]["total"] == 32 &&
                  initial["prompt_progress"]["cache"] == 12 &&
                  initial["prompt_progress"]["processed"] == 12 &&
                  initial["prompt_progress"]["time_ms"] == 0,
              "initial prompt progress begins at the admitted cache frontier");

    const Json middle = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 20,
        .elapsed_ns              = 57000000,
    }));
    failures += check(middle["prompt_progress"]["processed"] == 20 &&
                          middle["prompt_progress"]["time_ms"] == 57,
                      "prompt progress exposes a cumulative completed frontier");
    const Json complete = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 32,
        .elapsed_ns              = 100000000,
    }));
    failures +=
        check(complete["prompt_progress"]["processed"] == complete["prompt_progress"]["total"],
              "final prompt progress reaches the complete prompt");

    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens = 1, .prompt_elapsed_ns = 110000000, .generation_elapsed_ns = 0});
    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens      = 3,
        .prompt_elapsed_ns     = 110000000,
        .generation_elapsed_ns = 20000000,
    });
    const Json content = parse_sse(stream.content_delta("answer"));
    failures +=
        check(content["timings"]["prompt_n"] == 20 && content["timings"]["predicted_n"] == 3 &&
                  content["timings"]["predicted_per_second"] == 100.0,
              "visible output uses the latest independent commit observation");

    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning.clear();
    const std::vector<std::string> terminal = stream.finish(outcome);
    failures += check(parse_sse(terminal[1])["timings"]["predicted_n"] == 7,
                      "terminal usage replaces live timing with exact final accounting");
    return failures;
}

int test_common_objects() {
    int failures      = 0;
    const Json models = Json::parse(make_models_list("qwen", 7, 240000));
    failures +=
        check(models["data"][0]["id"] == "qwen" && models["data"][0]["max_model_len"] == 240000,
              "models list advertises the configured context limit");
    const Json model = Json::parse(make_model_object("qwen", 7, 240000));
    failures += check(model["max_model_len"] == 240000,
                      "model lookup advertises the configured context limit");
    const Json error = Json::parse(make_error_body(
        ApiError{.status = 400, .message = "bad", .param = "messages", .code = "invalid"}));
    failures += check(error["error"]["param"] == "messages" && error["error"]["code"] == "invalid",
                      "OpenAI common error shape remains stable");
    return failures;
}

int test_tool_call_demotion_signal() {
    using ninfer::ToolCallParseFallbackReason;
    int failures = 0;
    failures +=
        check(tool_call_demotion_signals(ToolCallParseFallbackReason::MalformedStructure, true) &&
                  !tool_call_demotion_signals(ToolCallParseFallbackReason::MalformedStructure,
                                              false) &&
                  !tool_call_demotion_signals(ToolCallParseFallbackReason::None, true),
              "demotion signal condition is not call-loss-only");

    struct Case {
        ToolCallParseFallbackReason reason;
        const char* code;
    };
    const Case cases[] = {
        {ToolCallParseFallbackReason::MalformedStructure, "tool_call_malformed_structure"},
        {ToolCallParseFallbackReason::DuplicateParameter, "tool_call_duplicate_parameter"},
        {ToolCallParseFallbackReason::InvalidToolName, "tool_call_invalid_tool_name"},
        {ToolCallParseFallbackReason::UndeclaredTool, "tool_call_undeclared_tool"},
        {ToolCallParseFallbackReason::TrailingContent, "tool_call_trailing_content"},
    };
    for (const Case& item : cases) {
        const ApiError error = tool_call_demotion_error(item.reason);
        const Json rendered  = Json::parse(make_error_body(error));
        failures += check(error.status == 409 && error.type == "tool_call_error" &&
                              error.code == item.code && rendered["error"]["code"] == item.code &&
                              rendered["error"]["message"].get<std::string>().find("tool call") !=
                                  std::string::npos,
                          std::string("demotion error shape is wrong for ") + item.code);
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_request_envelope_and_sampling();
    failures += test_standard_field_policy();
    failures += test_constrained_decoding_extensions();
    failures += test_constrained_thinking_default();
    failures += test_tools();
    failures += test_messages_and_media();
    failures += test_reasoning_and_extensions();
    failures += test_stops_and_ranges();
    failures += test_aggregate_response();
    failures += test_stream_response();
    failures += test_stream_observations();
    failures += test_common_objects();
    failures += test_tool_call_demotion_signal();
    if (failures == 0) { std::cout << "OpenAI Chat protocol tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
