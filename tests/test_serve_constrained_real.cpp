// Real-route constrained-serving test (wayfinder map #45, ticket #52).
//
// Drives the OpenAI chat route for constrained decoding end to end against a real artifact:
// parse the request body through the serve contract, translate it with the serve options, and
// run prepare/submit/wait on the public Engine. Covers enforced GBNF and JSON Schema
// `response_format`, the Engine compile rejection mapped back to `grammar_invalid`, the
// unsatisfiable-grammar rejection, and the serve-level DFlash/MTP backend policy.
//
// NINFER_TEST_ARTIFACT selects the artifact; without it the test skips (exit 77).

#include "ninfer/engine.h"
#include "serve/anthropic_messages.h"
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/translate.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    if (ok) { return; }
    ++g_failures;
    std::cerr << "FAIL: " << what << (detail.empty() ? "" : ": " + detail) << '\n';
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 64}; }

Json base_request() {
    return Json{
        {"model", "qwen"},
        {"enable_thinking", false},
        {"temperature", 0},
        {"max_tokens", 16},
        {"messages",
         Json::array({Json{{"role", "user"}, {"content", "Reply with the exact answer."}}})}};
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 4096;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                        = 1024;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_capacity_bytes    = 1ULL << 30;
    return options;
}

template <typename ParsedRequest>
ninfer::GenerationResult run_parsed(ninfer::Engine& engine, const ServeOptions& server,
                                    const ParsedRequest& parsed) {
    const ResolvedPromptSemantics res = resolve_prompt_semantics(parsed.generation, server);
    ninfer::RequestOptions options    = to_request_options(parsed.generation, server, res, false);
    ninfer::PreparedPrompt prompt = engine.prepare(to_prompt_input(parsed.generation, res, {}));
    ninfer::GenerationHandle handle = engine.submit(std::move(prompt), std::move(options));
    return handle.wait();
}

ninfer::GenerationResult run_route(ninfer::Engine& engine, const ServeOptions& server,
                                   const Json& body) {
    return run_parsed(engine, server, parse_chat_completion_request(body, limits()));
}

// The same constraint pipeline reached through the Anthropic route's output_config.format.
ninfer::GenerationResult run_anthropic_route(ninfer::Engine& engine, const ServeOptions& server,
                                             const Json& body) {
    return run_parsed(engine, server, parse_anthropic_messages_request(body, limits()));
}

int exercise(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));
    const ServeOptions server;

    // 1. GBNF literal through the full route: the answer is the literal, then the grammar's end.
    Json grammar_body       = base_request();
    grammar_body["grammar"] = "root ::= \"9999\"";
    const ninfer::GenerationResult literal = run_route(engine, server, grammar_body);
    check(literal.content == "9999", "grammar answer is the exact literal",
          "content=" + literal.content);
    check(literal.finish_reason == ninfer::FinishReason::StopToken,
          "grammar completion ends generation");

    // 2. JSON Schema response_format: the answer is a valid document of the schema.
    const Json answer_schema =
        Json{{"type", "object"},
             {"additionalProperties", false},
             {"required", Json::array({"answer"})},
             {"properties", Json{{"answer", Json{{"type", "number"}}}}}};
    Json schema_body               = base_request();
    schema_body["response_format"] = Json{
        {"type", "json_schema"},
        {"json_schema", Json{{"name", "answer"}, {"strict", true}, {"schema", answer_schema}}}};
    const ninfer::GenerationResult schema_result = run_route(engine, server, schema_body);
    bool schema_valid                            = false;
    std::string schema_detail;
    try {
        const Json document = Json::parse(schema_result.content);
        schema_valid = document.is_object() && document.contains("answer") &&
                       document.at("answer").is_number() && document.size() == 1;
        schema_detail = schema_result.content;
    } catch (const Json::exception& error) {
        schema_detail = std::string("not JSON: ") + error.what() + " content=" + schema_result.content;
    }
    check(schema_valid, "json_schema answer satisfies the schema", schema_detail);

    // 3. A grammar that overflows the producer's structural guards is rejected at compile with
    Json invalid_body       = base_request();
    invalid_body["grammar"] = "root ::= \"unterminated";
    const OpenAIChatRequest invalid_parsed =
        parse_chat_completion_request(invalid_body, limits());
    try {
        (void)run_route(engine, server, invalid_body);
        check(false, "an invalid grammar was accepted");
    } catch (const ninfer::RequestError& error) {
        const ApiError mapped =
            request_error_to_api_error(error, invalid_parsed.generation.constraint_source);
        check(mapped.status == 400 && mapped.code == "grammar_invalid" &&
                  mapped.param == "grammar",
              "an invalid grammar maps to grammar_invalid",
              mapped.code + " " + mapped.message);
    }

    // 4. Nested bounded repetition that the retired fork guard rejected is now compiled by the
    //    adopted XGrammar parser and executes; the output stays within the grammar's alphabet.
    Json nested_body       = base_request();
    nested_body["grammar"] = "root ::= ([ab]{0,5}){0,64}";
    try {
        const ninfer::GenerationResult nested = run_route(engine, server, nested_body);
        bool in_alphabet                      = true;
        for (const char byte : nested.content) {
            if (byte != 'a' && byte != 'b') { in_alphabet = false; }
        }
        check(in_alphabet, "a nested bounded-repetition grammar stays within its alphabet",
              nested.content);
    } catch (const ninfer::RequestError& error) {
        check(false, "a nested bounded-repetition grammar was rejected", error.what());
    }

    // 5. The adopted parser compiles the GBNF `"\x00"` literal, so there is no submit-time
    //    unsatisfiable surface as in the retired fork. If such a constraint ever admits no legal
    //    next token, generation fails closed as ConstraintDeadEnd (HTTP 400 constraint_dead_end).
    Json null_body       = base_request();
    null_body["grammar"] = "root ::= \"\\x00\"";
    try {
        (void)run_route(engine, server, null_body);
        std::cout << "note: the NUL-literal grammar is accepted by the adopted parser\n";
    } catch (const ninfer::RequestError& error) {
        check(error.kind() == ninfer::RequestErrorKind::ConstraintDeadEnd,
              "a NUL-literal grammar only fails closed as a generation dead end", error.what());
    }

    // 6. Serve-level backend policy: DFlash rejects a constrained request naming the backend;
    //    MTP keeps it.
    const OpenAIChatRequest parsed =
        parse_chat_completion_request(grammar_body, limits());
    ServeOptions mtp_server;
    mtp_server.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    const ninfer::RequestOptions mtp_options = to_request_options(
        parsed.generation, mtp_server, resolve_prompt_semantics(parsed.generation, mtp_server),
        false);
    check(mtp_options.constraint.has_value(), "the MTP backend keeps the serve constraint");

    ServeOptions dflash_server;
    dflash_server.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    bool dflash_rejected              = false;
    std::string dflash_code;
    std::string dflash_message;
    try {
        (void)to_request_options(parsed.generation, dflash_server,
                                 resolve_prompt_semantics(parsed.generation, dflash_server), false);
    } catch (const ApiException& error) {
        dflash_rejected = true;
        dflash_code     = error.error().code;
        dflash_message  = error.error().message;
    }
    check(dflash_rejected && dflash_code == "constrained_decoding_not_supported" &&
              dflash_message.find("dflash2") != std::string::npos,
          "a constrained request on DFlash2 is rejected naming the backend", dflash_message);

    // 7. Anthropic structured outputs: the equivalent json_schema through output_config.format
    //    reaches the same constraint pipeline and, at temperature 0, the same answer as the chat
    //    route. This is the ticket's Anthropic-vs-chat round-trip on a real artifact.
    Json anthropic_body = base_request();
    anthropic_body["output_config"] =
        Json{{"format", Json{{"type", "json_schema"}, {"schema", answer_schema}}}};
    const ninfer::GenerationResult anthropic_result =
        run_anthropic_route(engine, server, anthropic_body);
    bool anthropic_schema_valid = false;
    std::string anthropic_detail;
    try {
        const Json document = Json::parse(anthropic_result.content);
        anthropic_schema_valid = document.is_object() && document.contains("answer") &&
                                 document.at("answer").is_number() && document.size() == 1;
        anthropic_detail = anthropic_result.content;
    } catch (const Json::exception& error) {
        anthropic_detail = std::string("not JSON: ") + error.what() +
                           " content=" + anthropic_result.content;
    }
    check(anthropic_schema_valid, "Anthropic output_config.format answer satisfies the schema",
          anthropic_detail);
    check(anthropic_result.content == schema_result.content,
          "the Anthropic and chat constrained answers round-trip identically",
          "chat=" + schema_result.content + " anthropic=" + anthropic_result.content);

    // 8. Token log probabilities through the chat route. Greedy decoding makes the sampler pick
    //    the adjusted-logit argmax, so the reported top-1 id must equal the generated token and
    //    the chosen value must be the top-1 value. The full top-list must be finite, ordered, and
    //    inside the public token domain.
    Json logprobs_body           = base_request();
    logprobs_body["logprobs"]    = true;
    logprobs_body["top_logprobs"] = 3;
    const ninfer::GenerationResult logged = run_route(engine, server, logprobs_body);
    check(!logged.generated_token_ids.empty(), "logprobs request generated tokens");
    check(!logged.content_logprobs.empty(),
          "logprobs request produced content records");
    check(logged.content_logprobs.size() <= logged.generated_token_ids.size(),
          "content records do not exceed committed tokens");
    const auto check_records = [&](const std::vector<ninfer::TokenLogprob>& records,
                                   const char* label) {
        int failures = 0;
        for (const ninfer::TokenLogprob& record : records) {
            if (!(std::isfinite(record.logprob) && record.logprob <= 0.0F)) { ++failures; }
            if (record.top_ids[0] != record.id) { ++failures; }
            if (std::abs(record.top_values[0] - record.logprob) > 1e-4F) { ++failures; }
            if (record.bytes.empty()) { ++failures; }
            for (std::size_t k = 0; k + 1 < ninfer::kMaximumTokenLogprobs; ++k) {
                if (record.top_values[k] < record.top_values[k + 1]) { ++failures; }
                if (record.top_ids[k] < 0) { ++failures; }
            }
        }
        check(failures == 0, std::string(label) + " records satisfy the reported invariants",
              "violations=" + std::to_string(failures));
    };
    check_records(logged.content_logprobs, "chat");
    const ninfer::GenerationResult logged_again = run_route(engine, server, logprobs_body);
    check(logged_again.content_logprobs.size() == logged.content_logprobs.size() &&
              logged_again.content_logprobs.front().id == logged.content_logprobs.front().id &&
              logged_again.content_logprobs.front().logprob ==
                  logged.content_logprobs.front().logprob,
          "greedy logprobs are deterministic across runs");

    GenerationOutcome chat_outcome;
    chat_outcome.text                = logged.content;
    chat_outcome.reasoning           = logged.reasoning;
    chat_outcome.content_logprobs    = logged.content_logprobs;
    chat_outcome.completion_tokens   = static_cast<int>(logged.generated_token_ids.size());
    chat_outcome.finish_reason       = logged.finish_reason;
    const OpenAIChatResponseIdentity identity{.id = "chatcmpl-lp", .model = "qwen", .created = 1};
    const Json chat_wire = Json::parse(make_chat_completion_response(
        identity, chat_outcome, OpenAIChatLogprobs{true, 3}));
    check(chat_wire["choices"][0]["logprobs"]["content"].is_array() &&
              chat_wire["choices"][0]["logprobs"]["refusal"].is_array() &&
              !chat_wire["choices"][0]["logprobs"]["content"].empty(),
          "chat wire logprobs carry content and refusal arrays");
    std::cout << "chat logprobs[0] = "
              << chat_wire["choices"][0]["logprobs"]["content"][0].dump() << "\n";

    // 9. The same channel through the Responses route.
    const Json responses_body = {{"model", "qwen"},
                                {"input", "Reply with the exact answer."},
                                {"reasoning", Json{{"effort", "none"}}},
                                {"temperature", 0},
                                {"max_output_tokens", 16},
                                {"include", Json::array({"message.output_text.logprobs"})},
                                {"top_logprobs", 3}};
    const OpenAIResponsesCreateRequest responses_parsed =
        parse_openai_responses_create_request(responses_body, limits());
    check(responses_parsed.prompt.generation.logprobs,
          "Responses logprobs opt-in reaches the Engine");
    OpenAIResponsesStore responses_store(8, 1U << 20);
    const OpenAIResponsesResolvedPrompt responses_resolved = resolve_openai_responses_prompt(
        responses_parsed.prompt, responses_store, std::nullopt, false);
    check(responses_resolved.generation.logprobs,
          "Responses resolution preserves the logprobs opt-in");
    const ninfer::GenerationResult responses_result =
        run_parsed(engine, server, responses_resolved);
    std::cout << "responses content=" << responses_result.content.size()
              << " reasoning=" << responses_result.reasoning.size()
              << " records=" << responses_result.content_logprobs.size() << "\n";
    check_records(responses_result.content_logprobs, "Responses");
    check(!responses_result.content_logprobs.empty(),
          "Responses produced content logprob records");
    GenerationOutcome responses_outcome;
    responses_outcome.text              = responses_result.content;
    responses_outcome.reasoning         = responses_result.reasoning;
    responses_outcome.content_logprobs  = responses_result.content_logprobs;
    responses_outcome.completion_tokens = static_cast<int>(responses_result.generated_token_ids.size());
    responses_outcome.finish_reason     = responses_result.finish_reason;
    std::cout << "building responses wire\n";
    const BuiltOpenAIResponse responses_wire = make_openai_response_object(
        "resp_lp", 1, responses_parsed, {}, responses_outcome);
    const Json& output = responses_wire.body.at("output");
    int message_index  = -1;
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (output.at(index).at("type") == "message") { message_index = static_cast<int>(index); }
    }
    std::cout << "responses wire output items=" << output.size() << " message_index=" << message_index
              << "\n";
    const bool responses_wire_ok =
        message_index >= 0 &&
        output.at(static_cast<std::size_t>(message_index)).at("content").at(0).at("logprobs").is_array() &&
        !output.at(static_cast<std::size_t>(message_index)).at("content").at(0).at("logprobs").empty();
    check(responses_wire_ok, "Responses wire logprobs carry aggregate LogProb[]");
    if (responses_wire_ok) {
        std::cout << "responses logprobs[0] = "
                  << output.at(static_cast<std::size_t>(message_index))
                         .at("content")
                         .at(0)
                         .at("logprobs")
                         .at(0)
                         .dump()
                  << "\n";
    }

    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise(artifact);
    } catch (const std::exception& error) {
        std::cerr << "serve constrained route failed: " << error.what() << '\n';
        return 1;
    }
}
