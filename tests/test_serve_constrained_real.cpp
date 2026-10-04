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
#include "serve/translate.h"

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
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 512ULL << 20;
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

    // 4. An invalid grammar is rejected at submit and maps back to grammar_invalid.
    //     a diagnostic instead of crashing the serving engine mid-generation.
    Json pathological_body       = base_request();
    pathological_body["grammar"] = "root ::= ([ab]{0,5}){0,64}";
    try {
        (void)run_route(engine, server, pathological_body);
        check(false, "a pathological grammar was accepted");
    } catch (const ninfer::RequestError& error) {
        check(error.kind() == ninfer::RequestErrorKind::InvalidConstraint,
              "a pathological grammar fails closed as an invalid constraint", error.what());
    }

    // 5. A grammar whose initial mask admits nothing is rejected before generation.
    Json unsatisfiable_body       = base_request();
    unsatisfiable_body["grammar"] = "root ::= \"\\x00\"";
    try {
        (void)run_route(engine, server, unsatisfiable_body);
        check(false, "an unsatisfiable grammar was accepted");
    } catch (const ninfer::RequestError& error) {
        check(error.kind() == ninfer::RequestErrorKind::InvalidConstraint,
              "an unsatisfiable grammar fails closed as an invalid constraint");
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
