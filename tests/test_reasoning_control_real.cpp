// Real-model reasoning-control test (wayfinder map #175, ticket #187).
//
// This is the level the CPU-side tests cannot reach: it drives the public Engine against a real
// artifact and asserts that a real-time `reasoning_end` signal raised through
// `GenerationHandle::control()` is actually consumed by the request's output session. Every other
// test of this feature touches the frontend session or the serve registry directly, so deleting the
// Engine-side consumption leaves them green while the feature silently does nothing.
//
// It also exercises the serve registry against a live generation, which is where the registration
// lifetime is observable.
//
// NINFER_TEST_ARTIFACT selects the artifact; without it the test skips (exit 77).

#include "ninfer/engine.h"
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/reasoning_control.h"
#include "serve/translate.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    if (ok) { return; }
    ++g_failures;
    std::cerr << "FAIL: " << what << (detail.empty() ? "" : ": " + detail) << '\n';
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 8192;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_capacity_bytes    = 1ULL << 30;
    return options;
}

// A prompt that reliably opens a long thinking block: the model has to reason before it can answer.
Json thinking_request() {
    return Json{
        {"model", "qwen"},
        {"enable_thinking", true},
        {"temperature", 0},
        {"max_tokens", 512},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content",
              "A farmer has 17 sheep. One by one they each shake hands with every other sheep "
              "exactly once, and then each shakes hands with the farmer once. Work out the total "
              "number of handshakes, checking your arithmetic at least twice. Explain every step."}}})}};
}

ninfer::GenerationHandle submit(ninfer::Engine& engine, const ServeOptions& server,
                                const Json& body) {
    const OpenAIChatRequest parsed = parse_chat_completion_request(body, limits());
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(parsed.generation, server);
    ninfer::RequestOptions options =
        to_request_options(parsed.generation, server, semantics, false);
    ninfer::PreparedPrompt prompt = engine.prepare(to_prompt_input(parsed.generation, semantics, {}));
    return engine.submit(std::move(prompt), std::move(options));
}

// The load-bearing claim: a signal raised from outside the request reaches the output session, which
// commits the canonical early-close guidance and then continues generating content.
int exercise_engine_control(const char* artifact) {
    int failures_before = g_failures;
    ninfer::Engine engine(engine_options(artifact));
    const ServeOptions server;

    ninfer::GenerationHandle handle = submit(engine, server, thinking_request());
    const ninfer::GenerationControl surface = handle.control();
    check(surface.is_valid(), "a submitted request exposes a live control surface");
    // A copy shares the request's flag; end_reasoning() is a non-const signal, so this copy is the
    // one that raises it.
    ninfer::GenerationControl control = surface;

    // Signal while the model is still reasoning, on another thread so the wait is genuinely
    // concurrent with generation.
    std::atomic<bool> signalled{false};
    std::thread signaller([&control, &signalled] {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        control.end_reasoning();
        signalled.store(true, std::memory_order_release);
    });

    const ninfer::GenerationResult result = handle.wait();
    signaller.join();
    check(signalled.load(std::memory_order_acquire), "the control signal was raised");

    check(result.thinking.applied,
          "the Engine consumed the real-time signal and committed the early-close guidance",
          "configured_budget=" +
              (result.thinking.configured_budget
                   ? std::to_string(*result.thinking.configured_budget)
                   : std::string("none")) +
              " budget_thinking_tokens=" + std::to_string(result.thinking.budget_thinking_tokens) +
              " injected_tokens=" + std::to_string(result.thinking.injected_tokens));
    check(result.thinking.configured_budget == std::nullopt,
          "this request had no thinking budget, so only the real-time signal could have applied");
    check(result.thinking.injected_tokens > 1,
          "the committed control span is the complete canonical suffix, not a single token",
          "injected_tokens=" + std::to_string(result.thinking.injected_tokens));
    check(!result.reasoning.empty(),
          "the early-close guidance reached the reasoning stream",
          "reasoning chars=" + std::to_string(result.reasoning.size()));
    return g_failures - failures_before;
}

// The serve registry against a live generation: the id is reachable while the request runs and the
// signal it carries is the Engine's own surface, and both stop being true once the response ends.
int exercise_registry(const char* artifact) {
    int failures_before           = g_failures;
    ninfer::Engine engine(engine_options(artifact));
    const ServeOptions server;

    ReasoningControlRegistry registry;
    const std::string completion_id = "chatcmpl-test-1";
    ninfer::GenerationHandle handle  = submit(engine, server, thinking_request());
    std::shared_ptr<ReasoningControlRegistration> registration = arm_reasoning_control(
        registry, completion_id,
        {.control = handle.control(), .requested = true, .id_live_while_generating = true});
    check(registration != nullptr, "a streaming armed completion is registered");
    check(registry.size() == 1, "the live completion is reachable while it generates");

    // Fire the control request on another thread so it lands while the request is still generating.
    std::optional<ChatControlOutcome> delivered;
    std::thread client([&registry, &delivered, completion_id] {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        delivered = registry.request_reasoning_end(completion_id);
    });

    const ninfer::GenerationResult result = handle.wait();
    client.join();
    check(delivered.has_value() && delivered->success && delivered->message.empty(),
          "the control request was accepted while the completion was live",
          delivered.has_value() ? delivered->message : std::string("not delivered"));
    check(result.thinking.applied,
          "a signal delivered through the serve registry reached the Engine");

    // The response is over, so the registration that outlived the generation goes with it.
    registration.reset();
    check(registry.request_reasoning_end(completion_id).message ==
              "no active completion for this id",
          "a finished completion stops matching");
    check(registry.size() == 0, "the registry is empty once every response has ended");
    return g_failures - failures_before;
}

int run(const char* artifact) {
    const int engine_failures = exercise_engine_control(artifact);
    const int registry_failures = exercise_registry(artifact);
    std::cout << "reasoning control real-model failures: engine=" << engine_failures
              << " registry=" << registry_failures << '\n';
    return engine_failures + registry_failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return run(artifact) == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}