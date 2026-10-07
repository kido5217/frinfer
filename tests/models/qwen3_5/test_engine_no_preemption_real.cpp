// Fork product contract: no active-request preemption. Two concurrent requests run with history
// disabled and the preemption/recovery counters stay zero. The pool covers every resident lane at
// full context, so no resident can ever lose its legal execution unit; a pool below that bound is
// rejected at startup instead of failing a resident later. The rejection is asserted here and
// pinned by the message, which names the required and supplied capacity.
//
// NINFER_TEST_ARTIFACT selects the artifact; without it the test skips (exit 77).

#include "ninfer/engine.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kPromptTokens = 192;
constexpr std::uint32_t kOutputTokens = 256;
constexpr std::uint32_t kCapacity     = 1024;
// The pinned no-preemption contract sizes the pool for every resident lane at full context. The
// tight pool below (one lane's worth) is what the startup validator must reject.
constexpr std::uint32_t kPoolCapacity = 2 * kCapacity;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const std::filesystem::path& artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = kCapacity;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity(kPoolCapacity);
    options.prefill_chunk                     = 128;
    options.max_concurrency                   = 2;
    options.max_pending_requests              = 2;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    // History is deliberately disabled: the two requests exercise only their own execution
    // resources, so any pause/replay would be active-request preemption, not cache reclaim.
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    return options;
}

ninfer::RequestOptions request(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

// A pool that cannot cover every resident lane at full context must be rejected at startup, with a
// message naming the required and supplied capacity. This is the contract pin for option C: a
// compliant pool means a resident can never lose its completion ability at runtime.
void exercise_rejection(const std::filesystem::path& artifact) {
    auto options        = engine_options(artifact);
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    bool rejected       = false;
    std::string message;
    try {
        ninfer::Engine engine(options);
    } catch (const std::invalid_argument& error) {
        rejected = true;
        message  = error.what();
    }
    require(rejected, "a KV pool below max_concurrency lanes at full context was accepted");
    require(message.find("kv_capacity must cover every resident lane at full max_context") !=
                    std::string::npos &&
                message.find("32 Main KV pages") != std::string::npos &&
                message.find("1024 tokens") != std::string::npos,
            "the capacity rejection did not name the required and supplied capacity");
    std::cout << "no-preemption rejection: " << message << '\n';
}

void exercise(const std::filesystem::path& artifact) {
    ninfer::Engine engine(engine_options(artifact));
    const auto before = engine.runtime_stats();

    const std::vector<ninfer::TokenId> tokens(kPromptTokens, 1002);
    std::vector<ninfer::GenerationResult> results(2);
    std::vector<std::exception_ptr> errors(2);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> workers;
    workers.reserve(2);
    for (std::size_t row = 0; row < 2; ++row) {
        workers.emplace_back([&, row] {
            try {
                auto prompt = engine.prepare_tokens(tokens);
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                results[row] =
                    engine.generate(std::move(prompt), request(kOutputTokens));
            } catch (...) { errors[row] = std::current_exception(); }
        });
    }
    while (ready.load(std::memory_order_acquire) != 2) { std::this_thread::yield(); }
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    for (const auto& error : errors) {
        if (error) { std::rethrow_exception(error); }
    }

    for (const auto& result : results) {
        require(result.generated_token_ids.size() == kOutputTokens,
                "a concurrent request did not complete its full output");
        require(result.finish_reason == ninfer::FinishReason::OutputLimit,
                "a concurrent request finished for the wrong reason");
        require(result.scheduling.preemptions == 0 &&
                    result.scheduling.snapshot_restores == 0 &&
                    result.scheduling.replay_restores == 0 &&
                    result.scheduling.replayed_tokens == 0 && result.scheduling.paused_ns == 0,
                "a request reported active-request preemption/recovery work");
    }

    const auto memory = engine.memory_summary();
    (void)memory;
    const auto after = engine.runtime_stats();
    require(after.preemptions == before.preemptions,
            "Engine preemption counter advanced under fixed concurrency");
    require(after.snapshot_restores == before.snapshot_restores &&
                after.replay_restores == before.replay_restores &&
                after.replayed_tokens == before.replayed_tokens,
            "Engine preemption/recovery counters advanced under fixed concurrency");
    require(after.paused_requests == 0, "a resident request was left paused");
    require(after.decode_rounds > before.decode_rounds,
            "the fixture never executed a decode round");
    require(after.computed_prefill_tokens - before.computed_prefill_tokens == 2 * kPromptTokens &&
                after.committed_decode_tokens - before.committed_decode_tokens ==
                    2 * (kOutputTokens - 1),
            "preemption was charged to initial prefill or delivered decode tokens");

    std::cout << "no-preemption backend=none preemptions="
              << after.preemptions - before.preemptions << " snapshot_restores="
              << after.snapshot_restores - before.snapshot_restores << " replay_restores="
              << after.replay_restores - before.replay_restores << " replayed_tokens="
              << after.replayed_tokens - before.replayed_tokens << " decode_rounds="
              << after.decode_rounds - before.decode_rounds << " decode_row_rounds="
              << after.decode_row_rounds - before.decode_row_rounds << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        exercise(std::filesystem::path(artifact));
        exercise_rejection(std::filesystem::path(artifact));
    } catch (const std::exception& error) {
        std::cerr << "no-preemption real test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
