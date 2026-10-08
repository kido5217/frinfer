// Fork product contract: no active-request preemption. pause_resident never pauses a resident, so
// admission waits for capacity instead. This file pins the observable contract and the startup
// capacity rejection:
//
//   * exercise() runs two concurrent requests on two lanes with the context cache disabled and
//     asserts the request- and Engine-level preemption/recovery counters stay zero.
//   * exercise_cache_pressure() enables the context cache and drives the pool to its compliant
//     ceiling: two residents grow to full max_context (the whole shared pool) while a request that
//     must restore a cached context is admitted once a lane frees. It asserts no resident was paused
//     or preempted to admit the restore, and that no recovery work is charged.
//   * exercise_rejection() pins the startup validator: a pool that does not equal the full-resident
//     bound (max_concurrency lanes at full max_context) is rejected with a message naming the
//     required and supplied capacity, at both the undersized and the oversized end.
//
// The pin is enforced by construction, not by a runtime counter: the startup validator requires the
// shared Main KV pool to cover exactly max_concurrency lanes at full max_context
// (docs/adr/0003-full-resident-capacity-under-pinned-preemption.md), so every resident reservation
// succeeds and the reclaim/pause route is unreachable. The runtime's paused queue, and therefore
// the restore route through `reclaim(..., allow_pause = restoring && ...)`, is populated only by
// pause_resident succeeding, so it is empty under the pin. No public-Engine fixture can therefore
// distinguish a pinned from an un-pinned pause_resident: an un-pinned body is only reached under a
// pool pressure the compliant-pool validator forbids. These tests pin the behavior that is
// reachable; the validator and the rejection above are what rule out the un-pinned alternative.
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

// A pool that does not equal the pinned full-resident pool must be rejected at startup, with a
// message naming the required and supplied capacity. This is the contract pin for option C: a
// compliant pool means a resident can never lose its completion ability at runtime. Both ends of
// the usable range are covered: undersized (a resident could be starved) and oversized (the pool no
// longer matches the pinned bound; exactly max_concurrency lanes at full context is required).
// Returns the rejection message, or an empty string when the Engine accepted the capacity.
std::string rejection_message(const std::filesystem::path& artifact, std::uint32_t tokens) {
    auto options        = engine_options(artifact);
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(tokens);
    try {
        ninfer::Engine engine(options);
    } catch (const std::invalid_argument& error) { return std::string(error.what()); }
    return std::string{};
}

void exercise_rejection(const std::filesystem::path& artifact) {
    const std::string undersized = rejection_message(artifact, kCapacity);
    require(!undersized.empty(), "a KV pool below max_concurrency lanes at full context was accepted");
    require(undersized.find("kv_capacity must cover every resident lane at full max_context") !=
                    std::string::npos &&
                undersized.find("32 Main KV pages") != std::string::npos &&
                undersized.find("1024 tokens") != std::string::npos,
            "the undersized-capacity rejection did not name the required and supplied capacity");

    const std::string oversized = rejection_message(artifact, 3 * kCapacity);
    require(!oversized.empty(), "a KV pool above the full-resident pool was accepted");
    require(oversized.find("kv_capacity must equal the full-resident pool") != std::string::npos &&
                oversized.find("32 Main KV pages") != std::string::npos &&
                oversized.find("3072 tokens") != std::string::npos,
            "the oversized-capacity rejection did not name the required and supplied capacity");

    std::cout << "no-preemption rejection undersized: " << undersized << '\n';
    std::cout << "no-preemption rejection oversized: " << oversized << '\n';
}

// Context-cache pressure probe. The compliant pool cannot be exceeded by active residents — the
// startup validator requires exactly max_concurrency lanes at full max_context — so the engine's
// reclaim/pause route is only reachable when a materializing context competes with residents for
// the shared pool. This fixture establishes a restorable cached context, fills every lane with
// residents that grow to the full compliant context, and then admits a request that must restore
// the cached context once a lane frees. Under the pin no resident is paused or preempted to admit
// the restore and no recovery work is charged.
void exercise_cache_pressure(const std::filesystem::path& artifact) {
    auto options                              = engine_options(artifact);
    options.max_pending_requests              = 3;
    options.context_cache.enabled             = true;
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = 256ULL << 20;
    ninfer::Engine engine(std::move(options));
    const auto before = engine.runtime_stats();

    const auto reuse_request = [](std::uint32_t outputs) {
        ninfer::RequestOptions options       = request(outputs);
        options.execution.allow_prefix_reuse = true;
        return options;
    };

    // A completed half-context prompt leaves a restorable cached context: the source the later
    // request must restore while the pool is occupied.
    const std::vector<ninfer::TokenId> seed_prompt(kCapacity / 2, 1002);
    const ninfer::GenerationResult seed =
        engine.generate(engine.prepare_tokens(seed_prompt), reuse_request(4));
    require(seed.generated_token_ids.size() == 4 &&
                seed.prefix_reuse_path == ninfer::PrefixReusePath::Root,
            "cache-pressure seed did not establish a fresh cached context");

    // Two residents grow to the full compliant context (768 prompt + 256 output = max_context), so
    // together they need the entire shared pool. The restore request is submitted after them and
    // can only be admitted once a resident releases its lane.
    std::vector<ninfer::GenerationHandle> residents;
    residents.reserve(2);
    for (std::size_t row = 0; row < 2; ++row) {
        std::vector<ninfer::TokenId> prompt(768, 1003);
        prompt.front() = static_cast<ninfer::TokenId>(1004 + row);
        residents.push_back(engine.submit(engine.prepare_tokens(std::move(prompt)), request(256)));
    }
    auto restore = engine.submit(engine.prepare_tokens(seed_prompt), reuse_request(2));

    for (auto& resident : residents) {
        const ninfer::GenerationResult result = resident.wait();
        require(result.generated_token_ids.size() == 256 &&
                    result.finish_reason == ninfer::FinishReason::OutputLimit,
                "a cache-pressure resident did not complete its full output");
    }
    const ninfer::GenerationResult restored = restore.wait();
    require(restored.generated_token_ids.size() == 2,
            "the cache-pressure restore request did not complete");
    require(restored.reused_prompt_tokens != 0 &&
                restored.prefix_reuse_path == ninfer::PrefixReusePath::Checkpoint,
            "the cache-pressure request did not restore the cached context");

    const auto after = engine.runtime_stats();
    require(after.preemptions == before.preemptions, "cache pressure preempted a resident");
    require(after.snapshot_restores == before.snapshot_restores &&
                after.replay_restores == before.replay_restores &&
                after.replayed_tokens == before.replayed_tokens,
            "cache pressure triggered replay/snapshot recovery work");
    require(after.paused_requests == 0, "cache pressure left a resident paused");

    std::cout << "no-preemption cache-pressure reused=" << restored.reused_prompt_tokens
              << " preemptions=" << after.preemptions - before.preemptions
              << " paused=" << after.paused_requests << '\n';
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
        exercise_cache_pressure(std::filesystem::path(artifact));
        exercise_rejection(std::filesystem::path(artifact));
    } catch (const std::exception& error) {
        std::cerr << "no-preemption real test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
