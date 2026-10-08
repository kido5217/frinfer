#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "ninfer/engine.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationMetrics {
    std::uint64_t engine_request_id       = 0;
    std::uint32_t computed_prefill_tokens = 0;
    double prepare_seconds                = 0.0;
    double ttft_seconds                   = 0.0;
    double vision_seconds                 = 0.0;
    double prefill_seconds                = 0.0;
    double decode_seconds                 = 0.0;
    double prompt_wall_seconds            = 0.0;
    double generation_wall_seconds        = 0.0;
    double total_seconds                  = 0.0;
    ninfer::GenerationEngineTiming engine_timing;
    std::optional<ninfer::GenerationFirstOutputTiming> first_output_timing;
    ninfer::GenerationSchedulingStats scheduling;
    ninfer::GenerationAdmissionStats admission;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    ninfer::PrefixReusePath prefix_reuse_path = ninfer::PrefixReusePath::Root;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    // Per-content-token logprob records when the request opted in; empty otherwise.
    std::vector<ninfer::TokenLogprob> content_logprobs;
    std::vector<ninfer::TokenId> generated_token_ids;
    std::vector<ninfer::GeneratedToolCall> tool_calls;
    ninfer::ToolCallParseDiagnostics tool_call_parse;
    // Constrained-decoding diagnostics for a constrained request; null otherwise.
    std::optional<ninfer::ConstraintObservation> constraint;
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int reasoning_tokens  = 0;
    ninfer::ThinkingBudgetStats thinking;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    GenerationMetrics metrics;
};

// A transport may report an interrupted stream through a callback or is_cancelled().
// Generation still settles through Engine cancellation and returns its actual work statistics.
class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamSink {
    std::function<void(const ninfer::GenerationStart& start)> on_start;
    std::function<void(const ninfer::PromptProgress& progress)> on_progress;
    std::function<void(const ninfer::GenerationTimingObservation& timing)> on_timing;
    // Content delta with the logprob records that arrived in the same output delta (token order).
    // Passed together so a route can emit one event carrying both the text and its logprobs; the
    // records are empty when logprobs are not requested for this delta.
    std::function<void(const std::string& delta_text, std::vector<ninfer::TokenLogprob> logprobs)>
        on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    // ADR-0002: a demoted tool-call region (`text` holds only the region bytes). The route either
    // signals the class (call loss) or republishes the bytes as ordinary content (benign).
    std::function<void(const std::string& text, const ninfer::ToolCallDemotion& demotion)>
        on_tool_call_demoted;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
// `source` selects the contract code for a rejected constraint (grammar_invalid vs
// json_schema_invalid); other kinds ignore it.
ApiError request_error_to_api_error(const ninfer::RequestError& exception,
                                    ConstraintSource source = ConstraintSource::None);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    ninfer::GenerationHandle generation;
    ninfer::ResolvedSamplingParameters sampling;
    double prepare_seconds     = 0.0;
    double acquisition_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, StartupObserver startup_observer = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from ServeOptions.
    [[nodiscard]] const ninfer::EngineOptions& engine_options() const { return engine_->options(); }

    [[nodiscard]] ninfer::LoadSummary load_summary() const { return engine_->load_summary(); }

    [[nodiscard]] ninfer::MemorySummary memory_summary() const { return engine_->memory_summary(); }

    [[nodiscard]] ninfer::RuntimeStats runtime_stats() const { return engine_->runtime_stats(); }

    [[nodiscard]] bool is_available() const { return engine_->is_available(); }

    [[nodiscard]] ninfer::MediaCacheSummary media_cache_summary() const {
        return engine_->media_cache_summary();
    }

    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          ninfer::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled               = {},
                                          ContextCacheHints context_cache = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

private:
    enum class CacheParticipation : std::uint8_t {
        Disabled,
        ReadWrite,
    };

    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] PreparedRequest
    prepare_impl(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                 ninfer::GenerationObservationOptions observation,
                 std::function<bool()> is_cancelled, ContextCacheHints context_cache,
                 CacheParticipation cache_participation, DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;

    ServeOptions options_;
    std::unique_ptr<ninfer::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
};

} // namespace ninfer::serve
