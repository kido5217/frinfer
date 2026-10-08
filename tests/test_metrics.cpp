#include "serve/metrics.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

namespace {

// Collect the sampled metric family names, collapsing histogram _bucket/_sum/_count samples back to
// their base family so a scrape enumerates each advertised series once.
std::set<std::string> metric_families(const std::string& text) {
    std::set<std::string> names;
    const std::string prefix = "frinfer_";
    for (std::size_t pos = text.find(prefix); pos != std::string::npos;
         pos = text.find(prefix, pos)) {
        std::size_t end = pos + prefix.size();
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) != 0 || text[end] == '_')) {
            ++end;
        }
        std::string name = text.substr(pos, end - pos);
        for (const std::string_view suffix : {"_bucket", "_sum", "_count"}) {
            if (name.size() > suffix.size() && name.ends_with(suffix)) {
                name.resize(name.size() - suffix.size());
                break;
            }
        }
        names.insert(std::move(name));
        pos = end;
    }
    return names;
}

// The documented Prometheus families; docs/serving.md and the renderer must agree on this set.
const std::set<std::string>& documented_metric_families() {
    static const std::set<std::string> kFamilies{
        "frinfer_checkpoint_selections_total",
        "frinfer_constraint_cache_total",
        "frinfer_constraint_matcher_seconds_total",
        "frinfer_constraint_mask_positions_total",
        "frinfer_constraint_mask_seconds_total",
        "frinfer_constraint_mask_upload_bytes_total",
        "frinfer_constraint_prepare_seconds_total",
        "frinfer_constraint_requests_total",
        "frinfer_context_transfer_bytes_total",
        "frinfer_context_transfer_seconds_total",
        "frinfer_control_units_total",
        "frinfer_decode_rounds_total",
        "frinfer_decode_row_rounds_total",
        "frinfer_decode_tokens_total",
        "frinfer_device_backend_kv_used_pages",
        "frinfer_device_kv_capacity_pages",
        "frinfer_device_kv_used_pages",
        "frinfer_device_state_capacity_slots",
        "frinfer_device_state_used_slots",
        "frinfer_device_wait_seconds_total",
        "frinfer_engine_ready",
        "frinfer_generation_tokens_total",
        "frinfer_host_context_capacity_bytes",
        "frinfer_host_context_peak_bytes",
        "frinfer_host_context_reserved_bytes",
        "frinfer_host_context_used_bytes",
        "frinfer_host_kv_used_bytes",
        "frinfer_host_state_images",
        "frinfer_host_work_seconds_total",
        "frinfer_max_concurrency",
        "frinfer_max_context_tokens",
        "frinfer_model_info",
        "frinfer_preemptions_total",
        "frinfer_prefill_tokens_total",
        "frinfer_prefill_units_total",
        "frinfer_prompt_tokens_cached_total",
        "frinfer_prompt_tokens_total",
        "frinfer_replay_restores_total",
        "frinfer_replayed_tokens_total",
        "frinfer_request_duration_seconds",
        "frinfer_request_queue_seconds",
        "frinfer_requests_decode_ready",
        "frinfer_requests_materializing",
        "frinfer_requests_paused",
        "frinfer_requests_prefilling",
        "frinfer_requests_replaying",
        "frinfer_requests_running",
        "frinfer_requests_started_total",
        "frinfer_requests_total",
        "frinfer_requests_waiting",
        "frinfer_response_failures_total",
        "frinfer_root_selections_total",
        "frinfer_server_start_time_seconds",
        "frinfer_snapshot_restores_total",
        "frinfer_spec_decode_accepted_tokens_total",
        "frinfer_spec_decode_draft_tokens_total",
        "frinfer_spec_decode_draft_window",
        "frinfer_spec_decode_fallback_steps_total",
        "frinfer_spec_decode_rounds_total",
        "frinfer_time_to_first_token_seconds",
    };
    return kFamilies;
}

} // namespace

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                  = 2;
    options.context_cache.device_state_slots = 1;
    MemorySummary memory;
    memory.kv_capacity_page_groups     = 256;
    memory.host_context_capacity_bytes = 1048576;
    RuntimeStats baseline;
    baseline.generated_tokens         = 4;
    baseline.speculative_draft_tokens = 3;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, memory, baseline);
    RuntimeStats running = baseline;
    running.generated_tokens += 5;
    running.speculative_draft_tokens += 6;
    running.speculative_accepted_tokens = 3;
    running.running_requests            = 1;
    running.host_context_occupied_bytes = 4096;
    running.host_context_reserved_bytes = 1024;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true, 0);
    check(live.find("frinfer_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("frinfer_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("frinfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("frinfer_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("frinfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(live.find("ninfer_") == std::string::npos,
          "the renderer must publish only the frinfer_ namespace");
    check(metrics.render(running, true, 0) == live, "scrapes must not consume or reset counters");
    check(live.find("frinfer_host_context_used_bytes 4096\n") != std::string::npos,
          "reserved bytes must not be added to occupancy twice");
    check(live.find("le=\"0.05\"") != std::string::npos &&
              live.find("0.050000000000000003") == std::string::npos,
          "histogram bucket boundaries must use canonical short precision, not 17 digits");
    const auto emitted = metric_families(live);
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
    const auto& documented = documented_metric_families();
    std::set_difference(documented.begin(), documented.end(), emitted.begin(), emitted.end(),
                        std::back_inserter(missing));
    std::set_difference(emitted.begin(), emitted.end(), documented.begin(), documented.end(),
                        std::back_inserter(unexpected));
    if (!missing.empty() || !unexpected.empty()) {
        std::cerr << "metric family drift; documented-but-absent:";
        for (const auto& name : missing) { std::cerr << ' ' << name; }
        std::cerr << "; emitted-but-undocumented:";
        for (const auto& name : unexpected) { std::cerr << ' ' << name; }
        std::cerr << '\n';
        ++failures;
    }

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.metrics.total_seconds                    = 2.0;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false, 0);
    check(final.find("frinfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("frinfer_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("frinfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("frinfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("frinfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("frinfer_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("frinfer_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("frinfer_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("frinfer_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    outcome.constraint = ConstraintObservation{.complete          = true,
                                               .terminated        = false,
                                               .cache             = ConstraintCacheAccess::Built,
                                               .mask_positions    = 5,
                                               .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false, 0);
    check(constrained.find(
              "frinfer_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("frinfer_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("frinfer_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
