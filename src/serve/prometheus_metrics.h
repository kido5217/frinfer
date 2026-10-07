#pragma once

// Prometheus text exposition for the server's live counters. The renderer is a pure function of one
// snapshot so a scrape is reproducible and its values can be checked against the JSONL records the
// same snapshot feeds.

#include "ninfer/types.h"

#include <cstdint>
#include <string>

namespace ninfer::serve {

// The exact state a scrape publishes. `runtime` is the same Engine snapshot the JSONL `throughput`
// event carries as its `current` state; `requests_started` is the serve-side count of generation
// requests a protocol route has begun.
struct PrometheusSnapshot {
    ninfer::RuntimeStats runtime;
    ninfer::MemorySummary memory;
    std::uint64_t requests_started = 0;
};

// Prometheus text exposition format 0.0.4, terminated by a trailing newline.
std::string render_prometheus_metrics(const PrometheusSnapshot& snapshot);

} // namespace ninfer::serve
