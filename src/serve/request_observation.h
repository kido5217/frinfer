#pragma once

// Shared per-request observation derivations. The JSONL request log, the pretty operational log,
// and the throughput renderers consume the same runtime counters, so the monotonic-delta convention
// and the host-active time sum live here once instead of per renderer.

#include "ninfer/types.h"

#include <cstdint>

namespace ninfer::serve {

// A runtime statistic is monotonic; a non-monotonic readout (a reset counter or a raced snapshot)
// contributes zero rather than a negative delta.
template <class T>
[[nodiscard]] constexpr T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

// One interval's host-work counters.
[[nodiscard]] inline ninfer::RuntimeHostWorkStats
host_work_delta(const ninfer::RuntimeHostWorkStats& previous,
                const ninfer::RuntimeHostWorkStats& current) noexcept {
    return ninfer::RuntimeHostWorkStats{
        .engine_boundary_ns =
            monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns),
        .program_submit_ns = monotonic_delta(previous.program_submit_ns, current.program_submit_ns),
        .program_post_ns   = monotonic_delta(previous.program_post_ns, current.program_post_ns),
        .engine_commit_output_ns =
            monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns),
        .engine_maintenance_ns =
            monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns),
        .device_wait_ns = monotonic_delta(previous.device_wait_ns, current.device_wait_ns),
        .decode_host_ns = monotonic_delta(previous.decode_host_ns, current.decode_host_ns),
        .decode_device_wait_ns =
            monotonic_delta(previous.decode_device_wait_ns, current.decode_device_wait_ns),
        .prefill_host_ns = monotonic_delta(previous.prefill_host_ns, current.prefill_host_ns),
        .prefill_device_wait_ns =
            monotonic_delta(previous.prefill_device_wait_ns, current.prefill_device_wait_ns),
        .control_host_ns = monotonic_delta(previous.control_host_ns, current.control_host_ns),
        .control_device_wait_ns =
            monotonic_delta(previous.control_device_wait_ns, current.control_device_wait_ns),
        .prefill_units = monotonic_delta(previous.prefill_units, current.prefill_units),
        .control_units = monotonic_delta(previous.control_units, current.control_units),
        .stats_publication_ns =
            monotonic_delta(previous.stats_publication_ns, current.stats_publication_ns),
        .stats_publication_invocations = monotonic_delta(previous.stats_publication_invocations,
                                                         current.stats_publication_invocations),
    };
}

// The sum of the five host-active phases of one counter set.
[[nodiscard]] constexpr std::uint64_t
host_active_ns(const ninfer::RuntimeHostWorkStats& work) noexcept {
    return work.engine_boundary_ns + work.program_submit_ns + work.program_post_ns +
           work.engine_commit_output_ns + work.engine_maintenance_ns;
}

} // namespace ninfer::serve
