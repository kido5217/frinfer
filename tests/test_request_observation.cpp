// Focused test for the shared request-observation derivations: the monotonic-delta convention and
// the host-active time sum.

#include "serve/request_observation.h"

#include <cstdint>
#include <iostream>

int main() {
    int failures = 0;
    using ninfer::RuntimeHostWorkStats;
    using ninfer::serve::host_active_ns;
    using ninfer::serve::host_work_delta;
    using ninfer::serve::monotonic_delta;

    if (monotonic_delta<std::uint64_t>(4, 10) != 6) {
        std::cerr << "FAIL: an increasing counter yields its delta\n";
        ++failures;
    }
    if (monotonic_delta<std::uint64_t>(10, 4) != 0) {
        std::cerr << "FAIL: a non-monotonic readout yields zero\n";
        ++failures;
    }

    RuntimeHostWorkStats work{};
    work.engine_boundary_ns      = 1;
    work.program_submit_ns       = 2;
    work.program_post_ns         = 3;
    work.engine_commit_output_ns = 4;
    work.engine_maintenance_ns   = 5;
    work.device_wait_ns          = 100;
    if (host_active_ns(work) != 15) {
        std::cerr << "FAIL: host-active sums exactly the five host phases\n";
        ++failures;
    }
    if (host_active_ns(host_work_delta(RuntimeHostWorkStats{}, work)) != 15) {
        std::cerr << "FAIL: a zero-baseline delta preserves the host-active time\n";
        ++failures;
    }

    if (failures == 0) { std::cout << "Request observation tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
