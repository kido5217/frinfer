#pragma once

namespace ninfer::runtime {

// The fork product contract: one GPU, one resident model, startup-fixed concurrency of one to eight
// requests, bounded FIFO ingress, and NO active-request preemption (one compact decode batch per
// round). This type is the single owner of the pin: admission, reclaim, and the pause entry point
// consult `may_pause_resident()` instead of each encoding the decision.
//
// The upstream pause body is retained as the merge reference
// (`EngineCore::pause_resident_unreachable`) and is never wired in; lifting the pin means wiring
// that body back and re-validating full-resident capacity, not flipping this constant alone. See
// docs/adr/0003-full-resident-capacity-under-pinned-preemption.md.
struct PreemptionPolicy {
    static constexpr bool kResidentPauseEnabled = false;

    [[nodiscard]] static constexpr bool may_pause_resident() noexcept {
        return kResidentPauseEnabled;
    }
};

} // namespace ninfer::runtime
