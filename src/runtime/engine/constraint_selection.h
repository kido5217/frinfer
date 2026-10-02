#pragma once

#include "runtime/contract/request.h"

#include <cstdint>

namespace ninfer::runtime {

// Wrapper selection for constrained generation: a constrained request gets the full-stream
// thinking wrapper exactly when it resolves thinking enabled and its rendered prompt starts in
// reasoning. Then the constraint carries the reasoning stream and hands off to the answer grammar
// at the wire-format close. Every other constrained request compiles the plain answer grammar;
// the Program's per-lane constraint state engages the mask from token 0 (constraint_state.h).
[[nodiscard]] constexpr bool
constraint_carries_reasoning(const std::optional<GrammarConstraint>& constraint,
                             bool starts_in_reasoning) noexcept {
    return constraint.has_value() && constraint->thinking_enabled && starts_in_reasoning;
}

} // namespace ninfer::runtime
