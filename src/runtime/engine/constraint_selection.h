#pragma once

#include "runtime/contract/request.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::runtime {

// Wrapper selection for constrained generation: a constrained request gets the full-stream
// thinking wrapper exactly when it resolves thinking enabled and its rendered prompt starts in
// reasoning. Then the constraint carries the reasoning stream and hands off to the answer grammar
// at the wire-format close. Every other constrained request compiles the plain answer grammar;
// both engage their mask from token 0 (see constraint_mask_engagement).
[[nodiscard]] constexpr bool
constraint_carries_reasoning(const std::optional<GrammarConstraint>& constraint,
                             bool starts_in_reasoning) noexcept {
    return constraint.has_value() && constraint->thinking_enabled && starts_in_reasoning;
}

// Mask engagement for a constrained request: the grammar governs every column from token 0, so
// each column is an answer column (region 0) - the thinking wrapper admits the reasoning bytes
// itself, and an answer-only grammar is already at its answer position. `columns` is one for a
// prefill step, and the draft span plus the bonus column for a decode round. The phase-dependent
// deferral no longer exists for constrained requests.
struct ConstraintMaskEngagement {
    std::uint8_t column_count   = 0;
    std::uint8_t reasoning_mask = 0;
};

[[nodiscard]] constexpr ConstraintMaskEngagement
constraint_mask_engagement(std::size_t columns) noexcept {
    return ConstraintMaskEngagement{static_cast<std::uint8_t>(columns), 0};
}

} // namespace ninfer::runtime
