// The constrained-request policy seam: wrapper selection and mask engagement.
#include "runtime/engine/constraint_selection.h"

#include <cstddef>
#include <iostream>
#include <optional>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << message << '\n';
    }
}

} // namespace

int main() {
    using ninfer::GrammarConstraint;
    using ninfer::runtime::constraint_carries_reasoning;
    using ninfer::runtime::constraint_mask_engagement;

    constexpr auto kGrammar = "root ::= \"ok\"";
    const std::optional<GrammarConstraint> unconstrained;
    const std::optional<GrammarConstraint> answer =
        GrammarConstraint{.gbnf = kGrammar, .thinking_enabled = false};
    const std::optional<GrammarConstraint> thinking =
        GrammarConstraint{.gbnf = kGrammar, .thinking_enabled = true};

    // Wrapper selection: only an explicitly thinking constrained request whose rendered prompt
    // starts in reasoning carries the reasoning stream; everything else stays answer-only.
    check(!constraint_carries_reasoning(unconstrained, true),
          "an unconstrained request selected the thinking wrapper");
    check(!constraint_carries_reasoning(answer, true),
          "a thinking-off constrained request selected the thinking wrapper");
    check(!constraint_carries_reasoning(thinking, false),
          "a non-reasoning prompt selected the thinking wrapper");
    check(constraint_carries_reasoning(thinking, true),
          "an explicitly thinking reasoning prompt did not select the thinking wrapper");

    // Mask engagement: a constrained request masks every column as an answer column (region 0)
    // from token 0. A prefill step plans one column and a decode round plans the draft span plus
    // the bonus column, so the thinking-off plan is the pre-wrapper one for a normal template.
    const ninfer::runtime::ConstraintMaskEngagement prefill = constraint_mask_engagement(1);
    check(prefill.column_count == 1 && prefill.reasoning_mask == 0,
          "the constrained prefill plan is not a single answer column");
    for (const std::size_t columns : {std::size_t{2}, std::size_t{5}, std::size_t{9}}) {
        const ninfer::runtime::ConstraintMaskEngagement round = constraint_mask_engagement(columns);
        check(round.column_count == columns && round.reasoning_mask == 0,
              "a constrained decode round is not all answer columns");
    }
    return failures == 0 ? 0 : 1;
}
