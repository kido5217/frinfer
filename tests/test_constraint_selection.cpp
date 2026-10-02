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

    // Mask engagement: a constrained request masks every column as an answer column from token 0.
    // A prefill step plans one column and a decode round plans the draft span plus the bonus
    // column.
    check(constraint_mask_engagement(1) == 1, "the constrained prefill plan is not one column");
    for (const std::size_t columns : {std::size_t{2}, std::size_t{5}, std::size_t{9}}) {
        check(constraint_mask_engagement(columns) == columns,
              "a constrained decode round does not mask every column");
    }
    return failures == 0 ? 0 : 1;
}
