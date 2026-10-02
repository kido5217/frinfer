// The constrained-request policy seam: wrapper selection.
#include "runtime/engine/constraint_selection.h"

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

    return failures == 0 ? 0 : 1;
}
