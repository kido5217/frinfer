// Focused test for the shared constrained-turn compatibility rule: the gateway fail-fast and the
// Engine's submission boundary consult one validator.

#include "runtime/contract/constraint_compatibility.h"

#include <iostream>
#include <string>

namespace {

using ninfer::runtime::ConstrainedTurnOptions;
using ninfer::runtime::validate_constrained_turn;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    failures += check(!validate_constrained_turn({}).has_value(),
                      "an unconstrained turn is admissible");

    ConstrainedTurnOptions admissible;
    admissible.constrained = true;
    failures += check(!validate_constrained_turn(admissible).has_value(),
                      "a constrained turn with default EOS and no custom stops is admissible");

    auto violates = [](ConstrainedTurnOptions options) {
        return validate_constrained_turn(options).has_value();
    };

    {
        ConstrainedTurnOptions options;
        options.constrained       = true;
        options.model_default_eos = false;
        const auto violation      = validate_constrained_turn(options);
        failures += check(violation.has_value() && !violation->tools,
                          "a constraint without a model default EOS is rejected as a body constraint");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained_tools   = true;
        options.custom_stop_strings = true;
        const auto violation        = validate_constrained_turn(options);
        failures += check(violation.has_value() && violation->tools,
                          "constrained tools with custom stops is a tool violation");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained       = true;
        options.constrained_tools = true;
        options.tool_contract     = true;
        const auto violation      = validate_constrained_turn(options);
        failures += check(violation.has_value() && violation->tools,
                          "a body constraint with a constrained tool grammar is a tool violation");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained   = true;
        options.tool_contract = true;
        const auto violation  = validate_constrained_turn(options);
        failures += check(violation.has_value() && !violation->tools,
                          "a body constraint with a non-constrained tool contract is a body "
                          "violation");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained           = true;
        options.preserve_special_tokens = true;
        failures += check(violates(options), "preserved special tokens violate a constraint");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained         = true;
        options.publish_stop_token  = true;
        failures += check(violates(options), "published stop tokens violate a constraint");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained         = true;
        options.custom_stop_tokens  = true;
        failures += check(violates(options), "custom stop tokens violate a constraint");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained           = true;
        options.include_default_stops = false;
        failures += check(violates(options), "dropping the default stops violates a constraint");
    }
    {
        ConstrainedTurnOptions options;
        options.constrained  = true;
        options.raw_output   = true;
        failures += check(violates(options), "raw output violates a constraint");
    }

    // The constrained-turn speculative-backend rule: only the ordinary (None) and MTP backends can
    // carry a constraint; every other backend cannot (design #48).
    using ninfer::SpeculativeBackend;
    using ninfer::runtime::constraint_backend_supported;
    failures += check(constraint_backend_supported(SpeculativeBackend::None),
                      "the ordinary backend supports a constrained turn");
    failures += check(constraint_backend_supported(SpeculativeBackend::Mtp),
                      "MTP supports a constrained turn");
    failures += check(!constraint_backend_supported(SpeculativeBackend::DFlash),
                      "DFlash does not support a constrained turn");
    failures += check(!constraint_backend_supported(SpeculativeBackend::DFlash2),
                      "DFlash2 does not support a constrained turn");

    if (failures == 0) { std::cout << "Constrained-turn compatibility tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
