#pragma once

// The prompt-independent option compatibility a constrained turn requires: the model default EOS,
// text output, one output language, and no custom stops. The gateway fills this for an early
// fail-fast; the Engine's submission path (the frontend boundary) is the authority.

#include <optional>
#include <string>

#include "ninfer/types.h"

namespace ninfer::runtime {

struct ConstrainedTurnOptions {
    bool constrained             = false; // a body constraint (grammar / JSON schema / choice / regex)
    bool constrained_tools       = false; // a constrained tool grammar owns the turn
    bool tool_contract           = false; // any tool contract is active (constrained or not)
    bool model_default_eos       = true;  // the model declares a non-empty default EOS set
    bool include_default_stops   = true;  // the caller kept the model's default stop tokens/strings
    bool publish_stop_token      = false; // raw stop-token publication
    bool raw_output              = false;
    bool preserve_special_tokens = false;
    bool custom_stop_tokens      = false;
    bool custom_stop_strings     = false;
};

struct ConstrainedTurnViolation {
    bool tools = false; // the failing turn is reported as a constrained tool turn
    std::string message;
};

// Returns the first violated rule, or nullopt when the constrained turn is admissible. A turn that
// is not constrained is always admissible.
[[nodiscard]] std::optional<ConstrainedTurnViolation>
validate_constrained_turn(const ConstrainedTurnOptions& options);

// True when a speculative backend can carry a constrained turn: the ordinary backend (None) and
// MTP can, every other backend cannot. The gateway and the Engine both enforce this fail-closed
// rule (design #48), each in its own error shape.
[[nodiscard]] bool constraint_backend_supported(ninfer::SpeculativeBackend backend) noexcept;

} // namespace ninfer::runtime
