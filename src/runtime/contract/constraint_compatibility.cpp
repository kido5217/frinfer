#include "runtime/contract/constraint_compatibility.h"

namespace ninfer::runtime {

std::optional<ConstrainedTurnViolation>
validate_constrained_turn(const ConstrainedTurnOptions& options) {
    if (!options.constrained && !options.constrained_tools) { return std::nullopt; }

    // A body constraint cannot share the turn with any active tool contract; the constrained-tool
    // report kind additionally requires the tool grammar itself to own the turn.
    const bool violates =
        !options.model_default_eos || (options.constrained && options.tool_contract) ||
        !options.include_default_stops || options.publish_stop_token || options.raw_output ||
        options.preserve_special_tokens || options.custom_stop_tokens ||
        options.custom_stop_strings;
    if (!violates) { return std::nullopt; }

    if (options.constrained_tools) {
        return ConstrainedTurnViolation{
            true,
            "constrained tools require default EOS, text output, no custom stops, and one output "
            "language"};
    }
    return ConstrainedTurnViolation{
        false, "constraints require default EOS, text output, no active tools and no custom stops"};
}

bool constraint_backend_supported(ninfer::SpeculativeBackend backend) noexcept {
    return backend == ninfer::SpeculativeBackend::None || backend == ninfer::SpeculativeBackend::Mtp;
}

} // namespace ninfer::runtime
