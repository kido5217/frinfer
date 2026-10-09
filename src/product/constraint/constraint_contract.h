#pragma once

// Protocol-neutral JSON Schema constrained-decoding contract shared by every product gateway
// (CLI and HTTP serving; wayfinder #52, design #48). A schema is admitted only when every keyword
// it uses is one FrInfer enforces; violations raise the contract's fail-closed errors. The
// admitted document is forwarded as an OutputConstraint::json_schema source and compiled by the
// Engine's XGrammar converter (src/text/json_schema.*), not by this gateway.

#include <ninfer/types.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::constraint {

// 64 KiB per grammar/schema payload, nesting depth <= 64 (design #48).
inline constexpr std::size_t kConstraintPayloadLimit = 64U * 1024U;
inline constexpr int kConstraintNestingLimit         = 64;

// Protocol-neutral fail-closed rejection. Each protocol layer renders `code`/`param` into its own
// error body shape; the Engine's own rejection for an uncompilable grammar is reported separately
// by the caller.
class ConstraintError final : public std::invalid_argument {
public:
    ConstraintError(std::string message, std::string param, std::string code)
        : std::invalid_argument(std::move(message)), param_(std::move(param)),
          code_(std::move(code)) {}

    [[nodiscard]] const std::string& param() const noexcept { return param_; }

    [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
    std::string param_;
    std::string code_;
};

// Validates `schema` against the v1 allowlist and returns the document text carried to the Engine
// as a JSON-schema constraint. Throws ConstraintError with `json_schema_unsupported`,
// `json_schema_invalid`, or `constraint_too_large` when the document cannot be enforced exactly.
[[nodiscard]] std::string json_schema_constraint_source(const nlohmann::ordered_json& schema);

// Which constraint language a request field carried, protocol-neutral. The route maps this to its
// own error attribution (`ninfer::serve::ConstraintSource`).
enum class ConstraintOrigin : std::uint8_t {
    None,
    Grammar,
    Choice,
    Regex,
    JsonObject,
    JsonSchema,
};

// One admitted constraint language and the protocol field that carried it. `field` is the full
// field path used for error attribution (e.g. "structured_outputs.regex", "response_format",
// "output_config.format").
struct AdmittedConstraint {
    ninfer::OutputConstraint value;
    ConstraintOrigin origin = ConstraintOrigin::None;
    std::string field;
};

// The single selected option of a well-formed NInfer `structured_outputs` object ("grammar",
// "regex" or "choice"). Throws ConstraintError with the `structured_outputs.*` shape codes.
[[nodiscard]] std::string structured_outputs_option(const nlohmann::ordered_json& value);

// Parses the option `structured_outputs_option` selected into an admitted constraint, with the
// shared payload limit and codes. Throws ConstraintError on any violation.
[[nodiscard]] AdmittedConstraint
admit_structured_outputs_option(const nlohmann::ordered_json& value, std::string_view option);

// The whole `structured_outputs` extension: shape then option in one call.
[[nodiscard]] AdmittedConstraint admit_structured_outputs(const nlohmann::ordered_json& value);

// The llama.cpp-compatible top-level GBNF field. Empty text carries no constraint and yields no
// result; malformed or oversized text throws ConstraintError on `field`.
[[nodiscard]] std::optional<AdmittedConstraint> admit_grammar(const nlohmann::ordered_json& value,
                                                              std::string_view field);

// A `{"type":"json_object"}` response format.
[[nodiscard]] AdmittedConstraint admit_json_object(std::string_view field);

// Adopts a JSON Schema document through the shared allowlist and returns the admitted constraint
// carrying the document text the Engine compiles. A contract rejection is reported with
// `error_param`; an empty `error_param` keeps the contract's own (schema-relative) path.
[[nodiscard]] AdmittedConstraint admit_json_schema(const nlohmann::ordered_json& schema,
                                                   std::string_view field,
                                                   std::string_view error_param);

// The two cross-language rejections, with one message and code each. The route supplies the exact
// protocol field for `param` (chat and anthropic attribute these differently).
[[noreturn]] void reject_second_constraint(std::string_view param);
[[noreturn]] void reject_constraint_with_tools(std::string_view param);

} // namespace ninfer::constraint
