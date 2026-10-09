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

// One admitted constraint language and the protocol field it came from. `field` is the full field
// path used for error attribution (e.g. "structured_outputs.regex", "response_format",
// "output_config.format").
struct ConstraintSlot {
    std::string field;
    ConstraintOrigin origin = ConstraintOrigin::None;
    ninfer::OutputConstraint value;
};

// The single output constraint a request carries, or none.
struct ConstraintAdmission {
    std::optional<ninfer::OutputConstraint> constraint;
    ConstraintOrigin origin = ConstraintOrigin::None;
};

// Adopts one route's already-validated constraint fields in protocol order and enforces the
// cross-language rules every gateway shares: at most one constraint language, and never a
// constrained turn alongside a declared `tools` field. Throws ConstraintError with the offending
// field as `param` (`constrained_decoding_conflict` / `constrained_decoding_not_supported`).
[[nodiscard]] ConstraintAdmission admit_constraint(std::vector<ConstraintSlot> slots,
                                                   bool has_tools);

// The protocol-neutral `structured_outputs` extension: exactly one of grammar, regex or choice,
// with the shared payload limit and codes. Throws ConstraintError on any violation.
[[nodiscard]] ConstraintSlot admit_structured_outputs(const nlohmann::ordered_json& value);

// The llama.cpp-compatible top-level GBNF field. Empty text carries no constraint and yields no
// slot; malformed or oversized text throws ConstraintError on `field`.
[[nodiscard]] std::optional<ConstraintSlot> admit_grammar(const nlohmann::ordered_json& value,
                                                          std::string_view field);

// A `{"type":"json_object"}` response format.
[[nodiscard]] ConstraintSlot admit_json_object(std::string_view field);

// Adopts a JSON Schema document through the shared allowlist and returns the slot carrying the
// document text the Engine compiles. A contract rejection is reported with `error_param`; an empty
// `error_param` keeps the contract's own (schema-relative) path.
[[nodiscard]] ConstraintSlot admit_json_schema(const nlohmann::ordered_json& schema,
                                               std::string_view field,
                                               std::string_view error_param);

} // namespace ninfer::constraint
