#pragma once

// Protocol-neutral JSON Schema constrained-decoding contract shared by every product gateway
// (CLI and HTTP serving; wayfinder #52, design #48). A schema is admitted only when every keyword
// it uses is one FrInfer enforces; violations raise the contract's fail-closed errors. The
// admitted document is forwarded as an OutputConstraint::json_schema source and compiled by the
// Engine's XGrammar converter (src/text/json_schema.*), not by this gateway.

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

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

} // namespace ninfer::constraint
