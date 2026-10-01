#pragma once

// JSON Schema constrained-decoding contract for the serve route (wayfinder #52, design #48).
// A schema is admitted only when every keyword it uses is one the vendored converter genuinely
// enforces; violations raise the contract's fail-closed 400s. The admitted document converts to
// GBNF with the same vendored converter the Engine compiles.

#include "serve/request_json.h"

#include <cstddef>
#include <string>

namespace ninfer::serve {

// 64 KiB per grammar/schema payload, nesting depth <= 64 (design #48).
inline constexpr std::size_t kConstraintPayloadLimit  = 64U * 1024U;
inline constexpr int         kConstraintNestingLimit  = 64;

// Validates `schema` against the v1 allowlist and converts it to GBNF for the Engine constraint
// contract. Raises ApiException(400) with `json_schema_unsupported`, `json_schema_invalid`, or
// `constraint_too_large` when the document cannot be enforced exactly.
[[nodiscard]] std::string json_schema_constraint_grammar(const RequestJson& schema);

} // namespace ninfer::serve
