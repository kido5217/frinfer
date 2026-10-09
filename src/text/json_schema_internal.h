#pragma once

// Adapter-internal JSON-schema-to-grammar entry. Declared separately from json_schema.h so the
// vendor-typed signature stays out of headers that model code includes.

#include "ninfer/types.h"

namespace xgrammar {
class Grammar;
}

namespace ninfer::text {

// Builds the grammar for an admitted JSON object/schema body constraint.
xgrammar::Grammar build_json_grammar(const OutputConstraint& constraint);

} // namespace ninfer::text
