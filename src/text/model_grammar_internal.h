#pragma once

// Adapter-internal bridge: binds ModelGrammar to the vendored grammar it holds. Only src/text
// translation units may include this header; the facade (model_grammar.h) stays vendor-neutral so
// model code never names the vendored API.

#include "text/model_grammar.h"

#include <xgrammar/xgrammar.h>

#include <optional>

namespace ninfer::text {

class ModelGrammar::Impl {
public:
    // The vendored Grammar has no default constructor; an unset ModelGrammar is an empty handle.
    std::optional<xgrammar::Grammar> grammar;
};

} // namespace ninfer::text
