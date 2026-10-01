// Test-only access to the grammar module internals (wayfinder map #45, D4 validation):
// the vendored engine's full-vocabulary walk as a differential oracle for the compiled
// mask producer, and standalone producer instances over the same compiled backbone (so
// the differential walk can run with the producer's shape cache disabled, as the
// D4 decision requires for oracle runs).
#pragma once

#include <memory>
#include <vector>

#include "models/qwen3_5/frontend/grammar/grammar.h"

namespace ninfer::models::qwen3_5::frontend::detail {
class Producer;
}

namespace ninfer::models::qwen3_5::frontend {

class GrammarTestAccess {
public:
    // The vendored engine's fresh full-vocabulary walk for one state: no producer, no
    // caching - the independent oracle the producer's rows are compared against.
    static std::vector<std::uint32_t> engine_walk_row(const CompiledGrammar& grammar,
                                                      const GrammarState& state);

    // The grammar's production producer instance (diagnostics: shape cache, stats).
    static const detail::Producer& producer(const CompiledGrammar& grammar);

    // A standalone producer over the grammar's compiled backbone (trie + prepare
    // rebuilt at test time) with the shape cache set as requested.
    static std::shared_ptr<detail::Producer> make_producer(const CompiledGrammar& grammar,
                                                           bool shape_cache);
};

}  // namespace ninfer::models::qwen3_5::frontend
