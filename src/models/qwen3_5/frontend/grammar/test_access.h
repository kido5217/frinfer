// Test-only access to the grammar module internals (wayfinder map #45, D4 validation):
// standalone producer instances over the same compiled backbone, so the differential walk can
// run with the producer's shape cache disabled, as the D4 decision requires for oracle runs.
// The oracle itself is built directly on the vendored engine by the test.
#pragma once

#include <memory>

#include "models/qwen3_5/frontend/grammar/grammar.h"

namespace ninfer::models::qwen3_5::frontend::detail {
class Producer;
}

namespace ninfer::models::qwen3_5::frontend {

class GrammarTestAccess {
public:
    // A standalone producer over the grammar's compiled backbone (trie + prepare rebuilt at test
    // time) with the shape cache set as requested.
    static std::shared_ptr<detail::Producer> make_producer(const CompiledGrammar& grammar,
                                                           bool shape_cache);
};

} // namespace ninfer::models::qwen3_5::frontend
