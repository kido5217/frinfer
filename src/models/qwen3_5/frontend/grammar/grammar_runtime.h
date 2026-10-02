#pragma once

#include "models/qwen3_5/frontend/grammar/grammar.h"

#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::models::qwen3_5::frontend {

// A decode column whose token is unknown at planning time (the bonus column, or a draft the
// runtime cannot name).
inline constexpr int kUnknownConstraintToken = -1;

// Per-request constrained-generation runtime: owns the committed grammar state and turns a
// round's decode columns into mask rows. Host-only; row production shares the compiled grammar's
// row cache and therefore runs on the single serving worker.
//
// Every constrained column is an answer column (the thinking wrapper admits the reasoning bytes
// itself). Round semantics (wayfinder map #45, ticket #59):
//  - a masked column uses the row of the state reached by the committed prefix plus the answer
//    columns before it, and advances the state by that column's token;
//  - a token the grammar rejects ends the reachable prefix, so later masked columns also get
//    all-ones rows - the runtime samples against the preceding row, so a rejected draft is
//    replaced by a licensed correction token before the next column can be generated.
class GrammarRuntime {
public:
    explicit GrammarRuntime(std::shared_ptr<const CompiledGrammar> grammar);

    GrammarRuntime(const GrammarRuntime&)            = delete;
    GrammarRuntime& operator=(const GrammarRuntime&) = delete;
    GrammarRuntime(GrammarRuntime&&) noexcept;
    GrammarRuntime& operator=(GrammarRuntime&&) noexcept;
    ~GrammarRuntime();

    // Fills `rows` - [columns x words] row-major, `words` covering the grammar's token domain -
    // for one decode round. `tokens` has one entry per column; a column whose token is not known
    // yet passes `kUnknownConstraintToken`.
    void fill_round_rows(std::span<const int> tokens, std::span<std::uint32_t> rows,
                         std::size_t words) const;

    // Advances the committed state by the round's licensed tokens. Returns false without changing
    // the state when a licensed token cannot be consumed: the mask guarantees licensed tokens are
    // grammatical, so this is a fail-closed generation error (an out-of-domain token or a runtime
    // defect), not a sampling outcome.
    [[nodiscard]] bool commit(std::span<const int> accepted);

    [[nodiscard]] const CompiledGrammar& grammar() const noexcept;

    // True when the grammar can end at the committed state; end-of-generation ids are allowed
    // exactly there.
    [[nodiscard]] bool can_end() const noexcept;

private:
    std::shared_ptr<const CompiledGrammar> grammar_;
    GrammarState state_;
};

} // namespace ninfer::models::qwen3_5::frontend
