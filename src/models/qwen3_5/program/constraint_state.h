#pragma once

// Per-lane constrained-generation state: the compiled grammar attached to a request lane, the
// runtime tracking its committed position, and the mask plan of the round currently being filled.
// The Program owns one per lane; the engine only compiles the grammar and hands it over, so no
// mask plan crosses the Program interface (wayfinder map "concentrate the constrained-round
// lifecycle").
//
// Round planning follows the engagement rule: every column of a constrained lane is an answer
// column from token 0 (the thinking wrapper admits the reasoning bytes itself), so a prefill step
// plans one column and a decode round plans the draft span plus the bonus column. Filling drives
// the physical transport (mask_transport.h); the module never touches sequences or the ledger.

#include "models/qwen3_5/frontend/grammar/grammar_runtime.h"
#include "models/qwen3_5/program/mask_transport.h"

#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::models::qwen3_5 {

class ConstraintState {
public:
    ConstraintState() = default;

    // Attaches a compiled grammar (null clears) and builds its runtime. Every attach starts from
    // the grammar's initial state, even when an earlier request on this lane shared the same
    // compiled grammar. `carries_reasoning` marks the thinking wrapper: forced thinking-control
    // tokens then advance the grammar state as answer tokens.
    void attach(std::shared_ptr<const frontend::CompiledGrammar> grammar, bool carries_reasoning);
    void clear() noexcept;

    [[nodiscard]] bool constrained() const noexcept { return runtime_ != nullptr; }

    [[nodiscard]] bool carries_reasoning() const noexcept { return carries_reasoning_; }

    // Plans the round: a constrained lane masks the draft span plus the bonus column (a prefill
    // step passes no drafts and plans one column). Returns the planned column count; zero when the
    // lane is unconstrained.
    std::uint32_t plan_round(std::span<const TokenId> drafts);

    [[nodiscard]] std::uint32_t planned_columns() const noexcept { return planned_columns_; }

    // Fills the planned row into the transport staging at `batch_position`. Unconstrained lanes
    // and zero-column plans are no-ops.
    void fill_row(std::span<const TokenId> drafts, std::uint32_t batch_position,
                  MaskTransport& transport);

    // Advances the committed state by the round's licensed tokens. False means the mask licensed
    // an ungrammatical token: a fail-closed generation error. A lane without a planned round is a
    // no-op.
    [[nodiscard]] bool commit(std::span<const int> accepted);
    // The thinking wrapper's forced control span advances the grammar like answer tokens; lanes
    // without the wrapper no-op.
    [[nodiscard]] bool commit_forced(std::span<const int> forced);

private:
    std::shared_ptr<const frontend::CompiledGrammar> grammar_;
    std::unique_ptr<frontend::GrammarRuntime> runtime_;
    bool carries_reasoning_        = false;
    std::uint32_t planned_columns_ = 0;
};

} // namespace ninfer::models::qwen3_5
