#pragma once

#include "models/qwen3_5/frontend/grammar/vocabulary.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

class CompiledGrammar;

// A mask row is a dense bitmask over the whole token domain: bit i is set when token i is
// allowed at the state the row was produced for. Rows are immutable and shared, so a row
// only has to be transferred when a state has no cached row yet.
using GrammarMaskRow = std::span<const std::uint32_t>;

// Mutable per-request grammar state: the committed position in a compiled grammar. Copies
// derive independent states from the same grammar (MTP column previews use a copy and
// discard it); a copy is an engine-state clone, not a pointer alias.
//
// The state is a host-only value: all methods run on the serving worker, no CUDA.
class GrammarState {
public:
    GrammarState(const GrammarState&);
    GrammarState& operator=(const GrammarState&);
    GrammarState(GrammarState&&) noexcept;
    GrammarState& operator=(GrammarState&&) noexcept;
    ~GrammarState();

    [[nodiscard]] const CompiledGrammar& grammar() const noexcept;

    // Advances the state by one sampled token. Returns false without advancing when the
    // token cannot be consumed here (empty piece, a rejected token, or a token after the
    // grammar already ended) so callers can fail closed on unconstrained-sampler input.
    // End-of-generation ids are accepted exactly when can_end() holds and never move the
    // state, matching the runtime's terminal semantics.
    [[nodiscard]] bool accept(int token_id);

    // True when the grammar can end at this state; end-of-generation ids are allowed at
    // exactly those states and nowhere else.
    [[nodiscard]] bool can_end() const noexcept;

private:
    friend class CompiledGrammar;
    class Impl;
    explicit GrammarState(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

// An immutable compiled GBNF grammar plus its shared per-state mask-row cache. Compiled
// grammars and rows are shared by grammar identity (the GBNF text); every state and row
// references the same instance. Not internally synchronized: row production runs on the
// serving worker.
class CompiledGrammar {
public:
    // Parses `gbnf` with root symbol "root". Returns nullptr and, when `error` is
    // non-null, stores the parse diagnostic when the grammar is rejected.
    [[nodiscard]] static std::shared_ptr<const CompiledGrammar>
    compile(std::string_view gbnf, std::shared_ptr<const GrammarVocabulary> vocabulary,
            std::string* error = nullptr);

    // Converts a JSON Schema document with the vendored llama.cpp converter and compiles
    // the resulting GBNF; conversion failures surface like parse errors.
    [[nodiscard]] static std::shared_ptr<const CompiledGrammar>
    compile_from_schema(std::string_view json_schema,
                        std::shared_ptr<const GrammarVocabulary> vocabulary,
                        std::string* error = nullptr);

    ~CompiledGrammar();
    CompiledGrammar(const CompiledGrammar&)            = delete;
    CompiledGrammar& operator=(const CompiledGrammar&) = delete;

    // The initial state: the grammar at its start position.
    [[nodiscard]] GrammarState initial_state() const;

    // Token domain of every row (the vocabulary's token count).
    [[nodiscard]] int token_count() const noexcept;

    // The allowed-token row of `state`: produced by a full-vocabulary walk on the first
    // visit, served from the cache afterwards. The returned view stays valid for the
    // lifetime of this grammar.
    [[nodiscard]] GrammarMaskRow row_for(const GrammarState& state) const;

    // Row-cache counters. `row_fills` counts full-vocabulary walks, `row_hits` counts
    // states served from the cache, `distinct_rows` counts rows actually stored after
    // content dedup and `fill_micros` accumulates walk time.
    struct Stats {
        std::uint64_t row_fills     = 0;
        std::uint64_t row_hits      = 0;
        std::uint64_t distinct_rows = 0;
        std::uint64_t fill_micros   = 0;
    };

    [[nodiscard]] Stats stats() const noexcept;

private:
    class Impl;
    explicit CompiledGrammar(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_5::frontend
