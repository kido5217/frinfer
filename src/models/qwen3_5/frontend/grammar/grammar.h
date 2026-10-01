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
// allowed at the state the row was produced for. Rows are produced by the compiled
// token-trie producer (trie_producer.h) and shared from its bounded state-shape store;
// a row only has to be transferred when the state shape has not been seen before.
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
    // token cannot be consumed here: an out-of-domain id, a token whose piece the mask
    // declares illegal (empty or NUL-leading, which the engine's decoder reads as nothing),
    // a token the grammar rejects, or a token after the grammar already ended. Callers
    // sample against row_for(), so a rejection is fail-closed recovery for unconstrained
    // input. End-of-generation ids are accepted exactly when can_end() holds and never move
    // the state, matching the runtime's terminal semantics.
    [[nodiscard]] bool accept(int token_id);

    // True when the grammar can end at this state; end-of-generation ids are allowed at
    // exactly those states and nowhere else.
    [[nodiscard]] bool can_end() const noexcept;

private:
    friend class CompiledGrammar;
    friend class GrammarTestAccess;
    class Impl;
    explicit GrammarState(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

// An immutable compiled GBNF grammar plus its compiled mask producer. Compiled grammars
// are shared by grammar identity (the GBNF text); every state references the same
// instance. Not internally synchronized: row production runs on the serving worker.
class CompiledGrammar {
public:
    // Parses `gbnf` with root symbol "root". Returns nullptr and, when `error` is
    // non-null, stores the parse diagnostic when the grammar is rejected. The one-time
    // mask-producer build (the codepoint trie, the class structures, and the
    // fresh-round counter tables) runs here, at compile time (D2a).
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

    // The allowed-token row of `state`: produced by the compiled token-trie producer
    // (the vendored engine's full-vocabulary walk is retained as the differential
    // oracle, see GrammarTestAccess). The returned view is valid until the next row_for
    // call on this grammar - the producer serves it from a bounded store or its scratch
    // row - so the consumer must copy the row before the next fill (the serving runtime
    // copies it into the round buffer immediately).
    [[nodiscard]] GrammarMaskRow row_for(const GrammarState& state) const;

    // Producer counters. `row_fills` counts computed fills (fast path, fallback, or
    // pending-UTF-8 states), `row_hits` counts fills served by the producer's shape
    // cache, `distinct_rows` counts rows stored in the shape store, `fill_micros`
    // accumulates fill compute time, and `compile_micros` accumulates the one-time
    // compile-time producer build (D2a).
    struct Stats {
        std::uint64_t row_fills      = 0;
        std::uint64_t row_hits       = 0;
        std::uint64_t distinct_rows  = 0;
        std::uint64_t fill_micros    = 0;
        std::uint64_t compile_micros = 0;
    };

    [[nodiscard]] Stats stats() const noexcept;

private:
    friend class GrammarTestAccess;
    class Impl;
    explicit CompiledGrammar(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_5::frontend
