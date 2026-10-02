#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/test_access.h"
#include "models/qwen3_5/frontend/grammar/trie_producer.h"

#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {
namespace {

// Owning wrapper for the engine grammar object (llama_grammar has a const rules member and
// no move/copy assignment, so it is only ever created by init/clone and freed by us).
using GrammarOwner = std::unique_ptr<llama_grammar, void (*)(llama_grammar*)>;

void free_grammar(llama_grammar* grammar) { llama_grammar_free_impl(grammar); }

// ---------------------------------------------------------------------------
// Vocabulary facade
// ---------------------------------------------------------------------------

// Backs the vendored llama_vocab facade with the NInfer-owned GrammarVocabulary view. The
// piece callback copies into a scratch string because the engine expects a stable
// `const std::string&`; the engine consumes that reference before the next call, so a
// single scratch is sufficient (same lifetime contract as llama_vocab's piece cache).
struct VocabularyBridge {
    std::shared_ptr<const GrammarVocabulary> view;
    mutable std::string scratch;

    static const std::string& token_to_piece(void* context, llama_token id) {
        VocabularyBridge& self = *static_cast<VocabularyBridge*>(context);
        self.scratch.assign(self.view->piece(static_cast<int>(id)));
        return self.scratch;
    }

    static bool token_is_eog(void* context, llama_token id) {
        return static_cast<VocabularyBridge*>(context)->view->is_eog(static_cast<int>(id));
    }

    static int32_t tokenize(void* context, const char* text, int32_t text_len, llama_token* tokens,
                            int32_t n_tokens_max, bool add_special, bool parse_special) {
        (void)add_special;
        (void)parse_special;
        VocabularyBridge& self = *static_cast<VocabularyBridge*>(context);
        const std::vector<int> ids =
            self.view->encode(std::string_view(text, static_cast<std::size_t>(text_len)));
        const int32_t count   = static_cast<int32_t>(ids.size());
        const int32_t written = std::min(count, n_tokens_max);
        for (int32_t index = 0; index < written; ++index) {
            tokens[index] = static_cast<llama_token>(ids[static_cast<std::size_t>(index)]);
        }
        // llama.cpp convention: a negative count reports a truncated result.
        return count > n_tokens_max ? -count : count;
    }
};

// ---------------------------------------------------------------------------
// Shared compiled backbone
// ---------------------------------------------------------------------------

// Immutable per-grammar data shared by every state: the engine grammar whose rules
// the states' stacks refer to and the vocabulary facade. The compiled mask producer
// (trie_producer.h) is built over the same rule set in compile() (D2a).
struct GrammarBackbone {
    std::shared_ptr<const GrammarVocabulary> vocabulary;
    VocabularyBridge bridge;
    llama_vocab facade;
    GrammarOwner master{nullptr, &free_grammar};
};

} // namespace

// ---------------------------------------------------------------------------
// GrammarState
// ---------------------------------------------------------------------------

class GrammarState::Impl {
public:
    std::shared_ptr<GrammarBackbone> backbone;
    GrammarOwner engine{nullptr, &free_grammar};
};

GrammarState::GrammarState(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GrammarState::GrammarState(const GrammarState& other) {
    if (!other.impl_) { return; }
    auto impl      = std::make_unique<Impl>();
    impl->backbone = other.impl_->backbone;
    impl->engine   = GrammarOwner(llama_grammar_clone_impl(*other.impl_->engine), &free_grammar);
    impl_ = std::move(impl);
}

GrammarState& GrammarState::operator=(const GrammarState& other) {
    if (this == &other) { return *this; }
    GrammarState copy(other);
    impl_ = std::move(copy.impl_);
    return *this;
}

GrammarState::GrammarState(GrammarState&&) noexcept            = default;
GrammarState& GrammarState::operator=(GrammarState&&) noexcept = default;
GrammarState::~GrammarState()                                  = default;

bool GrammarState::accept(int token_id) {
    if (!impl_) {
        throw std::invalid_argument("GrammarState::accept called on a moved-from state");
    }
    Impl& self      = *impl_;
    const int count = self.backbone->vocabulary->token_count();
    if (token_id < 0 || token_id >= count) { return false; }
    if (self.backbone->vocabulary->is_eog(token_id)) { return can_end(); }
    const std::string& piece = VocabularyBridge::token_to_piece(&self.backbone->bridge, token_id);
    // The engine's decoder reads a piece as a NUL-terminated C string, so a NUL-leading
    // piece decodes to nothing and accepting it would silently not move the state; the mask
    // declares exactly these pieces (empty or NUL-leading) illegal, and so does accept.
    if (piece.empty() || piece[0] == '\0') { return false; }

    // The engine assigns the rebuilt stack set before validating it, so a rejected token
    // would leave the state empty; restore the snapshot on the rejection path.
    llama_grammar_stacks saved_stacks = self.engine->stacks;
    llama_partial_utf8 saved_partial  = self.engine->partial_utf8;
    try {
        llama_grammar_accept_token(*self.engine, static_cast<llama_token>(token_id), piece);
    } catch (const std::exception&) {
        self.engine->stacks       = std::move(saved_stacks);
        self.engine->partial_utf8 = saved_partial;
        return false;
    }
    return true;
}

bool GrammarState::can_end() const noexcept {
    if (!impl_) { return false; }
    for (const llama_grammar_stack& stack : impl_->engine->stacks) {
        if (stack.empty()) { return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// CompiledGrammar
// ---------------------------------------------------------------------------

class CompiledGrammar::Impl {
public:
    std::shared_ptr<GrammarBackbone> backbone;

    // The compiled mask producer (design #59): built and prepared at compile time
    // (D2a), one instance per grammar, row production on the serving worker.
    std::shared_ptr<detail::Producer> producer;
    // The master rules' index of the root rule (for the producer's preparation).
    std::uint32_t root_rule = 0;

    // The vendored engine's full-vocabulary walk for one state, without any caching:
    // the differential oracle for the producer (D4) and its test access. Fresh each
    // call: a walk must never be served from a producer cache.
    [[nodiscard]] std::vector<std::uint32_t> engine_walk_row(const GrammarState::Impl& state) const {
        const int token_count = backbone->vocabulary->token_count();
        const std::uint32_t row_words = static_cast<std::uint32_t>((token_count + 31) / 32);
        std::vector<llama_token_data> candidates(static_cast<std::size_t>(token_count));
        for (int id = 0; id < token_count; ++id) {
            candidates[static_cast<std::size_t>(id)] =
                llama_token_data{static_cast<llama_token>(id), 0.0f, 0.0f};
        }
        llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
        llama_grammar_apply_impl(*state.engine, &array);
        std::vector<std::uint32_t> row(row_words, 0);
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const float logit = candidates[index].logit;
            if (!(std::isinf(logit) && logit < 0.0f)) { row[index >> 5U] |= 1U << (index & 31U); }
        }
        return row;
    }
};

CompiledGrammar::CompiledGrammar(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

CompiledGrammar::~CompiledGrammar() = default;

std::shared_ptr<const CompiledGrammar>
CompiledGrammar::compile(std::string_view gbnf, std::shared_ptr<const GrammarVocabulary> vocabulary,
                         std::string* error) {
    if (!vocabulary) {
        throw std::invalid_argument("CompiledGrammar::compile requires a vocabulary");
    }
    const std::string source(gbnf);

    auto backbone         = std::make_shared<GrammarBackbone>();
    backbone->vocabulary  = std::move(vocabulary);
    backbone->bridge.view = backbone->vocabulary;
    backbone->facade      = llama_vocab{
             .context     = &backbone->bridge,
             .to_piece    = &VocabularyBridge::token_to_piece,
             .to_eog      = &VocabularyBridge::token_is_eog,
             .to_tokenize = &VocabularyBridge::tokenize,
    };

    // Parse separately from init so the diagnostic reaches the caller as a string.
    llama_grammar_parser parser(&backbone->facade);
    if (!parser.parse(source.c_str())) {
        if (error) {
            *error =
                parser.last_error.empty() ? std::string("invalid GBNF grammar") : parser.last_error;
        }
        return nullptr;
    }
    if (parser.symbol_ids.find("root") == parser.symbol_ids.end()) {
        if (error) { *error = "grammar does not define the 'root' rule"; }
        return nullptr;
    }
    // Init from this parser's own rule table (low-level overload): the master grammar's
    // rules and the root index below come from one parse session, so they are
    // consistent by construction (the string overload re-parses internally).
    std::vector<const llama_grammar_element*> rule_pointers;
    rule_pointers.reserve(parser.rules.size());
    for (const llama_grammar_rule& rule : parser.rules) {
        rule_pointers.push_back(rule.data());
    }
    backbone->master = GrammarOwner(
        llama_grammar_init_impl(&backbone->facade, rule_pointers.data(), rule_pointers.size(),
                                parser.symbol_ids.at("root")),
        &free_grammar);
    if (!backbone->master) {
        if (error) {
            *error = "grammar was rejected by the runtime (left recursion or invalid rule "
                     "references)";
        }
        return nullptr;
    }

    const int token_count = backbone->vocabulary->token_count();
    if (token_count <= 0) {
        if (error) { *error = "grammar vocabulary has no tokens"; }
        return nullptr;
    }

    auto impl = std::make_unique<Impl>();
    impl->backbone  = std::move(backbone);
    impl->root_rule = parser.symbol_ids.at("root");
    // The one-time producer work lands at compile time (D2a, ruled on #59): the
    // codepoint trie, the class structures, and the counter tables for every
    // fresh-round chain shape the rule graph can reach. Serving walks are pure
    // steady state from here on.
    impl->producer = std::make_shared<detail::Producer>();
    std::shared_ptr<const CompiledGrammar> compiled;
    try {
        impl->producer->build(*impl->backbone->vocabulary);
        impl->producer->prepare(impl->backbone->master->rules, impl->root_rule);
        compiled = std::shared_ptr<const CompiledGrammar>(new CompiledGrammar(std::move(impl)));
        // A constraint whose initial mask admits nothing (no token and no end-of-generation)
        // would sample a rejected token and fail closed mid-request; reject the grammar here.
        const GrammarState initial = compiled->initial_state();
        const GrammarMaskRow first = compiled->row_for(initial);
        bool admits_any            = false;
        for (const std::uint32_t word : first) { admits_any = admits_any || word != 0U; }
        if (!admits_any) {
            if (error) { *error = "grammar cannot produce any token with this vocabulary"; }
            return nullptr;
        }
    } catch (const std::exception& failure) {
        // Producer structural guards (expansion seen/stack caps) reject a pathological
        // user-supplied grammar here, as a validation failure, instead of throwing in the
        // serving worker mid-generation.
        if (error) { *error = failure.what(); }
        return nullptr;
    }
    return compiled;
}

std::shared_ptr<const CompiledGrammar>
CompiledGrammar::compile_from_schema(std::string_view json_schema,
                                     std::shared_ptr<const GrammarVocabulary> vocabulary,
                                     std::string* error) {
    std::string gbnf;
    try {
        const common_json schema = common_json::parse(std::string(json_schema));
        gbnf                     = json_schema_to_grammar(schema, /*force_gbnf=*/true);
    } catch (const std::exception& failure) {
        if (error) { *error = failure.what(); }
        return nullptr;
    }
    return compile(gbnf, std::move(vocabulary), error);
}

GrammarState CompiledGrammar::initial_state() const {
    auto impl      = std::make_unique<GrammarState::Impl>();
    impl->backbone = impl_->backbone;
    impl->engine = GrammarOwner(llama_grammar_clone_impl(*impl_->backbone->master), &free_grammar);
    return GrammarState(std::move(impl));
}

int CompiledGrammar::token_count() const noexcept {
    return impl_->backbone->vocabulary->token_count();
}

GrammarMaskRow CompiledGrammar::row_for(const GrammarState& state) const {
    if (!state.impl_) {
        throw std::invalid_argument("CompiledGrammar::row_for called with a moved-from state");
    }
    if (state.impl_->backbone != impl_->backbone) {
        throw std::invalid_argument("CompiledGrammar::row_for called with a foreign state");
    }
    // The compiled producer (design #59): bit v set iff the vendored engine accepts
    // token v at this state. The returned view is valid until the next row_for on
    // this grammar (the producer serves it from a bounded store or its scratch row).
    return impl_->producer->produce(state.impl_->engine->rules, state.impl_->engine->stacks,
                                    state.impl_->engine->partial_utf8);
}

CompiledGrammar::Stats CompiledGrammar::stats() const noexcept {
    const detail::ProducerStats& p = impl_->producer->stats;
    Stats s;
    s.row_fills      = p.fast_fills + p.fallback_fills + p.partial_states;
    s.row_hits       = p.shape_hits;
    s.distinct_rows  = p.shape_rows_stored;
    s.fill_micros    = p.fill_ns / 1000;
    s.compile_micros = (p.build_ns + p.prepare_ns) / 1000;
    return s;
}

// ---------------------------------------------------------------------------
// Test access (D4 differential oracle + producer diagnostics)
// ---------------------------------------------------------------------------

std::vector<std::uint32_t> GrammarTestAccess::engine_walk_row(const CompiledGrammar& grammar,
                                                              const GrammarState& state) {
    if (!state.impl_) {
        throw std::invalid_argument("GrammarTestAccess::engine_walk_row called with a "
                                    "moved-from state");
    }
    if (state.impl_->backbone != grammar.impl_->backbone) {
        throw std::invalid_argument("GrammarTestAccess::engine_walk_row called with a foreign "
                                    "state");
    }
    return grammar.impl_->engine_walk_row(*state.impl_);
}

const detail::Producer& GrammarTestAccess::producer(const CompiledGrammar& grammar) {
    return *grammar.impl_->producer;
}

std::shared_ptr<detail::Producer>
GrammarTestAccess::make_producer(const CompiledGrammar& grammar, bool shape_cache) {
    auto producer = std::make_shared<detail::Producer>(shape_cache);
    producer->build(*grammar.impl_->backbone->vocabulary);
    producer->prepare(grammar.impl_->backbone->master->rules, grammar.impl_->root_rule);
    return producer;
}

} // namespace ninfer::models::qwen3_5::frontend
