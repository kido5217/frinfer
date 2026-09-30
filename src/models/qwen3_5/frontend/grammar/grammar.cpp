#include "models/qwen3_5/frontend/grammar/grammar.h"

#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
// Rule addressing and counter normalization
// ---------------------------------------------------------------------------

// Maps a stack element pointer back to (rule id, element index). Rule element vectors are
// separately allocated and never move after construction, so ranges are stable.
class RuleAddressIndex {
public:
    struct Range {
        const llama_grammar_element* begin;
        const llama_grammar_element* end;
        std::uint32_t rule;
    };

    void build(const llama_grammar_rules& rules) {
        ranges_.clear();
        ranges_.reserve(rules.size());
        for (std::size_t rule = 0; rule < rules.size(); ++rule) {
            ranges_.push_back({rules[rule].data(), rules[rule].data() + rules[rule].size(),
                               static_cast<std::uint32_t>(rule)});
        }
        std::sort(ranges_.begin(), ranges_.end(), [](const Range& left, const Range& right) {
            return std::less<const llama_grammar_element*>{}(left.begin, right.begin);
        });
    }

    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t>
    locate(const llama_grammar_element* element) const {
        const auto found = std::upper_bound(
            ranges_.begin(), ranges_.end(), element,
            [](const llama_grammar_element* value, const Range& range) {
                return std::less<const llama_grammar_element*>{}(value, range.begin);
            });
        if (found == ranges_.begin()) {
            throw std::logic_error("grammar state references an element outside the rule set");
        }
        const Range& range = *(found - 1);
        if (element >= range.end) {
            throw std::logic_error("grammar state references an element outside the rule set");
        }
        return {range.rule, static_cast<std::uint32_t>(element - range.begin)};
    }

private:
    std::vector<Range> ranges_;
};

// Counter-normalized rule identity.
//
// A repetition expansion (`{m,n}`, `*`, `+`, `?`) generates a chain of exactly-one-round
// rules `f_p = body · f_{p-1}`, `f_1 = body`. Two chain positions p >= fold_limit have the
// same mask: a single accepted token consumes at most (max piece bytes) rounds, so beyond
// that limit every continuation is reachable from both. Positions below the limit (and the
// chain tail after the body) keep exact identities, because near the bound the mask is what
// enforces the repetition count — collapsing those would let a token overshoot `maxLength`.
//
// Frames are recognized structurally from the generated rules: first alternative
// `body [link]`, second alternative empty. A candidate link counts only when its target is
// itself a frame with the identical body (or the frame itself, the unbounded `S*` shape),
// which keeps hand-written look-alikes on the exact path unless they are genuine counters.
struct RepetitionNormalization {
    std::vector<std::uint32_t> canonical_rule; // per rule id; frames share clamped-position ids
    std::uint32_t fold_limit = 0;
};

constexpr std::uint64_t kInfinitePosition = std::numeric_limits<std::uint64_t>::max();

RepetitionNormalization normalize_repetitions(const llama_grammar_rules& rules,
                                              std::uint32_t fold_limit) {
    const std::size_t count = rules.size();
    RepetitionNormalization result;
    result.fold_limit = fold_limit;
    result.canonical_rule.resize(count);
    for (std::size_t rule = 0; rule < count; ++rule) {
        result.canonical_rule[rule] = static_cast<std::uint32_t>(rule);
    }

    struct Frame {
        std::size_t alternative = 0; // first-alternative element count
        std::uint32_t link      = 0;
        bool link_candidate     = false;
        bool linked             = false;
    };

    std::vector<Frame> frames(count);
    std::vector<bool> is_frame(count, false);
    for (std::size_t rule = 0; rule < count; ++rule) {
        const llama_grammar_rule& elements = rules[rule];
        if (elements.size() < 3) { continue; }
        if (elements.back().type != LLAMA_GRETYPE_END) { continue; }
        if (elements[elements.size() - 2].type != LLAMA_GRETYPE_ALT) { continue; }
        std::size_t alternatives = 0;
        for (const llama_grammar_element& element : elements) {
            if (element.type == LLAMA_GRETYPE_ALT) { ++alternatives; }
        }
        if (alternatives != 1) { continue; }
        const std::size_t alternative = elements.size() - 2;
        if (alternative == 0) { continue; }
        Frame frame;
        frame.alternative = alternative;
        if (alternative >= 2 && elements[alternative - 1].type == LLAMA_GRETYPE_RULE_REF) {
            frame.link_candidate = true;
            frame.link           = elements[alternative - 1].value;
        }
        frame.linked   = frame.link_candidate;
        frames[rule]   = frame;
        is_frame[rule] = true;
    }

    auto body_size = [&](std::size_t rule) {
        const Frame& frame = frames[rule];
        return frame.alternative - (frame.linked ? 1 : 0);
    };
    auto body_equal = [&](std::size_t left, std::size_t right) {
        const std::size_t size = body_size(left);
        if (size != body_size(right)) { return false; }
        for (std::size_t index = 0; index < size; ++index) {
            const llama_grammar_element& a = rules[left][index];
            const llama_grammar_element& b = rules[right][index];
            if (a.type != b.type || a.value != b.value) { return false; }
        }
        return true;
    };

    // A linked reading is only kept when the link target is a frame with an identical body;
    // the check is monotone (linked only ever turns off), so it converges.
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t rule = 0; rule < count; ++rule) {
            Frame& frame = frames[rule];
            if (!frame.linked) { continue; }
            const std::size_t target = frame.link;
            if (target == rule) { continue; } // the unbounded S* self-recursion
            if (target >= count || !is_frame[target]) {
                frame.linked = false;
                changed      = true;
                continue;
            }
            if (!body_equal(rule, target)) {
                frame.linked = false;
                changed      = true;
            }
        }
    }

    // Chain positions: base frames are 1, every link adds one round; a link cycle means the
    // chain can stop at any count, i.e. the saturated position.
    std::vector<std::uint64_t> position(count, 0);
    std::vector<std::uint32_t> on_path(count, 0); // 1-based index in the current walk path
    for (std::size_t start = 0; start < count; ++start) {
        if (!is_frame[start] || position[start] != 0) { continue; }
        std::vector<std::size_t> path;
        std::size_t current = start;
        bool cycle          = false;
        bool base           = false;
        while (true) {
            if (position[current] != 0) { break; } // known below: back-fill the path
            if (on_path[current] != 0) {
                cycle = true;
                break;
            }
            on_path[current] = static_cast<std::uint32_t>(path.size()) + 1;
            path.push_back(current);
            if (!frames[current].linked) {
                base = true;
                break;
            }
            current = frames[current].link;
        }
        if (cycle) {
            // The cycle loops forever, and everything leading into it inherits that.
            for (const std::size_t frame : path) { position[frame] = kInfinitePosition; }
        } else if (base) {
            // path.back() is the base (position 1); the frames above it count up.
            for (std::size_t index = path.size(); index-- > 0;) {
                position[path[index]] = static_cast<std::uint64_t>(path.size() - index);
            }
        } else {
            std::uint64_t next = position[current] + 1;
            for (std::size_t index = path.size(); index-- > 0;) { position[path[index]] = next++; }
        }
        for (const std::size_t frame : path) { on_path[frame] = 0; }
    }

    // Canonical ids: frames at the same clamped position with an identical body share one
    // id (`fold_limit` absorbs every saturated position, including infinite self-recursion).
    std::unordered_map<std::string, std::uint32_t> representatives;
    std::uint32_t next_id = static_cast<std::uint32_t>(count);
    for (std::size_t rule = 0; rule < count; ++rule) {
        if (!is_frame[rule]) { continue; }
        const std::uint64_t raw_position = position[rule] == 0 ? 1 : position[rule];
        const std::uint32_t clamped = raw_position == kInfinitePosition || raw_position > fold_limit
                                          ? fold_limit
                                          : static_cast<std::uint32_t>(raw_position);
        std::string key;
        key.reserve(body_size(rule) * 8 + 4);
        for (std::size_t index = 0; index < body_size(rule); ++index) {
            const llama_grammar_element& element = rules[rule][index];
            for (int shift = 24; shift >= 0; shift -= 8) {
                key.push_back(static_cast<char>((element.type >> shift) & 0xff));
            }
            for (int shift = 24; shift >= 0; shift -= 8) {
                key.push_back(static_cast<char>((element.value >> shift) & 0xff));
            }
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            key.push_back(static_cast<char>((clamped >> shift) & 0xff));
        }
        const auto [found, inserted] = representatives.try_emplace(std::move(key), next_id);
        if (inserted) { ++next_id; }
        result.canonical_rule[rule] = found->second;
    }
    return result;
}

// ---------------------------------------------------------------------------
// State identity
// ---------------------------------------------------------------------------

constexpr std::uint32_t kStackSeparator = std::numeric_limits<std::uint32_t>::max();

struct StateKey {
    std::uint32_t partial_value = 0;
    std::int32_t partial_remain = 0;
    // Canonical stacks: per stack, (canonical rule, element index) pairs followed by a
    // separator word; the stack encodings are sorted so stack order never splits the cache.
    std::vector<std::uint32_t> stacks;

    [[nodiscard]] bool operator==(const StateKey& other) const noexcept {
        return partial_value == other.partial_value && partial_remain == other.partial_remain &&
               stacks == other.stacks;
    }
};

struct StateKeyHash {
    [[nodiscard]] std::size_t operator()(const StateKey& key) const noexcept {
        return static_cast<std::size_t>(fnv1a(key));
    }

    static std::uint64_t fnv1a(const StateKey& key) noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        auto mix           = [&hash](std::uint32_t word) {
            for (int shift = 0; shift < 32; shift += 8) {
                hash ^= (word >> shift) & 0xffU;
                hash *= 1099511628211ULL;
            }
        };
        mix(key.partial_value);
        mix(static_cast<std::uint32_t>(key.partial_remain));
        for (const std::uint32_t word : key.stacks) { mix(word); }
        return hash;
    }
};

StateKey make_state_key(const llama_grammar_stacks& stacks, const llama_partial_utf8& partial,
                        const RuleAddressIndex& index,
                        const std::vector<std::uint32_t>& canonical_rule) {
    StateKey key;
    // The engine keeps the last decoded codepoint in partial_utf8.value after a sequence
    // completes, but reads that field only while n_remain != 0 (vendored decode_utf8 and
    // match_partial_char), so with no pending sequence it is dead state: canonicalize it or
    // every token that ends a multi-byte codepoint would mint a fresh cache key.
    key.partial_value  = partial.n_remain == 0 ? 0U : partial.value;
    key.partial_remain = partial.n_remain;

    std::vector<std::vector<std::uint32_t>> encoded;
    encoded.reserve(stacks.size());
    std::size_t words = 0;
    for (const llama_grammar_stack& stack : stacks) {
        std::vector<std::uint32_t> words_for_stack;
        words_for_stack.reserve(stack.size() * 2 + 1);
        for (const llama_grammar_element* element : stack) {
            const auto [rule, element_index] = index.locate(element);
            words_for_stack.push_back(canonical_rule[rule]);
            words_for_stack.push_back(element_index);
        }
        words_for_stack.push_back(kStackSeparator);
        words += words_for_stack.size();
        encoded.push_back(std::move(words_for_stack));
    }
    std::sort(encoded.begin(), encoded.end());
    key.stacks.reserve(words);
    for (const std::vector<std::uint32_t>& stack : encoded) {
        key.stacks.insert(key.stacks.end(), stack.begin(), stack.end());
    }
    return key;
}

// ---------------------------------------------------------------------------
// Shared compiled backbone
// ---------------------------------------------------------------------------

// Immutable per-grammar data shared by every state and by the row cache: the engine
// grammar whose rules the states' stacks refer to, the vocabulary facade, and the
// counter-normalized rule identity.
struct GrammarBackbone {
    std::shared_ptr<const GrammarVocabulary> vocabulary;
    VocabularyBridge bridge;
    llama_vocab facade;
    GrammarOwner master{nullptr, &free_grammar};
    std::vector<std::uint32_t> canonical_rule;
    std::uint32_t fold_limit = 0;
};

std::uint64_t content_hash(const std::vector<std::uint32_t>& words) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint32_t word : words) {
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (word >> shift) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

} // namespace

// ---------------------------------------------------------------------------
// GrammarState
// ---------------------------------------------------------------------------

class GrammarState::Impl {
public:
    std::shared_ptr<GrammarBackbone> backbone;
    GrammarOwner engine{nullptr, &free_grammar};
    RuleAddressIndex address_index;
};

GrammarState::GrammarState(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GrammarState::GrammarState(const GrammarState& other) {
    if (!other.impl_) { return; }
    auto impl      = std::make_unique<Impl>();
    impl->backbone = other.impl_->backbone;
    impl->engine   = GrammarOwner(llama_grammar_clone_impl(*other.impl_->engine), &free_grammar);
    impl->address_index.build(impl->engine->rules);
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
    std::uint32_t row_words = 0;

    // Persistent full-domain candidate array; the engine marks rejected entries with
    // -INFINITY, so every fill resets the logits first and every accepted token is exactly
    // one that keeps a zero logit.
    std::vector<llama_token_data> candidates;

    std::vector<std::vector<std::uint32_t>> rows;
    std::unordered_map<StateKey, std::uint32_t, StateKeyHash> rows_by_state;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> rows_by_hash;
    Stats stats;

    [[nodiscard]] GrammarMaskRow row_for(const GrammarState::Impl& state) {
        const StateKey key = make_state_key(state.engine->stacks, state.engine->partial_utf8,
                                            state.address_index, backbone->canonical_rule);
        const auto cached  = rows_by_state.find(key);
        if (cached != rows_by_state.end()) {
            ++stats.row_hits;
            return rows[cached->second];
        }

        const auto started = std::chrono::steady_clock::now();
        std::vector<std::uint32_t> row(row_words, 0);
        for (llama_token_data& candidate : candidates) { candidate.logit = 0.0f; }
        llama_token_data_array array{candidates.data(), candidates.size(), -1, false};
        llama_grammar_apply_impl(*state.engine, &array);
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const float logit = candidates[index].logit;
            if (!(std::isinf(logit) && logit < 0.0f)) { row[index >> 5U] |= 1U << (index & 31U); }
        }
        const auto finished = std::chrono::steady_clock::now();
        ++stats.row_fills;
        stats.fill_micros += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count());

        // Content dedup: distinct states (different counters, equivalent positions, ...)
        // regularly produce identical rows; store one copy and share it.
        const std::uint64_t hash = content_hash(row);
        for (const std::uint32_t row_id : rows_by_hash[hash]) {
            if (rows[row_id] == row) {
                rows_by_state.emplace(key, row_id);
                return rows[row_id];
            }
        }
        rows.push_back(std::move(row));
        const std::uint32_t row_id = static_cast<std::uint32_t>(rows.size() - 1);
        rows_by_hash[hash].push_back(row_id);
        ++stats.distinct_rows;
        rows_by_state.emplace(key, row_id);
        return rows[row_id];
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
    backbone->master = GrammarOwner(llama_grammar_init_impl(&backbone->facade, source.c_str(),
                                                            "root", false, nullptr, 0, nullptr, 0),
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
    std::size_t max_piece_bytes = 0;
    for (int id = 0; id < token_count; ++id) {
        max_piece_bytes = std::max(max_piece_bytes, backbone->vocabulary->piece(id).size());
    }
    backbone->fold_limit = static_cast<std::uint32_t>(max_piece_bytes) + 2;
    backbone->canonical_rule =
        normalize_repetitions(backbone->master->rules, backbone->fold_limit).canonical_rule;

    auto impl       = std::make_unique<Impl>();
    impl->backbone  = std::move(backbone);
    impl->row_words = static_cast<std::uint32_t>((token_count + 31) / 32);
    impl->candidates.resize(static_cast<std::size_t>(token_count));
    for (int id = 0; id < token_count; ++id) {
        impl->candidates[static_cast<std::size_t>(id)] =
            llama_token_data{static_cast<llama_token>(id), 0.0f, 0.0f};
    }
    return std::shared_ptr<const CompiledGrammar>(new CompiledGrammar(std::move(impl)));
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
    impl->address_index.build(impl->engine->rules);
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
    return impl_->row_for(*state.impl_);
}

CompiledGrammar::Stats CompiledGrammar::stats() const noexcept { return impl_->stats; }

} // namespace ninfer::models::qwen3_5::frontend
