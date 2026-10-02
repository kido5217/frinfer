// Compiled token-trie mask producer (wayfinder map #45, design #59, ticket #79).
//
// Produces the same dense bitmask row as the vendored engine's full-vocabulary walk
// (the module's oracle path in grammar.cpp, which drives llama_grammar_apply_impl),
// by walking a codepoint trie over the vocabulary's pieces and advancing the
// engine's own stack semantics. Ported from the verified prototype (the
// research/compiled-mask-producer branch, docs/research/compiled-mask-producer.md):
// 0 mismatches against the vendored engine on the G1-G4 fixture walks over the real
// 248k vocabulary, steady state 0.005-0.261 ms/step under the 0.4 ms/step bar.
//
// Exactness contract: for a state (stacks, partial_utf8), bit v is set iff the
// vendored engine accepts token v at that state; EOG ids exactly when some stack
// is empty. The row is a pure function of the state (verified differentially).
//
// Three cost layers:
//  1. Class-counter fast path - per-(body, C, frontier) counter tables: per-token
//     round thresholds + continuation handoff correction + partial thresholds.
//     Stack-based counter chains (the diary-class string/line counters) fill by
//     memcpy + threshold compare.
//  2. Exact trie fallback - bulk subtree acceptance with a capacity witness,
//     per-node descent, negated-class exclusion re-judgment, the partial-piece tail.
//  3. Producer-owned state-shape row cache - a 128-bit content hash of the state
//     shape keys a bounded row store; the row is copied only for repeated shapes
//     (first occurrences record the 16-byte signature only).
//
// Compile-time placement (D2a, ruled on #59 2026-10-02): the one-time work - the
// trie build, the class structures, and the counter tables for every chain shape
// the rule graph can reach at a fresh-round frontier - runs in build() +
// prepare(), called from CompiledGrammar::compile(); serving walks are pure
// steady state. A shape the enumeration does not cover (e.g. a mid-escape
// frontier) is still built on demand by the fast path, observable through
// stats.table_builds_on_demand.
//
// The counter tables are keyed by (C placement, body, frontier): the frontier is
// the set of body positions the parallel engine candidates sit at when the fill
// happens. The table's round thresholds are only valid from the positions in its
// own key, so a fill at a deeper frontier (e.g. mid-escape-sequence inside the
// body) uses its own table, built from the fill's actual position set.
//
// Engine semantics reproduced exactly (see the design record): the counter frames
// count pieces (tokens), not characters; the per-piece budget is unenforced
// (bounded overrun <= 127 chars past max); min is enforced via the eps/EOG
// empty-stack marker; max by termination.
#pragma once

#include "llama-grammar.h"
#include "models/qwen3_5/frontend/grammar/vocabulary.h"

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend::detail {

using Element = llama_grammar_element;

// Forward declaration; the definition lives in trie_producer.cpp (same namespace).
class RuleAddressIndex;

// Counter-normalized rule identity: frames of a repetition chain share clamped-position
// ids, so a state's content keyed by (canonical rule, element index) is a sound
// abstraction of the engine state (masks are provably equal at folded positions). This is
// the state-content the producer's shape signature mixes - a raw (type, value) mix is
// NOT sound: a self-referential counter keeps an invariant (type, value) stack across
// positions while the mask still changes at the bound.
struct RepetitionNormalization {
    std::vector<std::uint32_t> canonical_rule; // per rule id; frames share clamped-position ids
    std::uint32_t fold_limit = 0;
};

[[nodiscard]] RepetitionNormalization normalize_repetitions(const llama_grammar_rules& rules,
                                                            std::uint32_t fold_limit);

inline constexpr int kMaxStack = 24;   // stack depth cap (engine stacks: <= 6 on the diary)
inline constexpr int kMaxPos = 128;     // parallel-position cap: wide JSON enums (N-value
                                        // alternatives) run N parallel stacks; 128 covers
                                        // realistic enums, the throw is the fail-closed last
                                        // resort beyond it (checked before any state is copied)
inline constexpr std::uint32_t kNoEdge = 0xFFFFFFFFU;
inline constexpr int kCapFuel = 130;   // > max codepoints in any vocab piece

struct Stk {
    std::uint8_t n = 0;
    const Element* e[kMaxStack] = {};
};

struct Succ {
    std::uint8_t n = 0;
    bool has_empty = false;
    Stk s[kMaxPos];
};

struct PosSet {
    std::uint8_t n = 0;
    bool has_empty = false;
    Stk s[kMaxPos];
};

// Inclusive codepoint range set, optionally negated (complement over the full
// Unicode domain), mirroring one char-range element group of the engine.
struct CharClass {
    bool negated = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;  // sorted, merged

    [[nodiscard]] bool accepts(std::uint32_t cp) const noexcept;
    [[nodiscard]] bool operator==(const CharClass& other) const noexcept;
};

// One-time build accounting (D2a): everything recorded here ran at grammar
// compile time. The serving-side counters track the steady-state walk.
struct ProducerStats {
    std::uint64_t build_ns = 0;         // trie build
    std::uint64_t prepare_ns = 0;       // class structures + counter-table prebuilds
    std::uint64_t body_builds = 0;
    std::uint64_t body_build_ns = 0;
    std::uint64_t table_builds = 0;     // counter tables prebuilt in prepare()
    std::uint64_t table_builds_on_demand = 0;  // counter tables built during a fill
    std::uint64_t table_build_fail = 0;        // shapes the fast path cannot model
    std::uint64_t table_masks_words = 0;       // words allocated across cached tables
    std::uint64_t prebuilt_shapes = 0;         // distinct counter shapes the rule graph found
    // Fill accounting.
    std::uint64_t fills = 0;            // produce() calls
    std::uint64_t fast_fills = 0;       // computed fills served by the counter table
    std::uint64_t fallback_fills = 0;   // computed fills served by the trie walk
    std::uint64_t partial_states = 0;   // fills with a pending UTF-8 sequence
    std::uint64_t overflow_fills = 0;   // fills rejected (thrown) at the kMaxPos parallel-stack cap
    std::uint64_t fill_ns = 0;          // cumulative compute time (miss path only)
    std::uint64_t shape_hits = 0;
    std::uint64_t shape_misses = 0;
    std::uint64_t shape_evictions = 0;
    std::uint64_t shape_rows_stored = 0;
    // Trie-walk diagnostics.
    std::uint64_t nodes_visited = 0;
    std::uint64_t edges_visited = 0;
    std::uint64_t bulks = 0;
    std::uint64_t bulk_ids = 0;
    std::uint64_t bits_set = 0;
    std::uint64_t tail_judged = 0;
    std::uint64_t tail_steps = 0;
    std::uint64_t capacity_calls = 0;
    std::uint64_t capacity_steps = 0;
    std::uint64_t exception_judged = 0;
    std::uint64_t declined_capacity = 0;
    std::uint64_t declined_class = 0;
    std::uint64_t excl_judged = 0;
    // Fast-path recognition.
    std::uint64_t recog_attempts = 0;
    std::uint64_t recog_ok = 0;
    std::uint64_t bucket_tokens = 0;
    std::uint64_t correction_tokens = 0;
};

// Replicates the vendored engine's UTF-8 decoder over a piece (NUL-truncating,
// with a pending partial in/out). Exposed for the differential test's self-check.
[[nodiscard]] bool decode_utf8_bytes(const std::uint8_t* bytes, std::size_t len,
                                     const llama_partial_utf8& start,
                                     std::vector<std::uint32_t>& cps,
                                     llama_partial_utf8& end);

// The class accepted by one char-range element group of the engine, starting at
// `pos`. Exposed for the differential test's self-check.
[[nodiscard]] CharClass class_from_element(const Element* pos);

// Row production runs on the single serving worker; one producer per compiled
// grammar. The row returned by produce() is valid until the next produce() call
// on this producer: it points into the shape store (a hit, or a repeated shape)
// or into the producer's scratch row (a one-shot shape). Consumers must copy the
// row before the next fill (GrammarRuntime::fill_round_rows copies it into the
// round buffer immediately).
class Producer {
public:
    explicit Producer(bool shape_cache = true);

    // Builds the codepoint trie from the vocabulary. Pieces are decoded exactly
    // like the vendored decode_utf8 (NUL-truncating, first-byte-NUL filtered).
    // One-time work (D2a): called from CompiledGrammar::compile().
    void build(const GrammarVocabulary& vocabulary);

    // The one-time preparation for a compiled rule set (D2a): the class
    // structures for every char class the rules use, and the counter tables for
    // every chain shape the rule graph can reach at a fresh-round frontier.
    // `rules` is the master grammar's rule set; `root_rule` the index of the
    // grammar's root rule in it. Called from CompiledGrammar::compile().
    void prepare(const llama_grammar_rules& rules, std::uint32_t root_rule);

    // Produces the row for one engine state. `rules`/`stacks` must be the same
    // rule set the stacks point into (a state's cloned rules; the layout matches
    // the backbone's). See the class comment for the row's lifetime.
    [[nodiscard]] std::span<const std::uint32_t> produce(const llama_grammar_rules& rules,
                                                         const llama_grammar_stacks& stacks,
                                                         const llama_partial_utf8& partial);

    [[nodiscard]] std::uint32_t token_count() const noexcept { return token_count_; }
    [[nodiscard]] std::uint32_t row_words() const noexcept { return row_words_; }
    [[nodiscard]] std::uint64_t trie_nodes() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::uint64_t trie_edges() const noexcept { return edges_.size(); }
    [[nodiscard]] std::uint64_t trie_ids() const noexcept { return euler_ids_.size(); }
    [[nodiscard]] std::uint64_t partial_ids() const noexcept { return partial_ids_.size(); }
    [[nodiscard]] std::uint64_t exception_ids() const noexcept { return exceptions_.size(); }
    [[nodiscard]] std::uint64_t memory_bytes() const noexcept;

    ProducerStats stats;
    bool shape_cache_enabled;

    // Canonical rule identity (normalize_repetitions over the master rules): maps each
    // rule id to its clamped counter-position id. Sized to rules.size() by prepare();
    // the shape signature keys state content by (canonical rule, element index).
    std::vector<std::uint32_t> canonical_rule_;
    std::uint32_t fold_limit_ = 0;

private:
    struct Node {
        std::uint32_t first_edge = kNoEdge;
        std::uint32_t own_begin = kNoEdge;  // euler range of ids ending at this node
        std::uint32_t own_end = kNoEdge;
        std::uint32_t sub_begin = 0;  // euler range of ids in this node's subtree
        std::uint32_t sub_end = 0;
        std::uint16_t max_remain = 0;
        std::uint16_t depth = 0;
    };
    struct Edge {
        std::uint32_t cp;
        std::uint32_t child;
        std::uint32_t parent;
        std::uint32_t next;
    };
    struct PartialId {
        int id;
        std::uint32_t off;
        std::uint16_t len;
        std::uint32_t value;
        int remain;
    };
    struct ExceptionId {
        int id;
        std::uint32_t off;
        std::uint16_t len;
    };
    struct ClassCache {
        CharClass cls;
        std::vector<std::uint8_t> outside;  // per node (positive classes): subtree edge outside cls
        std::vector<std::uint8_t> excl_bitmap;   // per euler id: path hits an excluded codepoint
        std::vector<std::uint32_t> excl_euler;   // sorted euler positions of those ids
        bool has_exclusions = false;
        bool ready = false;
    };
    // Per-body region table: the counter rounds alone, without the continuation.
    struct BodyEntry {
        std::int32_t max_p = -1;                  // chain rounds the DFS could see
        std::vector<std::uint32_t> buckets;       // (max_p+1) * row_words, bucket[b] = thr == b
        std::vector<std::uint8_t> node_boundary;  // per node: min boundary rounds, 0xFF none
        std::vector<std::uint8_t> thr;            // per euler token: region-only threshold
    };
    // A full counter table: the body buckets with the continuation-handoff corrections
    // folded in (prefix OR'd downward), plus the per-partial-id round thresholds.
    struct TableEntry {
        std::int32_t table_max = -1;  // -1 = recognition/build failed for this key
        std::vector<std::uint32_t> masks;  // (table_max+1) * row_words
        std::vector<std::uint8_t> partial_thr;  // per partial id: threshold (0xFF = rejected)
    };
    // Class-counter fast path (threshold DFS).
    static constexpr int kFastPos = 8;   // position-set cap for the threshold DFS
    static constexpr int kFastStack = 10; // stack-depth cap for the threshold DFS
    struct KStk {
        std::uint8_t n = 0;
        const Element* e[kFastStack] = {};
    };
    struct KPos {
        KStk stack;
        std::uint8_t k = 0;
        bool region = true;
    };
    struct KSet {
        std::uint8_t n = 0;
        bool has_empty = false;
        std::uint8_t empty_k = 0xFF;  // sentinel: no boundary recorded yet
        KPos p[kFastPos];
    };
    struct KStkKey {
        std::uint8_t n = 0;
        const Element* e[kFastStack] = {};
    };
    struct FragEntry {
        std::uint64_t hash = 0;
        std::uint32_t gen = 0;
        KStkKey key;
        std::uint8_t k = 0;
        bool region = true;
        bool is_char = false;
        std::uint32_t class_id = 0;
        KSet succ;
    };
    // Open-addressing capacity memo (per fill, generation-cleared).
    struct CapEntry {
        std::uint64_t hash = 0;
        std::uint32_t gen = 0;
        int cap = 0;
        Stk stack;
    };
    struct PrepEntry {
        std::uint64_t hash = 0;
        std::uint32_t gen = 0;
        std::uint32_t class_id = 0;
        bool is_char = false;
        Succ succ;
        Stk key;
    };
    struct ShapeEntry {
        std::uint64_t h1 = 0;
        std::uint64_t h2 = 0;
        std::vector<std::uint32_t> row;
    };
    static constexpr std::size_t kShapeCacheMax = 1024;   // rows, <= 31 MB (D3 bound)
    static constexpr std::size_t kShapeSeenMax = 8192;    // signatures, 128 KB (D3 bound)

    // Per-depth scratch for the recursive trie walk. The position sets are producer-owned
    // and indexed by codepoint depth (sized by the trie's depth in build()), so one visit()
    // frame stays small: folding them into the frame costs sizeof(PosSet) * (1 + 4) bytes
    // per level (~128 KiB), and a real vocabulary's pieces descend up to their codepoint
    // length (measured 66+ frames on Qwen3.8-27B), overflowing an 8 MiB worker stack. The
    // arena is pre-sized once, so references into it survive the recursive calls.
    static constexpr int kVisitSigCache = 4;
    struct VisitScratch {
        std::uint64_t sigs[kVisitSigCache] = {};
        PosSet sets[kVisitSigCache];
        PosSet local;
        int nsigs = 0;
    };
    std::uint32_t trie_depth_ = 0;
    std::vector<VisitScratch> visit_scratch_;

    // --- static trie (built in build()) ---
    std::uint32_t token_count_ = 0;
    std::uint32_t row_words_ = 0;
    std::vector<int> eog_ids_;
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<int> euler_ids_;
    std::vector<std::uint32_t> euler_off_;
    std::vector<std::uint16_t> euler_len_;
    std::vector<std::uint32_t> path_data_;
    std::vector<PartialId> partial_ids_;
    std::vector<ExceptionId> exceptions_;
    std::vector<std::uint8_t> exception_bytes_;
    std::vector<std::uint32_t> euler_node_;  // trie end node per euler token
    std::vector<std::uint32_t> child_off_;   // per node: start index into child_edges_
    std::vector<std::uint32_t> child_edges_; // edge indices grouped by parent, cp-sorted
    std::vector<ClassCache> classes_;
    std::map<std::string, std::uint32_t> class_index_;
    // --- counter tables (prebuilt in prepare(); on-demand build is a backstop) ---
    std::map<std::string, TableEntry> tables_;
    std::map<std::string, BodyEntry> bodies_;
    std::vector<std::uint8_t> min_k_all_;
    std::vector<std::uint8_t> min_k_region_;
    std::vector<std::uint8_t> min_k_boundary_;
    std::vector<const Element*> chain_refs_;  // sorted, from the last recognition
    std::vector<FragEntry> frag_table_;
    std::uint32_t frag_gen_ = 0;
    // --- per-fill state (cleared or generation-bumped per produce) ---
    std::vector<std::uint32_t> row_;  // the scratch row being produced
    const llama_grammar_rules* rules_ = nullptr;
    PosSet root_;
    std::vector<CapEntry> cap_table_;
    std::uint32_t cap_gen_ = 0;
    std::vector<PrepEntry> prep_table_;
    std::uint32_t prep_gen_ = 0;
    std::map<const Element*, std::uint32_t> top_class_;
    std::vector<Stk> fast_c_set_;             // continuation positions of the recognized state
    std::string fast_body_key_;               // body signature (from the last recognition)
    const llama_grammar_rules* fast_rules_ = nullptr;
    bool fast_body_only_ = false;
    bool fast_abort_ = false;
    // --- state-shape row store (D3 bounds) ---
    std::vector<ShapeEntry> shape_cache_;
    // Signatures of shapes seen at least once. The row (31 KB) is stored only when a
    // shape repeats; first occurrences record the 16-byte signature only, so walks with
    // unique-per-step shapes (diary class) pay no row-copy overhead.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> shape_seen_;
    bool shape_miss_store_ = false;
    std::uint64_t shape_miss_h1_ = 0;
    std::uint64_t shape_miss_h2_ = 0;

    void build_child_index();
    std::uint32_t class_id(const CharClass& cls);
    std::uint32_t class_id_for_top(const Element* top);
    const PrepEntry* prepare_cached(const Stk& stack);
    void ensure_class_structures(std::uint32_t cid);
    const std::vector<std::uint8_t>& outside_for(std::uint32_t cid);
    void set_bit(int id);
    int capacity_of(const Stk& stack, const CharClass& cls, int fuel);
    int* cap_slot(const Stk& stack);
    bool try_bulk(std::uint32_t node, const PosSet& pos, std::uint32_t depth,
                  const PrepEntry* const* prepared);
    void visit(std::uint32_t node, const PosSet& pos, std::uint32_t depth,
               std::vector<int>& anc_pos, std::vector<int>& anc_neg);
    void judge_linear(const PosSet& start, const std::uint32_t* cps, std::size_t len,
                      llama_partial_utf8 tail, int id, bool count_tail);
    bool is_chain_ref(const Element* e) const noexcept;
    void expand_threshold(const Stk& stack, std::uint8_t k, bool region, KSet& out);
    const FragEntry* frag_cached(const KPos& q);
    void threshold_visit(std::uint32_t node, const KSet& cur);
    // Recognizes the state's counter chain and fills the key, rounds and
    // continuation out-parameters. The key includes the frontier (the body
    // positions the parallel candidates sit at) so a fill at a deeper frontier
    // gets its own table.
    bool recognize_chain(const llama_grammar_rules& rules, const PosSet& pos, std::string& key,
                         int& p, std::vector<Stk>& c_set);
    bool build_body_table(const std::string& body_key, int p);
    std::uint32_t trie_child(std::uint32_t node, std::uint32_t cp) const noexcept;
    bool c_accepts_suffix(const std::uint32_t* cps, std::size_t len);
    KSet initial_kset();
    std::uint8_t partial_threshold(const std::uint32_t* cps, std::size_t len,
                                   llama_partial_utf8 tail);
    bool build_counter_table(const std::string& key, int p, bool on_demand);
    void prepare_classes(const llama_grammar_rules& rules);
    void prepare_counter_shapes(const llama_grammar_rules& rules, std::uint32_t root_rule);
    void shape_signature(const llama_partial_utf8& partial, std::uint64_t& h1,
                         std::uint64_t& h2) const;
    const ShapeEntry* shape_lookup(std::uint64_t h1, std::uint64_t h2) const;
    void shape_store(std::uint64_t h1, std::uint64_t h2);
};

}  // namespace ninfer::models::qwen3_5::frontend::detail
