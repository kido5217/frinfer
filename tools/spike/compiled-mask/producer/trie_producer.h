// In-house token-trie mask producer prototype (wayfinder ticket #57, route (a)).
//
// Produces the same dense bitmask row as the module's full-vocabulary walk
// (src/models/qwen3_5/frontend/grammar/grammar.cpp, which drives the vendored
// llama.cpp grammar engine), by walking a codepoint trie over the vocabulary
// pieces and advancing the engine's own stack semantics.
//
// Exactness contract: for a state (stacks, partial_utf8), bit v is set iff the
// vendored engine accepts token v at that state, EOG ids exactly when some stack
// is empty, matching llama_grammar_apply_impl bit for bit.
#pragma once

#include "llama-grammar.h"

#include "models/qwen3_5/frontend/grammar/vocabulary.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace trie_probe {

using Element = llama_grammar_element;

inline constexpr int kMaxStack = 24;   // stack depth cap (engine stacks: <= 6 on the diary)
inline constexpr int kMaxPos = 24;      // position-set cap (engine stacks per state: <= 6 on the diary)
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

struct ProducerStats {
    std::uint64_t body_builds = 0;
    std::uint64_t body_build_ns = 0;
    std::uint64_t fills = 0;
    std::uint64_t nodes_visited = 0;
    std::uint64_t edges_visited = 0;
    std::uint64_t bulks = 0;
    std::uint64_t bulk_ids = 0;
    std::uint64_t bits_set = 0;
    std::uint64_t tail_judged = 0;
    std::uint64_t tail_steps = 0;
    std::uint64_t capacity_calls = 0;
    std::uint64_t capacity_steps = 0;
    std::uint64_t partial_states = 0;
    std::uint64_t exception_judged = 0;
    std::uint64_t declined_capacity = 0;
    std::uint64_t declined_class = 0;
    std::uint64_t excl_judged = 0;
    std::uint64_t overflow_fills = 0;
    // Class-counter fast path
    std::uint64_t fast_fills = 0;        // fills served by the counter table
    std::uint64_t fallback_fills = 0;    // fills served by the trie walk
    std::uint64_t recog_attempts = 0;
    std::uint64_t recog_ok = 0;
    std::uint64_t table_builds = 0;
    std::uint64_t table_build_fail = 0;
    std::uint64_t table_masks_words = 0; // words allocated across cached tables
    std::uint64_t bucket_tokens = 0;     // tokens placed in a table
    std::uint64_t correction_tokens = 0; // tokens whose threshold needs the continuation C
    std::uint64_t fast_ns = 0;
    std::uint64_t build_ns = 0;
    // State-shape row cache (producer-owned compiled masks)
    std::uint64_t shape_hits = 0;
    std::uint64_t shape_misses = 0;
    std::uint64_t shape_evictions = 0;
    // rdtsc phase attribution (cycles)
    std::uint64_t cyc_tail = 0;
    std::uint64_t cyc_bulk_set = 0;
    std::uint64_t cyc_capacity = 0;
    std::uint64_t cyc_prepare = 0;
    std::uint64_t cyc_visit = 0;
};

class Producer {
public:
    Producer();

    // Builds the codepoint trie from the vocabulary. Pieces are decoded exactly
    // like the vendored decode_utf8 (NUL-truncating, first-byte-NUL filtered).
    void build(const ninfer::models::qwen3_5::frontend::GrammarVocabulary& vocabulary);

    // Produces the row for one engine state. `rules`/`stacks` must be the same
    // rule set the stacks point into. Zeroes `row` (size row_words()).
    void produce(const llama_grammar_rules& rules, const llama_grammar_stacks& stacks,
                 const llama_partial_utf8& partial, std::vector<std::uint32_t>& row);

    [[nodiscard]] std::uint32_t token_count() const noexcept { return token_count_; }
    [[nodiscard]] std::uint32_t row_words() const noexcept { return row_words_; }
    [[nodiscard]] std::uint64_t trie_nodes() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::uint64_t trie_edges() const noexcept { return edges_.size(); }
    [[nodiscard]] std::uint64_t trie_ids() const noexcept { return euler_ids_.size(); }
    [[nodiscard]] std::uint64_t partial_ids() const noexcept { return partial_ids_.size(); }
    [[nodiscard]] std::uint64_t exception_ids() const noexcept { return exceptions_.size(); }
    [[nodiscard]] std::uint64_t memory_bytes() const noexcept;

    // Cumulative counters.
    ProducerStats stats;

    // Ablation switches.
    bool bulk_enabled = true;
    bool tail_enabled = true;
    bool writes_enabled = true;
    bool fast_path_enabled = true;

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
    struct Prepared {
        bool is_char = false;
        std::uint32_t class_id = 0;
        Succ succ;
    };

    // --- static trie ---
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
    std::vector<ClassCache> classes_;
    std::map<std::string, std::uint32_t> class_index_;

    // --- per-fill state ---
    const llama_grammar_rules* rules_ = nullptr;
    std::vector<std::uint32_t>* row_ = nullptr;
    std::map<std::vector<const Element*>, int> cap_memo_;
    PosSet root_;

    // --- state-shape row cache (producer-owned compiled masks) ---
    // The row is a pure function of (root-position content, can_end, pending partial,
    // ablation switches): the producer reads no other per-state input. A 128-bit content
    // hash keys the row; on a hit the row is a memcpy.
    struct ShapeEntry {
        std::uint64_t h1 = 0;
        std::uint64_t h2 = 0;
        std::vector<std::uint32_t> row;
    };
    std::vector<ShapeEntry> shape_cache_;
    static constexpr std::size_t kShapeCacheMax = 1024;
    // Signatures of shapes seen at least once. The row (31 KB) is stored only when a
    // shape repeats; first occurrences record the 16-byte signature only, so walks with
    // unique-per-step shapes (diary class) pay no row-copy overhead.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> shape_seen_;
    static constexpr std::size_t kShapeSeenMax = 8192;
    bool shape_cache_enabled_ = true;
    bool shape_miss_store_ = false;
    std::uint64_t shape_miss_h1_ = 0;
    std::uint64_t shape_miss_h2_ = 0;

    void shape_signature(const llama_partial_utf8& partial, std::uint64_t& h1, std::uint64_t& h2) const;
    const ShapeEntry* shape_lookup(std::uint64_t h1, std::uint64_t h2) const;
    void shape_store(std::uint64_t h1, std::uint64_t h2);

    // Open-addressing capacity memo (per fill, generation-cleared).
    struct CapEntry {
        std::uint64_t hash = 0;
        std::uint32_t gen = 0;
        int cap = 0;
        Stk stack;
    };
    std::vector<CapEntry> cap_table_;
    std::uint32_t cap_gen_ = 0;
    std::map<const Element*, std::uint32_t> top_class_;
    int* cap_slot(const Stk& stack);

    // Open-addressing prepared-successor cache (per fill): position stack -> class + expansion.
    struct PrepEntry {
        std::uint64_t hash = 0;
        std::uint32_t gen = 0;
        std::uint32_t class_id = 0;
        bool is_char = false;
        Succ succ;
        Stk key;
    };
    std::vector<PrepEntry> prep_table_;  // sized lazily; big entries, keep modest
    std::uint32_t prep_gen_ = 0;
    const PrepEntry* prepare_cached(const Stk& stack);

    std::uint32_t class_id(const CharClass& cls);
    std::uint32_t class_id_for_top(const Element* top);
    // --- class-counter fast path (fastpath_* function bodies below) ---
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
    struct TableEntry {
        std::int32_t table_max = -1;  // -1 = recognition/build failed for this key
        std::vector<std::uint32_t> masks;  // (table_max+1) * row_words
        std::vector<std::uint8_t> partial_thr;  // per partial id: threshold (0xFF = rejected)
    };
    // Per-body region table: the counter rounds alone, without the continuation.
    struct BodyEntry {
        std::int32_t max_p = -1;                  // chain rounds the DFS could see
        std::vector<std::uint32_t> buckets;       // 129 * row_words, bucket[b] = thr == b
        std::vector<std::uint8_t> node_boundary;  // per node: min boundary rounds, 0xFF none
        std::vector<std::uint8_t> thr;            // per euler token: region-only threshold
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
    std::vector<FragEntry> frag_table_;
    std::uint32_t frag_gen_ = 0;
    const FragEntry* frag_cached(const KPos& q);
    std::map<std::string, TableEntry> tables_;
    std::vector<std::uint32_t> euler_node_;  // trie end node per euler token
    std::vector<std::uint32_t> child_off_;   // per node: start index into child_edges_
    std::vector<std::uint32_t> child_edges_; // edge indices grouped by parent, cp-sorted
    std::vector<std::uint8_t> min_k_all_;
    std::vector<std::uint8_t> min_k_region_;
    std::vector<const Element*> chain_refs_;  // sorted, for binary-search membership
    std::vector<Stk> fast_c_set_;             // continuation positions of the recognized state
    std::string fast_body_key_;               // body signature (from the last recognition)
    std::map<std::string, BodyEntry> bodies_;
    std::vector<std::uint8_t> min_k_boundary_;
    bool fast_body_only_ = false;
    const llama_grammar_rules* fast_rules_ = nullptr;
    bool fast_abort_ = false;
    bool is_chain_ref(const Element* e) const noexcept;
    void expand_threshold(const Stk& stack, std::uint8_t k, bool region, KSet& out);
    void threshold_visit(std::uint32_t node, const KSet& cur);
    bool recognize_chain(const llama_grammar_rules& rules, const PosSet& pos, std::string& key,
                         int& p, std::vector<Stk>& c_set);
    bool build_counter_table(const std::string& key, int p);
    bool build_body_table(const std::string& body_key, int p);
    bool c_accepts_suffix(const std::uint32_t* cps, std::size_t len);
    std::uint32_t trie_child(std::uint32_t node, std::uint32_t cp) const noexcept;
    void build_child_index();
    KSet initial_kset();
    std::uint8_t partial_threshold(const std::uint32_t* cps, std::size_t len,
                                   llama_partial_utf8 tail);
    void ensure_class_structures(std::uint32_t class_id);
    const std::vector<std::uint8_t>& outside_for(std::uint32_t class_id);
    bool try_bulk(std::uint32_t node, const PosSet& pos, std::uint32_t depth,
                  const PrepEntry* const* prepared);
    void visit(std::uint32_t node, const PosSet& pos, std::uint32_t depth,
               std::vector<int>& anc_pos, std::vector<int>& anc_neg);
    int capacity_of(const Stk& stack, const CharClass& cls, int fuel);
    void judge_linear(const PosSet& start, const std::uint32_t* cps, std::size_t len,
                      llama_partial_utf8 tail, int id, bool count_tail);
    void set_bit(int id);
};

// Exposed for the probe's self-check.
[[nodiscard]] bool decode_utf8_bytes(const std::uint8_t* bytes, std::size_t len,
                                     const llama_partial_utf8& start, std::vector<std::uint32_t>& cps,
                                     llama_partial_utf8& end);
[[nodiscard]] CharClass class_from_element(const Element* pos);

}  // namespace trie_probe
