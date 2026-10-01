// Producer tests (wayfinder map #45, ticket #79).
//
// The compiled token-trie producer is verified against the vendored engine's
// full-vocabulary walk on the fixture grammars. Per the D4 ruling (#59) the
// differential gate runs with the producer's shape cache OFF (every fill is a
// fresh compute, so a stale cached row can never mask a real mismatch); the D2a
// prebuilds still run, so this also exercises the prebuilt counter tables. A
// separate test covers the shape cache itself, and a cost smoke bounds the
// steady-state fill time on host.
#include "json-schema-to-grammar.h"
#include "json.h"
#include "llama-grammar.h"
#include "llama-vocab.h"
#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/test_access.h"
#include "models/qwen3_5/frontend/grammar/trie_producer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace frontend = ninfer::models::qwen3_5::frontend;
namespace detail   = ninfer::models::qwen3_5::frontend::detail;

namespace {

// ---------------------------------------------------------------------------
// Check framework (same style as test_grammar.cpp)
// ---------------------------------------------------------------------------

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = "") {
    if (ok) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s%s%s\n", what.c_str(), detail.empty() ? "" : ": ", detail.c_str());
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open fixture: " + path); }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// ---------------------------------------------------------------------------
// Standard fixture vocabulary: every single byte plus multi-byte pieces and two
// end-of-generation ids (identical to test_grammar.cpp's standard_vocabulary).
// ---------------------------------------------------------------------------

class StandardVocab final : public frontend::GrammarVocabulary {
public:
    StandardVocab() {
        for (int byte = 0; byte < 256; ++byte) { pieces_.emplace_back(1, static_cast<char>(byte)); }
        const char* extras[] = {"ab",  "abc", "ba",     "ca",     "-",       "=",       ";",
                                "0.",  "0.5", "\n\n",   "да",     "нет",     "ет",      "т",
                                "д",   "Hello", " world", "日记",  "记",      "！",      "。",
                                "{",   "}",   "\":" ,   "\",\"",  "title",   "text",    "facts",
                                "people", "when", "name"};
        for (const char* extra : extras) { pieces_.emplace_back(extra); }
        const int eos = static_cast<int>(pieces_.size());
        pieces_.emplace_back("<eos>");
        const int im_end = static_cast<int>(pieces_.size());
        pieces_.emplace_back("<|im_end|>");
        eog_ = {eos, im_end};
        for (std::size_t id = 0; id < pieces_.size(); ++id) {
            piece_to_id_.emplace(pieces_[id], static_cast<int>(id));
            max_length_ = std::max(max_length_, static_cast<int>(pieces_[id].size()));
        }
    }

    [[nodiscard]] int token_count() const noexcept override {
        return static_cast<int>(pieces_.size());
    }
    [[nodiscard]] std::string_view piece(int id) const override {
        return pieces_[static_cast<std::size_t>(id)];
    }
    [[nodiscard]] bool is_eog(int id) const override {
        return std::find(eog_.begin(), eog_.end(), id) != eog_.end();
    }
    [[nodiscard]] std::vector<int> encode(std::string_view text) const override {
        std::vector<int> ids;
        std::size_t offset = 0;
        while (offset < text.size()) {
            int matched = 0;
            const int limit =
                std::min<int>(max_length_, static_cast<int>(text.size() - offset));
            for (int length = limit; length >= 1; --length) {
                const auto found = piece_to_id_.find(
                    std::string(text.substr(offset, static_cast<std::size_t>(length))));
                if (found != piece_to_id_.end()) {
                    matched = length;
                    ids.push_back(found->second);
                    break;
                }
            }
            if (matched == 0) {
                throw std::runtime_error("test vocabulary cannot encode input");
            }
            offset += static_cast<std::size_t>(matched);
        }
        return ids;
    }

private:
    std::vector<std::string> pieces_;
    std::vector<int> eog_;
    std::unordered_map<std::string, int> piece_to_id_;
    int max_length_ = 0;
};

// ---------------------------------------------------------------------------
// Engine facade and the independent engines
// ---------------------------------------------------------------------------

struct OracleBridge {
    const StandardVocab* view = nullptr;
    mutable std::string scratch;

    static const std::string& piece(void* context, llama_token id) {
        auto& self   = *static_cast<OracleBridge*>(context);
        self.scratch = std::string(self.view->piece(static_cast<int>(id)));
        return self.scratch;
    }
    static bool eog(void* context, llama_token id) {
        return static_cast<OracleBridge*>(context)->view->is_eog(static_cast<int>(id));
    }
    static int32_t tokenize(void*, const char*, int32_t, llama_token*, int32_t, bool, bool) {
        throw std::runtime_error("oracle facade does not implement tokenize");
    }
};

// A raw engine built the same way the module builds its master (low-level init
// from a parser's rule table), so rule indices agree with the producer's
// prebuilt tables.
llama_grammar* make_engine(const llama_vocab& facade, const std::string& gbnf) {
    llama_grammar_parser parser(&facade);
    if (!parser.parse(gbnf.c_str())) {
        throw std::runtime_error("engine parse failed: " + parser.last_error);
    }
    std::vector<const llama_grammar_element*> rule_ptrs;
    for (const auto& rule : parser.rules) { rule_ptrs.push_back(rule.data()); }
    auto* grammar = llama_grammar_init_impl(&facade, rule_ptrs.data(), rule_ptrs.size(),
                                            parser.symbol_ids.at("root"));
    if (!grammar) { throw std::runtime_error("engine init failed"); }
    return grammar;
}

class EngineOwner {
public:
    explicit EngineOwner(llama_grammar* grammar) : grammar_(grammar) {}
    ~EngineOwner() {
        if (grammar_) { llama_grammar_free_impl(grammar_); }
    }
    EngineOwner(const EngineOwner&)            = delete;
    EngineOwner& operator=(const EngineOwner&) = delete;
    llama_grammar* get() const noexcept { return grammar_; }
    void reset(llama_grammar* grammar) {
        if (grammar_) { llama_grammar_free_impl(grammar_); }
        grammar_ = grammar;
    }

private:
    llama_grammar* grammar_;
};

// The independent oracle: a raw engine driven in parallel; token_allowed samples
// the sampler's validate path (a one-element apply) without disturbing the walk.
class Oracle {
public:
    Oracle(const StandardVocab& vocabulary, const std::string& gbnf)
        : vocabulary_(vocabulary) {
        bridge_.view = &vocabulary_;
        facade_      = llama_vocab{.context     = &bridge_,
                                   .to_piece    = &OracleBridge::piece,
                                   .to_eog      = &OracleBridge::eog,
                                   .to_tokenize = &OracleBridge::tokenize};
        owner_ = std::make_unique<EngineOwner>(make_engine(facade_, gbnf));
    }

    [[nodiscard]] bool token_allowed(int id) const {
        llama_token_data data{static_cast<llama_token>(id), 0.0f, 0.0f};
        llama_token_data_array array{&data, 1, -1, false};
        llama_grammar_apply_impl(*owner_->get(), &array);
        return !(std::isinf(array.data[0].logit) && array.data[0].logit < 0.0f);
    }

    // Advances the oracle; returns false (and leaves state unchanged) on reject.
    bool accept(int id) {
        const llama_grammar_stacks saved_stacks = owner_->get()->stacks;
        const llama_partial_utf8 saved_partial  = owner_->get()->partial_utf8;
        try {
            llama_grammar_accept_token(*owner_->get(), static_cast<llama_token>(id),
                                       std::string(vocabulary_.piece(id)));
            return true;
        } catch (const std::exception&) {
            owner_->get()->stacks       = saved_stacks;
            owner_->get()->partial_utf8 = saved_partial;
            return false;
        }
    }

private:
    const StandardVocab& vocabulary_;
    OracleBridge bridge_;
    llama_vocab facade_;
    std::unique_ptr<EngineOwner> owner_;
};

bool row_bit(const std::span<const std::uint32_t>& row, int id) {
    const auto word = static_cast<std::size_t>(id) >> 5U;
    return word < row.size() && (row[word] & (1U << (static_cast<unsigned>(id) & 31U))) != 0U;
}

// A differential feed: fresh walk + oracle engines, then at every step compare
// the producer's full row against the oracle's per-token verdicts and advance
// both on the pick.
void differential_feed(detail::Producer& producer, const StandardVocab& vocabulary,
                       const std::string& gbnf, const std::vector<int>& feed,
                       const std::string& label) {
    OracleBridge bridge;
    bridge.view = &vocabulary;
    const llama_vocab facade{.context = &bridge,
                             .to_piece = &OracleBridge::piece,
                             .to_eog = &OracleBridge::eog,
                             .to_tokenize = &OracleBridge::tokenize};
    EngineOwner walk_owner(make_engine(facade, gbnf));
    Oracle oracle(vocabulary, gbnf);
    const int vocab_size = vocabulary.token_count();
    for (std::size_t step = 0; step < feed.size(); ++step) {
        const auto row = producer.produce(walk_owner.get()->rules, walk_owner.get()->stacks,
                                          walk_owner.get()->partial_utf8);
        for (int id = 0; id < vocab_size; ++id) {
            const bool expected = oracle.token_allowed(id);
            const bool actual   = row_bit(row, id);
            if (actual != expected) {
                check(false, label + " differential mismatch at step " + std::to_string(step),
                      "id=" + std::to_string(id) + " producer=" + std::to_string((int)actual) +
                          " oracle=" + std::to_string((int)expected));
                return;
            }
        }
        const int pick = feed[step];
        check(oracle.accept(pick),
              label + " oracle advances at step " + std::to_string(step));
        llama_grammar_accept_token(*walk_owner.get(), static_cast<llama_token>(pick),
                                   std::string(vocabulary.piece(pick)));
    }
}

// ---------------------------------------------------------------------------
// Test 1: D4 differential gate (shape cache OFF, prebuilds ON)
// ---------------------------------------------------------------------------

void test_differential_cache_off() {
    const std::string g1 = read_file(std::string(NINFER_SOURCE_DIR) +
                                     "/tests/fixtures/grammar/g1_literal_alternation.gbnf");
    const std::string g2 = read_file(std::string(NINFER_SOURCE_DIR) +
                                     "/tests/fixtures/grammar/g2_bounded_line.gbnf");
    const std::string g3 = read_file(std::string(NINFER_SOURCE_DIR) +
                                     "/tests/fixtures/grammar/g3_key_value_record.gbnf");
    const std::string diary_gbnf =
        json_schema_to_grammar(
            common_json::parse(read_file(std::string(NINFER_SOURCE_DIR) +
                                         "/tests/fixtures/grammar/diary.json")),
            true);

    StandardVocab vocabulary;
    const struct {
        const char* name;
        const std::string& gbnf;
        std::vector<std::vector<int>> feeds;
    } cases[] = {
        {"g1", g1,
         {vocabulary.encode("да"), vocabulary.encode("нет")}},
        {"g2", g2,
         {vocabulary.encode(std::string(240, 'x')), vocabulary.encode("abc"),
          vocabulary.encode(std::string(3, 'x'))}},
        {"g3", g3,
         {vocabulary.encode("act=view;base_amount=0.5;say=All clear;tone=calm"),
          vocabulary.encode("act=none;say=hi;tone=ok"),
          vocabulary.encode("act=gesture;presay=yo;base_amount=1;say=done;tone=calm")}},
        {"diary", diary_gbnf,
         {vocabulary.encode(
              std::string("{\"title\":\"t\",\"text\":\"hello world\",\"facts\":[\"a\",\"b\"],"
                           "\"people\":[{\"name\":\"p\",\"when\":\"w\"}]}"))}},
    };

    for (const auto& case_def : cases) {
        std::string error;
        auto grammar = frontend::CompiledGrammar::compile(case_def.gbnf,
                                                          std::make_shared<StandardVocab>(),
                                                          &error);
        check(grammar != nullptr, std::string(case_def.name) + " compiled", error);
        if (!grammar) { continue; }
        auto producer = frontend::GrammarTestAccess::make_producer(*grammar, /*shape_cache=*/false);
        for (std::size_t feed = 0; feed < case_def.feeds.size(); ++feed) {
            differential_feed(*producer, vocabulary, case_def.gbnf, case_def.feeds[feed],
                              case_def.name);
        }
        if (std::string(case_def.name) == "diary") {
            // The diary's string counters run the class-counter fast path (differentially
            // verified above); pin it so a silent recognition regression (fast path always
            // declining, suite green via the exact fallback) cannot land unnoticed.
            check(producer->stats.fast_fills > 0, "diary walk exercised the fast path",
                  std::to_string(producer->stats.fast_fills) + " fast fills");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: one-time build property (the counter tables are built once per distinct
// shape and reused, not rebuilt per fill). The D2a prebuild enumeration is
// best-effort: shapes it reaches are prebuilt at compile time; shapes it misses
// (the static walk's continuation is wider than the runtime's, hitting the
// advance_expand caps) are built on demand at first use. Either way each shape's
// table is built once, so a long walk's on-demand builds stay bounded.
// ---------------------------------------------------------------------------

void test_one_time_builds() {
    const std::string diary_gbnf =
        json_schema_to_grammar(
            common_json::parse(read_file(std::string(NINFER_SOURCE_DIR) +
                                         "/tests/fixtures/grammar/diary.json")),
            true);

    std::string error;
    auto grammar = frontend::CompiledGrammar::compile(diary_gbnf, std::make_shared<StandardVocab>(),
                                                      &error);
    check(grammar != nullptr, "diary compiled for one-time-build test", error);
    if (!grammar) { return; }

    auto producer = frontend::GrammarTestAccess::make_producer(*grammar, /*shape_cache=*/false);
    StandardVocab vocabulary;

    // Walk the diary; the distinct counter shapes are built once and reused, so the
    // on-demand build count stays at the distinct-shape count: the first walk pays it,
    // the repeated walks add nothing.
    const auto feed = vocabulary.encode(
        std::string("{\"title\":\"t\",\"text\":\"hello world\",\"facts\":[\"a\"],"
                    "\"people\":[{\"name\":\"p\",\"when\":\"w\"}]}"));
    const std::size_t base = producer->stats.table_builds_on_demand;
    differential_feed(*producer, vocabulary, diary_gbnf, feed, "diary-onetime-1");
    const std::size_t first_walk = producer->stats.table_builds_on_demand - base;
    for (int rep = 2; rep <= 4; ++rep) {
        differential_feed(*producer, vocabulary, diary_gbnf, feed, "diary-onetime");
    }
    const std::size_t repeat_walks = producer->stats.table_builds_on_demand - base - first_walk;
    check(repeat_walks == 0,
          "counter tables built in the first walk are reused by later walks",
          std::to_string(repeat_walks) + " rebuilt on " + std::to_string(3) + " repeated walks");
    std::fprintf(stderr,
                 "[onetime] diary on-demand table builds: first walk=%zu, repeated walks=%zu\n",
                 first_walk, repeat_walks);
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// Test 3: the shape cache serves repeated shapes and stays correct
// ---------------------------------------------------------------------------

void test_shape_cache() {
    StandardVocab vocabulary;
    const std::string gbnf = "root ::= [ab]{0,64}";
    std::string error;
    auto grammar = frontend::CompiledGrammar::compile(gbnf, std::make_shared<StandardVocab>(),
                                                      &error);
    check(grammar != nullptr, "fold grammar compiled", error);
    if (!grammar) { return; }

    Oracle oracle(vocabulary, gbnf);
    auto state = grammar->initial_state();
    // Walk 64 single-byte steps; the counter shape repeats (positions at/above the
    // fold limit share a canonical signature), so the cache stores and serves it.
    for (int step = 0; step < 64; ++step) {
        const auto row = grammar->row_for(state);
        for (int id = 0; id < vocabulary.token_count(); ++id) {
            check(row_bit(row, id) == oracle.token_allowed(id),
                  "shape cache row matches the oracle at step " + std::to_string(step));
        }
        const int pick = (step % 2 == 0) ? 97 : 98;  // 'a' / 'b'
        check(state.accept(pick), "fold walk advances at step " + std::to_string(step));
        check(oracle.accept(pick), "fold oracle advances at step " + std::to_string(step));
    }
    const frontend::CompiledGrammar::Stats stats = grammar->stats();
    check(stats.row_hits > 0, "shape cache served repeated shapes",
          std::to_string(stats.row_hits));
}

// ---------------------------------------------------------------------------
// Test 4: cost smoke (steady-state fill time on host, generous budget)
// ---------------------------------------------------------------------------

void test_cost_smoke() {
    StandardVocab vocabulary;
    const std::string g3 = read_file(std::string(NINFER_SOURCE_DIR) +
                                     "/tests/fixtures/grammar/g3_key_value_record.gbnf");
    std::string error;
    auto grammar = frontend::CompiledGrammar::compile(g3, std::make_shared<StandardVocab>(),
                                                      &error);
    check(grammar != nullptr, "g3 compiled for cost smoke", error);
    if (!grammar) { return; }

    auto state = grammar->initial_state();
    const auto feed = vocabulary.encode("act=view;base_amount=0.5;say=All clear;tone=calm");
    std::vector<std::chrono::nanoseconds> samples;
    for (const int id : feed) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto row = grammar->row_for(state);
        (void)row;
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(t1 - t0);
        state.accept(id);
    }
    std::sort(samples.begin(), samples.end());
    const auto mean_ns =
        std::accumulate(samples.begin(), samples.end(), std::chrono::nanoseconds{0}) /
        static_cast<std::int64_t>(samples.size());
    const auto p50_ns = samples[samples.size() / 2];
    // Host smoke only (the real 0.4 ms/step bar is measured on the target in #54):
    // bound against a generous 2 ms/step to catch pathological regressions.
    check(p50_ns < std::chrono::nanoseconds(2'000'000),
          "g3 steady-state p50 under the host smoke budget", std::to_string(p50_ns.count()));
    check(mean_ns < std::chrono::nanoseconds(2'000'000),
          "g3 steady-state mean under the host smoke budget", std::to_string(mean_ns.count()));
    std::fprintf(stderr, "[cost] g3 fills=%zu p50=%ld ns mean=%ld ns\n", samples.size(),
                 static_cast<long>(p50_ns.count()), static_cast<long>(mean_ns.count()));
}

// ---------------------------------------------------------------------------
// Test 5: wide parallel stacks (a 25-alternative REF runs 25 parallel stacks -
// beyond the old kMaxPos of 24). Pins the produce() pre-check: the row must be
// computed correctly (no overflow, no throw) and match the engine oracle.
// ---------------------------------------------------------------------------

void test_wide_stacks() {
    StandardVocab vocabulary;
    std::string gbnf = "root ::= \"start\" val\nval ::=";
    for (int i = 0; i < 25; ++i) {
        gbnf += std::string(i ? " | " : " ") + "\"v" + std::to_string(i) + "\"";
    }
    std::string error;
    auto grammar = frontend::CompiledGrammar::compile(gbnf, std::make_shared<StandardVocab>(),
                                                      &error);
    check(grammar != nullptr, "wide-stack grammar compiled", error);
    if (!grammar) { return; }

    Oracle oracle(vocabulary, gbnf);
    auto state = grammar->initial_state();
    const int start_chars[] = {115, 116, 97, 114, 116};  // "start"
    for (const int c : start_chars) {
        check(state.accept(c), "wide-stack state advances");
        check(oracle.accept(c), "wide-stack oracle advances");
    }
    const auto row = grammar->row_for(state);
    for (int id = 0; id < vocabulary.token_count(); ++id) {
        check(row_bit(row, id) == oracle.token_allowed(id),
              "wide-stack row matches the oracle for id " + std::to_string(id));
    }
    // 'v' plus the single-byte digits 0-9 that begin the alternatives.
    check(row_bit(row, 118), "wide-stack row allows the shared leading 'v'");
}

}  // namespace

int main() {
    test_differential_cache_off();
    test_one_time_builds();
    test_shape_cache();
    test_cost_smoke();
    test_wide_stacks();
    if (g_failures == 0) {
        std::fprintf(stderr, "producer tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d producer test failure(s)\n", g_failures);
    return 1;
}
