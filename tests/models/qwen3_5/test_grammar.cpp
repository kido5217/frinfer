// Grammar module tests: the host-side GBNF mask engine
// (src/models/qwen3_5/frontend/grammar/).
//
// Three layers:
//  - the vendored engine's structural vector and GBNF language cases, ported from
//    llama.cpp tests/test-llama-grammar.cpp and tests/test-grammar-integration.cpp at the
//    vendored baseline, drive the same expectations through this module's compile and
//    state API;
//  - a differential mask oracle replays every vocabulary token through the engine
//    (one-element apply, the sampler's validate path) and through an accept replay at the
//    states the module produced rows for, so cached and counter-normalized rows are checked
//    against the exact state they are served at, not against each other;
//  - cache, repetition-bound and construction-path behavior (row fills, dedup, the
//    maxLength 2400 diary schema, the tokenizer-backed vocabulary).
//
// Vocabularies are synthetic: all 256 single bytes plus multi-byte pieces (Latin, Cyrillic,
// CJK) and end-of-generation ids, so byte-level pieces, partial UTF-8 sequences and unicode
// terminals are exercised without a model artifact.

#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/grammar_runtime.h"
#include "models/qwen3_5/frontend/grammar/thinking_wrapper.h"
#include "models/qwen3_5/frontend/grammar/tokenizer_vocabulary.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include "json-schema-to-grammar.h"
#include "llama-grammar.h"
#include "llama-vocab.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace frontend = ninfer::models::qwen3_5::frontend;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s%s%s\n", what.c_str(), detail.empty() ? "" : ": ",
                     detail.c_str());
    }
}

// ---------------------------------------------------------------------------
// Synthetic vocabulary
// ---------------------------------------------------------------------------

// Explicit piece table (id = index), explicit end-of-generation ids and a greedy
// longest-match encoder over the pieces. The byte table is complete, so any text encodes.
class SyntheticVocabulary final : public frontend::GrammarVocabulary {
public:
    SyntheticVocabulary(std::vector<std::string> pieces, std::vector<int> eog)
        : pieces_(std::move(pieces)), eog_(std::move(eog)) {
        std::size_t max_length = 0;
        for (std::size_t id = 0; id < pieces_.size(); ++id) {
            piece_to_id_.emplace(pieces_[id], static_cast<int>(id));
            max_length = std::max(max_length, pieces_[id].size());
        }
        max_length_ = static_cast<int>(max_length);
    }

    [[nodiscard]] int token_count() const noexcept override {
        return static_cast<int>(pieces_.size());
    }

    [[nodiscard]] std::string_view piece(int id) const override {
        if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size()) {
            throw std::out_of_range("synthetic vocabulary piece id out of range");
        }
        return pieces_[static_cast<std::size_t>(id)];
    }

    [[nodiscard]] bool is_eog(int id) const override {
        return std::find(eog_.begin(), eog_.end(), id) != eog_.end();
    }

    [[nodiscard]] std::vector<int> encode(std::string_view text) const override {
        std::vector<int> ids;
        std::size_t offset = 0;
        while (offset < text.size()) {
            int matched     = 0;
            const int limit = std::min<int>(max_length_, static_cast<int>(text.size() - offset));
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
                throw std::runtime_error("synthetic vocabulary cannot encode input");
            }
            offset += static_cast<std::size_t>(matched);
        }
        return ids;
    }

    [[nodiscard]] std::size_t max_piece_bytes() const noexcept {
        return static_cast<std::size_t>(max_length_);
    }

private:
    std::vector<std::string> pieces_;
    std::vector<int> eog_;
    std::unordered_map<std::string, int> piece_to_id_;
    int max_length_ = 0;
};

// The standard fixture vocabulary: every single byte (id = byte value) plus multi-byte
// pieces for Latin, Cyrillic and CJK text, and two end-of-generation ids.
std::shared_ptr<const SyntheticVocabulary> make_standard_vocabulary() {
    std::vector<std::string> pieces;
    for (int byte = 0; byte < 256; ++byte) { pieces.emplace_back(1, static_cast<char>(byte)); }
    const char* extras[] = {
        "ab",  "abc", "ba",    "ca",    "-",     "=",      ";",      "0.",   "0.5", "\n\n", "да",
        "нет", "ет",  "т",     "д",     "Hello", " world", "日记",   "记",   "！",  "。",   "{",
        "}",   "\":", "\",\"", "title", "text",  "facts",  "people", "when", "name"};
    for (const char* extra : extras) { pieces.emplace_back(extra); }
    const int eos = static_cast<int>(pieces.size());
    pieces.emplace_back("<eos>");
    const int im_end = static_cast<int>(pieces.size());
    pieces.emplace_back("<|im_end|>");
    return std::make_shared<const SyntheticVocabulary>(std::move(pieces),
                                                       std::vector<int>{eos, im_end});
}

const std::shared_ptr<const SyntheticVocabulary>& standard_vocabulary() {
    static const std::shared_ptr<const SyntheticVocabulary> vocabulary = make_standard_vocabulary();
    return vocabulary;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

bool row_bit(frontend::GrammarMaskRow row, int id) {
    const auto word = static_cast<std::size_t>(id) >> 5U;
    return word < row.size() && (row[word] & (1U << (static_cast<unsigned>(id) & 31U))) != 0U;
}

std::string render_row(frontend::GrammarMaskRow row, const frontend::GrammarVocabulary& vocab) {
    std::string text;
    int shown = 0;
    for (int id = 0; id < vocab.token_count(); ++id) {
        if (!row_bit(row, id)) { continue; }
        if (shown++ == 12) {
            text += " ...";
            break;
        }
        text += " ";
        text += std::to_string(id);
        text += ":";
        text += vocab.is_eog(id) ? std::string("<eog>") : std::string(vocab.piece(id));
    }
    return text;
}

// True when `piece` is a sequence of well-formed UTF-8 codepoints. The mask's walk
// rejects pieces that are not, while the accept path tolerates them as a poisoned partial
// sequence, so the differential oracle only demands the replay to agree on clean pieces.
bool decodes_cleanly(std::string_view piece) {
    std::size_t offset = 0;
    while (offset < piece.size()) {
        const auto lead    = static_cast<unsigned char>(piece[offset]);
        std::size_t length = 0;
        std::uint32_t code = 0;
        if (lead < 0x80) {
            offset += 1;
            continue;
        }
        if (lead >= 0xC2 && lead <= 0xDF) {
            length = 2;
            code   = lead & 0x1FU;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            code   = lead & 0x0FU;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            code   = lead & 0x07U;
        } else {
            return false;
        }
        if (offset + length > piece.size()) { return false; }
        for (std::size_t index = 1; index < length; ++index) {
            const auto next = static_cast<unsigned char>(piece[offset + index]);
            if ((next & 0xC0U) != 0x80U) { return false; }
            code = (code << 6U) | (next & 0x3FU);
        }
        if ((length == 3 && code < 0x800) || (length == 4 && code < 0x10000) || code > 0x10FFFF ||
            (code >= 0xD800 && code <= 0xDFFF)) {
            return false;
        }
        offset += length;
    }
    return true;
}

// ---------------------------------------------------------------------------
// The independent engine oracle
// ---------------------------------------------------------------------------

// Test-owned vocabulary facade, independent of the module's bridge.
struct OracleBridge {
    const SyntheticVocabulary* view = nullptr;
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

// An engine grammar built and driven directly by the test: the differential reference for
// the module's rows and states.
class EngineOracle {
public:
    EngineOracle(const std::string& gbnf, const SyntheticVocabulary& vocabulary)
        : vocabulary_(vocabulary) {
        bridge_.view = &vocabulary;
        facade_      = llama_vocab{.context     = &bridge_,
                                   .to_piece    = &OracleBridge::piece,
                                   .to_eog      = &OracleBridge::eog,
                                   .to_tokenize = &OracleBridge::tokenize};
        llama_grammar_parser parser(&facade_);
        if (!parser.parse(gbnf.c_str())) {
            throw std::runtime_error("oracle parse failed: " + parser.last_error);
        }
        grammar_.reset(
            llama_grammar_init_impl(&facade_, gbnf.c_str(), "root", false, nullptr, 0, nullptr, 0));
        if (!grammar_) { throw std::runtime_error("oracle init failed"); }
    }

    // True while a partial UTF-8 sequence is pending: the mask then applies a look-ahead
    // the accept path does not, so replay verdicts are not comparable.
    [[nodiscard]] bool partial_pending() const { return grammar_->partial_utf8.n_remain != 0; }

    [[nodiscard]] bool can_end() const {
        for (const llama_grammar_stack& stack : grammar_->stacks) {
            if (stack.empty()) { return true; }
        }
        return false;
    }

    // The sampler's validate path: a one-element candidate array through the same apply the
    // batch mask uses.
    [[nodiscard]] bool token_allowed(int id) const {
        llama_token_data data{static_cast<llama_token>(id), 0.0f, 0.0f};
        llama_token_data_array array{&data, 1, -1, false};
        llama_grammar_apply_impl(*grammar_, &array);
        return !(std::isinf(array.data[0].logit) && array.data[0].logit < 0.0f);
    }

    // Accept replay through llama_grammar_accept_token, restoring the engine state
    // afterwards so the verdict can be sampled per token without disturbing the walk.
    [[nodiscard]] bool accept_replay(int id) {
        const llama_grammar_stacks saved_stacks = grammar_->stacks;
        const llama_partial_utf8 saved_partial  = grammar_->partial_utf8;
        bool accepted                           = false;
        try {
            llama_grammar_accept_token(*grammar_, static_cast<llama_token>(id),
                                       std::string(vocabulary_.piece(id)));
            accepted = true;
        } catch (const std::exception&) { accepted = false; }
        grammar_->stacks       = saved_stacks;
        grammar_->partial_utf8 = saved_partial;
        return accepted;
    }

    [[nodiscard]] bool accept(int id) {
        const llama_grammar_stacks saved_stacks = grammar_->stacks;
        const llama_partial_utf8 saved_partial  = grammar_->partial_utf8;
        try {
            llama_grammar_accept_token(*grammar_, static_cast<llama_token>(id),
                                       std::string(vocabulary_.piece(id)));
            return true;
        } catch (const std::exception&) {
            grammar_->stacks       = saved_stacks;
            grammar_->partial_utf8 = saved_partial;
            return false;
        }
    }

private:
    const SyntheticVocabulary& vocabulary_;
    OracleBridge bridge_;
    llama_vocab facade_;

    struct GrammarDeleter {
        void operator()(llama_grammar* grammar) const { llama_grammar_free_impl(grammar); }
    };

    std::unique_ptr<llama_grammar, GrammarDeleter> grammar_;
};

// ---------------------------------------------------------------------------
// Feed helpers
// ---------------------------------------------------------------------------

int piece_id(const frontend::GrammarVocabulary& vocabulary, std::string_view piece) {
    for (int id = 0; id < vocabulary.token_count(); ++id) {
        if (vocabulary.piece(id) == piece) { return id; }
    }
    throw std::runtime_error("test vocabulary has no piece: " + std::string(piece));
}

// Feeds token ids through `state`, requiring every token to be allowed by the mask first.
bool feed_ids(frontend::GrammarState& state, const frontend::CompiledGrammar& grammar,
              const std::vector<int>& ids) {
    for (const int id : ids) {
        if (!row_bit(grammar.row_for(state), id)) { return false; }
        if (!state.accept(id)) { return false; }
    }
    return true;
}

bool feed_text(frontend::GrammarState& state, const frontend::CompiledGrammar& grammar,
               const SyntheticVocabulary& vocabulary, std::string_view text) {
    return feed_ids(state, grammar, vocabulary.encode(text));
}

// ---------------------------------------------------------------------------
// GBNF language cases (ported from tests/test-grammar-integration.cpp)
// ---------------------------------------------------------------------------

struct LanguageCase {
    std::string name;
    std::string grammar;
    std::vector<std::string> passing;
    std::vector<std::string> failing;
};

void run_language_case(const LanguageCase& test) {
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar =
        frontend::CompiledGrammar::compile(test.grammar, vocabulary, &error);
    check(grammar != nullptr, "language case " + test.name + " compiled", error);
    if (!grammar) { return; }

    for (const std::string& text : test.passing) {
        auto state     = grammar->initial_state();
        const bool fed = feed_text(state, *grammar, *vocabulary, text);
        check(fed && state.can_end(), "language case " + test.name + " accepts \"" + text + "\"");
    }
    for (const std::string& text : test.failing) {
        auto state     = grammar->initial_state();
        const bool fed = feed_text(state, *grammar, *vocabulary, text);
        check(!(fed && state.can_end()),
              "language case " + test.name + " rejects \"" + text + "\"");
    }
}

void test_language_cases() {
    const std::string many_x(240, 'x');
    const std::string too_many_x(241, 'x');

    const std::vector<LanguageCase> cases = {
        // Fixture G1: literal alternation over Cyrillic text.
        {"g1 literal alternation",
         "root ::= \"да\" | \"нет\"",
         {"да", "нет"},
         {"", "д", "не", "нето", "дада", "ет"}},
        // Fixture G2: bounded character class.
        {"g2 bounded line",
         "root ::= [^\\n]{3,240}",
         {std::string(3, 'x'), many_x, "abc def", "да!"},
         {std::string(2, 'x'), too_many_x, "ab\nc", ""}},
        // Fixture G3: key=value record.
        {"g3 key value record",
         "root  ::= \"act=\" act rest \";say=\" said \";tone=\" text\n"
         "act   ::= \"none\" | \"gesture\" | \"base\" | \"view\"\n"
         "rest  ::= (\";\" pair)*\n"
         "pair  ::= key \"=\" text\n"
         "key   ::= \"presay\" | \"base\" | \"base_amount\" | \"needs_view\"\n"
         "text  ::= [^;\\x0A]*\n"
         "said  ::= [^;\\x0A]+",
         {"act=base;say=Ready;tone=calm", "act=view;base_amount=0.5;say=All clear;tone=calm",
          "act=base;say=Ready;tone="},
         {"act=base;say=Ready", "act=unknown;say=Ready;tone=calm", "act=base;say=;tone=calm"}},
        // Ported: simple grammar.
        {"simple grammar",
         "root ::= expr\n"
         "expr ::= term (\"+\" term)*\n"
         "term ::= number\n"
         "number ::= [0-9]+",
         {"42", "1+2+3+4+5", "123+456"},
         {"+", "/ 3", "1+2+3+4+5+", "12a45"}},
        // Ported: medium complexity grammar.
        {"medium complexity grammar",
         "root ::= expression\n"
         "expression ::= term ws ((\"+\"|\"-\") ws term)*\n"
         "term ::= factor ws ((\"*\"|\"/\") ws factor)*\n"
         "factor ::= number | variable | \"(\" expression \")\" | function-call\n"
         "number ::= [0-9]+\n"
         "variable ::= [a-zA-Z_][a-zA-Z0-9_]*\n"
         "function-call ::= variable ws \"(\" (expression (\",\" ws expression)*)? \")\"\n"
         "ws ::= [ \\t\\n\\r]?",
         {"42", "1*2*3*4*5", "x", "x+10", "x1+y2", "(a+b)*(c-d)", "func()", "func(x,y+2)",
          "a*(b+c)-d/e", "f(g(x),h(y,z))", "x + 10", "123+456*789-123/456"},
         {"+", "/ 3x", "x + + y", "a * / b", "func(,)", "func(x y)", "(a + b", "x + y)",
          "a + b * (c - d", "42 +", "func("}},
        // Ported: special characters.
        {"special characters",
         "root ::= ... \"abc\" ...",
         {"abcabcabc", "aaaabcccc", "🔵🟠✅abc❌🟠🔵"},
         {"aaabcccc", "aaaaabcccc", "aaaabccc", "aaaabccccc", "🔵🟠✅❌abc❌✅🟠🔵",
          "🔵🟠abc🟠🔵"}},
        // Ported: quantifiers.
        {"* quantifier",
         "root ::= \"a\"*",
         {"", "a", "aaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
         {"b", "ab", "aab", "ba",
          "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"}},
        {"+ quantifier",
         "root ::= \"a\"+",
         {"a", "aaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
         {"", "b", "ab", "aab", "ba",
          "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"}},
        {"? quantifier", "root ::= \"a\"?", {"", "a"}, {"b", "ab", "aa", "ba"}},
        {"mixed quantifiers",
         "root ::= cons+ vowel* cons? (vowel cons)*\n"
         "vowel ::= [aeiouy]\n"
         "cons ::= [bcdfghjklmnpqrstvwxyz]",
         {"yes", "no", "noyes", "crwth", "four", "bryyyy"},
         {"yess", "yesno", "forty", "catyyy"}},
        {"simple exact repetition",
         "root ::= [ab]{4}",
         {"aaaa", "bbbb", "abab"},
         {"a", "b", "aaaaa"}},
        {"simple min repetition",
         "root ::= [ab]{4,}",
         {"aaaa", "aaaaab", "bbbb", "ababab"},
         {"", "aba"}},
        {"simple max repetition", "root ::= [ab]{0,4}", {"", "a", "aa", "aaa", "aaab"}, {"aaaaa"}},
        {"min / max repetition",
         "root ::= (\"0x\" [A-F0-9]{2} \" \"?){3,5}",
         {"0xFF 0x12 0xAB", "0xFF 0x12 0xAB 0x00 0x00"},
         {"", "0xFF", "0xFF 0x12", "0xFF 0x12 0xAB 0x00 0x00 0x00"}},
        {"segfault", "root ::= ( [x]* )*", {"", "x", "xx"}, {"y", "yy"}},
        // Fixture CJK/unicode terminal: multibyte text directly in the grammar.
        {"cjk literal alternation",
         "root ::= \"日记\" | \"記録\"",
         {"日记", "記録"},
         {"日", "日记记", "記"}},
    };
    for (const LanguageCase& test : cases) { run_language_case(test); }

    // `<token>` literals tokenize through the vocabulary's encode hook; the text between
    // the brackets must encode to exactly one id.
    {
        std::vector<std::string> pieces;
        for (int byte = 0; byte < 256; ++byte) { pieces.emplace_back(1, static_cast<char>(byte)); }
        pieces.emplace_back("<hello>");
        const auto vocabulary =
            std::make_shared<const SyntheticVocabulary>(std::move(pieces), std::vector<int>{});
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= <hello>", vocabulary, &error);
        check(grammar != nullptr, "token literal grammar compiled", error);
        if (grammar) {
            auto state   = grammar->initial_state();
            const int id = piece_id(*vocabulary, "<hello>");
            check(row_bit(grammar->row_for(state), id) && state.accept(id) && state.can_end(),
                  "token literal matches its encoded id");
        }
        std::shared_ptr<const frontend::CompiledGrammar> invalid =
            frontend::CompiledGrammar::compile("root ::= <hel>", vocabulary, &error);
        check(invalid == nullptr && !error.empty(),
              "token literal that does not encode to one id fails", error);
    }

    // Token terminals: `<[id]>` matches by token id, `!<[id]>` excludes it (ported case).
    {
        const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
        const std::string grammar_text = "root ::= <[10]> content <[11]>\ncontent ::= (!<[11]>)*";
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile(grammar_text, vocabulary, &error);
        check(grammar != nullptr, "token terminal grammar compiled", error);
        if (grammar) {
            const auto text_ids = [](std::string_view text) {
                return standard_vocabulary()->encode(text);
            };
            const auto run = [&](const std::vector<int>& ids) {
                auto state = grammar->initial_state();
                return feed_ids(state, *grammar, ids) && state.can_end();
            };
            std::vector<int> passing = {10};
            const auto body          = text_ids("hello world");
            passing.insert(passing.end(), body.begin(), body.end());
            passing.push_back(11);
            check(run(passing), "token terminal: complete region accepted");

            std::vector<int> nested = {10};
            const auto part         = text_ids("a");
            nested.insert(nested.end(), part.begin(), part.end());
            nested.push_back(12); // an arbitrary non-terminator token id
            nested.push_back(11);
            check(run(nested), "token terminal: nested other tokens accepted");

            check(!run({10, 11, 11}), "token terminal: double terminator rejected");
            check(!run({10}), "token terminal: missing terminator rejected");
        }
    }
}

// ---------------------------------------------------------------------------
// JSON Schema cases (ported from tests/test-grammar-integration.cpp)
// ---------------------------------------------------------------------------

// The vendored JSON-schema converter, the production channel's first half (the protocol adapter
// uses the same converter before the compile seam).
std::string schema_to_gbnf(const std::string& schema) {
    return json_schema_to_grammar(common_json::parse(schema), /*force_gbnf=*/true);
}

void run_schema_case(const std::string& name, const std::string& schema,
                     const std::vector<std::string>& passing,
                     const std::vector<std::string>& failing) {
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar =
        frontend::CompiledGrammar::compile(schema_to_gbnf(schema), vocabulary, &error);
    check(grammar != nullptr, "schema case " + name + " compiled", error);
    if (!grammar) { return; }

    for (const std::string& text : passing) {
        auto state     = grammar->initial_state();
        const bool fed = feed_text(state, *grammar, *vocabulary, text);
        check(fed && state.can_end(), "schema case " + name + " accepts " + text);
    }
    for (const std::string& text : failing) {
        auto state     = grammar->initial_state();
        const bool fed = feed_text(state, *grammar, *vocabulary, text);
        check(!(fed && state.can_end()), "schema case " + name + " rejects " + text);
    }
}

void test_json_schema_cases() {
    run_schema_case("empty schema (any value)", "{}",
                    {"{}", "{\"foo\": \"bar\"}", "[]", "null", "\"\"", "true"},
                    {"", "{\"foo\"}", "foo"});
    run_schema_case("string", "{\"type\": \"string\"}", {"\"foo\"", "\"\""},
                    {"{}", "\"foo\": \"bar\""});
    run_schema_case("string maxLength 3", "{\"type\": \"string\", \"maxLength\": 3}",
                    {"\"foo\"", "\"\"", "\"f\""}, {"\"foobar\""});
    run_schema_case("string min+max length 1..4",
                    "{\"type\": \"string\", \"minLength\": 1, \"maxLength\": 4}",
                    {"\"f\"", "\"barf\""}, {"\"\"", "\"barfo\""});
    run_schema_case("integer minimum 0", "{\"type\": \"integer\", \"minimum\": 0}",
                    {"0", "10", "10000"}, {"-1", "-10000", "00", "01", "-0"});
    run_schema_case("integer min 5 max 30",
                    "{\"type\": \"integer\", \"minimum\": 5, \"maximum\": 30}", {"5", "10", "30"},
                    {"05", "4", "-1", "31", "123"});
    run_schema_case("boolean", "{\"type\": \"boolean\"}", {"true", "false"},
                    {"\"\"", "\"true\"", "True", "FALSE"});
    run_schema_case("string const", "{\"const\": \"foo\"}", {"\"foo\""}, {"foo", "\"bar\""});
    run_schema_case("enum", "{\"enum\": [\"red\", \"amber\", \"green\", null, 42]}",
                    {"\"red\"", "null", "42"}, {"", "420", "true", "foo"});
    run_schema_case("simple pattern", "{\"pattern\": \"^[a-zA-Z0-9_-]*$\"}",
                    {"\"\"", "\"He_llo-12\""}, {"\"!\"", "\"Hello World\""});
    run_schema_case("min+max items",
                    "{\"items\": {\"type\": \"number\"}, \"minItems\": 3, \"maxItems\": 5}",
                    {"[1, 2, 3]", "[1, 2, 3, 4, 5]"}, {"[1, 2]", "[1, 2, 3, 4, 5, 6]", "1"});
    run_schema_case("object properties, additionalProperties false",
                    "{\"type\": \"object\", \"properties\": {\"number\": {\"type\": \"number\"}, "
                    "\"street_name\": {\"type\": \"string\"}}, \"additionalProperties\": false}",
                    {"{\"number\": 1600, \"street_name\": \"Pennsylvania\"}",
                     "{\"street_name\": \"Pennsylvania\"}"},
                    {"{\"number\": \"1600\"}", "{\"street_name\": \"P\", \"number\": 1600}"});
    run_schema_case("required props in order",
                    "{\"properties\": {\"b\": {\"type\": \"string\"}, \"a\": {\"type\": "
                    "\"string\"}}, \"required\": [\"a\", \"b\"], \"additionalProperties\": false}",
                    {"{\"b\": \"foo\", \"a\": \"bar\"}"},
                    {"{\"a\": \"foo\", \"b\": \"bar\"}", "{\"b\": \"bar\"}"});
}

// ---------------------------------------------------------------------------
// Repetition bound (the documented deviation: no silent unbounded clamping)
// ---------------------------------------------------------------------------

void test_repetition_bound() {
    check(LLAMA_GRAMMAR_MAX_REPETITION_BOUND == 65536, "exported repetition bound is 65536");

    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    const int a                                                  = 97; // 'a'

    // Upstream clamps any finite bound above 2000 to `unbounded`; this port must enforce it.
    const auto count_accepted = [&](const std::string& gbnf, std::string* error) {
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile(gbnf, vocabulary, error);
        if (!grammar) { return std::size_t{0}; }
        auto state    = grammar->initial_state();
        std::size_t n = 0;
        while (state.accept(a)) { ++n; }
        if (!state.can_end()) { return std::size_t{0}; }
        return n;
    };

    std::string error;
    check(count_accepted("root ::= [a]{0,2400}", &error) == 2400, "maxLength 2400 is enforced",
          error);
    check(count_accepted("root ::= [a]{0,2401}", &error) == 2401,
          "bound one past the old clamp is enforced", error);
    check(count_accepted("root ::= [a]{2400}", &error) == 2400, "exact repetition 2400 is enforced",
          error);

    // Above the exported bound the parse fails closed instead of loosening the grammar.
    {
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= [a]{0,65537}", vocabulary, &error);
        check(grammar == nullptr && !error.empty(), "repetition above the exported bound fails",
              error);
    }
    {
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= [a]{65537,}", vocabulary, &error);
        check(grammar == nullptr && !error.empty(), "minimum above the exported bound fails",
              error);
    }
    {
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= ([ab]{0,300}){300}", vocabulary, &error);
        check(grammar == nullptr && !error.empty(), "rule-count product above the bound fails",
              error);
    }
}

// ---------------------------------------------------------------------------
// Differential mask oracle
// ---------------------------------------------------------------------------

struct DifferentialResult {
    std::size_t steps       = 0; // visited states (one terminal state may add no advance)
    std::size_t advances    = 0;
    std::size_t rejected    = 0;
    std::size_t allowed_eog = 0;
    bool exhausted          = false;
};

DifferentialResult run_differential(const std::string& name, const std::string& gbnf,
                                    std::uint32_t seed, std::size_t max_steps,
                                    bool from_schema = false, bool single_step = false) {
    DifferentialResult result;
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar =
        from_schema ? frontend::CompiledGrammar::compile(schema_to_gbnf(gbnf), vocabulary, &error)
                    : frontend::CompiledGrammar::compile(gbnf, vocabulary, &error);
    check(grammar != nullptr, "differential " + name + " compiled", error);
    if (!grammar) { return result; }

    EngineOracle oracle(from_schema ? schema_to_gbnf(gbnf) : gbnf, *vocabulary);
    frontend::GrammarState state = grammar->initial_state();
    std::mt19937 rng(seed);

    for (std::size_t step = 0; step < max_steps; ++step) {
        const frontend::GrammarMaskRow row = grammar->row_for(state);

        for (int id = 0; id < vocabulary->token_count(); ++id) {
            const bool expected = oracle.token_allowed(id);
            const bool actual   = row_bit(row, id);
            if (actual != expected) {
                check(false,
                      "differential " + name + " step " + std::to_string(step) + " token " +
                          std::to_string(id),
                      "mask=" + std::string(actual ? "allow" : "deny") +
                          " oracle=" + (expected ? "allow" : "deny") + " row=[" +
                          render_row(row, *vocabulary) + "]");
                return result;
            }
            // The accept replay is the mask's oracle where the two paths are comparable:
            // non-EOG tokens carry text, and a pending partial UTF-8 sequence (or a piece the
            // mask declares illegal) is where the mask applies a look-ahead instead.
            const std::string_view piece = vocabulary->piece(id);
            const bool legal_piece       = !vocabulary->is_eog(id) && !piece.empty() &&
                                     piece[0] != '\0' && decodes_cleanly(piece);
            const bool comparable = legal_piece && !oracle.partial_pending();
            const bool replay     = oracle.accept_replay(id);
            if (actual) {
                if (comparable) {
                    check(replay, "differential " + name + ": mask-allowed token is replayable",
                          std::to_string(id));
                }
            } else {
                ++result.rejected;
                if (comparable && replay) {
                    check(false, "differential " + name + ": replay accepted a mask-rejected token",
                          std::to_string(id));
                    return result;
                }
            }
        }
        for (int id = 0; id < vocabulary->token_count(); ++id) {
            if (vocabulary->is_eog(id) && row_bit(row, id)) { ++result.allowed_eog; }
        }

        std::vector<int> allowed;
        for (int id = 0; id < vocabulary->token_count(); ++id) {
            if (!vocabulary->is_eog(id) && row_bit(row, id)) { allowed.push_back(id); }
        }
        ++result.steps;
        if (allowed.empty()) {
            result.exhausted = true;
            check(oracle.can_end(), "differential " + name + ": exhausted state can end");
            return result;
        }
        // `single_step` drives the walk deterministically through the smallest allowed
        // token, which for byte-level vocabularies advances one byte per step and so
        // reaches deep counter positions the random walk may skip.
        const int pick = single_step ? allowed.front() : allowed[rng() % allowed.size()];
        if (!state.accept(pick)) {
            check(false, "differential " + name + ": module rejected a mask-allowed token",
                  std::to_string(pick));
            return result;
        }
        check(oracle.accept(pick), "differential " + name + ": oracle rejected an advance");
        ++result.advances;
    }
    return result;
}

void test_differential_oracle() {
    const DifferentialResult g1 =
        run_differential("g1 literal alternation", "root ::= \"да\" | \"нет\"", 0x51u, 8);
    check(g1.rejected > 0, "g1 constrains");
    check(g1.allowed_eog > 0, "g1 can end at some state");

    const DifferentialResult g2 =
        run_differential("g2 bounded line", "root ::= [^\\n]{3,240}", 0x52u, 260);
    check(g2.steps > 20, "g2 walk reached the long tail", std::to_string(g2.steps));
    check(g2.rejected > 0, "g2 constrains");

    // A deterministic one-byte-per-step walk reaches the maxLength boundary exactly.
    const DifferentialResult g2_deep =
        run_differential("g2 deep walk", "root ::= [^\\n]{3,240}", 0x52u, 260, false, true);
    check(g2_deep.advances == 240 && g2_deep.exhausted, "g2 deep walk reached the 240 boundary",
          std::to_string(g2_deep.advances) +
              " advances, exhausted=" + std::to_string(static_cast<int>(g2_deep.exhausted)));

    const DifferentialResult g3 =
        run_differential("g3 key value record",
                         "root  ::= \"act=\" act rest \";say=\" said \";tone=\" text\n"
                         "act   ::= \"none\" | \"gesture\" | \"base\" | \"view\"\n"
                         "rest  ::= (\";\" pair)*\n"
                         "pair  ::= key \"=\" text\n"
                         "key   ::= \"presay\" | \"base\" | \"base_amount\" | \"needs_view\"\n"
                         "text  ::= [^;\\x0A]*\n"
                         "said  ::= [^;\\x0A]+",
                         0x53u, 60);
    check(g3.rejected > 0, "g3 constrains");

    const DifferentialResult cjk =
        run_differential("cjk literals", "root ::= \"日记\" \"！\"? \"。\"", 0x54u, 12);
    check(cjk.rejected > 0, "cjk constrains");
}

// ---------------------------------------------------------------------------
// Row cache behavior
// ---------------------------------------------------------------------------

bool rows_equal(frontend::GrammarMaskRow left, frontend::GrammarMaskRow right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

void test_row_cache() {
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();

    // The producer keys its shape cache by canonical state (not by mask content), so two
    // distinct states that happen to share a mask are each computed and stored separately.
    // The mask contents are still verified here against the expected branch structure.
    {
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= \"xa\" | \"ya\"", vocabulary, &error);
        check(grammar != nullptr, "dedup grammar compiled", error);
        if (grammar) {
            const int x                      = 120;
            const int y                      = 121;
            const frontend::GrammarMaskRow initial = grammar->row_for(grammar->initial_state());
            std::vector<std::uint32_t> initial_row(initial.begin(), initial.end());
            auto first                       = grammar->initial_state();
            check(first.accept(x), "dedup: first branch advances");
            const frontend::GrammarMaskRow first_row = grammar->row_for(first);
            std::vector<std::uint32_t> first_row_copy(first_row.begin(), first_row.end());
            auto second                              = grammar->initial_state();
            check(second.accept(y), "dedup: second branch advances");
            const frontend::GrammarMaskRow second_row = grammar->row_for(second);
            std::vector<std::uint32_t> second_row_copy(second_row.begin(), second_row.end());

            const frontend::GrammarMaskRow first_span = std::span(first_row_copy);
            const frontend::GrammarMaskRow second_span = std::span(second_row_copy);
            check(rows_equal(first_span, second_span), "dedup: branch rows are identical");
            check(row_bit(std::span(initial_row), x) && row_bit(std::span(initial_row), y),
                  "dedup: initial row allows branches");
            check(!row_bit(first_span, y) && !row_bit(second_span, x),
                  "dedup: branch rows are exact");
            // The compile-time initial-row validation computes the initial shape once, and
            // one-shot shapes are not stored, so this test's own initial-state fill computes it
            // again; the two branch shapes are unique and computed once each: four fills.
            const frontend::CompiledGrammar::Stats stats = grammar->stats();
            check(stats.row_fills == 4, "dedup: one fill per unique shape",
                  std::to_string(stats.row_fills));
            check(stats.row_hits == 0, "dedup: no repeated shape to serve from the cache",
                  std::to_string(stats.row_hits));
        }
    }

    // Counter-normalized keys: a long repetition chain is walked with a bounded number of
    // full-vocabulary fills (the fold limit is the longest piece plus two).
    {
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= [ab]{0,64}", vocabulary, &error);
        check(grammar != nullptr, "fold grammar compiled", error);
        if (grammar) {
            EngineOracle oracle("root ::= [ab]{0,64}", *vocabulary);
            auto state        = grammar->initial_state();
            const int ab      = piece_id(*vocabulary, "ab");
            std::size_t steps = 0;
            while (true) {
                const frontend::GrammarMaskRow row = grammar->row_for(state);
                for (int id = 0; id < vocabulary->token_count(); ++id) {
                    check(row_bit(row, id) == oracle.token_allowed(id),
                          "fold: row matches the oracle at step " + std::to_string(steps));
                }
                check(rows_equal(row, grammar->row_for(state)), "fold: cached row is stable");
                if (!state.accept(ab)) { break; }
                check(oracle.accept(ab), "fold: oracle advances");
                ++steps;
                if (steps > 64) { break; }
            }
            check(steps == 32, "fold: chain walked to its bound", std::to_string(steps));
            check(state.can_end(), "fold: exhausted chain can end");

            const frontend::CompiledGrammar::Stats stats = grammar->stats();
            const std::size_t fold_limit                 = vocabulary->max_piece_bytes() + 2;
            check(stats.row_fills <= fold_limit + 3, "fold: fills bounded by the fold limit",
                  std::to_string(stats.row_fills) + " fills, limit " + std::to_string(fold_limit));
            check(stats.row_fills * 2 < steps, "fold: fills far below visited states",
                  std::to_string(stats.row_fills) + " fills for " + std::to_string(steps) +
                      " states");
            check(stats.row_hits >= steps - stats.row_fills, "fold: later states are cache hits");
            check(stats.distinct_rows <= stats.row_fills, "fold: rows dedupe");
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture G4: diary JSON schema through the vendored converter
// ---------------------------------------------------------------------------

const char* kDiarySchema =
    R"({"type":"object","additionalProperties":false,"required":["title","text","facts","people"],)"
    R"("properties":{"title":{"type":"string","maxLength":60},)"
    R"("text":{"type":"string","maxLength":2400},)"
    R"("facts":{"type":"array","maxItems":14,"items":{"type":"string","maxLength":120}},)"
    R"("people":{"type":"array","maxItems":4,"items":{"type":"object","additionalProperties":false,)"
    R"("required":["name","when"],"properties":{"name":{"type":"string","maxLength":30},)"
    R"("when":{"type":"string","maxLength":5}}}}}})";

void test_diary_schema() {
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar =
        frontend::CompiledGrammar::compile(schema_to_gbnf(kDiarySchema), vocabulary, &error);
    check(grammar != nullptr, "diary schema compiles", error);
    if (!grammar) { return; }

    const std::string minimal =
        "{\"title\": \"Day one\", \"text\": \"The park was quiet\", \"facts\": [], "
        "\"people\": []}";
    {
        auto state = grammar->initial_state();
        check(feed_text(state, *grammar, *vocabulary, minimal), "diary: minimal document accepted");
        check(state.can_end(), "diary: minimal document complete");
    }
    {
        // maxLength 2400 is enforced on `text` (upstream silently dropped bounds above 2000).
        const std::string body(2400, 'q');
        const std::string document =
            "{\"title\": \"D\", \"text\": \"" + body + "\", \"facts\": [], \"people\": []}";
        auto state = grammar->initial_state();
        check(feed_text(state, *grammar, *vocabulary, document), "diary: 2400-char text accepted");
        check(state.can_end(), "diary: 2400-char document complete");
    }
    {
        const std::string body(2400, 'q');
        const std::string overflow =
            "{\"title\": \"D\", \"text\": \"" + body + "q\", \"facts\": [], \"people\": []}";
        auto state = grammar->initial_state();
        check(!feed_text(state, *grammar, *vocabulary, overflow), "diary: 2401-char text rejected");
    }
    {
        // The title carries its own bound (60).
        const std::string title(61, 't');
        const std::string document =
            "{\"title\": \"" + title + "\", \"text\": \"x\", \"facts\": [], \"people\": []}";
        auto state = grammar->initial_state();
        check(!feed_text(state, *grammar, *vocabulary, document), "diary: 61-char title rejected");
    }
    {
        // maxItems 14 on `facts`.
        std::string facts = "[";
        for (int index = 0; index < 15; ++index) {
            if (index != 0) { facts += ", "; }
            facts += "\"f\"";
        }
        facts += "]";
        const std::string document =
            "{\"title\": \"D\", \"text\": \"x\", \"facts\": " + facts + ", \"people\": []}";
        auto state = grammar->initial_state();
        check(!feed_text(state, *grammar, *vocabulary, document),
              "diary: 15 facts rejected (maxItems 14)");
    }
    {
        // The differential oracle over the real schema.
        const DifferentialResult result =
            run_differential("g4 diary schema", kDiarySchema, 0x55u, 40, /*from_schema=*/true);
        check(result.rejected > 0, "g4 constrains", std::to_string(result.rejected));
    }
}

// ---------------------------------------------------------------------------
// Tokenizer-backed vocabulary (the frontend construction path)
// ---------------------------------------------------------------------------

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            std::string encoded;
            const std::uint32_t code = codepoint;
            if (code < 0x80) {
                encoded.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
                encoded.push_back(static_cast<char>(0xC0 | (code >> 6)));
                encoded.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
                encoded.push_back(static_cast<char>(0xE0 | (code >> 12)));
                encoded.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                encoded.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            return encoded;
        }
    }
    throw std::logic_error("byte-level symbol outside one byte");
}

void test_tokenizer_vocabulary() {
    // A minimal byte-level BPE tokenizer fixture with two added tokens.
    const auto added = [](int id, const std::string& content, bool special) {
        return nlohmann::json{{"id", id},          {"content", content}, {"single_word", false},
                              {"lstrip", false},   {"rstrip", false},    {"normalized", false},
                              {"special", special}};
    };
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(value))] = value;
    }
    nlohmann::json added_tokens =
        nlohmann::json::array({added(300, "<|endoftext|>", true), added(301, "hello", false)});
    const std::string tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"}, {"vocab", std::move(vocab)}, {"merges", nlohmann::json::array()}}},
        {"added_tokens", added_tokens}}.dump();
    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : added_tokens) {
        nlohmann::json entry = token;
        entry.erase("id");
        decoder[std::to_string(token.at("id").get<int>())] = std::move(entry);
    }
    const std::string tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    const std::string generation_config_json = R"({"eos_token_id":[300]})";

    const auto tokenizer = std::make_shared<const frontend::Tokenizer>(frontend::TokenizerResources{
        tokenizer_json, tokenizer_config_json, generation_config_json});
    const frontend::TokenizerVocabulary vocabulary(tokenizer, std::vector<int>{302});

    check(vocabulary.token_count() == static_cast<int>(tokenizer->vocab_size()),
          "tokenizer vocabulary covers the token domain");
    for (int id = 0; id < vocabulary.token_count(); ++id) {
        if (!tokenizer->is_valid_token(id)) { continue; }
        check(vocabulary.piece(id) == tokenizer->decode_token_bytes(id, false),
              "tokenizer vocabulary piece matches the tokenizer", std::to_string(id));
    }
    check(vocabulary.is_eog(300) && vocabulary.is_eog(302) && !vocabulary.is_eog(301),
          "tokenizer vocabulary end-of-generation set");
    check(vocabulary.encode("<|endoftext|>") == std::vector<int>{300},
          "tokenizer vocabulary encodes added tokens");
    check(vocabulary.encode("hi") == std::vector<int>({104, 105}),
          "tokenizer vocabulary encodes bytes");

    // End to end: the real construction path drives a compiled grammar's row.
    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar = frontend::CompiledGrammar::compile(
        "root ::= \"hello\"", std::make_shared<frontend::TokenizerVocabulary>(tokenizer), &error);
    check(grammar != nullptr, "tokenizer vocabulary grammar compiles", error);
    if (grammar) {
        auto state     = grammar->initial_state();
        const auto row = grammar->row_for(state);
        check(row_bit(row, 301), "tokenizer vocabulary row allows the added token");
        check(row_bit(row, 104), "tokenizer vocabulary row allows a byte piece");
        check(state.accept(301), "tokenizer vocabulary state advances on the added token");
        check(state.can_end(), "tokenizer vocabulary grammar completes");
    }
}

// ---------------------------------------------------------------------------
// Cache-key canonicalization, look-alike chains and accept/mask agreement
// ---------------------------------------------------------------------------

void test_partial_key_canonicalization() {
    // A walk that feeds tokens ending in different multi-byte codepoints leaves a different
    // dead `partial_utf8.value` after every accepted token (the engine keeps the last decoded
    // codepoint there once n_remain reaches 0). With that field in the cache key, each of the
    // twenty distinct trailing codepoints mints its own key even though every state sits far
    // inside the same folded counter; canonicalized, the whole walk is one fill. The oracle
    // comparison keeps the rows exact at every step.
    {
        std::vector<std::string> pieces;
        for (int byte = 0; byte < 256; ++byte) { pieces.emplace_back(1, static_cast<char>(byte)); }
        const char* extras[] = {"д", "е", "ж", "з", "и", "й", "к", "л", "м", "н",
                                "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч"};
        for (const char* extra : extras) { pieces.emplace_back(extra); }
        const auto vocabulary =
            std::make_shared<const SyntheticVocabulary>(std::move(pieces), std::vector<int>{});
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= [^\"]{0,400}", vocabulary, &error);
        check(grammar != nullptr, "partial-key grammar compiled", error);
        if (grammar) {
            EngineOracle oracle("root ::= [^\"]{0,400}", *vocabulary);
            std::vector<int> cycle;
            for (const char* extra : extras) { cycle.push_back(piece_id(*vocabulary, extra)); }
            auto state        = grammar->initial_state();
            std::size_t steps = 0;
            while (steps < 200) {
                const frontend::GrammarMaskRow row = grammar->row_for(state);
                for (int id = 0; id < vocabulary->token_count(); ++id) {
                    check(row_bit(row, id) == oracle.token_allowed(id),
                          "partial-key row matches the oracle at step " + std::to_string(steps));
                }
                const int pick = cycle[steps % cycle.size()];
                if (!state.accept(pick)) { break; }
                check(oracle.accept(pick), "partial-key oracle advances");
                ++steps;
            }
            const std::size_t fold_limit                 = vocabulary->max_piece_bytes() + 2;
            const frontend::CompiledGrammar::Stats stats = grammar->stats();
            check(steps == 200, "partial-key walk reached the step bound", std::to_string(steps));
            check(stats.row_fills <= fold_limit + 3, "dead partial values do not mint cache keys",
                  std::to_string(stats.row_fills) + " fills for " + std::to_string(steps) +
                      " states (limit " + std::to_string(fold_limit + 3) + ")");
        }
    }

    // Tokens the mask declares illegal (empty or NUL-leading pieces) are rejected by accept
    // without moving the state, so the mask and accept cannot disagree on them; the standard
    // vocabulary's id 0 is a single NUL byte.
    {
        const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile("root ::= [^\"]+", vocabulary, &error);
        check(grammar != nullptr, "nul-piece grammar compiled", error);
        if (grammar) {
            auto state = grammar->initial_state();
            check(!row_bit(grammar->row_for(state), 0), "mask clears the NUL-leading piece");
            check(!state.accept(0), "accept rejects the NUL-leading piece");
            check(state.accept(120), "the state advances after the NUL rejection");
            check(state.can_end(), "the state is consistent after the NUL rejection");
        }
    }
}

void test_lookalike_chain_guard() {
    // The repetition normalizer accepts a candidate link only when its target is a frame with
    // an identical body. This fixture makes the guard observable: `A` and `B` are hand-written
    // frame look-alikes (their second alternative is the empty literal, which is how a source
    // rule carries the frame shape) with the same body, each linking into a repetition of a
    // different body. Without the guard both fold to the same saturated canonical position,
    // the states after "1" and after "2" share one cached row, and the oracle comparison
    // below detects it; the explicit "xy"/"xz" expectations show which row leaked.
    const char* gbnf                = "root ::= \"1\" A | \"2\" B\n"
                                      "A ::= \"x\" \"y\"{0,12} | \"\"\n"
                                      "B ::= \"x\" \"z\"{0,12} | \"\"\n";
    std::vector<std::string> pieces = {"1", "2", "x", "y", "z", "xy", "xz", "<eos>"};
    const auto vocabulary           = std::make_shared<const SyntheticVocabulary>(
        std::move(pieces), std::vector<int>{static_cast<int>(7)});
    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar =
        frontend::CompiledGrammar::compile(gbnf, vocabulary, &error);
    check(grammar != nullptr, "look-alike chain grammar compiled", error);
    if (!grammar) { return; }

    const auto walk = [&](const std::string& prefix) {
        EngineOracle oracle(gbnf, *vocabulary);
        auto state = grammar->initial_state();
        for (const int id : vocabulary->encode(prefix)) {
            const frontend::GrammarMaskRow row = grammar->row_for(state);
            for (int token = 0; token < vocabulary->token_count(); ++token) {
                check(row_bit(row, token) == oracle.token_allowed(token),
                      "look-alike chain row matches the oracle after " + prefix);
            }
            check(state.accept(id), "look-alike chain prefix advances: " + prefix);
            check(oracle.accept(id), "look-alike chain oracle advances: " + prefix);
        }
        const frontend::GrammarMaskRow row = grammar->row_for(state);
        for (int token = 0; token < vocabulary->token_count(); ++token) {
            check(row_bit(row, token) == oracle.token_allowed(token),
                  "look-alike chain final row matches the oracle after " + prefix);
        }
        // The producer's row view is transient (valid until the next row_for), so copy it:
        // the next walk overwrites the backing storage.
        return std::vector<std::uint32_t>(row.begin(), row.end());
    };
    const std::vector<std::uint32_t> row_one = walk("1");
    const std::vector<std::uint32_t> row_two = walk("2");

    const frontend::GrammarMaskRow one_span = std::span(row_one);
    const frontend::GrammarMaskRow two_span = std::span(row_two);
    const int xy                            = piece_id(*vocabulary, "xy");
    const int xz                            = piece_id(*vocabulary, "xz");
    check(row_bit(one_span, xy) && !row_bit(one_span, xz),
          "the first look-alike accepts xy and rejects xz");
    check(row_bit(two_span, xz) && !row_bit(two_span, xy),
          "the second look-alike accepts xz and rejects xy");
    check(!rows_equal(one_span, two_span), "the look-alike branches keep distinct rows");
}

// ---------------------------------------------------------------------------
// The thinking wrapper (constrained thinking design; ADR 0001)
// ---------------------------------------------------------------------------

// Builds, compiles and returns a wrapped grammar; `grammar` is null on failure.
std::optional<std::string>
wrap_and_compile(std::string_view gbnf, std::string_view close_marker,
                 const std::shared_ptr<const frontend::GrammarVocabulary>& vocabulary,
                 std::shared_ptr<const frontend::CompiledGrammar>* grammar) {
    std::string error;
    std::optional<std::string> wrapped =
        frontend::wrap_thinking_constraint_grammar(gbnf, close_marker, &error);
    check(wrapped.has_value(), "wrapper built", error);
    if (!wrapped) { return std::nullopt; }
    *grammar = frontend::CompiledGrammar::compile(*wrapped, vocabulary, &error);
    check(*grammar != nullptr, "wrapper compiled", error);
    return wrapped;
}

void test_thinking_wrapper() {
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    constexpr std::string_view kClose                            = "</think>";

    // Construction: the wrapper renames the input root and its references, takes the close
    // marker as given, and avoids every rule-name prefix that already occurs in the input.
    {
        std::string error;
        const std::string input = "root ::= \"a\"\nref ::= root\ntcw0-body ::= \"q\"\n";
        const std::optional<std::string> wrapped =
            frontend::wrap_thinking_constraint_grammar(input, kClose, &error);
        check(wrapped.has_value(), "wrapper: colliding-prefix grammar wraps", error);
        if (wrapped) {
            check(wrapped->find("tcw0-body ::= \"q\"") != std::string::npos,
                  "wrapper: the input's look-alike rule survives");
            check(wrapped->find("tcw1-answer ::= \"a\"") != std::string::npos,
                  "wrapper: the input root is renamed with a free prefix");
            check(wrapped->find("ref ::= tcw1-answer") != std::string::npos,
                  "wrapper: root references follow the rename");
            check(wrapped->find("tcw1-close ::= \"</think>\"") != std::string::npos,
                  "wrapper: the close rule is the wire marker");
        }
    }

    // Token references (`<[id]>`, `<text>`) are operands: the rename leaves their inner text
    // untouched while a bare `root` rule is still renamed.
    {
        std::string error;
        const std::string input = "root ::= <[42]> <root> <root_x>\nroots ::= \"z\"\n";
        const std::optional<std::string> wrapped =
            frontend::wrap_thinking_constraint_grammar(input, kClose, &error);
        check(wrapped.has_value(), "wrapper: token-reference grammar wraps", error);
        if (wrapped) {
            check(wrapped->find("tcw0-answer ::= <[42]> <root> <root_x>") != std::string::npos,
                  "wrapper: token references are left untouched");
            check(wrapped->find("roots ::= \"z\"") != std::string::npos,
                  "wrapper: a rule name that only starts with root is untouched");
        }
    }

    // Token references survive the wrap and compile in the answer grammar.
    {
        std::shared_ptr<const frontend::CompiledGrammar> grammar;
        (void)wrap_and_compile("root ::= <[42]>", kClose, vocabulary, &grammar);
        check(grammar != nullptr, "wrapper: a token-reference answer grammar compiles");
    }

    // Construction errors.
    {
        std::string error;
        check(!frontend::wrap_thinking_constraint_grammar("root ::= \"a\"", "", &error) &&
                  !error.empty(),
              "wrapper: an empty close marker is rejected");
        error.clear();
        check(!frontend::wrap_thinking_constraint_grammar("x ::= \"a\"", kClose, &error) &&
                  error.find("'root'") != std::string::npos,
              "wrapper: a grammar without a root declaration is rejected", error);
        error.clear();
        check(!frontend::wrap_thinking_constraint_grammar("x ::= root", kClose, &error),
              "wrapper: a referenced but undeclared root is rejected");
    }

    // Scripted walk: free-form reasoning, forced close, whitespace, then the answer grammar.
    // Every visited state's row is compared against the independent engine oracle.
    {
        std::shared_ptr<const frontend::CompiledGrammar> grammar;
        const std::optional<std::string> wrapped =
            wrap_and_compile("root ::= \"ok\"", kClose, vocabulary, &grammar);
        if (wrapped && grammar) {
            EngineOracle oracle(*wrapped, *vocabulary);
            frontend::GrammarState state = grammar->initial_state();
            const auto verify            = [&](const std::string& where) {
                const frontend::GrammarMaskRow row = grammar->row_for(state);
                for (int id = 0; id < vocabulary->token_count(); ++id) {
                    if (row_bit(row, id) != oracle.token_allowed(id)) {
                        check(false, "wrapper walk: row matches the oracle " + where);
                        return;
                    }
                }
            };
            const auto feed = [&](std::string_view text) {
                for (const int id : vocabulary->encode(text)) {
                    const bool accepted = state.accept(id);
                    check(accepted, "wrapper walk: accepts its scripted text");
                    if (accepted) {
                        check(oracle.accept(id), "wrapper walk: oracle advances with the text");
                    }
                }
            };
            verify("at the initial state");
            check(state.can_end(), "wrapper walk: EOG is admitted at the empty body");
            feed("Hello, world! <t </thi");
            verify("mid-prefix");
            check(!state.can_end(), "wrapper walk: no EOG mid-prefix");
            feed("x");
            verify("after the prefix resolves");
            check(state.can_end(), "wrapper walk: EOG returns after the prefix resolves");
            feed(kClose);
            verify("after the close");
            check(!state.can_end(), "wrapper walk: no EOG after the close");
            feed("\n\n");
            verify("after the whitespace");
            check(!state.can_end(), "wrapper walk: no EOG before a complete answer");
            feed("ok");
            verify("after the answer");
            check(state.can_end(), "wrapper walk: EOG after a complete answer");

            // A full marker anywhere in the reasoning text closes the stream: the abort path
            // cannot consume it, so no quoted `</think>` stays reasoning.
            frontend::GrammarState quoted = grammar->initial_state();
            check(feed_text(quoted, *grammar, *vocabulary, "he said </think>"),
                  "wrapper walk: a quoted marker is consumed through the close");
            check(!quoted.can_end(), "wrapper walk: the quoted marker leaves no abort path");
        }
    }

    // Corner probes with a multi-byte close piece: while the body is mid-prefix, a close token
    // that completes the marker is rejected (the model must resolve the prefix with one byte
    // first), a longer partial token is admitted, and the close then fires.
    {
        std::vector<std::string> pieces;
        for (int byte = 0; byte < 256; ++byte) {
            pieces.emplace_back(1, static_cast<char>(byte));
        }
        pieces.emplace_back("</think>");
        pieces.emplace_back("</thin");
        const int eos = static_cast<int>(pieces.size());
        pieces.emplace_back("<eos>");
        const auto corner_vocabulary = std::make_shared<const SyntheticVocabulary>(
            std::move(pieces), std::vector<int>{eos});

        std::shared_ptr<const frontend::CompiledGrammar> grammar;
        const std::optional<std::string> wrapped =
            wrap_and_compile("root ::= \"ok\"", kClose, corner_vocabulary, &grammar);
        if (wrapped && grammar) {
            const int close_piece = piece_id(*corner_vocabulary, "</think>");
            const int thin_piece  = piece_id(*corner_vocabulary, "</thin");
            frontend::GrammarState state = grammar->initial_state();
            check(feed_text(state, *grammar, *corner_vocabulary, "reasoning<"),
                  "wrapper corner: a trailing marker-start byte is body text");
            const frontend::GrammarMaskRow pending = grammar->row_for(state);
            check(!row_bit(pending, close_piece),
                  "wrapper corner: the completing close token is rejected mid-prefix");
            check(row_bit(pending, thin_piece),
                  "wrapper corner: a partial close token continues the chain mid-prefix");
            check(row_bit(pending, piece_id(*corner_vocabulary, "x")) &&
                      row_bit(pending, piece_id(*corner_vocabulary, ">")),
                  "wrapper corner: fallback bytes are admitted mid-prefix");
            check(!state.can_end(), "wrapper corner: no EOG mid-prefix");
            check(feed_text(state, *grammar, *corner_vocabulary, "x"),
                  "wrapper corner: a fallback byte resolves the prefix");
            check(state.can_end(), "wrapper corner: EOG returns after the fallback");
            check(feed_text(state, *grammar, *corner_vocabulary, kClose),
                  "wrapper corner: the close fires after the fallback");
            check(feed_text(state, *grammar, *corner_vocabulary, "\n\nok"),
                  "wrapper corner: whitespace and the answer complete");
            check(state.can_end(), "wrapper corner: EOG after the answer");
        }
    }

    // Marker shapes: single-byte, two-byte and escape-bearing markers exercise the general
    // chain and the character-class encoding.
    for (const std::string& marker : {"#", "<>", "]-", "</think>"}) {
        std::shared_ptr<const frontend::CompiledGrammar> grammar;
        const std::optional<std::string> wrapped =
            wrap_and_compile("root ::= \"ok\"", marker, vocabulary, &grammar);
        if (!wrapped || !grammar) { continue; }
        auto state = grammar->initial_state();
        check(feed_text(state, *grammar, *vocabulary, "reasoning text. "),
              "wrapper marker " + marker + ": reasoning is free-form");
        check(state.can_end(), "wrapper marker " + marker + ": EOG during reasoning");
        check(feed_text(state, *grammar, *vocabulary, marker),
              "wrapper marker " + marker + ": the close is admitted");
        check(!state.can_end(), "wrapper marker " + marker + ": no EOG before the answer");
        check(feed_text(state, *grammar, *vocabulary, "\n") &&
                  feed_text(state, *grammar, *vocabulary, "ok"),
              "wrapper marker " + marker + ": whitespace and the answer complete");
        check(state.can_end(), "wrapper marker " + marker + ": EOG after the answer");
    }

    // A converted non-object-root schema (array) wraps and answers through the same path.
    {
        const std::string schema_gbnf =
            schema_to_gbnf(R"({"type":"array","items":{"type":"integer"}})");
        std::shared_ptr<const frontend::CompiledGrammar> grammar;
        const std::optional<std::string> wrapped =
            wrap_and_compile(schema_gbnf, kClose, vocabulary, &grammar);
        if (wrapped && grammar) {
            auto state = grammar->initial_state();
            check(feed_text(state, *grammar, *vocabulary, "counting: ") &&
                      feed_text(state, *grammar, *vocabulary, kClose) &&
                      feed_text(state, *grammar, *vocabulary, "\n"),
                  "wrapper array schema: reasoning and close are admitted");
            check(feed_text(state, *grammar, *vocabulary, "[1,2]"),
                  "wrapper array schema: an array answer is admitted");
            check(state.can_end(), "wrapper array schema: EOG after the array");
        }
    }
}

// ---------------------------------------------------------------------------
// Compile and parse error propagation (the serve layer turns these into 400s)
// ---------------------------------------------------------------------------

void test_compile_errors() {
    const std::shared_ptr<const SyntheticVocabulary>& vocabulary = standard_vocabulary();
    const auto compile                                           = [&](const std::string& gbnf) {
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar =
            frontend::CompiledGrammar::compile(gbnf, vocabulary, &error);
        return std::make_pair(grammar, error);
    };

    {
        // A syntax error carries the parser's own diagnostic (patch 0002).
        const auto [grammar, error] = compile("root ::= \"unterminated");
        check(grammar == nullptr && error.find("end of input") != std::string::npos,
              "syntax error carries the parser diagnostic", error);
    }
    {
        const auto [grammar, error] = compile("root ::= expr\nexpr ::= term\nterm ::= numero");
        check(grammar == nullptr && error.find("numero") != std::string::npos,
              "undefined reference names the rule", error);
    }
    {
        const auto [grammar, error] = compile("root ::= \"a\" | root \"a\"");
        check(grammar == nullptr && !error.empty(), "left recursion is rejected", error);
    }
    {
        const auto [grammar, error] = compile("other ::= \"a\"");
        check(grammar == nullptr && error.find("root") != std::string::npos,
              "missing root symbol is reported", error);
    }
    {
        std::string error;
        std::shared_ptr<const frontend::CompiledGrammar> grammar;
        try {
            grammar = frontend::CompiledGrammar::compile(schema_to_gbnf("not a schema"), vocabulary,
                                                         &error);
        } catch (const std::exception& failure) { error = failure.what(); }
        check(grammar == nullptr && !error.empty(), "conversion failure is reported", error);
    }
}

// ---------------------------------------------------------------------------
// Structural vector (ported from llama.cpp tests/test-llama-grammar.cpp)
// ---------------------------------------------------------------------------

void test_engine_structural_vector() {
    // The vector pins the engine's rule expansion, initial stacks and per-stack rejection
    // through the vendored adapter: rules are injected directly (as upstream does), then
    // the stacks built by init and the rejects of a synthetic code-point candidate set are
    // compared against upstream's recorded expectations.
    llama_grammar_parser parsed;
    const std::vector<std::pair<std::string, uint32_t>> expected_symbols = {
        {"expr", 2}, {"expr_6", 6},  {"expr_7", 7}, {"ident", 8},  {"ident_10", 10},
        {"num", 9},  {"num_11", 11}, {"root", 0},   {"root_1", 1}, {"root_5", 5},
        {"term", 4}, {"ws", 3},      {"ws_12", 12},
    };
    const std::vector<std::vector<llama_grammar_element>> expected_rules = {
        {{LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_CHAR, 10},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 4}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_END, 0}},
        {{LLAMA_GRETYPE_RULE_REF, 12}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 40},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 41},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 1},
         {LLAMA_GRETYPE_RULE_REF, 5},
         {LLAMA_GRETYPE_ALT, 0},
         {LLAMA_GRETYPE_RULE_REF, 1},
         {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 45},
            {LLAMA_GRETYPE_CHAR_ALT, 43},
            {LLAMA_GRETYPE_CHAR_ALT, 42},
            {LLAMA_GRETYPE_CHAR_ALT, 47},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 6},
         {LLAMA_GRETYPE_RULE_REF, 7},
         {LLAMA_GRETYPE_ALT, 0},
         {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 11}, {LLAMA_GRETYPE_RULE_REF, 3}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_CHAR_ALT, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_CHAR_ALT, 95},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_RULE_REF, 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 32},
            {LLAMA_GRETYPE_CHAR_ALT, 9},
            {LLAMA_GRETYPE_CHAR_ALT, 10},
            {LLAMA_GRETYPE_RULE_REF, 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
    };
    for (const auto& [name, id] : expected_symbols) { parsed.symbol_ids[name] = id; }
    for (const auto& rule : expected_rules) { parsed.rules.push_back(rule); }

    std::vector<const llama_grammar_element*> grammar_rules = parsed.c_rules();
    llama_grammar* grammar                                  = llama_grammar_init_impl(
        nullptr, grammar_rules.data(), grammar_rules.size(), parsed.symbol_ids.at("root"));
    check(grammar != nullptr, "structural vector: grammar initialized");
    if (grammar == nullptr) { return; }

    const std::vector<std::vector<llama_grammar_element>> expected_stacks = {
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        }};
    const llama_grammar_stacks& stacks = llama_grammar_get_stacks(grammar);
    check(stacks.size() == expected_stacks.size(), "structural vector: stack count",
          std::to_string(stacks.size()) + " vs " + std::to_string(expected_stacks.size()));
    for (std::size_t index = 0; index < stacks.size() && index < expected_stacks.size(); ++index) {
        const llama_grammar_stack& stack                   = stacks[index];
        const std::vector<llama_grammar_element>& expected = expected_stacks[index];
        bool ok                                            = stack.size() == expected.size();
        if (ok) {
            for (std::size_t element = 0; element < stack.size(); ++element) {
                ok = ok && stack[element]->type == expected[element].type &&
                     stack[element]->value == expected[element].value;
            }
        }
        check(ok, "structural vector: stack " + std::to_string(index));
    }

    std::vector<std::vector<uint32_t>> points(23);
    std::vector<llama_grammar_candidate> candidates(23);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        points[index]     = {static_cast<uint32_t>(37 + index), 0U};
        candidates[index] = {index, points[index].data(), {}, 0};
    }
    const std::vector<std::vector<std::pair<uint32_t, uint16_t>>> expected_rejects = {
        {
            {0, 37},  {1, 38},  {2, 39},  {4, 41},  {5, 42},  {6, 43},  {7, 44},  {8, 45},
            {9, 46},  {10, 47}, {11, 48}, {12, 49}, {13, 50}, {14, 51}, {15, 52}, {16, 53},
            {17, 54}, {18, 55}, {19, 56}, {20, 57}, {21, 58}, {22, 59}, {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},  {1, 38},  {2, 39},  {3, 40},  {4, 41},  {5, 42},  {6, 43},  {7, 44},
            {8, 45},  {9, 46},  {10, 47}, {11, 48}, {12, 49}, {13, 50}, {14, 51}, {15, 52},
            {16, 53}, {17, 54}, {18, 55}, {19, 56}, {20, 57}, {21, 58}, {22, 59},
        },
        {
            {0, 37},  {1, 38},  {2, 39},  {4, 41},  {5, 42},  {6, 43},  {7, 44},  {8, 45},
            {9, 46},  {10, 47}, {11, 48}, {12, 49}, {13, 50}, {14, 51}, {15, 52}, {16, 53},
            {17, 54}, {18, 55}, {19, 56}, {20, 57}, {21, 58}, {22, 59}, {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},  {1, 38},  {2, 39},  {3, 40},  {4, 41},  {5, 42},  {6, 43},  {7, 44},
            {8, 45},  {9, 46},  {10, 47}, {11, 48}, {12, 49}, {13, 50}, {14, 51}, {15, 52},
            {16, 53}, {17, 54}, {18, 55}, {19, 56}, {20, 57}, {21, 58}, {22, 59},
        },
    };
    for (std::size_t index = 0; index < stacks.size() && index < expected_rejects.size(); ++index) {
        const llama_grammar_candidates rejects = llama_grammar_reject_candidates_for_stack(
            llama_grammar_get_rules(grammar), stacks[index], candidates);
        // Upstream's loop iterates over the actual rejects and indexes the expected list,
        // so its recorded rows may carry trailing entries the engine never returns.
        const std::vector<std::pair<uint32_t, uint16_t>>& expected = expected_rejects[index];
        bool ok = rejects.size() <= expected.size();
        if (ok) {
            for (std::size_t entry = 0; entry < rejects.size(); ++entry) {
                ok = ok && rejects[entry].index == expected[entry].first &&
                     *rejects[entry].code_points == expected[entry].second;
            }
        }
        check(ok, "structural vector: rejects for stack " + std::to_string(index),
              std::to_string(rejects.size()) + " vs " + std::to_string(expected.size()));
    }
    llama_grammar_free_impl(grammar);
}


// The per-request constrained-generation runtime: the round advance, the reachable prefix and the
// atomic commit.
void test_grammar_runtime() {
    const std::shared_ptr<const frontend::CompiledGrammar> grammar =
        frontend::CompiledGrammar::compile("root ::= \"ab\" (\"c\" | \"d\")",
                                           standard_vocabulary(), nullptr);
    check(grammar != nullptr, "grammar runtime fixture did not compile");
    if (grammar == nullptr) { return; }
    const frontend::GrammarVocabulary& vocab = *standard_vocabulary();
    const int ab                             = vocab.encode("ab").front();
    const int c                              = vocab.encode("c").front();
    const int z                              = vocab.encode("z").front();
    const std::size_t words = (static_cast<std::size_t>(grammar->token_count()) + 31U) / 32U;
    const auto all_ones     = [](std::span<const std::uint32_t> row) {
        return std::all_of(row.begin(), row.end(),
                           [](std::uint32_t word) { return word == 0xFFFFFFFFU; });
    };
    const auto same = [](std::span<const std::uint32_t> left,
                         std::span<const std::uint32_t> right) {
        return std::equal(left.begin(), left.end(), right.begin());
    };
    const auto column = [words](const std::vector<std::uint32_t>& rows, std::size_t index) {
        return std::span<const std::uint32_t>(rows).subspan(index * words, words);
    };

    std::vector<std::uint32_t> initial(words, 0);
    std::vector<std::uint32_t> rows(3 * words, 0);
    frontend::GrammarRuntime runtime(grammar);
    {
        const std::array<int, 1> tokens{frontend::kUnknownConstraintToken};
        runtime.fill_round_rows(tokens, rows, words);
        std::copy(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(words), initial.begin());
        check(render_row(column(rows, 0), vocab) ==
                  render_row(grammar->row_for(grammar->initial_state()), vocab),
              "the committed-state row is not the fixture grammar's initial row",
              render_row(column(rows, 0), vocab));
    }

    // Masked columns advance the preview state: after "ab" only "c"/"d" remain.
    const std::array<int, 2> answer_tokens{ab, frontend::kUnknownConstraintToken};
    runtime.fill_round_rows(answer_tokens, rows, words);
    check(same(column(rows, 0), initial), "the first masked column did not use the initial row");
    frontend::GrammarState after_ab = grammar->initial_state();
    check(after_ab.accept(ab), "the fixture's \"ab\" token did not advance a fresh state");
    check(render_row(column(rows, 1), vocab) == render_row(grammar->row_for(after_ab), vocab),
          "the second masked column did not advance by the first answer token",
          render_row(column(rows, 1), vocab));

    // An ungrammatical draft ends the reachable prefix; later masked columns keep all-ones.
    const std::array<int, 2> rejected_tokens{z, frontend::kUnknownConstraintToken};
    runtime.fill_round_rows(rejected_tokens, rows, words);
    check(all_ones(column(rows, 1)), "a rejected draft did not end the reachable prefix");

    // An unknown column token cannot advance the preview either: later masked columns keep
    // all-ones rows while the column itself still uses the committed state's row.
    const std::array<int, 2> unknown_tokens{frontend::kUnknownConstraintToken, ab};
    runtime.fill_round_rows(unknown_tokens, rows, words);
    check(same(column(rows, 0), initial),
          "a masked column with an unknown token did not use the committed row");
    check(all_ones(column(rows, 1)), "an unknown column token did not end the reachable prefix");

    // Words past the grammar's row width are padding and must stay all-ones.
    const std::size_t wide_words = words + 2;
    std::vector<std::uint32_t> wide_rows(wide_words, 0);
    const std::array<int, 1> wide_tokens{frontend::kUnknownConstraintToken};
    runtime.fill_round_rows(wide_tokens, wide_rows, wide_words);
    bool wide_padding_ok = true;
    for (std::size_t word = words; word < wide_words; ++word) {
        if (wide_rows[word] != 0xFFFFFFFFU) { wide_padding_ok = false; }
    }
    check(wide_padding_ok, "mask row padding words were not all-ones");

    // Commit is atomic: a later rejection leaves the committed state exactly where it was.
    {
        frontend::GrammarRuntime atomic_runtime(grammar);
        const std::array<int, 2> partial_tokens{ab, z};
        check(!atomic_runtime.commit(partial_tokens),
              "a partially invalid answer round was committed");
        std::vector<std::uint32_t> probe_rows(words, 0);
        const std::array<int, 1> probe_tokens{frontend::kUnknownConstraintToken};
        atomic_runtime.fill_round_rows(probe_tokens, probe_rows, words);
        check(same(std::span<const std::uint32_t>(probe_rows), initial),
              "a failed commit moved the committed state");
    }

    // Commit is atomic; a token after the grammar ended is refused.
    const std::array<int, 2> commit_tokens{ab, c};
    check(runtime.commit(commit_tokens), "committing the answer tokens failed");
    check(runtime.can_end(), "the grammar could not end after the complete answer");
    const std::array<int, 1> late_tokens{ab};
    check(!runtime.commit(late_tokens), "a token after the grammar ended was committed");
    check(runtime.can_end(), "a failed commit moved the committed state");
    check(!runtime.commit(std::array<int, 1>{frontend::kUnknownConstraintToken}),
          "an answer column without a token was committed");
}

} // namespace

int main() {
    test_partial_key_canonicalization();
    test_lookalike_chain_guard();
    test_compile_errors();
    test_engine_structural_vector();
    test_language_cases();
    test_json_schema_cases();
    test_repetition_bound();
    test_differential_oracle();
    test_row_cache();
    test_diary_schema();
    test_tokenizer_vocabulary();
    test_grammar_runtime();
    test_thinking_wrapper();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d grammar test failure(s)\n", g_failures);
        return 1;
    }
    std::printf("grammar tests passed\n");
    return 0;
}
