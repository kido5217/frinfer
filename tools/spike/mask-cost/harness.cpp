// Mask-cost spike harness.
//
// Times llama.cpp's GBNF runtime (upstream/llama-grammar.{h,cpp}, byte-identical
// to llama.cpp at baseline 05af0d2b1398394cfa67e1918fee7feabccaa9bc) over the real
// Qwen3.8-27B vocabulary dump (see dump_vocab.py):
//
//   - llama_grammar_init_impl       compile (once-per-request cost)
//   - llama_grammar_apply_impl      full-vocab candidate filter ("mask fill")
//   - the one-element validate check common_sampler_sample uses
//   - llama_grammar_accept_token    per-token state advance
//
// from fresh / mid-sequence / near-completion states of four fixture grammars.
// Single-threaded; median/min/max over --iters iterations; logits are reset
// outside the timed region by memcpy-ing a pristine candidate array (full fill)
// or reinitializing the one-element array (validate).
//
// Build: tools/spike/mask-cost/build.sh <build-dir>
// Usage: mask-cost-harness --vocab <vocab.tsv> --grammars <dir> [--eog a,b]
//                          [--iters N] [--json out.json]

#include "llama-grammar.h"
#include "llama-vocab.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double now_us() {
    return std::chrono::duration<double, std::micro>(clock_type::now().time_since_epoch()).count();
}

struct Stats {
    size_t n      = 0;
    double median = 0.0;
    double min    = 0.0;
    double max    = 0.0;
    double mean   = 0.0;
};

Stats summarize(std::vector<double> values) {
    Stats stats;
    stats.n = values.size();
    if (values.empty()) { return stats; }
    std::sort(values.begin(), values.end());
    stats.min    = values.front();
    stats.max    = values.back();
    stats.median = values.size() % 2 == 1
                       ? values[values.size() / 2]
                       : 0.5 * (values[values.size() / 2 - 1] + values[values.size() / 2]);
    double sum   = 0.0;
    for (const double value : values) { sum += value; }
    stats.mean = sum / static_cast<double>(values.size());
    return stats;
}

std::string hex_encode(const std::string& bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const unsigned char byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0xF]);
    }
    return out;
}

std::string hex_decode(const std::string& hex) {
    auto value = [](char c) -> int {
        if ('0' <= c && c <= '9') { return c - '0'; }
        if ('a' <= c && c <= 'f') { return c - 'a' + 10; }
        if ('A' <= c && c <= 'F') { return c - 'A' + 10; }
        throw std::runtime_error(std::string("bad hex character: ") + c);
    };
    if (hex.size() % 2 != 0) { throw std::runtime_error("odd-length hex string"); }
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<char>((value(hex[i]) << 4) | value(hex[i + 1])));
    }
    return out;
}

// Printable rendering for report readability; non-printables as \xNN.
std::string printable(const std::string& bytes) {
    std::string out;
    for (const unsigned char byte : bytes) {
        if (byte >= 0x20 && byte < 0x7F) {
            out.push_back(static_cast<char>(byte));
        } else {
            char buffer[8];
            std::snprintf(buffer, sizeof(buffer), "\\x%02x", byte);
            out += buffer;
        }
    }
    return out;
}

size_t utf8_seq_len(unsigned char lead) {
    if (lead < 0x80) { return 1; }
    if ((lead & 0xE0) == 0xC0) { return 2; }
    if ((lead & 0xF0) == 0xE0) { return 3; }
    if ((lead & 0xF8) == 0xF0) { return 4; }
    return 1;
}

std::vector<std::string> split_ids(const std::string& text) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const size_t end   = comma == std::string::npos ? text.size() : comma;
        if (end > start) { parts.push_back(text.substr(start, end - start)); }
        if (comma == std::string::npos) { break; }
        start = comma + 1;
    }
    return parts;
}

// ---------------------------------------------------------------- vocab dump

std::vector<std::string> load_dump(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open vocab dump " + path); }
    std::vector<std::pair<size_t, std::string>> rows;
    size_t max_id = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) { throw std::runtime_error("malformed dump line: " + line); }
        const size_t id = static_cast<size_t>(std::stoul(line.substr(0, tab)));
        rows.emplace_back(id, hex_decode(line.substr(tab + 1)));
        max_id = std::max(max_id, id);
    }
    std::vector<std::string> pieces(max_id + 1);
    for (auto& [id, bytes] : rows) { pieces[id] = std::move(bytes); }
    return pieces;
}

struct PieceIndex {
    std::unordered_map<std::string, int> piece_to_id;
    size_t max_piece_len = 0;
};

PieceIndex build_piece_index(const std::vector<std::string>& pieces) {
    PieceIndex index;
    for (size_t id = 0; id < pieces.size(); ++id) {
        const std::string& piece = pieces[id];
        if (piece.empty()) { continue; }
        index.piece_to_id.emplace(piece, static_cast<int>(id)); // lowest id wins
        index.max_piece_len = std::max(index.max_piece_len, piece.size());
    }
    return index;
}

struct TokenRef {
    int id = -1; // -1: no vocab piece matches; fed through accept_str
    std::string piece;
};

// Greedy longest-match spelling of `text` in vocab pieces.
std::vector<TokenRef> tokenize_text(const PieceIndex& index, const std::string& text) {
    std::vector<TokenRef> out;
    size_t pos = 0;
    while (pos < text.size()) {
        int best_id        = -1;
        size_t best_len    = 0;
        const size_t limit = std::min(index.max_piece_len, text.size() - pos);
        for (size_t len = limit; len >= 1; --len) {
            const auto it = index.piece_to_id.find(text.substr(pos, len));
            if (it != index.piece_to_id.end()) {
                best_id  = it->second;
                best_len = len;
                break;
            }
        }
        if (best_id < 0) {
            best_len =
                std::min(utf8_seq_len(static_cast<unsigned char>(text[pos])), text.size() - pos);
        }
        out.push_back(TokenRef{best_id, text.substr(pos, best_len)});
        pos += best_len;
    }
    return out;
}

void feed_text(llama_grammar& grammar, const PieceIndex& index, const std::string& text) {
    for (const TokenRef& token : tokenize_text(index, text)) {
        if (token.id >= 0) {
            llama_grammar_accept_token(grammar, token.id, token.piece);
        } else {
            llama_grammar_accept_str(grammar, token.piece);
        }
    }
}

// ---------------------------------------------------------------- timing

template <typename Setup, typename Timed>
std::vector<double> bench_raw(size_t iters, size_t warmup, Setup setup, Timed timed) {
    for (size_t i = 0; i < warmup; ++i) {
        setup();
        timed();
    }
    std::vector<double> times;
    times.reserve(iters);
    for (size_t i = 0; i < iters; ++i) {
        setup();
        const double t0 = now_us();
        timed();
        const double t1 = now_us();
        times.push_back(t1 - t0);
    }
    return times;
}

std::string describe_stacks(llama_grammar& grammar) {
    std::string out;
    for (const auto& stack : llama_grammar_get_stacks(&grammar)) {
        if (!out.empty()) { out += ","; }
        out += std::to_string(stack.size());
    }
    return out;
}

bool has_empty_stack(llama_grammar& grammar) {
    for (const auto& stack : llama_grammar_get_stacks(&grammar)) {
        if (stack.empty()) { return true; }
    }
    return false;
}

// One-element apply, exactly common_sampler_sample's fast validate path.
bool token_allowed(llama_grammar& grammar, int id) {
    llama_token_data td{id, 1.0f, 0.0f};
    llama_token_data_array array{&td, 1, -1, false};
    llama_grammar_apply_impl(grammar, &array);
    return td.logit != -INFINITY;
}

// ---------------------------------------------------------------------------
// Auxiliary attribution: instrumented decomposition of llama_grammar_apply_impl
// (upstream/llama-grammar.cpp, baseline 05af0d2b, lines 1359-1400) into its two
// stages, so the full-fill cost can be split between them:
//   decode stage  = per-candidate piece decode + candidate array construction
//   reject stage  = llama_grammar_reject_candidates (walk of every stack against
//                   every candidate)
// The decode copies below mirror upstream decode_utf8 line for line; the reject
// composition mirrors the static llama_grammar_reject_candidates using the
// engine's exported llama_grammar_reject_candidates_for_stack.
// ---------------------------------------------------------------------------

static std::pair<uint32_t, const char*> decode_utf8_one(const char* src) {
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t first_byte        = static_cast<uint8_t>(*src);
    uint8_t highbits          = first_byte >> 4;
    int len                   = lookup[highbits];
    uint8_t mask              = (1 << (8 - len)) - 1;
    uint32_t value            = first_byte & mask;
    const char* end           = src + len; // may overrun!
    const char* pos           = src + 1;
    for (; pos < end && *pos; pos++) { value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F); }
    return std::make_pair(value, pos);
}

static std::pair<std::vector<uint32_t>, llama_partial_utf8>
decode_utf8_piece(const std::string& src, llama_partial_utf8 partial_start) {
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
    const char* pos           = src.c_str();
    std::vector<uint32_t> code_points;

    code_points.reserve(src.size() + 1);
    uint32_t value = partial_start.value;
    int n_remain   = partial_start.n_remain;

    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{0, -1});
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) { code_points.push_back(value); }

    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint8_t highbits   = first_byte >> 4;
        n_remain           = lookup[highbits] - 1;

        if (n_remain < 0) {
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{0, n_remain});
        }

        uint8_t mask = (1 << (7 - n_remain)) - 1;
        value        = first_byte & mask;

        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) { code_points.push_back(value); }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_partial_utf8{value, n_remain});
}

struct DecodeStage {
    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> decoded;
    std::vector<llama_grammar_candidate> candidates;
};

// The apply loop up to (excluding) the reject call.
void apply_decode_stage(const llama_grammar& grammar, llama_token_data_array* cur_p,
                        DecodeStage* stage) {
    bool allow_eog = false;
    for (const auto& stack : grammar.stacks) {
        if (stack.empty()) {
            allow_eog = true;
            break;
        }
    }

    stage->decoded.clear();
    stage->candidates.clear();
    stage->decoded.reserve(cur_p->size);
    stage->candidates.reserve(cur_p->size);

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id     = cur_p->data[i].id;
        const std::string& piece = grammar.vocab->token_to_piece(id);

        if (grammar.vocab->is_eog(id)) {
            if (!allow_eog) { cur_p->data[i].logit = -INFINITY; }
        } else if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            stage->decoded.push_back(decode_utf8_piece(piece, grammar.partial_utf8));
            stage->candidates.push_back(
                {i, stage->decoded.back().first.data(), stage->decoded.back().second, id});
        }
    }
}

// The reject call: same composition as the static llama_grammar_reject_candidates.
llama_grammar_candidates reject_stage(const llama_grammar& grammar,
                                      const llama_grammar_candidates& candidates) {
    if (candidates.empty()) { return {}; }

    auto rejects = llama_grammar_reject_candidates_for_stack(grammar.rules, grammar.stacks.front(),
                                                             candidates);
    for (size_t i = 1, size = grammar.stacks.size(); i < size; ++i) {
        rejects =
            llama_grammar_reject_candidates_for_stack(grammar.rules, grammar.stacks[i], rejects);
    }
    return rejects;
}

// ---------------------------------------------------------------- report

struct StateReport {
    std::string name;
    std::string prefix;
    size_t stacks         = 0;
    bool allow_eog        = false;
    size_t allowed_tokens = 0;
    Stats fill;
    Stats decode_only; // auxiliary attribution (instrumented decode stage)
    Stats reject_only; // auxiliary attribution (engine reject composition)
    // validate fast path
    int good_id = -1;
    std::string good_piece;
    bool good_finite = false;
    Stats validate_good;
    int bad_id = -1;
    std::string bad_piece;
    Stats validate_bad;
};

struct AcceptReport {
    std::string state;
    std::vector<TokenRef> tokens;
    std::vector<Stats> per_token;
};

struct GrammarReport {
    std::string name;
    std::string desc;
    std::string file;
    size_t grammar_bytes = 0;
    size_t rules         = 0;
    Stats init;
    std::vector<StateReport> states;
    std::vector<AcceptReport> accepts;
};

struct Case {
    std::string name;
    std::string desc;
    std::string file;
    std::string mid;  // text fed to reach the mid-sequence state
    std::string near; // text fed to reach the near-completion state
    std::string cont_fresh;
    std::string cont_mid;
    std::string cont_near;
};

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (const char c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                out += buffer;
            } else {
                out.push_back(c);
            }
        }
    }
    return out;
}

void json_stats(std::ostream& out, const Stats& stats) {
    out << "{\"n\": " << stats.n << ", \"median\": " << std::setprecision(6) << stats.median
        << ", \"min\": " << stats.min << ", \"max\": " << stats.max << ", \"mean\": " << stats.mean
        << "}";
}

void write_json(const std::string& path, size_t vocab_size, size_t mask_row_bytes,
                const std::vector<GrammarReport>& reports, size_t iters, const Stats& mask_memset,
                const Stats& mask_memcpy) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"mask_row_bytes\": " << mask_row_bytes << ",\n";
    out << "  \"iters\": " << iters << ",\n";
    out << "  \"mask_row_memset_us\": ";
    json_stats(out, mask_memset);
    out << ",\n";
    out << "  \"mask_row_memcpy_us\": ";
    json_stats(out, mask_memcpy);
    out << ",\n";
    out << "  \"grammars\": [\n";
    for (size_t g = 0; g < reports.size(); ++g) {
        const GrammarReport& report = reports[g];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(report.name) << "\",\n";
        out << "      \"desc\": \"" << json_escape(report.desc) << "\",\n";
        out << "      \"file\": \"" << json_escape(report.file) << "\",\n";
        out << "      \"grammar_bytes\": " << report.grammar_bytes << ",\n";
        out << "      \"rules\": " << report.rules << ",\n";
        out << "      \"init_us\": ";
        json_stats(out, report.init);
        out << ",\n";
        out << "      \"states\": [\n";
        for (size_t s = 0; s < report.states.size(); ++s) {
            const StateReport& state = report.states[s];
            out << "        {\n";
            out << "          \"name\": \"" << json_escape(state.name) << "\",\n";
            out << "          \"prefix_printable\": \"" << json_escape(printable(state.prefix))
                << "\",\n";
            out << "          \"prefix_hex\": \"" << hex_encode(state.prefix) << "\",\n";
            out << "          \"stacks\": " << state.stacks << ",\n";
            out << "          \"allow_eog\": " << (state.allow_eog ? "true" : "false") << ",\n";
            out << "          \"allowed_tokens\": " << state.allowed_tokens << ",\n";
            out << "          \"fill_us\": ";
            json_stats(out, state.fill);
            out << ",\n";
            out << "          \"decode_only_us\": ";
            json_stats(out, state.decode_only);
            out << ",\n";
            out << "          \"reject_only_us\": ";
            json_stats(out, state.reject_only);
            out << ",\n";
            out << "          \"validate_good\": {\"id\": " << state.good_id
                << ", \"piece_hex\": \"" << hex_encode(state.good_piece)
                << "\", \"accepted\": " << (state.good_finite ? "true" : "false") << ", \"us\": ";
            json_stats(out, state.validate_good);
            out << "},\n";
            out << "          \"validate_bad\": {\"id\": " << state.bad_id << ", \"piece_hex\": \""
                << hex_encode(state.bad_piece) << "\", \"us\": ";
            json_stats(out, state.validate_bad);
            out << "}\n";
            out << "        }" << (s + 1 < report.states.size() ? "," : "") << "\n";
        }
        out << "      ],\n";
        out << "      \"accept\": [\n";
        for (size_t a = 0; a < report.accepts.size(); ++a) {
            const AcceptReport& accept = report.accepts[a];
            out << "        {\n";
            out << "          \"state\": \"" << json_escape(accept.state) << "\",\n";
            out << "          \"tokens\": [\n";
            for (size_t t = 0; t < accept.tokens.size(); ++t) {
                out << "            {\"id\": " << accept.tokens[t].id << ", \"piece_hex\": \""
                    << hex_encode(accept.tokens[t].piece)
                    << "\", \"bytes\": " << accept.tokens[t].piece.size() << ", \"us\": ";
                json_stats(out, accept.per_token[t]);
                out << "}" << (t + 1 < accept.tokens.size() ? "," : "") << "\n";
            }
            out << "          ]\n";
            out << "        }" << (a + 1 < report.accepts.size() ? "," : "") << "\n";
        }
        out << "      ]\n";
        out << "    }" << (g + 1 < reports.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void print_stats_line(const std::string& label, const Stats& stats) {
    std::printf("  %-22s n=%zu median=%9.3f us  min=%9.3f  max=%9.3f  mean=%9.3f\n", label.c_str(),
                stats.n, stats.median, stats.min, stats.max, stats.mean);
}

} // namespace

int main(int argc, char** argv) {
    std::string vocab_path;
    std::string grammars_dir;
    std::string json_path;
    std::string eog_arg = "248044,248046";
    size_t iters        = 200;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next             = [&](const char* name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string("missing value for ") + name);
                }
                return argv[++i];
            };
            if (arg == "--vocab") {
                vocab_path = next("--vocab");
            } else if (arg == "--grammars") {
                grammars_dir = next("--grammars");
            } else if (arg == "--eog") {
                eog_arg = next("--eog");
            } else if (arg == "--iters") {
                iters = std::stoul(next("--iters"));
            } else if (arg == "--json") {
                json_path = next("--json");
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        if (vocab_path.empty() || grammars_dir.empty()) {
            std::fprintf(stderr,
                         "usage: %s --vocab <vocab.tsv> --grammars <dir> [--eog a,b] "
                         "[--iters N] [--json out.json]\n",
                         argv[0]);
            return 2;
        }

        const std::vector<std::string> pieces = load_dump(vocab_path);
        const size_t vocab_size               = pieces.size();

        llama_vocab vocab;
        vocab.pieces = pieces;
        vocab.eog.assign(vocab_size, false);
        for (const std::string& id : split_ids(eog_arg)) {
            vocab.eog.at(static_cast<size_t>(std::stoul(id))) = true;
        }

        const PieceIndex piece_index = build_piece_index(vocab.pieces);

        std::string grammars_base = grammars_dir;
        while (!grammars_base.empty() && grammars_base.back() == '/') { grammars_base.pop_back(); }

        const std::vector<Case> cases = {
            {"G1", "literal alternation", grammars_base + "/g1_literal_alternation.gbnf",
             /*mid=*/"н", /*near=*/"не", /*cont_fresh=*/"нет", /*cont_mid=*/"ет",
             /*cont_near=*/"т"},
            {"G2", "bounded line [^\\n]{3,240}", grammars_base + "/g2_bounded_line.gbnf",
             /*mid=*/std::string(120, 'x'), /*near=*/std::string(238, 'x'),
             /*cont_fresh=*/"Hello world! The quick brown fox.",
             /*cont_mid=*/"Hello world! The quick brown fox.",
             /*cont_near=*/"xy"},
            {"G3", "key=value record", grammars_base + "/g3_key_value_record.gbnf",
             /*mid=*/"act=base;base=forward;base_amount=0.5;",
             /*near=*/"act=view;say=All clear;tone=calm",
             /*cont_fresh=*/"act=base;say=Ready;tone=calm", /*cont_mid=*/"say=Ready;tone=calm",
             /*cont_near=*/" for now"},
            {"G4", "diary JSON schema (converted)", grammars_base + "/g4_diary_schema.gbnf",
             /*mid=*/"{\"title\": \"Day one\", \"text\": \"The park",
             /*near=*/
             "{\"title\": \"Day one\", \"text\": \"ok\", \"facts\": [\"a\"], "
             "\"people\": [{\"name\": \"Ann\", \"when\": \"9am\"}]",
             /*cont_fresh=*/"{\"title\": \"T\", \"text\": \"x\", \"facts\": [], \"people\": []}",
             /*cont_mid=*/" was quiet\"", /*cont_near=*/"}"},
        };

        const size_t mask_row_bytes = ((vocab_size + 31) / 32) * 4;

        std::printf("mask-cost spike harness\n");
        std::printf("vocab ids:        %zu\n", vocab_size);
        std::printf("mask row:         ceil(%zu/32)*4 = %zu bytes\n", vocab_size, mask_row_bytes);
        std::printf("eog ids:          %s\n", eog_arg.c_str());
        std::printf("iters:            %zu\n\n", iters);

        // Auxiliary floor: steady-state cost of an xgrammar-style cached mask row
        // (write / copy one ceil(N/32)*4-byte row) on this host.
        Stats mask_memset;
        Stats mask_memcpy;
        {
            std::vector<uint32_t> row_a((vocab_size + 31) / 32, 0);
            std::vector<uint32_t> row_b((vocab_size + 31) / 32, 0);
            mask_memset = summarize(bench_raw(
                iters * 5, 50, [&]() {}, [&]() { std::memset(row_a.data(), 0, mask_row_bytes); }));
            mask_memcpy = summarize(bench_raw(
                iters * 5, 50, [&]() {},
                [&]() { std::memcpy(row_b.data(), row_a.data(), mask_row_bytes); }));
        }
        print_stats_line("mask row memset", mask_memset);
        print_stats_line("mask row memcpy", mask_memcpy);
        std::printf("\n");

        std::vector<GrammarReport> reports;
        reports.reserve(cases.size());

        for (const Case& test : cases) {
            const std::string source = read_file(test.file);
            GrammarReport report;
            report.name          = test.name;
            report.desc          = test.desc;
            report.file          = test.file.substr(grammars_base.size() + 1);
            report.grammar_bytes = source.size();

            std::printf("== %s: %s (%s) ==\n", test.name.c_str(), test.desc.c_str(),
                        report.file.c_str());

            // Compile once to validate and to serve as the fresh state.
            llama_grammar* fresh = llama_grammar_init_impl(&vocab, source.c_str(), "root",
                                                           /*lazy=*/false, nullptr, 0, nullptr, 0);
            if (fresh == nullptr) {
                throw std::runtime_error("grammar failed to compile: " + test.file);
            }
            report.rules = llama_grammar_get_rules(fresh).size();

            // Init timing: compile + free, free outside the timed region.
            {
                llama_grammar* init_grammar = nullptr;
                auto times                  = bench_raw(
                    iters, 10,
                    [&]() {
                        llama_grammar_free_impl(init_grammar);
                        init_grammar = nullptr;
                    },
                    [&]() {
                        init_grammar = llama_grammar_init_impl(&vocab, source.c_str(), "root",
                                                                                false, nullptr, 0, nullptr, 0);
                    });
                llama_grammar_free_impl(init_grammar);
                report.init = summarize(times);
            }
            print_stats_line("init (compile+free)", report.init);

            // Build states: fresh = as compiled; mid/near = fresh + fed text.
            llama_grammar* state_grammars[3] = {fresh, nullptr, nullptr};
            const std::string state_names[3] = {"fresh", "mid", "near"};
            const std::string state_texts[3] = {"", test.mid, test.near};
            for (int s = 1; s < 3; ++s) {
                state_grammars[s] = llama_grammar_clone_impl(*fresh);
                feed_text(*state_grammars[s], piece_index, state_texts[s]);
            }

            // Full-vocab candidate array, reset by memcpy outside the timed region.
            std::vector<llama_token_data> pristine(vocab_size);
            std::vector<llama_token_data> work(vocab_size);
            for (size_t id = 0; id < vocab_size; ++id) {
                pristine[id] = llama_token_data{static_cast<llama_token>(id), 0.0f, 0.0f};
            }
            llama_token_data_array array{nullptr, vocab_size, -1, false};

            for (int s = 0; s < 3; ++s) {
                llama_grammar& grammar = *state_grammars[s];
                StateReport state;
                state.name   = state_names[s];
                state.prefix = state_texts[s];

                auto stacks_ref = llama_grammar_get_stacks(&grammar);
                state.stacks    = stacks_ref.size();
                state.allow_eog = has_empty_stack(grammar);

                // Full mask fill.
                auto times = bench_raw(
                    iters, 5,
                    [&]() {
                        std::memcpy(work.data(), pristine.data(),
                                    vocab_size * sizeof(llama_token_data));
                        array = llama_token_data_array{work.data(), vocab_size, -1, false};
                    },
                    [&]() { llama_grammar_apply_impl(grammar, &array); });
                state.fill = summarize(times);
                for (const llama_token_data& candidate : work) {
                    if (candidate.logit != -INFINITY) { ++state.allowed_tokens; }
                }

                // Auxiliary attribution: decode stage and reject stage timed apart.
                {
                    DecodeStage stage;
                    auto decode_times = bench_raw(
                        iters, 5,
                        [&]() {
                            std::memcpy(work.data(), pristine.data(),
                                        vocab_size * sizeof(llama_token_data));
                            array = llama_token_data_array{work.data(), vocab_size, -1, false};
                        },
                        [&]() { apply_decode_stage(grammar, &array, &stage); });
                    state.decode_only = summarize(decode_times);

                    // Prebuild one candidate list for the reject-only timing.
                    std::memcpy(work.data(), pristine.data(),
                                vocab_size * sizeof(llama_token_data));
                    array = llama_token_data_array{work.data(), vocab_size, -1, false};
                    apply_decode_stage(grammar, &array, &stage);
                    DecodeStage reject_source = stage; // deep copy, untimed
                    // Rebinding: candidate code_points must point into the copy's
                    // decoded buffers, not the original's (1:1 push order).
                    for (size_t c = 0; c < reject_source.candidates.size(); ++c) {
                        reject_source.candidates[c].code_points =
                            reject_source.decoded[c].first.data();
                    }
                    auto reject_times = bench_raw(
                        iters, 5, [&]() {},
                        [&]() {
                            const auto rejects = reject_stage(grammar, reject_source.candidates);
                            if (rejects.size() > vocab_size) { std::abort(); }
                        });
                    state.reject_only = summarize(reject_times);
                }

                // Validate fast path from the continuation's first token.
                const std::vector<TokenRef> continuation =
                    tokenize_text(piece_index, s == 0 ? test.cont_fresh
                                                      : (s == 1 ? test.cont_mid : test.cont_near));
                if (continuation.empty() || continuation[0].id < 0) {
                    throw std::runtime_error("continuation for " + test.name + "/" + state.name +
                                             " has no vocab token");
                }
                state.good_id     = continuation[0].id;
                state.good_piece  = continuation[0].piece;
                state.good_finite = token_allowed(grammar, state.good_id);
                if (!state.good_finite) {
                    throw std::runtime_error("continuation head is rejected from " + test.name +
                                             "/" + state.name);
                }
                {
                    llama_token_data td{state.good_id, 1.0f, 0.0f};
                    llama_token_data_array one{&td, 1, -1, false};
                    state.validate_good = summarize(bench_raw(
                        iters * 5, 50,
                        [&]() {
                            td  = llama_token_data{state.good_id, 1.0f, 0.0f};
                            one = llama_token_data_array{&td, 1, -1, false};
                        },
                        [&]() { llama_grammar_apply_impl(grammar, &one); }));
                }

                // A token the grammar rejects from this state (first non-empty,
                // non-EOG piece that the reject walk kills).
                for (size_t id = 0; id < vocab_size; ++id) {
                    const std::string& piece = vocab.pieces[id];
                    if (piece.empty() || piece[0] == 0 || vocab.is_eog(static_cast<int>(id))) {
                        continue;
                    }
                    if (!token_allowed(grammar, static_cast<int>(id))) {
                        state.bad_id    = static_cast<int>(id);
                        state.bad_piece = piece;
                        break;
                    }
                }
                if (state.bad_id >= 0) {
                    llama_token_data td{state.bad_id, 1.0f, 0.0f};
                    llama_token_data_array one{&td, 1, -1, false};
                    state.validate_bad = summarize(bench_raw(
                        iters * 5, 50,
                        [&]() {
                            td  = llama_token_data{state.bad_id, 1.0f, 0.0f};
                            one = llama_token_data_array{&td, 1, -1, false};
                        },
                        [&]() { llama_grammar_apply_impl(grammar, &one); }));
                }

                std::printf("  state %-5s prefix=\"%s\" (%zu bytes) stacks=%zu allow_eog=%d "
                            "allowed_tokens=%zu\n",
                            state.name.c_str(), printable(state.prefix).c_str(),
                            state.prefix.size(), state.stacks, state.allow_eog ? 1 : 0,
                            state.allowed_tokens);
                std::printf("    stack sizes: %s\n", describe_stacks(grammar).c_str());
                print_stats_line("fill (full vocab)", state.fill);
                print_stats_line("  |- decode stage", state.decode_only);
                print_stats_line("  |- reject stage", state.reject_only);
                print_stats_line("validate good", state.validate_good);
                print_stats_line("validate bad", state.validate_bad);
                report.states.push_back(std::move(state));
            }

            // Accept timing: clone the state per iteration, accept the
            // continuation token by token, time each accept call.
            for (int s = 0; s < 3; ++s) {
                AcceptReport accept;
                accept.state = state_names[s];
                accept.tokens =
                    tokenize_text(piece_index, s == 0 ? test.cont_fresh
                                                      : (s == 1 ? test.cont_mid : test.cont_near));
                accept.per_token.assign(accept.tokens.size(), Stats{});

                std::vector<std::vector<double>> times(accept.tokens.size());
                const size_t warmup = 20;
                for (size_t iteration = 0; iteration < iters + warmup; ++iteration) {
                    llama_grammar* grammar = llama_grammar_clone_impl(*state_grammars[s]);
                    for (size_t t = 0; t < accept.tokens.size(); ++t) {
                        const TokenRef& token = accept.tokens[t];
                        if (token.id >= 0) {
                            const double t0 = now_us();
                            llama_grammar_accept_token(*grammar, token.id, token.piece);
                            const double t1 = now_us();
                            if (iteration >= warmup) { times[t].push_back(t1 - t0); }
                        } else {
                            const double t0 = now_us();
                            llama_grammar_accept_str(*grammar, token.piece);
                            const double t1 = now_us();
                            if (iteration >= warmup) { times[t].push_back(t1 - t0); }
                        }
                    }
                    llama_grammar_free_impl(grammar);
                }
                for (size_t t = 0; t < accept.tokens.size(); ++t) {
                    accept.per_token[t] = summarize(times[t]);
                }

                std::printf("  accept from %-5s:", accept.state.c_str());
                for (size_t t = 0; t < accept.tokens.size(); ++t) {
                    std::printf(" [%s](%zub)=%.3fus", printable(accept.tokens[t].piece).c_str(),
                                accept.tokens[t].piece.size(), accept.per_token[t].median);
                }
                std::printf("\n");
                report.accepts.push_back(std::move(accept));
            }

            llama_grammar_free_impl(state_grammars[1]);
            llama_grammar_free_impl(state_grammars[2]);
            llama_grammar_free_impl(fresh);
            std::printf("\n");
            reports.push_back(std::move(report));
        }

        if (!json_path.empty()) {
            write_json(json_path, vocab_size, mask_row_bytes, reports, iters, mask_memset,
                       mask_memcpy);
            std::printf("json report: %s\n", json_path.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mask-cost harness: %s\n", e.what());
        return 1;
    }
}
