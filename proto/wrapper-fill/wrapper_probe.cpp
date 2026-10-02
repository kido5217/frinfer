// Prototype driver for ticket #96 (branch proto/wrapper-thinking-boundary): throwaway.
//
// Builds the two wrapper-GBNF variants around the vendored JSON-schema conversion,
// compiles them through the in-tree producer, and walks a realistic re-tokenized
// reasoning stream followed by a scripted </think> + whitespace + schema-answer.
// Reports row fills/hits/cost per phase and the boundary row's admit/forbid behavior.
//
// Variants:
//   a ("full encoding"): a marker-avoiding DFA body that FORCES the close: after the
//     full "</think>" the only continuation is format whitespace, then the schema.
//   c ("permissive prefix"): body is `.*` (single full-vocab row per step), close +
//     REQUIRED whitespace + schema; admits the whitespace at the deciding column but
//     does not force it (the permissive path stays alive).
//   plain: the bare converted schema (reference).
//
// Not a product artifact.

#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/tokenizer_vocabulary.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include "json-schema-to-grammar.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace frontend = ninfer::models::qwen3_5::frontend;

static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

static void write_file(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    out << text;
}

static bool row_bit(frontend::GrammarMaskRow row, int id) {
    const auto word = static_cast<std::size_t>(id) >> 5U;
    return word < row.size() &&
           (row[word] & (1U << (static_cast<unsigned>(id) & 31U))) != 0U;
}

// The engine's decoder reads an empty or NUL-leading piece as nothing and the mask never
// carries those ids, so a "full-vocab" row means every structurally maskable id is set.
static bool is_structurally_maskable(std::string_view piece) {
    return !piece.empty() && piece.front() != '\0';
}

static bool row_is_full(frontend::GrammarMaskRow row, int token_count,
                        const std::vector<std::uint8_t>& maskable) {
    for (int id = 0; id < token_count; ++id) {
        if (maskable[static_cast<std::size_t>(id)] != 0 && !row_bit(row, id)) { return false; }
    }
    return true;
}

static std::uint64_t row_hash(frontend::GrammarMaskRow row) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint32_t word : row) {
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (word >> shift) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static int count_unset(frontend::GrammarMaskRow row, int token_count,
                       const std::vector<std::uint8_t>& maskable) {
    int total = 0;
    for (int id = 0; id < token_count; ++id) {
        if (maskable[static_cast<std::size_t>(id)] != 0 && !row_bit(row, id)) { ++total; }
    }
    return total;
}

static void report_unset(frontend::GrammarMaskRow row, int token_count,
                         const std::vector<std::uint8_t>& maskable,
                         const frontend::GrammarVocabulary& vocabulary) {
    int shown = 0;
    std::string first;
    for (int id = 0; id < token_count && shown < 5; ++id) {
        if (maskable[static_cast<std::size_t>(id)] != 0 && !row_bit(row, id)) {
            const std::string piece(vocabulary.piece(id));
            std::string escaped;
            for (char raw : piece) {
                const unsigned char byte = static_cast<unsigned char>(raw);
                if (byte >= 0x20 && byte < 0x7f) {
                    escaped += raw;
                } else {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\x%02x", byte);
                    escaped += buffer;
                }
            }
            first += std::to_string(id) + "=\"" + escaped + "\" ";
            ++shown;
        }
    }
    long total = 0;
    for (int id = 0; id < token_count; ++id) {
        if (maskable[static_cast<std::size_t>(id)] != 0 && !row_bit(row, id)) { ++total; }
    }
    std::printf("unset_maskable=%d first=[%s]\n", total, first.c_str());
}

struct Percentiles {
    double mn = 0, p50 = 0, p90 = 0, mx = 0;
    long n = 0;
};

static Percentiles percentiles(std::vector<double> values) {
    Percentiles out;
    out.n = static_cast<long>(values.size());
    if (values.empty()) { return out; }
    std::sort(values.begin(), values.end());
    out.mn = values.front();
    out.p50 = values[values.size() / 2];
    out.p90 = values[(values.size() * 9) / 10 < values.size() ? (values.size() * 9) / 10
                                                             : values.size() - 1];
    out.mx = values.back();
    return out;
}

static std::string rename_converter_root(std::string gbnf) {
    const std::string needle = "root ::= ";
    const auto pos = gbnf.find(needle);
    if (pos == std::string::npos) {
        throw std::runtime_error("converter output has no 'root ::= ' rule");
    }
    if (gbnf.find("schema ::= ") != std::string::npos) {
        throw std::runtime_error("converter output already defines 'schema'");
    }
    for (const char* reserved : {"body0 ::= ", "ws ::= ", "close ::= "}) {
        if (gbnf.find(reserved) != std::string::npos) {
            throw std::runtime_error(std::string("converter output defines reserved rule ") +
                                     reserved);
        }
    }
    gbnf.replace(pos, needle.size(), "schema ::= ");
    return gbnf;
}

// Variant (a): marker-avoiding body states. The chain B -> B1 -> ... -> B7 is the DFA for
// "text avoiding </think": at state Bk the consumed suffix is the first k chars of "</think".
// A '<' always (re)enters B1; a failed continuation consumes one fallback char and completes
// back into the B loop. The marker's '>' has no body transition: the only way to consume it
// is the root-level close, so the close+required whitespace are forced. Corner: when the
// text immediately before the marker leaves a partial prefix open (e.g. "..." ends with
// "</"), the chain is mid-state and the root close cannot start there; the '>' is then
// rejected and the model must fall back. The chain never empties mid-prefix (that would
// let the marker be split across body elements and kill the forcing). EOG stays available
// at every completed body element (the `| body` root alternative) and only after a full
// answer value once the close is taken.
static std::string wrapper_a(const std::string& schema_gbnf) {
    std::string out = rename_converter_root(schema_gbnf);
    out +=
        "root ::= body close ws schema | body\n"
        "body ::= B*\n"
        "B ::= [^<] | \"<\" B1\n"
        "B1 ::= \"<\"* ( \"/\" B2 | [^/<] )\n"
        "B2 ::= [^t<] | \"t\" B3 | \"<\" B1\n"
        "B3 ::= [^h<] | \"h\" B4 | \"<\" B1\n"
        "B4 ::= [^i<] | \"i\" B5 | \"<\" B1\n"
        "B5 ::= [^n<] | \"n\" B6 | \"<\" B1\n"
        "B6 ::= [^k<] | \"k\" B7 | \"<\" B1\n"
        "B7 ::= [^><] | \"<\" B1\n"
        "close ::= \"</think>\"\n"
        "ws ::= [ \\t\\r\\n]+\n";
    return out;
}

// Variant (c): permissive body (`.*`), then the close, then REQUIRED format whitespace,
// then the schema. The permissive path is never killed by the close, so this admits the
// deciding whitespace but leaves the answer unconstrained (the producer's rows are the
// union over live stacks).
static std::string wrapper_c(const std::string& schema_gbnf) {
    std::string out = rename_converter_root(schema_gbnf);
    out +=
        "root ::= .* close ws schema | .*\n"
        "close ::= \"</think>\"\n"
        "ws ::= [ \\t\\r\\n]+\n";
    return out;
}

struct PhaseReport {
    std::string name;
    long steps = 0;
    long full_rows = 0;
    long non_full = 0;
    std::uint64_t fills = 0;
    std::uint64_t hits = 0;
    double fill_ms = 0;
    double wall_ms = 0;
    Percentiles us;
};

int main(int argc, char** argv) {
    std::string snapshot;
    std::string schema_path;
    std::string reasoning_path;
    std::string answer_path;
    std::string variant = "a";
    std::string emit_path;
    std::string dump_path;
    long max_reasoning_steps = 1000000;

    for (int index = 1; index < argc; ++index) {
        const std::string_view flag = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag.data());
                std::exit(2);
            }
            return argv[++index];
        };
        if (flag == "--snapshot") {
            snapshot = value();
        } else if (flag == "--schema") {
            schema_path = value();
        } else if (flag == "--reasoning") {
            reasoning_path = value();
        } else if (flag == "--answer") {
            answer_path = value();
        } else if (flag == "--variant") {
            variant = value();
        } else if (flag == "--emit-grammar") {
            emit_path = value();
        } else if (flag == "--dump-steps") {
            dump_path = value();
        } else if (flag == "--max-reasoning-steps") {
            max_reasoning_steps = std::stol(value());
        } else {
            std::fprintf(stderr, "unknown flag %s\n", flag.data());
            return 2;
        }
    }
    if (snapshot.empty() || schema_path.empty() || reasoning_path.empty() || answer_path.empty()) {
        std::fprintf(stderr,
                     "usage: wrapper_probe --snapshot DIR --schema FILE --reasoning FILE "
                     "--answer FILE [--variant a|c|plain] [--emit-grammar FILE]\n");
        return 2;
    }

    const auto tokenizer = std::make_shared<const frontend::Tokenizer>(frontend::TokenizerResources{
        read_file(snapshot + "/tokenizer.json"), read_file(snapshot + "/tokenizer_config.json"),
        read_file(snapshot + "/generation_config.json")});
    const auto vocabulary = std::make_shared<const frontend::TokenizerVocabulary>(
        tokenizer, std::vector<int>{});

    const std::string schema_gbnf =
        json_schema_to_grammar(common_json::parse(read_file(schema_path)), /*force_gbnf=*/true);
    std::string wrapper;
    if (variant == "a") {
        wrapper = wrapper_a(schema_gbnf);
    } else if (variant == "c") {
        wrapper = wrapper_c(schema_gbnf);
    } else if (variant == "plain") {
        wrapper = schema_gbnf;
    } else if (variant == "dot") {
        wrapper = "root ::= .*\n";
    } else {
        std::fprintf(stderr, "unknown variant %s\n", variant.c_str());
        return 2;
    }
    if (!emit_path.empty()) { write_file(emit_path, wrapper); }

    std::string error;
    const auto grammar =
        frontend::CompiledGrammar::compile(wrapper, vocabulary, &error);
    if (!grammar) {
        std::fprintf(stderr, "compile failed: %s\n", error.c_str());
        return 1;
    }
    const int token_count = grammar->token_count();
    std::vector<std::uint8_t> maskable(static_cast<std::size_t>(token_count));
    long unmaskable = 0;
    for (int id = 0; id < token_count; ++id) {
        maskable[static_cast<std::size_t>(id)] =
            is_structurally_maskable(vocabulary->piece(id)) ? 1 : 0;
        if (maskable[static_cast<std::size_t>(id)] == 0) { ++unmaskable; }
    }
    std::printf("variant=%s grammar_bytes=%zu token_count=%d row_words=%d unmaskable_ids=%ld "
                "compile_us=%llu\n",
                variant.c_str(), wrapper.size(), token_count, (token_count + 31) / 32, unmaskable,
                static_cast<unsigned long long>(grammar->stats().compile_micros));

    // Realistic stream: reasoning text re-tokenized, then scripted close + ws + answer.
    const auto reasoning_ids = tokenizer->encode(read_file(reasoning_path));
    const auto close_ids    = tokenizer->encode("</think>");
    const auto ws_ids       = tokenizer->encode("\n\n");
    const auto space_ids    = tokenizer->encode(" ");
    const auto brace_ids    = tokenizer->encode("{");
    const auto answer_ids   = tokenizer->encode(read_file(answer_path));
    const auto pieces_of    = [&](const std::vector<int>& ids) {
        std::string s;
        for (int id : ids) { s += "\"" + std::string(vocabulary->piece(id)) + "\" "; }
        return s;
    };
    std::printf("tokens reasoning=%zu close=%zu(%s) ws=%zu space=%zu brace=%zu(%s) answer=%zu\n",
                reasoning_ids.size(), close_ids.size(), pieces_of(close_ids).c_str(), ws_ids.size(),
                space_ids.size(), brace_ids.size(), pieces_of(brace_ids).c_str(),
                answer_ids.size());

    auto state = grammar->initial_state();
    std::ofstream dump;
    if (!dump_path.empty()) { dump.open(dump_path); }
    if (variant == "plain" || variant == "dot") {
        const frontend::GrammarMaskRow row = grammar->row_for(state);
        std::printf("plain_root: can_end=%d ws_admitted=%d brace_admitted=%d unset_maskable=%d "
                    "row_hash=%016llx\n",
                    state.can_end() ? 1 : 0,
                    (!ws_ids.empty() && row_bit(row, ws_ids.front())) ? 1 : 0,
                    (!brace_ids.empty() && row_bit(row, brace_ids.front())) ? 1 : 0,
                    count_unset(row, token_count, maskable),
                    static_cast<unsigned long long>(row_hash(row)));
        return 0;
    }
    const auto run_phase = [&](const std::string& name, const std::vector<int>& ids, long limit,
                               bool check_full, bool must_admit) {
        PhaseReport report;
        report.name    = name;
        const auto s0  = grammar->stats();
        const auto t0  = std::chrono::steady_clock::now();
        std::vector<double> us;
        us.reserve(ids.size());
        for (int id : ids) {
            if (report.steps >= limit) { break; }
            const auto before_fills = grammar->stats().row_fills;
            const auto started      = std::chrono::steady_clock::now();
            const frontend::GrammarMaskRow row = grammar->row_for(state);
            const auto finished = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(finished - started).count());
            report.steps += 1;
            if (grammar->stats().row_fills != before_fills) { report.fills += 1; } else { report.hits += 1; }
            if (dump.is_open()) {
                dump << "phase=" << name << " step=" << report.steps << " token=" << id
                     << " can_end=" << (state.can_end() ? 1 : 0)
                     << " unset=" << count_unset(row, token_count, maskable) << " hash=" << std::hex
                     << row_hash(row) << std::dec << "\n";
            }
            if (check_full) {
                if (row_is_full(row, token_count, maskable)) {
                    report.full_rows += 1;
                } else {
                    report.non_full += 1;
                    if (report.non_full == 1) {
                        std::printf("first_non_full_row phase=%s step=%ld: ", name.c_str(),
                                    report.steps);
                        report_unset(row, token_count, maskable, *vocabulary);
                        std::printf("first_row_hash=%016llx\n",
                                    static_cast<unsigned long long>(row_hash(row)));
                    }
                }
            }
            if (must_admit && !row_bit(row, id)) {
                std::fprintf(stderr, "variant=%s phase=%s token=%d piece=\"%s\" NOT ADMITTED\n",
                             variant.c_str(), name.c_str(), id, std::string(vocabulary->piece(id)).c_str());
                std::exit(3);
            }
            if (!state.accept(id)) {
                std::fprintf(stderr, "variant=%s phase=%s token=%d piece=\"%s\" REJECTED\n",
                             variant.c_str(), name.c_str(), id, std::string(vocabulary->piece(id)).c_str());
                std::exit(3);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const auto s1 = grammar->stats();
        report.us        = percentiles(std::move(us));
        report.fills     = s1.row_fills - s0.row_fills;
        report.hits      = s1.row_hits - s0.row_hits;
        report.fill_ms   = static_cast<double>(s1.fill_micros - s0.fill_micros) / 1000.0;
        report.wall_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf(
            "phase=%s steps=%ld full_rows=%ld non_full=%ld fills=%llu hits=%llu fill_ms=%.2f "
            "wall_ms=%.2f per_call_us(min=%.2f p50=%.2f p90=%.2f max=%.2f)\n",
            name.c_str(), report.steps, report.full_rows, report.non_full,
            static_cast<unsigned long long>(report.fills),
            static_cast<unsigned long long>(report.hits), report.fill_ms, report.wall_ms,
            report.us.mn, report.us.p50, report.us.p90, report.us.mx);
        return report;
    };

    run_phase("reasoning", reasoning_ids, max_reasoning_steps, /*check_full=*/true,
              /*must_admit=*/true);

    // Boundary checks: the state after the full </think>.
    {
        for (int id : close_ids) {
            if (!state.accept(id)) {
                std::fprintf(stderr, "close token %d rejected\n", id);
                return 3;
            }
        }
        const frontend::GrammarMaskRow row = grammar->row_for(state);
        const bool ws_ok    = !ws_ids.empty() && row_bit(row, ws_ids.front());
        const bool space_ok = !space_ids.empty() && row_bit(row, space_ids.front());
        const bool brace_ok = !brace_ids.empty() && row_bit(row, brace_ids.front());
        std::printf("after_close: can_end=%d ws_admitted=%d space_admitted=%d brace_admitted=%d ",
                    state.can_end() ? 1 : 0, ws_ok ? 1 : 0, space_ok ? 1 : 0, brace_ok ? 1 : 0);
        report_unset(row, token_count, maskable, *vocabulary);
        std::printf("close_row_hash=%016llx\n", static_cast<unsigned long long>(row_hash(row)));
        std::fflush(stdout);
    }

    run_phase("ws", ws_ids, ws_ids.size(), /*check_full=*/false, /*must_admit=*/true);
    {
        const frontend::GrammarMaskRow row = grammar->row_for(state);
        const bool first_answer_ok = !answer_ids.empty() && row_bit(row, answer_ids.front());
        std::printf("after_ws: can_end=%d first_answer_token_admitted=%d ",
                    state.can_end() ? 1 : 0, first_answer_ok ? 1 : 0);
        report_unset(row, token_count, maskable, *vocabulary);
        std::printf("schema_root_row_hash=%016llx\n",
                    static_cast<unsigned long long>(row_hash(row)));
    }
    run_phase("answer", answer_ids, answer_ids.size(), /*check_full=*/false, /*must_admit=*/true);
    std::printf("end: can_end=%d\n", state.can_end() ? 1 : 0);

    const auto total = grammar->stats();
    std::printf("stats: fills=%llu hits=%llu distinct=%llu fill_ms=%.2f compile_ms=%.2f\n",
                static_cast<unsigned long long>(total.row_fills),
                static_cast<unsigned long long>(total.row_hits),
                static_cast<unsigned long long>(total.distinct_rows),
                static_cast<double>(total.fill_micros) / 1000.0,
                static_cast<double>(total.compile_micros) / 1000.0);
    return 0;
}
