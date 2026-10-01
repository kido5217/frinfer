// Trie mask producer probe (wayfinder ticket #57, route (a)).
//
// Drives the same fixture walks as tools/spike/grammar-fill/probe.cpp but
// produces the row for every state with the in-house codepoint-trie producer and
// compares it bit for bit against the module's vendored-walk row. Times the trie
// production per module-fill state.
#include "trie_producer.h"

#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/tokenizer_vocabulary.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include "llama-grammar.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace frontend = ninfer::models::qwen3_5::frontend;
using trie_probe::Producer;

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

static bool row_bit(const std::vector<std::uint32_t>& row, int id) {
    const auto word = static_cast<std::size_t>(id) >> 5U;
    return word < row.size() && (row[word] & (1U << (static_cast<unsigned>(id) & 31U))) != 0U;
}

static std::string escape_piece(std::string_view piece) {
    std::string out;
    for (const char raw : piece) {
        const unsigned char byte = static_cast<unsigned char>(raw);
        if (byte >= 0x20 && byte < 0x7f && byte != '"' && byte != '\\') {
            out.push_back(static_cast<char>(byte));
        } else {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02x", byte);
            out += buf;
        }
    }
    return out;
}

struct Percentiles {
    double min = 0, p50 = 0, p90 = 0, max = 0, mean = 0;
};

static Percentiles percentiles(std::vector<double> values) {
    Percentiles out;
    if (values.empty()) { return out; }
    std::sort(values.begin(), values.end());
    out.min = values.front();
    out.p50 = values[values.size() / 2];
    out.p90 = values[(values.size() * 9) / 10 < values.size() ? (values.size() * 9) / 10
                                                              : values.size() - 1];
    out.max = values.back();
    double sum = 0;
    for (const double v : values) { sum += v; }
    out.mean = sum / static_cast<double>(values.size());
    return out;
}

int main(int argc, char** argv) {
    std::string snapshot;
    std::string schema_path;
    std::string grammar_path;
    std::string mode = "lowest";
    std::uint64_t seed = 0x5eed;
    long max_steps = 20000;
    int repeat = 1;
    long dump = 0;
    bool no_bulk = false;
    bool no_tail = false;
    bool no_writes = false;
    bool no_fast = false;

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
        } else if (flag == "--grammar") {
            grammar_path = value();
        } else if (flag == "--mode") {
            mode = value();
        } else if (flag == "--seed") {
            seed = std::stoull(value());
        } else if (flag == "--steps") {
            max_steps = std::stol(value());
        } else if (flag == "--repeat") {
            repeat = std::stoi(value());
        } else if (flag == "--dump") {
            dump = std::stol(value());
        } else if (flag == "--no-bulk") {
            no_bulk = true;
        } else if (flag == "--no-tail") {
            no_tail = true;
        } else if (flag == "--no-writes") {
            no_writes = true;
        } else if (flag == "--no-fast") {
            no_fast = true;
        } else {
            std::fprintf(stderr, "unknown flag %s\n", flag.data());
            return 2;
        }
    }
    if (snapshot.empty() || (schema_path.empty() == grammar_path.empty())) {
        std::fprintf(stderr,
                     "usage: trie_probe --snapshot DIR (--schema FILE | --grammar FILE) "
                     "[--mode lowest|random] [--seed N] [--steps N] [--repeat K] [--dump N]\n");
        return 2;
    }

    const std::string tokenizer_json = read_file(snapshot + "/tokenizer.json");
    const std::string tokenizer_config = read_file(snapshot + "/tokenizer_config.json");
    const std::string generation_config = read_file(snapshot + "/generation_config.json");

    const auto tokenizer =
        std::make_shared<const frontend::Tokenizer>(frontend::TokenizerResources{
            tokenizer_json, tokenizer_config, generation_config});
    const auto vocabulary = std::make_shared<const frontend::TokenizerVocabulary>(
        tokenizer, std::vector<int>{});

    std::string error;
    std::shared_ptr<const frontend::CompiledGrammar> grammar;
    if (!schema_path.empty()) {
        grammar = frontend::CompiledGrammar::compile_from_schema(read_file(schema_path),
                                                                 vocabulary, &error);
    } else {
        grammar =
            frontend::CompiledGrammar::compile(read_file(grammar_path), vocabulary, &error);
    }
    if (!grammar) {
        std::fprintf(stderr, "compile failed: %s\n", error.c_str());
        return 1;
    }

    Producer producer;
    producer.bulk_enabled = !no_bulk;
    producer.tail_enabled = !no_tail;
    producer.writes_enabled = !no_writes;
    producer.fast_path_enabled = !no_fast;
    const auto build_started = std::chrono::steady_clock::now();
    producer.build(*vocabulary);
    const auto build_finished = std::chrono::steady_clock::now();
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_finished - build_started).count();
    std::printf(
        "trie: token_count=%u row_words=%u nodes=%llu edges=%llu registered=%llu "
        "partial_ids=%llu exceptions=%llu build_ms=%.1f mem_mib=%.1f\n",
        producer.token_count(), producer.row_words(),
        static_cast<unsigned long long>(producer.trie_nodes()),
        static_cast<unsigned long long>(producer.trie_edges()),
        static_cast<unsigned long long>(producer.trie_ids()),
        static_cast<unsigned long long>(producer.partial_ids()),
        static_cast<unsigned long long>(producer.exception_ids()), build_ms,
        static_cast<double>(producer.memory_bytes()) / (1024.0 * 1024.0));

    std::mt19937_64 rng(seed);
    for (int run = 1; run <= repeat; ++run) {
        auto state = grammar->initial_state();
        const auto before = grammar->stats();
        std::vector<double> fill_us;
        std::vector<double> fill_nodes;
        std::vector<double> fill_edges;
        std::vector<double> fill_bulks;
        long steps = 0;
        long compared = 0;
        long mismatches = 0;
        long mismatches_reported = 0;
        const auto wall_start = std::chrono::steady_clock::now();

        while (steps < max_steps) {
            const auto fills_before = grammar->stats().row_fills;
            const auto ref_started = std::chrono::steady_clock::now();
            const frontend::GrammarMaskRow ref_row = grammar->row_for(state);
            const auto ref_finished = std::chrono::steady_clock::now();
            const bool was_fill = grammar->stats().row_fills != fills_before;

            if (std::getenv("TRIE_DEBUG_REFROW") != nullptr && steps < 8) {
                std::uint64_t h1 = 1469598103934665603ULL, h2 = 1099511628211ULL;
                for (const std::uint32_t w : ref_row) {
                    h1 ^= w; h1 *= 1099511628211ULL;
                    h2 = (h2 ^ (w + 0x9e3779b9u)) * 1469598103934665603ULL;
                }
                std::fprintf(stderr, "[refrow] step=%ld md5=%016llx%016llx\n", steps,
                             (unsigned long long)h1, (unsigned long long)h2);
                std::fflush(stderr);
            }
            const auto* engine = static_cast<const llama_grammar*>(state.probe_engine());
            if (engine == nullptr) {
                std::fprintf(stderr, "probe_engine returned null\n");
                return 1;
            }
            if (std::getenv("TRIE_DEBUG_REC") != nullptr && steps < 3) {
                std::fprintf(stderr, "[eng] step=%ld nstacks=%zu:", steps, engine->stacks.size());
                for (const auto& st : engine->stacks) {
                    std::fprintf(stderr, " [%zu:", st.size());
                    for (std::size_t j = 0; j < st.size(); ++j) {
                        std::fprintf(stderr, " (%d,%u)", (int)st[j]->type, st[j]->value);
                    }
                }
                std::fprintf(stderr, "]\n");
                std::fflush(stderr);
            }
            const trie_probe::ProducerStats stats_before = producer.stats;
            std::vector<std::uint32_t> trie_row;
            const auto trie_started = std::chrono::steady_clock::now();
            try {
                producer.produce(engine->rules, engine->stacks, engine->partial_utf8, trie_row);
            } catch (const std::exception& failure) {
                std::fprintf(stderr, "producer failure at step %ld: %s\n", steps, failure.what());
                return 1;
            }
            const auto trie_finished = std::chrono::steady_clock::now();
            ++compared;

            bool equal = trie_row.size() == ref_row.size();
            if (equal) {
                equal = std::memcmp(trie_row.data(), ref_row.data(),
                                    trie_row.size() * sizeof(std::uint32_t)) == 0;
            }
            if (!equal) {
                ++mismatches;
                if (mismatches_reported < 5) {
                    ++mismatches_reported;
                    long differing_bits = 0;
                    std::string detail;
                    for (std::size_t word = 0;
                         word < std::min(trie_row.size(), ref_row.size()); ++word) {
                        std::uint32_t diff = trie_row[word] ^ ref_row[word];
                        while (diff != 0U) {
                            const int bit = __builtin_ctz(diff);
                            diff &= diff - 1U;
                            const int id = static_cast<int>(word * 32U + static_cast<unsigned>(bit));
                            ++differing_bits;
                            if (differing_bits <= 16) {
                                char buf[128];
                                const std::string_view piece = vocabulary->piece(id);
                                std::snprintf(buf, sizeof(buf), " id=%d(trie=%d,vendored=%d,piece=", id,
                                              (trie_row[word] >> bit) & 1U,
                                              (ref_row[word] >> bit) & 1U);
                                detail += buf;
                                detail += escape_piece(piece);
                                detail += ")";
                            }
                        }
                    }
                    std::printf(
                        "MISMATCH step=%ld trie_row_words=%zu ref_row_words=%zu differing_bits=%ld%s\n",
                        steps, trie_row.size(), ref_row.size(), differing_bits, detail.c_str());
                    if (mismatches_reported <= 3) {
                        std::printf("  state partial=(%u,%d) stacks=%zu\n",
                                    engine->partial_utf8.value, engine->partial_utf8.n_remain,
                                    engine->stacks.size());
                        for (const auto& st : engine->stacks) {
                            std::printf("    stack n=%zu:", st.size());
                            for (const auto* el : st) {
                                std::printf(" (%d,%u)", static_cast<int>(el->type), el->value);
                            }
                            std::printf("\n");
                        }
                    }
                }
            }

            {
                fill_us.push_back(std::chrono::duration<double, std::micro>(trie_finished -
                                                                            trie_started)
                                      .count());
                fill_nodes.push_back(
                    static_cast<double>(producer.stats.nodes_visited - stats_before.nodes_visited));
                fill_edges.push_back(
                    static_cast<double>(producer.stats.edges_visited - stats_before.edges_visited));
                fill_bulks.push_back(
                    static_cast<double>(producer.stats.bulks - stats_before.bulks));
            }

            int pick = -1;
            if (mode == "lowest") {
                for (int id = 0; id < grammar->token_count(); ++id) {
                    if (!vocabulary->is_eog(id) && row_bit(std::vector<std::uint32_t>(ref_row.begin(), ref_row.end()), id)) {
                        pick = id;
                        break;
                    }
                }
            } else {
                std::vector<int> allowed;
                for (int id = 0; id < grammar->token_count(); ++id) {
                    if (!vocabulary->is_eog(id) &&
                        (ref_row[static_cast<std::size_t>(id) >> 5U] &
                         (1U << (static_cast<unsigned>(id) & 31U))) != 0U) {
                        allowed.push_back(id);
                    }
                }
                if (!allowed.empty()) { pick = allowed[rng() % allowed.size()]; }
            }
            if (pick < 0) { break; }
            if (!state.accept(pick)) {
                std::fprintf(stderr, "module rejected a mask-allowed token %d\n", pick);
                return 1;
            }
            if (steps < dump) {
                const std::string_view piece = vocabulary->piece(pick);
                std::printf("step=%ld pick=%d %s piece=\"%s\"\n", steps, pick,
                            was_fill ? "FILL" : "hit ", escape_piece(piece).c_str());
            }
            ++steps;
        }
        const auto wall_finished = std::chrono::steady_clock::now();
        const auto after = grammar->stats();

        const Percentiles tp = percentiles(fill_us);
        const Percentiles np = percentiles(fill_nodes);
        const Percentiles ep = percentiles(fill_edges);
        const Percentiles bp = percentiles(fill_bulks);
        double fill_total_ms = 0;
        for (const double v : fill_us) { fill_total_ms += v; }
        fill_total_ms /= 1000.0;
        const double module_fill_ms =
            static_cast<double>(after.fill_micros - before.fill_micros) / 1000.0;
        const double wall_ms = std::chrono::duration<double, std::milli>(wall_finished - wall_start).count();

        std::printf(
            "run=%d mode=%s steps=%ld module_fills=%llu module_hits=%llu module_distinct=%llu "
            "module_fill_ms=%.1f module_wall_ms=%.1f\n",
            run, mode.c_str(), steps,
            static_cast<unsigned long long>(after.row_fills - before.row_fills),
            static_cast<unsigned long long>(after.row_hits - before.row_hits),
            static_cast<unsigned long long>(after.distinct_rows), module_fill_ms, wall_ms);
        std::printf(
            "trie_fills=%zu trie_fill_ms=%.2f mean_us=%.1f min_us=%.1f p50_us=%.1f p90_us=%.1f "
            "max_us=%.1f nodes(min=%.0f p50=%.0f p90=%.0f max=%.0f) edges(p50=%.0f p90=%.0f "
            "max=%.0f) bulks(p50=%.0f p90=%.0f max=%.0f)\n",
            fill_us.size(), fill_total_ms, tp.mean, tp.min, tp.p50, tp.p90, tp.max, np.min,
            np.p50, np.p90, np.max, ep.p50, ep.p90, ep.max, bp.p50, bp.p90, bp.max);
        std::printf(
            "phases_cyc: visit=%llu prepare=%llu capacity=%llu bulk_set=%llu tail=%llu\n"
            "compare: states=%ld mismatches=%ld partial_states=%llu bulks=%llu bulk_ids=%llu "
            "bits_set=%llu nodes=%llu edges=%llu tail_judged=%llu tail_steps=%llu "
            "capacity_calls=%llu capacity_steps=%llu declined_cap=%llu declined_class=%llu "
            "excl_judged=%llu exceptions_judged=%llu\n"
            "fastpath: recog_attempts=%llu recog_ok=%llu fast_fills=%llu fallback_fills=%llu "
            "table_builds=%llu table_build_fail=%llu build_ms=%.2f table_mib=%.2f bucket_tokens=%llu "
            "correction_tokens=%llu shape_hits=%llu shape_misses=%llu shape_evict=%llu\n",
            static_cast<unsigned long long>(producer.stats.cyc_visit),
            static_cast<unsigned long long>(producer.stats.cyc_prepare),
            static_cast<unsigned long long>(producer.stats.cyc_capacity),
            static_cast<unsigned long long>(producer.stats.cyc_bulk_set),
            static_cast<unsigned long long>(producer.stats.cyc_tail),
            compared, mismatches, static_cast<unsigned long long>(producer.stats.partial_states),
            static_cast<unsigned long long>(producer.stats.bulks),
            static_cast<unsigned long long>(producer.stats.bulk_ids),
            static_cast<unsigned long long>(producer.stats.bits_set),
            static_cast<unsigned long long>(producer.stats.nodes_visited),
            static_cast<unsigned long long>(producer.stats.edges_visited),
            static_cast<unsigned long long>(producer.stats.tail_judged),
            static_cast<unsigned long long>(producer.stats.tail_steps),
            static_cast<unsigned long long>(producer.stats.capacity_calls),
            static_cast<unsigned long long>(producer.stats.capacity_steps),
            static_cast<unsigned long long>(producer.stats.declined_capacity),
            static_cast<unsigned long long>(producer.stats.declined_class),
            static_cast<unsigned long long>(producer.stats.excl_judged),
            static_cast<unsigned long long>(producer.stats.exception_judged),
            static_cast<unsigned long long>(producer.stats.recog_attempts),
            static_cast<unsigned long long>(producer.stats.recog_ok),
            static_cast<unsigned long long>(producer.stats.fast_fills),
            static_cast<unsigned long long>(producer.stats.fallback_fills),
            static_cast<unsigned long long>(producer.stats.table_builds),
            static_cast<unsigned long long>(producer.stats.table_build_fail),
            static_cast<double>(producer.stats.build_ns) / 1e6,
            static_cast<double>(producer.stats.table_masks_words) * 4.0 / (1024.0 * 1024.0),
            static_cast<unsigned long long>(producer.stats.bucket_tokens),
            static_cast<unsigned long long>(producer.stats.correction_tokens),
            static_cast<unsigned long long>(producer.stats.shape_hits),
            static_cast<unsigned long long>(producer.stats.shape_misses),
            static_cast<unsigned long long>(producer.stats.shape_evictions));
        std::fflush(stdout);
    }
    return 0;
}
