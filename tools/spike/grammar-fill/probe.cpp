// Ad-hoc probe: drive the grammar module's row cache over the real model
// vocabulary and report fills/hits/time for a walk. Not a product artifact
// (investigation for ticket #57).
#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/tokenizer_vocabulary.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include <algorithm>
#include <chrono>
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

static std::string read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

static bool row_bit(frontend::GrammarMaskRow row, int id) {
  const auto word = static_cast<std::size_t>(id) >> 5U;
  return word < row.size() &&
         (row[word] & (1U << (static_cast<unsigned>(id) & 31U))) != 0U;
}

struct Percentiles {
  double min = 0, p50 = 0, p90 = 0, max = 0;
};

static Percentiles percentiles(std::vector<double> values) {
  Percentiles out;
  if (values.empty()) {
    return out;
  }
  std::sort(values.begin(), values.end());
  out.min = values.front();
  out.p50 = values[values.size() / 2];
  out.p90 =
      values[(values.size() * 9) / 10 < values.size() ? (values.size() * 9) / 10
                                                      : values.size() - 1];
  out.max = values.back();
  return out;
}

int main(int argc, char **argv) {
  std::string snapshot;
  std::string schema_path;
  std::string grammar_path;
  std::string mode = "lowest";
  std::uint64_t seed = 0x5eed;
  long max_steps = 20000;
  int repeat = 1;
  long dump = 0;
  std::string diag_path;

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
    } else if (flag == "--diag") {
      diag_path = value();
    } else {
      std::fprintf(stderr, "unknown flag %s\n", flag.data());
      return 2;
    }
  }
  if (snapshot.empty() || (schema_path.empty() == grammar_path.empty())) {
    std::fprintf(
        stderr, "usage: probe --snapshot DIR (--schema FILE | --grammar FILE) "
                "[--mode lowest|random] [--seed N] [--steps N] [--repeat K]\n");
    return 2;
  }

  const std::string tokenizer_json = read_file(snapshot + "/tokenizer.json");
  const std::string tokenizer_config =
      read_file(snapshot + "/tokenizer_config.json");
  const std::string generation_config =
      read_file(snapshot + "/generation_config.json");

  const auto tokenizer =
      std::make_shared<const frontend::Tokenizer>(frontend::TokenizerResources{
          tokenizer_json, tokenizer_config, generation_config});
  const auto vocabulary = std::make_shared<const frontend::TokenizerVocabulary>(
      tokenizer, std::vector<int>{});

  std::string error;
  std::shared_ptr<const frontend::CompiledGrammar> grammar;
  if (!schema_path.empty()) {
    grammar = frontend::CompiledGrammar::compile_from_schema(
        read_file(schema_path), vocabulary, &error);
  } else {
    grammar = frontend::CompiledGrammar::compile(read_file(grammar_path),
                                                 vocabulary, &error);
  }
  if (!grammar) {
    std::fprintf(stderr, "compile failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("token_count=%d row_words=%d\n", grammar->token_count(),
              (grammar->token_count() + 31) / 32);
  {
    std::size_t max_piece = 0;
    long over16 = 0, over32 = 0, over64 = 0, over128 = 0;
    for (int id = 0; id < vocabulary->token_count(); ++id) {
      const std::size_t size = vocabulary->piece(id).size();
      max_piece = std::max(max_piece, size);
      if (size > 16)
        ++over16;
      if (size > 32)
        ++over32;
      if (size > 64)
        ++over64;
      if (size > 128)
        ++over128;
    }
    std::printf(
        "max_piece_bytes=%zu over16=%ld over32=%ld over64=%ld over128=%ld\n",
        max_piece, over16, over32, over64, over128);
  }

  std::mt19937_64 rng(seed);
  for (int run = 1; run <= repeat; ++run) {
    auto state = grammar->initial_state();
    const auto before = grammar->stats();
    std::vector<double> per_call_us;
    per_call_us.reserve(static_cast<std::size_t>(max_steps));
    long steps = 0;
    long misses_before = static_cast<long>(before.row_fills);
    (void)misses_before;
    const auto wall_start = std::chrono::steady_clock::now();
    while (steps < max_steps) {
      const auto before_fills = grammar->stats().row_fills;
      const auto started = std::chrono::steady_clock::now();
      const frontend::GrammarMaskRow row = grammar->row_for(state);
      const auto finished = std::chrono::steady_clock::now();
      const bool was_fill = grammar->stats().row_fills != before_fills;
      per_call_us.push_back(
          std::chrono::duration<double, std::micro>(finished - started)
              .count());

      int pick = -1;
      if (mode == "lowest") {
        for (int id = 0; id < grammar->token_count(); ++id) {
          if (!vocabulary->is_eog(id) && row_bit(row, id)) {
            pick = id;
            break;
          }
        }
      } else {
        std::vector<int> allowed;
        for (int id = 0; id < grammar->token_count(); ++id) {
          if (!vocabulary->is_eog(id) && row_bit(row, id)) {
            allowed.push_back(id);
          }
        }
        if (!allowed.empty()) {
          pick = allowed[rng() % allowed.size()];
        }
      }
      if (pick < 0) {
        break;
      }
      if (!state.accept(pick)) {
        std::fprintf(stderr, "module rejected a mask-allowed token %d\n", pick);
        return 1;
      }
      if (steps < dump) {
        const std::string_view piece = vocabulary->piece(pick);
        std::uint64_t mask_hash = 1469598103934665603ULL;
        for (const std::uint32_t word : row) {
          for (int shift = 0; shift < 32; shift += 8) {
            mask_hash ^= (word >> shift) & 0xffU;
            mask_hash *= 1099511628211ULL;
          }
        }
        std::printf("step=%ld pick=%d %s mask=%016llx piece=\"", steps, pick,
                    was_fill ? "FILL" : "hit ",
                    static_cast<unsigned long long>(mask_hash));
        for (const char raw : piece) {
          const unsigned char byte = static_cast<unsigned char>(raw);
          if (byte >= 0x20 && byte < 0x7f) {
            std::printf("%c", byte);
          } else {
            std::printf("\\x%02x", byte);
          }
        }
        std::printf("\"\n");
      }
      ++steps;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const auto after = grammar->stats();
    const Percentiles pct = percentiles(per_call_us);
    const double row_bytes =
        static_cast<double>((grammar->token_count() + 31) / 32) * 4.0;
    std::printf(
        "run=%d mode=%s steps=%ld fills=%llu hits=%llu distinct=%llu "
        "fill_ms=%.1f "
        "wall_ms=%.1f per_call_us(min=%.1f p50=%.1f p90=%.1f max=%.1f) "
        "row_bytes=%.0f distinct_row_mib=%.1f\n",
        run, mode.c_str(), steps,
        static_cast<unsigned long long>(after.row_fills - before.row_fills),
        static_cast<unsigned long long>(after.row_hits - before.row_hits),
        static_cast<unsigned long long>(after.distinct_rows),
        static_cast<double>(after.fill_micros - before.fill_micros) / 1000.0,
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count(),
        pct.min, pct.p50, pct.p90, pct.max, row_bytes,
        static_cast<double>(after.distinct_rows) * row_bytes /
            (1024.0 * 1024.0));
    std::fflush(stdout);
  }
#ifdef NINFER_GRAMMAR_DIAG
  if (!diag_path.empty()) {
    grammar->diag_write(diag_path);
  }
#endif
  return 0;
}
