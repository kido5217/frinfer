#include "serve/metrics.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::serve;

namespace {

// Collect the sampled metric family names, collapsing histogram _bucket/_sum/_count samples back to
// their base family so a scrape enumerates each advertised series once.
std::set<std::string> metric_families(const std::string& text) {
    std::set<std::string> names;
    const std::string prefix = "frinfer_";
    for (std::size_t pos = text.find(prefix); pos != std::string::npos;
         pos = text.find(prefix, pos)) {
        std::size_t end = pos + prefix.size();
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) != 0 || text[end] == '_')) {
            ++end;
        }
        std::string name = text.substr(pos, end - pos);
        for (const std::string_view suffix : {"_bucket", "_sum", "_count"}) {
            if (name.size() > suffix.size() && name.ends_with(suffix)) {
                name.resize(name.size() - suffix.size());
                break;
            }
        }
        names.insert(std::move(name));
        pos = end;
    }
    return names;
}

// Read a repository-relative documentation file. NINFER_SOURCE_DIR is supplied by CMake for tests
// declared with NEEDS_SOURCE_DIR.
std::string read_repository_file(const std::string& relative_path) {
    std::ifstream input(std::string(NINFER_SOURCE_DIR) + "/" + relative_path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

// The documented Prometheus family set, parsed from docs/serving.md's metrics table. The table is
// the oracle: the renderer and the documentation must agree in both directions. Handles backticked
// names, `{a,b}` brace groups, and the `frinfer_requests_*` row whose later entries are bare
// suffixes; `frinfer_*` wildcard rows are tracked as prefixes.
struct DocumentedFamilies {
    std::set<std::string> names;
    std::set<std::string> wildcards;
};

DocumentedFamilies parse_documented_families(const std::string& document) {
    DocumentedFamilies documented;
    const auto table_begin = document.find("| Metrics | Meaning |");
    if (table_begin == std::string::npos) { return documented; }
    const auto after_table   = document.find("Histograms expose", table_begin);
    const std::string table  = document.substr(
        table_begin, after_table == std::string::npos ? std::string::npos : after_table - table_begin);
    for (std::size_t line_begin = 0; line_begin < table.size();) {
        const auto line_end = table.find('\n', line_begin);
        const std::string line =
            table.substr(line_begin, line_end == std::string::npos ? std::string::npos
                                                                   : line_end - line_begin);
        line_begin = line_end == std::string::npos ? table.size() : line_end + 1;
        if (line.empty() || line.front() != '|') { continue; }
        const auto cell_end = line.find('|', 1);
        const std::string cell =
            line.substr(1, cell_end == std::string::npos ? std::string::npos : cell_end - 1);
        std::string namespace_prefix;
        for (std::size_t search = 0;;) {
            const auto open = cell.find('`', search);
            if (open == std::string::npos) { break; }
            const auto close = cell.find('`', open + 1);
            if (close == std::string::npos) { break; }
            const std::string token = cell.substr(open + 1, close - open - 1);
            search                  = close + 1;
            if (token.rfind("frinfer_", 0) != 0) {
                const bool bare =
                    !token.empty() &&
                    token.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") ==
                        std::string::npos;
                if (!namespace_prefix.empty() && bare) {
                    documented.names.insert(namespace_prefix + token);
                }
                continue;
            }
            if (token.back() == '*') {
                documented.wildcards.insert(token.substr(0, token.size() - 1));
                continue;
            }
            const auto brace = token.find('{');
            const auto brace_end =
                brace == std::string::npos ? std::string::npos : token.find('}', brace);
            if (brace != std::string::npos && brace_end != std::string::npos &&
                brace_end + 1 < token.size()) {
                const std::string head    = token.substr(0, brace);
                const std::string tail    = token.substr(brace_end + 1);
                const std::string options = token.substr(brace + 1, brace_end - brace - 1);
                for (std::size_t option_begin = 0;;) {
                    const auto comma = options.find(',', option_begin);
                    documented.names.insert(
                        head + options.substr(option_begin, comma == std::string::npos
                                                              ? std::string::npos
                                                              : comma - option_begin) +
                        tail);
                    if (comma == std::string::npos) { break; }
                    option_begin = comma + 1;
                }
                namespace_prefix = head.substr(0, head.rfind('_') + 1);
            } else {
                const std::string family =
                    brace == std::string::npos ? token : token.substr(0, brace);
                documented.names.insert(family);
                namespace_prefix = family.substr(0, family.rfind('_') + 1);
            }
        }
    }
    return documented;
}

} // namespace

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                  = 2;
    options.context_cache.device_state_slots = 1;
    MemorySummary memory;
    memory.kv_capacity_page_groups     = 256;
    memory.host_context_capacity_bytes = 1048576;
    RuntimeStats baseline;
    baseline.generated_tokens         = 4;
    baseline.speculative_draft_tokens = 3;
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, memory, baseline);
    RuntimeStats running = baseline;
    running.generated_tokens += 5;
    running.speculative_draft_tokens += 6;
    running.speculative_accepted_tokens = 3;
    running.running_requests            = 1;
    running.host_context_occupied_bytes = 4096;
    running.host_context_reserved_bytes = 1024;
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true, 0);
    check(live.find("frinfer_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("frinfer_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("frinfer_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("frinfer_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("frinfer_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(live.find("ninfer_") == std::string::npos,
          "the renderer must publish only the frinfer_ namespace");
    check(metrics.render(running, true, 0) == live, "scrapes must not consume or reset counters");
    check(live.find("frinfer_host_context_used_bytes 4096\n") != std::string::npos,
          "reserved bytes must not be added to occupancy twice");
    check(live.find("le=\"0.05\"") != std::string::npos &&
              live.find("0.050000000000000003") == std::string::npos,
          "histogram bucket boundaries must use canonical short precision, not 17 digits");
    const auto emitted = metric_families(live);
    const DocumentedFamilies documented =
        parse_documented_families(read_repository_file("docs/serving.md"));
    if (documented.names.empty()) {
        std::cerr << "could not parse the metrics table from docs/serving.md\n";
        ++failures;
    }
    // docs/serving.md is the oracle: the renderer must emit exactly the documented families.
    std::vector<std::string> undocumented;
    std::vector<std::string> absent;
    for (const auto& name : emitted) {
        if (documented.names.count(name) != 0) { continue; }
        bool wildcard_match = false;
        for (const auto& wildcard : documented.wildcards) {
            if (name.rfind(wildcard, 0) == 0) {
                wildcard_match = true;
                break;
            }
        }
        if (!wildcard_match) { undocumented.push_back(name); }
    }
    for (const auto& name : documented.names) {
        if (emitted.count(name) == 0) { absent.push_back(name); }
    }
    for (const auto& wildcard : documented.wildcards) {
        const bool matched =
            std::any_of(emitted.begin(), emitted.end(), [&](const std::string& name) {
                return name.rfind(wildcard, 0) == 0;
            });
        if (!matched) { absent.push_back(wildcard + "*"); }
    }
    if (!undocumented.empty() || !absent.empty()) {
        std::cerr << "docs/serving.md metric drift; emitted-but-undocumented:";
        for (const auto& name : undocumented) { std::cerr << ' ' << name; }
        std::cerr << "; documented-but-absent:";
        for (const auto& name : absent) { std::cerr << ' ' << name; }
        std::cerr << '\n';
        ++failures;
    }

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.metrics.total_seconds                    = 2.0;
    outcome.metrics.engine_timing.queue_wait_seconds = 0.1;
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false, 0);
    check(final.find("frinfer_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("frinfer_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("frinfer_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("frinfer_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("frinfer_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("frinfer_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("frinfer_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("frinfer_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("frinfer_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    outcome.constraint = ConstraintObservation{.complete          = true,
                                               .terminated        = false,
                                               .cache             = ConstraintCacheAccess::Built,
                                               .mask_positions    = 5,
                                               .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false, 0);
    check(constrained.find(
              "frinfer_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("frinfer_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("frinfer_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
