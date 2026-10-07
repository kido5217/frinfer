// Scrape-shape contract for GET /metrics: a Prometheus text exposition renders every declared
// metric exactly once with a HELP and a TYPE line, and its values are the snapshot it was given.

#include "serve/prometheus_metrics.h"
#include "serve/serve_options.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::serve::PrometheusSnapshot;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

struct ParsedExposition {
    // Sample name (without the frinfer_ prefix) -> published value string, in exposition order.
    std::vector<std::pair<std::string, std::string>> order;
    std::map<std::string, std::string> help;
    std::map<std::string, std::string> type;
    std::vector<std::string> unterminated;
    std::size_t help_lines    = 0;
    std::size_t type_lines    = 0;
    std::size_t sample_lines  = 0;
    bool ends_with_newline    = false;
    bool help_before_type     = true;
    bool type_before_sample   = true;
};

// Rejects the exposition shapes a Prometheus parser would reject: a sample without HELP/TYPE, a
// missing trailing newline, an unparsable value, or a duplicated series.
ParsedExposition parse_exposition(const std::string& body) {
    // "# HELP frinfer_" / "# TYPE frinfer_" / "frinfer_" are 15/15/8 characters long.
    constexpr std::size_t kPrefixLength = 15;
    ParsedExposition parsed;
    parsed.ends_with_newline = !body.empty() && body.back() == '\n';
    std::istringstream stream(body);
    std::string line;
    std::string previous;
    while (std::getline(stream, line)) {
        if (line.rfind("# HELP frinfer_", 0) == 0) {
            ++parsed.help_lines;
            const std::size_t split = line.find(' ', kPrefixLength);
            parsed.help[line.substr(kPrefixLength, split - kPrefixLength)] = line.substr(split + 1);
            parsed.order.emplace_back(line.substr(kPrefixLength, split - kPrefixLength), "");
            previous = "help";
            continue;
        }
        if (line.rfind("# TYPE frinfer_", 0) == 0) {
            ++parsed.type_lines;
            const std::size_t split = line.find(' ', kPrefixLength);
            parsed.type[line.substr(kPrefixLength, split - kPrefixLength)] = line.substr(split + 1);
            if (previous != "help") { parsed.help_before_type = false; }
            previous = "type";
            continue;
        }
        if (line.empty()) { continue; }
        ++parsed.sample_lines;
        if (previous != "type") { parsed.type_before_sample = false; }
        const std::size_t split = line.find(' ');
        if (split == std::string::npos) { parsed.unterminated.push_back(line); continue; }
        const std::string name = line.substr(8, split - 8);
        const std::string value = line.substr(split + 1);
        const auto entry       = std::find_if(parsed.order.begin(), parsed.order.end(),
                                       [&](const auto& pair) { return pair.first == name; });
        if (entry == parsed.order.end()) {
            parsed.order.emplace_back(name, value);
        } else {
            entry->second = value; // a duplicate series would silently overwrite
        }
        previous = "sample";
    }
    return parsed;
}

PrometheusSnapshot populated_snapshot() {
    PrometheusSnapshot snapshot;
    snapshot.runtime.computed_prefill_tokens      = 4096;
    snapshot.runtime.reused_prompt_tokens         = 1024;
    snapshot.runtime.committed_decode_tokens      = 777;
    snapshot.runtime.decode_rounds                = 33;
    snapshot.runtime.decode_row_rounds            = 99;
    snapshot.runtime.running_requests             = 2;
    snapshot.runtime.prefilling_requests          = 1;
    snapshot.runtime.decode_ready_requests        = 3;
    snapshot.runtime.waiting_requests             = 4;
    snapshot.runtime.materializing_requests       = 5;
    snapshot.runtime.capture_pending_requests     = 6;
    snapshot.runtime.terminal_pending_requests    = 7;
    snapshot.runtime.host_work.prefill_units      = 8;
    snapshot.runtime.host_work.control_units      = 9;
    snapshot.runtime.host_work.prefill_host_ns    = 2'000'000'000ULL;
    snapshot.runtime.host_work.decode_host_ns     = 1'000'000'000ULL;
    snapshot.runtime.main_kv_d2h_pages            = 11;
    snapshot.runtime.main_kv_h2d_pages            = 12;
    snapshot.runtime.main_kv_d2d_pages            = 13;
    snapshot.runtime.host_kv_occupied_bytes       = 4096;
    snapshot.runtime.device_state_occupied_slots  = 3;
    snapshot.runtime.host_state_occupied_slots    = 2;
    snapshot.runtime.device_main_kv_occupied_pages = 64;
    snapshot.runtime.device_backend_kv_occupied_pages = 8;
    snapshot.runtime.pressure_checkpoints_dropped = 2;
    snapshot.runtime.pressure_searches            = 40;
    snapshot.memory.max_context                   = 8192;
    snapshot.memory.kv_capacity_page_groups       = 128;
    snapshot.memory.host_kv_capacity_bytes        = 8ULL << 30;
    snapshot.requests_started                     = 11;
    return snapshot;
}

int test_exposition_shape() {
    int failures = 0;
    const std::string body = ninfer::serve::render_prometheus_metrics(populated_snapshot());
    const ParsedExposition parsed = parse_exposition(body);

    failures += check(parsed.ends_with_newline, "the exposition must end with a newline");
    failures += check(parsed.unterminated.empty(),
                      "every sample line needs a value after its name");
    failures += check(parsed.help_before_type && parsed.type_before_sample,
                      "HELP must precede TYPE, which must precede the sample");
    failures += check(parsed.help_lines == parsed.order.size() &&
                          parsed.type_lines == parsed.order.size() &&
                          parsed.sample_lines == parsed.order.size(),
                      "HELP, TYPE and sample counts must agree");
    failures += check(parsed.help.size() == parsed.order.size(),
                      "a metric name appeared more than once");
    const auto lookup = [](const std::map<std::string, std::string>& table,
                           const std::string& name) -> std::string {
        const auto entry = table.find(name);
        return entry == table.end() ? std::string() : entry->second;
    };
    for (const auto& [name, value] : parsed.order) {
        failures += check(!lookup(parsed.help, name).empty(), name + ": HELP text is empty");
        const std::string type = lookup(parsed.type, name);
        failures += check(type == "counter" || type == "gauge",
                          name + ": TYPE must be counter or gauge");
        failures += check(!value.empty(), name + ": value is empty");
    }
    return failures;
}

int test_snapshot_values() {
    int failures = 0;
    const std::string body = ninfer::serve::render_prometheus_metrics(populated_snapshot());
    const ParsedExposition parsed = parse_exposition(body);
    const auto value = [&](const std::string& name) -> std::string {
        const auto entry = std::find_if(parsed.order.begin(), parsed.order.end(),
                                        [&](const auto& pair) { return pair.first == name; });
        return entry == parsed.order.end() ? std::string("<missing>") : entry->second;
    };

    failures += check(value("prompt_tokens_total") == "4096", "prompt_tokens_total");
    failures += check(value("prompt_tokens_reused_total") == "1024", "prompt_tokens_reused_total");
    failures += check(value("generation_tokens_total") == "777", "generation_tokens_total");
    failures += check(value("prefill_units_total") == "8", "prefill_units_total");
    failures += check(value("control_units_total") == "9", "control_units_total");
    failures += check(value("decode_rounds_total") == "33", "decode_rounds_total");
    failures += check(value("decode_row_rounds_total") == "99", "decode_row_rounds_total");
    failures += check(value("requests_started_total") == "11", "requests_started_total");
    failures += check(value("main_kv_pages_transferred_d2h_total") == "11", "kv d2h pages");
    failures += check(value("host_kv_pressure_checkpoints_dropped_total") == "2",
                      "pressure_checkpoints_dropped_total");
    failures += check(value("requests_running") == "2", "requests_running");
    failures += check(value("requests_waiting") == "4", "requests_waiting");
    failures += check(value("requests_terminal_pending") == "7", "requests_terminal_pending");
    failures += check(value("context_capacity_tokens") == "8192", "context_capacity_tokens");
    failures += check(value("main_kv_capacity_pages") == "128", "main_kv_capacity_pages");
    failures += check(value("host_kv_capacity_bytes") == "8589934592", "host_kv_capacity_bytes");

    // 99 rows over 33 rounds is exactly 3; both rates are per second over the Engine host time.
    failures += check(value("decode_batch_average_rows") == "3", "decode_batch_average_rows");
    failures += check(value("generation_tokens_per_second") == "777",
                      "generation_tokens_per_second");
    failures += check(value("prompt_tokens_per_second") == "2048", "prompt_tokens_per_second");
    const auto type_of = [&](const std::string& name) -> std::string {
        const auto entry = parsed.type.find(name);
        return entry == parsed.type.end() ? std::string("<missing>") : entry->second;
    };
    failures += check(type_of("decode_batch_average_rows") == "gauge" &&
                          type_of("generation_tokens_total") == "counter",
                      "counters and gauges must be typed as such");

    // A fresh server has no work and no host time; the renderer must publish zeros, not NaN or a
    // division by zero.
    const ParsedExposition empty = parse_exposition(ninfer::serve::render_prometheus_metrics({}));
    const auto empty_value        = [&](const std::string& name) -> std::string {
        const auto entry = std::find_if(empty.order.begin(), empty.order.end(),
                                        [&](const auto& pair) { return pair.first == name; });
        return entry == empty.order.end() ? std::string("<missing>") : entry->second;
    };
    failures += check(empty_value("decode_batch_average_rows") == "0",
                      "an idle server must report a zero decode batch average, not NaN");
    failures += check(empty_value("generation_tokens_per_second") == "0" &&
                          empty_value("prompt_tokens_per_second") == "0",
                      "an idle server must report zero throughput, not NaN");
    failures += check(empty.order.size() == parsed.order.size(),
                      "the metric set must not depend on the snapshot's values");
    return failures;
}

// --metrics is opt-in and default-off: the flag is what publishes the route.
int test_metrics_flag() {
    int failures  = 0;
    char program[] = "frinfer-serve";
    char model[]   = "model.ninfer";
    char* argv[]   = {program, model, nullptr};
    char enabled[] = "--metrics";
    char* with_flag[] = {program, model, enabled, nullptr};

    const auto defaults = ninfer::serve::parse_serve_options(2, argv);
    failures += check(!defaults.enable_metrics, "--metrics must default to off");
    const auto metrics = ninfer::serve::parse_serve_options(3, with_flag);
    failures += check(metrics.enable_metrics, "--metrics did not enable the scrape route");
    failures += check(
        ninfer::serve::serve_usage_text("frinfer-serve").find("--metrics") != std::string::npos,
        "serve help omits --metrics");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_exposition_shape();
    failures += test_snapshot_values();
    failures += test_metrics_flag();
    if (failures == 0) { std::cout << "Prometheus metrics tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
