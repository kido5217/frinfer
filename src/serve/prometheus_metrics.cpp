#include "serve/prometheus_metrics.h"

#include <array>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

// Prometheus parses a Go float; integral values are published without a fractional part so a counter
// reads as the integer it is.
std::string format_value(double value) {
    if (value == static_cast<double>(static_cast<std::uint64_t>(value))) {
        return std::to_string(static_cast<std::uint64_t>(value));
    }
    std::string text = std::to_string(value);
    // std::to_string emits six fractional digits; trim the trailing zeros it pads short ratios with.
    while (!text.empty() && text.back() == '0') { text.pop_back(); }
    if (!text.empty() && text.back() == '.') { text.pop_back(); }
    return text;
}

double rate(std::uint64_t tokens, std::uint64_t nanoseconds) {
    if (nanoseconds == 0) { return 0.0; }
    return static_cast<double>(tokens) / (static_cast<double>(nanoseconds) / 1e9);
}

class Exposition {
public:
    void counter(std::string_view name, std::string_view help, double value) {
        metric("counter", name, help, format_value(value));
    }

    void gauge(std::string_view name, std::string_view help, double value) {
        metric("gauge", name, help, format_value(value));
    }

    [[nodiscard]] std::string finish() const { return std::move(out_); }

private:
    void metric(std::string_view type, std::string_view name, std::string_view help,
                std::string value) {
        out_ += "# HELP frinfer_";
        out_ += name;
        out_ += ' ';
        out_ += help;
        out_ += "\n# TYPE frinfer_";
        out_ += name;
        out_ += ' ';
        out_ += type;
        out_ += "\nfrinfer_";
        out_ += name;
        out_ += ' ';
        out_ += value;
        out_ += '\n';
    }

    std::string out_;
};

} // namespace

std::string render_prometheus_metrics(const PrometheusSnapshot& snapshot) {
    const ninfer::RuntimeStats& runtime = snapshot.runtime;
    const ninfer::MemorySummary& memory = snapshot.memory;

    Exposition out;
    // Work counters: cumulative since load, so a rate over them is an average, not an interval.
    out.counter("prompt_tokens_total",
                "Prompt tokens evaluated by prefill, excluding reused checkpoint prefixes.",
                static_cast<double>(runtime.computed_prefill_tokens));
    out.counter("prompt_tokens_reused_total",
                "Prompt tokens served from an existing context-cache checkpoint.",
                static_cast<double>(runtime.reused_prompt_tokens));
    out.counter("generation_tokens_total", "Generation tokens committed by decode rounds.",
                static_cast<double>(runtime.committed_decode_tokens));
    out.counter("prefill_units_total", "Prefill units executed.",
                static_cast<double>(runtime.host_work.prefill_units));
    out.counter("control_units_total", "Thinking-control units committed.",
                static_cast<double>(runtime.host_work.control_units));
    out.counter("decode_rounds_total", "Compact decode batch executions.",
                static_cast<double>(runtime.decode_rounds));
    out.counter("decode_row_rounds_total",
                "Decode rows summed over every compact decode batch execution.",
                static_cast<double>(runtime.decode_row_rounds));
    out.counter("requests_started_total",
                "Generation requests a protocol route has begun since startup.",
                static_cast<double>(snapshot.requests_started));
    out.counter("main_kv_pages_transferred_d2h_total",
                "Main KV pages copied device to host.",
                static_cast<double>(runtime.main_kv_d2h_pages));
    out.counter("main_kv_pages_transferred_h2d_total",
                "Main KV pages copied host to device.",
                static_cast<double>(runtime.main_kv_h2d_pages));
    out.counter("main_kv_pages_transferred_d2d_total",
                "Main KV pages copied device to device.",
                static_cast<double>(runtime.main_kv_d2d_pages));
    out.counter("host_kv_pressure_checkpoints_dropped_total",
                "Context-cache checkpoints dropped under KV pressure.",
                static_cast<double>(runtime.pressure_checkpoints_dropped));
    out.counter("host_kv_pressure_searches_total",
                "Context-cache checkpoints examined under KV pressure.",
                static_cast<double>(runtime.pressure_searches));

    // Queue depth by request state, straight from the Engine's scheduler snapshot.
    out.gauge("requests_running", "Requests currently executing model work.",
              static_cast<double>(runtime.running_requests));
    out.gauge("requests_prefilling", "Requests consuming a prompt suffix.",
              static_cast<double>(runtime.prefilling_requests));
    out.gauge("requests_decode_ready", "Requests waiting to join the next decode batch.",
              static_cast<double>(runtime.decode_ready_requests));
    out.gauge("requests_waiting", "Requests waiting for admission.",
              static_cast<double>(runtime.waiting_requests));
    out.gauge("requests_materializing", "Requests materializing their logical context claim.",
              static_cast<double>(runtime.materializing_requests));
    out.gauge("requests_capture_pending", "Requests with a pending context capture.",
              static_cast<double>(runtime.capture_pending_requests));
    out.gauge("requests_terminal_pending", "Finished requests still holding active resources.",
              static_cast<double>(runtime.terminal_pending_requests));

    out.gauge("decode_batch_average_rows",
              "Mean rows per compact decode batch since startup.",
              runtime.decode_rounds == 0
                  ? 0.0
                  : static_cast<double>(runtime.decode_row_rounds) /
                        static_cast<double>(runtime.decode_rounds));
    out.gauge("prompt_tokens_per_second",
              "Mean prefill throughput over Engine prefill host time since startup.",
              rate(runtime.computed_prefill_tokens, runtime.host_work.prefill_host_ns));
    out.gauge("generation_tokens_per_second",
              "Mean decode throughput over Engine decode host time since startup.",
              rate(runtime.committed_decode_tokens, runtime.host_work.decode_host_ns));

    out.gauge("context_capacity_tokens", "Logical context ceiling of one sequence.",
              static_cast<double>(memory.max_context));
    out.gauge("main_kv_capacity_pages", "Resolved Main KV capacity in pages.",
              static_cast<double>(memory.kv_capacity_page_groups));
    out.gauge("main_kv_occupied_pages", "Main KV pages currently resident on the device.",
              static_cast<double>(runtime.device_main_kv_occupied_pages));
    out.gauge("host_kv_capacity_bytes", "Host KV budget in bytes.",
              static_cast<double>(memory.host_kv_capacity_bytes));
    out.gauge("host_kv_occupied_bytes", "Host KV bytes currently retained.",
              static_cast<double>(runtime.host_kv_occupied_bytes));
    out.gauge("device_state_occupied_slots", "Device StateImage slots currently occupied.",
              static_cast<double>(runtime.device_state_occupied_slots));
    out.gauge("host_state_occupied_slots", "Host State slots currently occupied.",
              static_cast<double>(runtime.host_state_occupied_slots));
    out.gauge("device_backend_kv_occupied_pages", "Backend KV pages resident on the device.",
              static_cast<double>(runtime.device_backend_kv_occupied_pages));
    return out.finish();
}

} // namespace ninfer::serve
