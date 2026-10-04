#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

// Which CLI flag produced a constrained-decoding request, so an Engine compile/validation
// rejection maps to the same fail-closed code the serve route uses (`grammar_invalid` vs
// `json_schema_invalid`).
enum class ConstraintSource : std::uint8_t {
    None,
    Grammar,
    JsonSchema,
};

struct ConstraintOptions {
    ConstraintSource source = ConstraintSource::None;
    // Resolved GBNF: the `--grammar`/`--grammar-file` text, or the GBNF the serve's JSON-Schema
    // contract produced from `--json-schema`/`--json-schema-file`.
    std::string gbnf;
    // The request's resolved thinking flag under the serve's constrained-request default (issue
    // #86). The Engine wraps the grammar for the reasoning stream only when this is set and the
    // rendered prompt starts in reasoning.
    bool thinking_enabled = false;
};

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;

    ConstraintOptions constraint;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

// The fail-closed code for a constraint rejection at Engine submission, selected by the flag that
// requested it (grammar_invalid for --grammar/--grammar-file, json_schema_invalid for
// --json-schema/--json-schema-file).
[[nodiscard]] const char* constraint_error_code(ConstraintSource source) noexcept;

} // namespace ninfer::cli
