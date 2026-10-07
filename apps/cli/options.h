#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
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

struct ModelAcquisitionOptions {
    std::optional<std::string> hf_repo;
    std::optional<std::string> hf_file;
    std::optional<std::string> hf_revision;
    std::optional<std::string> hf_token;
    std::optional<std::string> model_url;
    std::filesystem::path cache_dir;
    bool offline    = false;
    bool cache_list = false;
};

struct Options {
    bool help_requested    = false;
    bool version_requested = false;

    // Which flag supplied the prompt. Exactly one source is admitted; an empty --prompt-file or an
    // empty --prompt-stdin is still a supplied source, which an empty --prompt string is not.
    enum class PromptSource : std::uint8_t {
        None,
        Inline,
        File,
        Stdin,
        Messages,
    };

    std::filesystem::path artifact_path;
    ModelAcquisitionOptions acquisition;
    std::filesystem::path chat_template_path;
    PromptSource prompt_source = PromptSource::None;
    // The resolved prompt text for Inline, File and Stdin; empty for Messages.
    std::string prompt;
    std::filesystem::path prompt_file_path;
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
    bool show_prompt     = false;
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

// Human-readable flag name for a prompt source, for the exactly-one diagnostic.
[[nodiscard]] std::string_view prompt_source_flag(Options::PromptSource source) noexcept;

} // namespace ninfer::cli
