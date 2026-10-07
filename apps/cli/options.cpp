#include "options.h"
#include "product/constraint/constraint_contract.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

std::string read_constraint_file(const char* path, std::string_view flag) {
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
        throw std::invalid_argument(std::string(flag) + " is a directory: " + path);
    }
    error.clear();
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (!error && size > ninfer::constraint::kConstraintPayloadLimit) {
        throw std::invalid_argument(std::string(flag) + ": constraint_too_large: file exceeds " +
                                    std::to_string(ninfer::constraint::kConstraintPayloadLimit) +
                                    " bytes");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::invalid_argument(std::string(flag) + " cannot open file: " + path); }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (input.bad()) {
        throw std::invalid_argument(std::string(flag) + " failed to read file: " + path);
    }
    return buffer.str();
}

// Validates a JSON Schema document through the shared protocol-neutral constraint contract, so the
// CLI and the serve route admit exactly the same documents and report the same fail-closed codes
// (`json_schema_invalid`, `json_schema_unsupported`, `constraint_too_large`). The returned text is
// the schema document the Engine's XGrammar converter compiles.
std::string convert_json_schema(const std::string& text, std::string_view flag) {
    nlohmann::ordered_json schema;
    try {
        schema = nlohmann::ordered_json::parse(text);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string(flag) + ": json_schema_invalid: " +
                                    "JSON Schema is not valid JSON: " + error.what());
    }
    try {
        return ninfer::constraint::json_schema_constraint_source(schema);
    } catch (const ninfer::constraint::ConstraintError& error) {
        throw std::invalid_argument(std::string(flag) + ": " + error.code() + ": " + error.what());
    }
}

// The prompt body, read verbatim. No escape processing: a backslash is a backslash, so the same
// bytes read here produce exactly the prompt the inline flag would for the identical bytes.
std::string read_prompt_file(const char* path) {
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
        throw std::invalid_argument(std::string("--prompt-file is a directory: ") + path);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::invalid_argument(std::string("--prompt-file cannot open file: ") + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (input.bad()) {
        throw std::invalid_argument(std::string("--prompt-file failed to read file: ") + path);
    }
    return buffer.str();
}

std::string read_prompt_stdin() {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    if (std::cin.bad()) { throw std::invalid_argument("--prompt-stdin failed to read stdin"); }
    return buffer.str();
}

void set_prompt_source(Options& options, Options::PromptSource source) {
    if (options.prompt_source != Options::PromptSource::None) {
        throw std::invalid_argument(
            "pass exactly one prompt source: " +
            std::string(prompt_source_flag(options.prompt_source)) + " conflicts with " +
            std::string(prompt_source_flag(source)));
    }
    options.prompt_source = source;
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "none") { return ReasoningEffort::None; }
    if (text == "minimal") { return ReasoningEffort::Minimal; }
    if (text == "high") { return ReasoningEffort::High; }
    if (text == "max") { return ReasoningEffort::Max; }
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

const char* constraint_error_code(ConstraintSource source) noexcept {
    return source == ConstraintSource::JsonSchema ? "json_schema_invalid" : "grammar_invalid";
}

std::string_view prompt_source_flag(Options::PromptSource source) noexcept {
    switch (source) {
    case Options::PromptSource::Inline: return "--prompt";
    case Options::PromptSource::File: return "--prompt-file";
    case Options::PromptSource::Stdin: return "--prompt-stdin";
    case Options::PromptSource::Messages: return "--messages";
    case Options::PromptSource::None: break;
    }
    return "<none>";
}

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--prompt-file FILE|--prompt-stdin|--messages "
           "<messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--spec mtp|dflash|dflash2 --draft-tokens "
           "N]\n"
           "       [--lm-head-draft]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--chat-template FILE]\n"
           "       [--grammar GBNF|--grammar-file FILE|--json-schema JSON|--json-schema-file "
           "FILE]\n"
           "       [--raw-output] [--print-token-ids] [--show-prompt] [--no-thinking] "
           "[--thinking-budget N]\n"
           "       [--reasoning-effort none|minimal|low|medium|high|xhigh|max] [--vision]\n"
           "       [--no-cuda-graph]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       [--hf-repo OWNER/REPO --hf-file FILE [--hf-revision REV] [--hf-token TOKEN]]\n"
           "       [--model-url https://.../model.ninfer] [--cache-dir DIR] [--offline] "
           "[--cache-list]\n"
           "       [--version]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "--prompt-file FILE and --prompt-stdin read the plain-text prompt verbatim, with no\n"
           "escape processing, so the same bytes give the same tokens as --prompt. Exactly one\n"
           "of --prompt, --prompt-file, --prompt-stdin and --messages is required; naming two\n"
           "is an error rather than a silent winner.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "--show-prompt dumps the exact rendered prompt to stderr before generation; stdout "
           "stays the answer channel.\n"
           "--thinking-budget caps model-origin thinking tokens; inserted control tokens count "
           "toward --max-new.\n"
           "--kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom.\n"
           "--max-context above the model's native position capacity activates the YaRN context "
           "extension (spec defaults); the DFlash draft backend rejects it.\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n"
           "Model acquisition takes exactly one source: positional <model.ninfer>, --hf-repo +\n"
           "--hf-file (cached under --cache-dir, default ~/.cache/frinfer), or --model-url.\n"
           "--hf-revision defaults to main; --hf-token defaults to $HF_TOKEN/\n"
           "$HUGGING_FACE_HUB_TOKEN. --offline reuses the cache without network.\n"
           "--cache-list prints cached .ninfer artifacts and exits without a prompt.\n"
           "Constrained decoding takes exactly one of --grammar (GBNF text), --grammar-file, "
           "--json-schema (a JSON Schema document), or --json-schema-file; the Engine compiles "
           "GBNF and the serve contract converts and validates JSON Schema. A constrained "
           "request resolves thinking off unless --reasoning-effort enables it.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "--version") {
        options.version_requested = true;
        return options;
    }
    int positional_end = 2;
    if (argc >= 2 && !std::string_view(argv[1]).starts_with("--")) {
        options.artifact_path = argv[1];
        positional_end        = 2;
    } else {
        positional_end = 1;
    }
    bool kv_capacity_explicit = false;
    std::optional<std::string> grammar_source_text;
    std::string grammar_flag;
    std::optional<std::string> schema_source_text;
    std::string schema_flag;

    for (int i = positional_end; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            set_prompt_source(options, Options::PromptSource::Inline);
            options.prompt = value(arg);
        } else if (arg == "--prompt-file") {
            set_prompt_source(options, Options::PromptSource::File);
            options.prompt_file_path = value(arg);
        } else if (arg == "--prompt-stdin") {
            set_prompt_source(options, Options::PromptSource::Stdin);
        } else if (arg == "--chat-template") {
            options.chat_template_path = value(arg);
        } else if (arg == "--messages") {
            set_prompt_source(options, Options::PromptSource::Messages);
            options.messages_path = value(arg);
        } else if (arg == "--grammar" || arg == "--grammar-file") {
            if (grammar_source_text) {
                throw std::invalid_argument("--grammar and --grammar-file are mutually exclusive");
            }
            grammar_flag        = std::string(arg);
            grammar_source_text = arg == "--grammar" ? std::string(value(arg))
                                                     : read_constraint_file(value(arg), arg);
        } else if (arg == "--json-schema" || arg == "--json-schema-file") {
            if (schema_source_text) {
                throw std::invalid_argument(
                    "--json-schema and --json-schema-file are mutually exclusive");
            }
            schema_flag        = std::string(arg);
            schema_source_text = arg == "--json-schema" ? std::string(value(arg))
                                                        : read_constraint_file(value(arg), arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--show-prompt") {
            options.show_prompt = true;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > 20) { throw std::invalid_argument("--top-k must be in [0,20]"); }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(value(arg));
        } else if (arg == "--hf-repo") {
            options.acquisition.hf_repo = value(arg);
        } else if (arg == "--hf-file") {
            options.acquisition.hf_file = value(arg);
        } else if (arg == "--hf-revision") {
            options.acquisition.hf_revision = value(arg);
        } else if (arg == "--hf-token") {
            options.acquisition.hf_token = value(arg);
        } else if (arg == "--model-url") {
            options.acquisition.model_url = value(arg);
        } else if (arg == "--cache-dir") {
            options.acquisition.cache_dir = value(arg);
        } else if (arg == "--offline") {
            options.acquisition.offline = true;
        } else if (arg == "--cache-list") {
            options.acquisition.cache_list = true;
        } else if (arg == "--version") {
            options.version_requested = true;
            return options;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }

    const bool has_hf  = options.acquisition.hf_repo.has_value();
    const bool has_url = options.acquisition.model_url.has_value();
    if (has_hf && has_url) {
        throw std::invalid_argument(
            "hf_spec_invalid: --hf-repo and --model-url are mutually exclusive");
    }
    if (options.acquisition.hf_file && !has_hf) {
        throw std::invalid_argument("hf_spec_invalid: --hf-file needs --hf-repo");
    }
    if (options.acquisition.hf_revision && !has_hf) {
        throw std::invalid_argument("hf_spec_invalid: --hf-revision needs --hf-repo");
    }
    if (options.acquisition.hf_token && !has_hf) {
        throw std::invalid_argument("hf_spec_invalid: --hf-token needs --hf-repo");
    }
    if (has_hf && !options.acquisition.hf_file) {
        throw std::invalid_argument("hf_spec_invalid: --hf-repo needs --hf-file");
    }
    if ((has_hf || has_url) && !options.artifact_path.empty()) {
        throw std::invalid_argument(
            "hf_spec_invalid: positional <model.ninfer> and --hf-repo/--model-url "
            "are mutually exclusive");
    }
    if (!has_hf && !has_url && !options.acquisition.cache_list &&
        options.artifact_path.empty()) {
        throw std::invalid_argument(".ninfer model path is required");
    }
    if (!options.acquisition.cache_list) {
        // Exactly one prompt source. This is a precedence rule, not a silent winner: a command that
        // supplies two names both flags in the diagnostic instead of dropping one.
        if (options.prompt_source == Options::PromptSource::None) {
            throw std::invalid_argument(
                "pass exactly one prompt source: --prompt, --prompt-file, --prompt-stdin or "
                "--messages");
        }
        // An empty file or an empty stdin is a supplied prompt, not an absent one; the Engine
        // reports it as a context that cannot hold a turn.
        if (options.prompt_source == Options::PromptSource::File) {
            options.prompt = read_prompt_file(options.prompt_file_path.string().c_str());
        } else if (options.prompt_source == Options::PromptSource::Stdin) {
            options.prompt = read_prompt_stdin();
        }
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.enable_thinking == false && options.reasoning_effort &&
        *options.reasoning_effort != ReasoningEffort::None) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.reasoning_effort == ReasoningEffort::None) options.enable_thinking = false;
    if (options.enable_thinking == false && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }

    if (grammar_source_text && grammar_source_text->empty()) {
        throw std::invalid_argument(std::string(grammar_flag) +
                                    ": grammar_invalid: grammar must not be empty");
    }
    const bool has_grammar = grammar_source_text.has_value();
    const bool has_schema  = schema_source_text.has_value();
    if (has_grammar && has_schema) {
        throw std::invalid_argument(
            "constrained_decoding_conflict: --grammar/--grammar-file cannot be combined with "
            "--json-schema/--json-schema-file");
    }
    if (has_grammar) {
        if (grammar_source_text->size() > ninfer::constraint::kConstraintPayloadLimit) {
            throw std::invalid_argument(
                std::string(grammar_flag) + ": constraint_too_large: grammar exceeds " +
                std::to_string(ninfer::constraint::kConstraintPayloadLimit) + " bytes");
        }
        options.constraint.source = ConstraintSource::Grammar;
        options.constraint.gbnf   = std::move(*grammar_source_text);
    } else if (has_schema) {
        options.constraint.source = ConstraintSource::JsonSchema;
        options.constraint.gbnf   = convert_json_schema(*schema_source_text, schema_flag);
    }

    if (options.constraint.source != ConstraintSource::None) {
        // A backend that cannot carry a constraint is rejected before submission, mirroring the
        // serve policy and code (design #48) instead of the Engine's generic InvalidConstraint.
        const SpeculativeBackend backend = options.speculative.backend;
        if (backend != SpeculativeBackend::None && backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument(
                "constrained_decoding_not_supported: constrained generation is not supported "
                "with the " +
                std::string(product::speculative_backend_name(backend)) +
                " speculative backend; use the ordinary or MTP backend");
        }
        // A constrained request resolves thinking off unless it explicitly enables it through a
        // non-none reasoning effort (the serve's issue #86 rule). The resolved flag drives both
        // prompt rendering and the Engine's reasoning-wrapper selection.
        const bool explicitly_enabled =
            options.enable_thinking.value_or(false) ||
            (options.reasoning_effort && *options.reasoning_effort != ReasoningEffort::None);
        // Match the CLI's own `--no-thinking --thinking-budget` rule: a budget is meaningless when
        // the resolved mode is non-thinking, so reject it here rather than leaving it inert.
        if (!explicitly_enabled && options.thinking_budget) {
            throw std::invalid_argument(
                "--thinking-budget cannot be combined with a constraint that resolves thinking "
                "off; pass --reasoning-effort or drop --thinking-budget");
        }
        options.enable_thinking             = explicitly_enabled;
        options.constraint.thinking_enabled = explicitly_enabled;
    }
    return options;
}

} // namespace ninfer::cli
