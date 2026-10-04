#include "options.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

std::string rejection_message(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

std::filesystem::path write_temp(std::string_view name, std::string_view content) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / std::string(name);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << content;
    output.close();
    return path;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    // --- Constrained decoding ---
    const std::string grammar = "root ::= \"ok\"";
    const ninfer::cli::Options grammar_options =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar", grammar});
    failures += check(grammar_options.constraint.source == ninfer::cli::ConstraintSource::Grammar &&
                          grammar_options.constraint.gbnf == grammar &&
                          !grammar_options.constraint.thinking_enabled,
                      "--grammar did not select an answer-only GBNF constraint");
    for (const char* flag :
         {"--grammar", "--grammar-file", "--json-schema", "--json-schema-file"}) {
        failures += check(help.find(flag) != std::string::npos,
                          "CLI help omits a constrained-decoding flag");
    }
    const std::string conflict = rejection_message([&] {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar", grammar,
                     "--json-schema", "{}"});
    });
    failures += check(conflict.find("constrained_decoding_conflict") != std::string::npos,
                      "a grammar/schema conflict omitted constrained_decoding_conflict");
    const std::string too_large = rejection_message([&] {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar",
                     std::string(70U * 1024U, ' ')});
    });
    failures += check(too_large.find("constraint_too_large") != std::string::npos,
                      "an oversized grammar omitted constraint_too_large");
    failures += check(rejects([&] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--json-schema", "{}", "--json-schema-file", "x.json"});
                      }),
                      "CLI accepted both --json-schema and --json-schema-file");
    const std::string dflash_message = rejection_message([&] {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar", grammar,
                     "--spec", "dflash", "--draft-tokens", "7"});
    });
    failures +=
        check(dflash_message.find("constrained_decoding_not_supported") != std::string::npos,
              "a constrained DFlash request omitted the serve's fail-closed code");

    const ninfer::cli::Options thinking_options =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar", grammar,
               "--reasoning-effort", "medium"});
    failures += check(thinking_options.constraint.thinking_enabled &&
                          thinking_options.enable_thinking.value_or(false),
                      "a constrained request with reasoning effort did not resolve thinking on");

    constexpr std::string_view kSchema =
        R"({"type":"object","additionalProperties":false,"required":["answer"],"properties":{"answer":{"type":"number"}}})";
    const ninfer::cli::Options schema_options = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--json-schema", std::string(kSchema)});
    failures +=
        check(schema_options.constraint.source == ninfer::cli::ConstraintSource::JsonSchema &&
                  !schema_options.constraint.gbnf.empty(),
              "--json-schema did not convert to a GBNF constraint");
    const std::string bad_json = rejection_message([&] {
        (void)parse(
            {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--json-schema", "not json"});
    });
    failures += check(bad_json.find("json_schema_invalid") != std::string::npos,
                      "a malformed JSON Schema omitted json_schema_invalid");
    const std::string unsupported = rejection_message([&] {
        (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--json-schema",
                     R"({"type":"string","pattern":"x"})"});
    });
    failures += check(unsupported.find("json_schema_unsupported") != std::string::npos,
                      "an unsupported JSON Schema keyword omitted json_schema_unsupported");
    const std::string empty_grammar = rejection_message(
        [&] { (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar", ""}); });
    failures += check(empty_grammar.find("grammar_invalid") != std::string::npos,
                      "an empty --grammar omitted grammar_invalid");

    const std::filesystem::path grammar_path = write_temp("ninfer_cli_constraint.gbnf", grammar);
    const std::filesystem::path schema_path  = write_temp("ninfer_cli_constraint.json", kSchema);
    const ninfer::cli::Options grammar_file =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--grammar-file",
               grammar_path.string()});
    failures += check(grammar_file.constraint.source == ninfer::cli::ConstraintSource::Grammar &&
                          grammar_file.constraint.gbnf == grammar,
                      "--grammar-file did not read the GBNF text");
    const ninfer::cli::Options schema_file =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--json-schema-file",
               schema_path.string()});
    failures += check(schema_file.constraint.source == ninfer::cli::ConstraintSource::JsonSchema &&
                          !schema_file.constraint.gbnf.empty(),
                      "--json-schema-file did not convert the schema");
    std::error_code ignored;
    std::filesystem::remove(grammar_path, ignored);
    std::filesystem::remove(schema_path, ignored);
    return failures == 0 ? 0 : 1;
}
