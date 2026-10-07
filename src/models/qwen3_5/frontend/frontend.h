#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/registry.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

[[nodiscard]] ModelSamplingDefaults default_sampling(Architecture architecture);

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    std::size_t grammar_cache_bytes        = 256ULL * 1024 * 1024;
    Architecture architecture              = Architecture::Qwen3_5;
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] std::string_view rendered_text() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class Frontend {
public:
    // Constraint compilation scope. `Answer` compiles the grammar as the answer grammar (the
    // pre-wrapper behavior). `Thinking` first wraps the grammar in the family's full-stream
    // thinking wrapper - a marker-avoiding reasoning body, the wire-format close marker, forced
    // format whitespace, then the answer grammar - so a request whose rendered prompt starts in
    // reasoning is constrained from token 0 and hands off at the close.
    enum class ConstraintScope : std::uint8_t {
        Answer,
        Thinking,
    };

    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    // Artifact-tokenizer decoding, the inverse of tokenize_text up to the NFC normalisation
    // tokenize_text applies. `skip_special_tokens` drops control-token spellings, which is what a
    // caller rendering generated text wants and what a caller auditing a prompt's framing does not.
    [[nodiscard]] std::string detokenize(std::span<const TokenId> ids,
                                         bool skip_special_tokens = false) const;
    // One id's vocabulary spelling and whether it is a special token. The token must be in the
    // checkpoint vocabulary.
    [[nodiscard]] TokenPiece token_piece(TokenId id) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output               = {},
                        const ThinkingControlOptions& thinking    = {},
                        const std::optional<std::string>& grammar = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;

    // Grammar constraints: the shared grammar vocabulary over this frontend's tokenizer (built on
    // first use) and the compiled grammars keyed by GBNF text. Compiled grammars and their row
    // caches are shared by grammar identity for as long as a caller holds them; compiling invalid
    // text returns nullptr and, when `error` is non-null, stores the parse diagnostic. The
    // protocol adapter converts JSON Schema documents before calling here.
    //
    // Safe to call from concurrent request threads: the lazy caches are serialized. Row
    // production against a compiled grammar still belongs to the single serving worker.
    [[nodiscard]] std::shared_ptr<const frontend::GrammarVocabulary> grammar_vocabulary() const;
    [[nodiscard]] std::shared_ptr<const frontend::CompiledGrammar>
    compile_grammar(std::string_view gbnf, ConstraintScope scope, std::string* error = nullptr) const;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::models::qwen3_5
