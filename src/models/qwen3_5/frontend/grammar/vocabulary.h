#pragma once

#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// The vocabulary as the GBNF runtime sees it: raw piece bytes per token id, the
// end-of-generation set, and the encoder used by `<token>` GBNF literals. The interface is
// host-only and free of engine types, so callers can supply synthetic vocabularies (tests)
// while the model path uses the frontend tokenizer's data (TokenizerVocabulary).
class GrammarVocabulary {
public:
    GrammarVocabulary(const GrammarVocabulary&)            = delete;
    GrammarVocabulary& operator=(const GrammarVocabulary&) = delete;
    virtual ~GrammarVocabulary()                           = default;

    // Number of token ids in the domain; ids are [0, token_count()). The mask rows of a
    // compiled grammar cover exactly this domain.
    [[nodiscard]] virtual int token_count() const noexcept = 0;

    // Raw piece bytes of `id`, empty when the id carries no piece (the grammar filter
    // rejects empty pieces, matching llama.cpp). The returned view must stay valid for the
    // lifetime of this vocabulary and must not be mutated while any compiled grammar or
    // grammar state references it.
    [[nodiscard]] virtual std::string_view piece(int id) const = 0;

    // True when `id` ends generation for the constraining request (model end-of-generation
    // ids plus any request stop ids the caller folded into the view).
    [[nodiscard]] virtual bool is_eog(int id) const = 0;

    // Encodes literal text for a `<token>` GBNF terminal. The grammar runtime requires
    // exactly one id; any other count is a grammar error.
    [[nodiscard]] virtual std::vector<int> encode(std::string_view text) const = 0;

protected:
    GrammarVocabulary() = default;
};

} // namespace ninfer::models::qwen3_5::frontend
