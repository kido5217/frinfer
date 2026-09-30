#pragma once

#include "models/qwen3_5/frontend/grammar/vocabulary.h"

#include <memory>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;

// GrammarVocabulary over the frontend tokenizer's loaded data: the raw decoded piece bytes
// (byte-level BPE pieces are decoded to bytes at load, added tokens keep their literal
// text), the model end-of-generation ids plus any request stop ids folded in through
// `additional_eog_ids`, and the tokenizer's encoder for `<token>` GBNF literals.
//
// This is the production construction path of the grammar module; synthetic vocabularies
// implement GrammarVocabulary directly. The instance must outlive every compiled grammar
// that references it.
class TokenizerVocabulary final : public GrammarVocabulary {
public:
    explicit TokenizerVocabulary(std::shared_ptr<const Tokenizer> tokenizer,
                                 std::vector<int> additional_eog_ids = {});

    [[nodiscard]] int token_count() const noexcept override;
    [[nodiscard]] std::string_view piece(int id) const override;
    [[nodiscard]] bool is_eog(int id) const override;
    [[nodiscard]] std::vector<int> encode(std::string_view text) const override;

private:
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::vector<int> eog_ids_;
};

} // namespace ninfer::models::qwen3_5::frontend
