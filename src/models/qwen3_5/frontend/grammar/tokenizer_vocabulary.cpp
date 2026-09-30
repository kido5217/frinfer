#include "models/qwen3_5/frontend/grammar/tokenizer_vocabulary.h"

#include "models/qwen3_5/frontend/tokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {

TokenizerVocabulary::TokenizerVocabulary(std::shared_ptr<const Tokenizer> tokenizer,
                                         std::vector<int> additional_eog_ids)
    : tokenizer_(std::move(tokenizer)) {
    if (!tokenizer_) { throw std::invalid_argument("TokenizerVocabulary requires a tokenizer"); }
    eog_ids_ = tokenizer_->default_stop_token_ids();
    for (const int id : additional_eog_ids) {
        if (id >= 0 && std::find(eog_ids_.begin(), eog_ids_.end(), id) == eog_ids_.end()) {
            eog_ids_.push_back(id);
        }
    }
}

int TokenizerVocabulary::token_count() const noexcept {
    return static_cast<int>(tokenizer_->vocab_size());
}

std::string_view TokenizerVocabulary::piece(int id) const {
    if (!tokenizer_->is_valid_token(id)) { return {}; }
    return tokenizer_->decode_token_bytes(id, false);
}

bool TokenizerVocabulary::is_eog(int id) const {
    return std::find(eog_ids_.begin(), eog_ids_.end(), id) != eog_ids_.end();
}

std::vector<int> TokenizerVocabulary::encode(std::string_view text) const {
    return tokenizer_->encode(text, EncodeOptions{.parse_added_tokens = true});
}

} // namespace ninfer::models::qwen3_5::frontend
