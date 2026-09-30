#pragma once

// NInfer compat shim for llama.cpp's src/llama-vocab.h, reduced to the three llama_vocab
// calls llama-grammar.cpp makes: token_to_piece (apply/accept), is_eog (apply/accept) and
// tokenize (only reachable through `<token>` GBNF literals). The facade is a plain callback
// table; the NInfer grammar module
// (src/models/qwen3_5/frontend/grammar/grammar.cpp) backs it with its decoupled
// GrammarVocabulary view, so the vendored grammar logic stays untouched and the vocabulary
// is the frontend tokenizer's data. See README.ninfer.md.

#include "llama.h"

#include <cstdint>
#include <string>

// Callback shapes for the backing vocabulary. `context` is the owner's opaque pointer.
// token_to_piece returns a reference that must stay stable at least until the next call.
using llama_vocab_piece_fn    = const std::string& (*)(void* context, llama_token id);
using llama_vocab_eog_fn      = bool (*)(void* context, llama_token id);
using llama_vocab_tokenize_fn = int32_t (*)(void* context, const char* text, int32_t text_len,
                                            llama_token* tokens, int32_t n_tokens_max,
                                            bool add_special, bool parse_special);

struct llama_vocab {
    void* context                       = nullptr;
    llama_vocab_piece_fn to_piece       = nullptr;
    llama_vocab_eog_fn to_eog           = nullptr;
    llama_vocab_tokenize_fn to_tokenize = nullptr;

    const std::string& token_to_piece(llama_token id) const { return to_piece(context, id); }

    bool is_eog(llama_token id) const { return to_eog(context, id); }

    int32_t tokenize(const char* text, int32_t text_len, llama_token* tokens, int32_t n_tokens_max,
                     bool add_special, bool parse_special) const {
        return to_tokenize(context, text, text_len, tokens, n_tokens_max, add_special,
                           parse_special);
    }
};
