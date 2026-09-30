#pragma once
// Vocab adapter for the mask-cost spike: exactly the three calls
// upstream/llama-grammar.cpp makes on llama_vocab, backed by a token-piece dump
// (dump_vocab.py output) and an EOG id list. Mirrors the NInfer mapping in
// docs/research/constrained-decoding-sources.md section 1.5.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct llama_vocab {
    std::vector<std::string> pieces; // pieces[id] = raw bytes; empty = no legitimate piece
    std::vector<bool> eog;           // eog[id] = end-of-generation id

    const std::string& token_to_piece(llama_token id) const { return pieces[id]; }

    bool is_eog(llama_token id) const {
        return id >= 0 && static_cast<size_t>(id) < eog.size() && eog[id];
    }

    // Only reachable through `<token>` literals (llama-grammar.cpp parse_token);
    // the spike fixtures do not use them.
    int32_t tokenize(const char* text, int32_t text_len, llama_token* tokens, int32_t n_tokens_max,
                     bool add_special, bool parse_special) const {
        (void)text;
        (void)text_len;
        (void)tokens;
        (void)n_tokens_max;
        (void)add_special;
        (void)parse_special;
        std::fprintf(stderr, "mask-cost shim: tokenize() is not backed in this spike\n");
        std::abort();
    }
};
