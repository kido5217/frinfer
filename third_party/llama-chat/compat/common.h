#pragma once

// FrInfer compat shim for llama.cpp's common/common.h.
//
// The vendored chat-parsing sources include "common.h" for a small, fixed slice of
// llama.cpp's common layer. This shim provides exactly that slice so the vendored
// files stay byte-identical to upstream. The llama.cpp runtime API (vocab/model
// backed helpers, legacy template application) has no implementation in the
// minimal port; its stubs in excluded_stubs.cpp throw when reached.

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using llama_token  = std::int32_t;
using llama_tokens = std::vector<llama_token>;

constexpr llama_token LLAMA_TOKEN_NULL = -1;

struct llama_vocab;
struct llama_model;

struct llama_chat_message {
    const char* role;
    const char* content;
};

// common/common.h: grammar triggers (declared there, defined in common/grammar.h upstream)
enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

// common/common.h: reasoning API response format (not to be confused with the
// chat template's reasoning format)
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,
    COMMON_REASONING_FORMAT_DEEPSEEK,
};

// common/common.h: string helpers (same definitions as upstream, kept inline)
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// common/common.h: string helpers (definitions in common_helpers.cpp, matching upstream)
std::string string_join(const std::vector<std::string>& values, const std::string& separator);
std::vector<std::string> string_split(const std::string& str, const std::string& delimiter);
std::string string_repeat(const std::string& str, std::size_t n);
void string_replace_all(std::string& s, const std::string& search, const std::string& replace);

// llama.cpp runtime API used by the vendored chat layer (stubbed, see excluded_stubs.cpp)
const struct llama_vocab* llama_model_get_vocab(const struct llama_model* model);
const char* llama_model_chat_template(const struct llama_model* model, const char* name);
llama_token llama_vocab_bos(const struct llama_vocab* vocab);
llama_token llama_vocab_eos(const struct llama_vocab* vocab);
bool llama_vocab_get_add_bos(const struct llama_vocab* vocab);
bool llama_vocab_get_add_eos(const struct llama_vocab* vocab);

int32_t llama_chat_apply_template(const char* tmpl, const struct llama_chat_message* chat,
                                  std::size_t n_msg, bool add_ass, char* buf, int32_t length);

std::vector<llama_token> common_tokenize(const struct llama_vocab* vocab, const std::string& text,
                                         bool add_special, bool parse_special = false);

std::string common_token_to_piece(const struct llama_vocab* vocab, llama_token token,
                                  bool special = true);
