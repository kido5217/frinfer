// NInfer compat: definitions for symbols the vendored chat layer references but whose
// upstream implementations sit in subsystems the minimal port deliberately excludes
// (see README.ninfer.md; design ticket #28: no differential auto-parser, no other
// specialized parsers, no GGUF/llama runtime init, no GBNF-driven sampling plumbing).
//
// Excluded entry points throw, so reaching one is loud, never silent.

#include "chat-auto-parser.h"
#include "common.h"
#include "parsers/parsers.h"

#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void throw_excluded(const char* what) {
    throw std::runtime_error(std::string("llama-chat minimal port: ") + what +
                             " is not part of the vendored parser-only subset");
}

} // namespace

// clang-format off

// --- llama.cpp runtime API (upstream: llama.cpp + common/common.cpp) ----------

const struct llama_vocab * llama_model_get_vocab(const struct llama_model *) {
    throw_excluded("llama_model_get_vocab");
}
const char * llama_model_chat_template(const struct llama_model *, const char *) {
    throw_excluded("llama_model_chat_template");
}
llama_token llama_vocab_bos(const struct llama_vocab *) {
    throw_excluded("llama_vocab_bos");
}
llama_token llama_vocab_eos(const struct llama_vocab *) {
    throw_excluded("llama_vocab_eos");
}
bool llama_vocab_get_add_bos(const struct llama_vocab *) {
    throw_excluded("llama_vocab_get_add_bos");
}
bool llama_vocab_get_add_eos(const struct llama_vocab *) {
    throw_excluded("llama_vocab_get_add_eos");
}
int32_t llama_chat_apply_template(const char *, const struct llama_chat_message *, std::size_t, bool, char *, int32_t) {
    throw_excluded("llama_chat_apply_template");
}
std::vector<llama_token> common_tokenize(const struct llama_vocab *, const std::string &, bool, bool) {
    throw_excluded("common_tokenize");
}
std::string common_token_to_piece(const struct llama_vocab *, llama_token, bool) {
    throw_excluded("common_token_to_piece");
}

// --- other specialized parsers (upstream: common/parsers/*.cpp) ---------------

bool is_lfm2_template(const std::string & /*src*/) {
    // LFM2 is out of the minimal port; no supported template is an LFM2 template.
    return false;
}

namespace workaround {
void convert_tool_responses_gemma4(json & /*messages*/) {
    throw_excluded("workaround::convert_tool_responses_gemma4");
}
}  // namespace workaround

#define NINFER_EXCLUDED_PARSER(name)                                                        \
    common_chat_params common_chat_params_init_##name(const common_chat_template &,         \
                                                      const autoparser::generation_params &) { \
        throw_excluded("common_chat_params_init_" #name);                                   \
    }

NINFER_EXCLUDED_PARSER(cohere2moe)
NINFER_EXCLUDED_PARSER(deepseek_v3_2)
NINFER_EXCLUDED_PARSER(functionary_v3_2)
NINFER_EXCLUDED_PARSER(gemma4)
NINFER_EXCLUDED_PARSER(gigachat_v3)
NINFER_EXCLUDED_PARSER(gpt_oss)
NINFER_EXCLUDED_PARSER(kimi_k2)
NINFER_EXCLUDED_PARSER(kimi_k3)
NINFER_EXCLUDED_PARSER(ling3)
NINFER_EXCLUDED_PARSER(minicpm5)
NINFER_EXCLUDED_PARSER(minimax_m3)
NINFER_EXCLUDED_PARSER(ministral_3)
NINFER_EXCLUDED_PARSER(muse_glimmer)

#undef NINFER_EXCLUDED_PARSER

common_chat_params common_chat_params_init_lfm2(const common_chat_template &,
                                                const autoparser::generation_params &,
                                                bool /*tool_list_tokens*/) {
    throw_excluded("common_chat_params_init_lfm2");
}

// --- differential auto-parser (upstream: chat-diff-analyzer.cpp,
//     chat-auto-parser-generator.cpp, chat-auto-parser-helpers.cpp) -------------

namespace autoparser {

void autoparser::analyze_template(const common_chat_template & /*tmpl*/) {
    throw_excluded("autoparser::analyze_template");
}

common_chat_params peg_generator::generate_parser(const common_chat_template &,
                                                  const generation_params &,
                                                  const autoparser &) {
    throw_excluded("autoparser::peg_generator::generate_parser");
}

// Virtual members: anchor the vtables so default-constructed autoparser members link.
common_peg_parser analyze_reasoning::build_parser(parser_build_context &) const {
    throw_excluded("autoparser::analyze_reasoning::build_parser");
}

common_peg_parser analyze_content::build_parser(parser_build_context &) const {
    throw_excluded("autoparser::analyze_content::build_parser");
}

common_peg_parser analyze_tools::build_parser(parser_build_context &) const {
    throw_excluded("autoparser::analyze_tools::build_parser");
}

}  // namespace autoparser

// clang-format on
