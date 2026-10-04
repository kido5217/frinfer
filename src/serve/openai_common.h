#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

// Serializes an OpenAI wire object. Token logprob entries carry tokenizer byte strings, which can
// be partial UTF-8 for a byte-level BPE token; nlohmann's default dump throws on those, so the
// replacement handler is used (the exact bytes remain available in the `bytes` array).
inline std::string dump_openai_json(const nlohmann::json& payload) {
    return payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             std::uint32_t max_model_len);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

void validate_openai_model(std::string_view requested, std::string_view available);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
