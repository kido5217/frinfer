#pragma once

// Real-time reasoning control for in-flight chat completions. The registry is the only mutable
// state: a chat completion registers its Engine control surface for exactly as long as its HTTP
// response lives, and the control route looks the completion id up to arm or reject the request.

#include "ninfer/engine.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ninfer::serve {

struct ChatControlRequest {
    std::string id;
    std::string action;
};

// Rejects a missing `id` or an action other than `reasoning_end`.
ChatControlRequest parse_chat_control_request(const RequestJson& body);

// The control route's answer: `success` plus the reason whenever it is false.
std::string make_chat_control_response(bool success, std::string message = {});

struct ChatControlOutcome {
    bool success = false;
    std::string message;
};

// Live chat completions that can be acted on. An absent id, or one whose completion already
// finished, matches nothing and reports failure without touching the Engine.
class ReasoningControlRegistry {
public:
    // Arms the completion's reasoning close. `control` is empty for a completion that did not opt
    // in, which stays registered so the route can name that reason instead of "no live completion".
    ChatControlOutcome request_reasoning_end(const std::string& id);

    void register_completion(std::string id, ninfer::GenerationControl control);
    void unregister_completion(const std::string& id);

    [[nodiscard]] std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ninfer::GenerationControl> completions_;
};

} // namespace ninfer::serve
