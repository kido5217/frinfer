#pragma once

// Real-time reasoning control for in-flight chat completions. The registry is the only mutable
// state: a chat completion registers its Engine control surface for exactly as long as its response
// lives, and the control route looks the completion id up to arm or reject the request.

#include "ninfer/engine.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace ninfer::serve {

struct ChatControlRequest {
    std::string id;
    std::string action;
};

// Rejects a body that cannot name a completion or cannot name a supported action. The two
// rejections are the ones llama.cpp makes at this route, with its messages: an `id` that is absent
// or empty is "missing completion id", and any `action` that is not exactly `reasoning_end` —
// absent, not a string, or an unknown name — is "unknown control action". A non-empty id that names
// nothing is deliberately not a rejection: it is a lookup miss the route answers 200, so a client
// that padded or mistyped an id gets the same readable answer as one that raced the completion.
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

// Owns one registry entry for exactly its own lifetime. This is the whole release rule: the control
// route is reachable while a response that can still be read is alive, so holding this handle is
// what makes a completion controllable, and destroying it is what stops that.
class ReasoningControlRegistration {
public:
    ReasoningControlRegistration(ReasoningControlRegistry& registry, std::string id,
                                 ninfer::GenerationControl control);
    ~ReasoningControlRegistration();

    ReasoningControlRegistration(ReasoningControlRegistration&&) noexcept;
    ReasoningControlRegistration& operator=(ReasoningControlRegistration&&) noexcept;
    ReasoningControlRegistration(const ReasoningControlRegistration&)            = delete;
    ReasoningControlRegistration& operator=(const ReasoningControlRegistration&) = delete;

    [[nodiscard]] const std::string& id() const noexcept { return id_; }

private:
    ReasoningControlRegistry* registry_ = nullptr;
    std::string id_;
};

// What a request handler knows about one completion's control surface.
struct ReasoningControlArming {
    // Valid only when the Engine will actually consume a signal for this request. A request that
    // never entered the Engine has no decode boundary to consume one, so its control is empty.
    ninfer::GenerationControl control;
    // The request asked for `reasoning_control`.
    bool requested = false;
    // The client can see the completion id before generation ends, which is the streaming case.
    bool id_live_while_generating = false;
};

// The publication policy, kept out of the request handler so it is verifiable on its own.
//
// The control route can only reach a completion the client can still be reading, so an aggregate
// completion is never registered: its id arrives in the response that ends generation, and
// registering it would leave a completion reachable after it is over.
//
// A request that asked for control but whose generation ended before any boundary could consume a
// signal is likewise not registered, and reports the same "no active completion" as an unknown id —
// which is true. Saying "not enabled" there would be a false claim about a request that did enable
// it.
//
// A live request that did not ask for control is still registered, with an empty control, so the
// route can name that reason instead of claiming no completion exists.
//
// A registration is handed back as a shared_ptr because the streaming response outlives the handler
// that armed it.
[[nodiscard]] std::shared_ptr<ReasoningControlRegistration>
arm_reasoning_control(ReasoningControlRegistry& registry, std::string id,
                      ReasoningControlArming arming);

} // namespace ninfer::serve