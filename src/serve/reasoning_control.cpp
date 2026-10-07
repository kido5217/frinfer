#include "serve/reasoning_control.h"
#include "serve/request_validation.h"

#include <utility>

namespace ninfer::serve {

ChatControlRequest parse_chat_control_request(const RequestJson& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    if (!body.contains("id") || !body.at("id").is_string() ||
        body.at("id").get_ref<const std::string&>().find_first_not_of(" \t\r\n") ==
            std::string::npos) {
        // A blank id can never name a completion; naming it "missing" keeps the caller from reading
        // "no active completion" for an obviously malformed body.
        bad_request("missing completion id", "id");
    }
    ChatControlRequest request{.id = body.at("id").get<std::string>()};
    if (!body.contains("action") || !body.at("action").is_string()) {
        bad_request("action must be a string", "action");
    }
    request.action = body.at("action").get<std::string>();
    if (request.action != "reasoning_end") { bad_request("unknown control action", "action"); }
    return request;
}

std::string make_chat_control_response(bool success, std::string message) {
    RequestJson response{{"success", success}};
    if (!message.empty()) { response["message"] = std::move(message); }
    return response.dump();
}

ChatControlOutcome ReasoningControlRegistry::request_reasoning_end(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto entry = completions_.find(id);
    if (entry == completions_.end()) {
        return {.success = false, .message = "no active completion for this id"};
    }
    if (!entry->second.is_valid()) {
        return {.success = false, .message = "reasoning control not enabled for this completion"};
    }
    // The Engine decides at its next decode boundary whether the close is admissible, exactly as it
    // does for a budget boundary: a request no longer in reasoning, or one whose remaining output
    // budget cannot fit the control span, ignores the signal.
    entry->second.end_reasoning();
    return {.success = true};
}

void ReasoningControlRegistry::register_completion(std::string id,
                                                   ninfer::GenerationControl control) {
    std::lock_guard lock(mutex_);
    completions_.insert_or_assign(std::move(id), std::move(control));
}

void ReasoningControlRegistry::unregister_completion(const std::string& id) {
    std::lock_guard lock(mutex_);
    completions_.erase(id);
}

std::size_t ReasoningControlRegistry::size() const {
    std::lock_guard lock(mutex_);
    return completions_.size();
}

} // namespace ninfer::serve
