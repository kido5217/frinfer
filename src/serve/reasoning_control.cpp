#include "serve/reasoning_control.h"
#include "serve/request_validation.h"

#include <utility>

namespace ninfer::serve {
namespace {

// llama.cpp reads both fields through a defaulting accessor, so an absent field and a wrong-typed
// one are the same empty string. Reproducing that is what keeps the route's rejections identical.
std::string string_field(const RequestJson& body, const char* name) {
    if (!body.is_object() || !body.contains(name) || !body.at(name).is_string()) { return {}; }
    return body.at(name).get<std::string>();
}

} // namespace

ChatControlRequest parse_chat_control_request(const RequestJson& body) {
    ChatControlRequest request{.id = string_field(body, "id"),
                               .action = string_field(body, "action")};
    if (request.id.empty()) { bad_request("missing completion id", "id"); }
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

ReasoningControlRegistration::ReasoningControlRegistration(ReasoningControlRegistry& registry,
                                                           std::string id,
                                                           ninfer::GenerationControl control)
    : registry_(&registry), id_(std::move(id)) {
    registry_->register_completion(id_, std::move(control));
}

ReasoningControlRegistration::~ReasoningControlRegistration() {
    if (registry_ != nullptr) { registry_->unregister_completion(id_); }
}

ReasoningControlRegistration::ReasoningControlRegistration(
    ReasoningControlRegistration&& other) noexcept
    : registry_(std::exchange(other.registry_, nullptr)), id_(std::move(other.id_)) {}

ReasoningControlRegistration&
ReasoningControlRegistration::operator=(ReasoningControlRegistration&& other) noexcept {
    if (this != &other) {
        if (registry_ != nullptr) { registry_->unregister_completion(id_); }
        registry_ = std::exchange(other.registry_, nullptr);
        id_       = std::move(other.id_);
    }
    return *this;
}

std::shared_ptr<ReasoningControlRegistration>
arm_reasoning_control(ReasoningControlRegistry& registry, std::string id,
                      ReasoningControlArming arming) {
    if (!arming.id_live_while_generating) { return nullptr; }
    if (arming.requested && !arming.control.is_valid()) { return nullptr; }
    return std::make_shared<ReasoningControlRegistration>(registry, std::move(id),
                                                          std::move(arming.control));
}

} // namespace ninfer::serve