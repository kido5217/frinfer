#include "serve/tool_call_signal.h"

#include <string>

namespace ninfer::serve {

bool tool_call_demotion_signals(ninfer::ToolCallParseFallbackReason reason,
                                bool call_attempted) noexcept {
    return reason != ninfer::ToolCallParseFallbackReason::None && call_attempted;
}

std::string_view tool_call_demotion_code(ninfer::ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ninfer::ToolCallParseFallbackReason::MalformedStructure:
        return "tool_call_malformed_structure";
    case ninfer::ToolCallParseFallbackReason::DuplicateParameter:
        return "tool_call_duplicate_parameter";
    case ninfer::ToolCallParseFallbackReason::InvalidToolName:
        return "tool_call_invalid_tool_name";
    case ninfer::ToolCallParseFallbackReason::UndeclaredTool:
        return "tool_call_undeclared_tool";
    case ninfer::ToolCallParseFallbackReason::TrailingContent:
        return "tool_call_trailing_content";
    case ninfer::ToolCallParseFallbackReason::None:
        break;
    }
    return {};
}

namespace {

std::string_view demotion_detail(ninfer::ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ninfer::ToolCallParseFallbackReason::MalformedStructure:
        return "malformed structure";
    case ninfer::ToolCallParseFallbackReason::DuplicateParameter:
        return "a conflicting duplicate parameter";
    case ninfer::ToolCallParseFallbackReason::InvalidToolName:
        return "an invalid tool name";
    case ninfer::ToolCallParseFallbackReason::UndeclaredTool:
        return "an undeclared tool name";
    case ninfer::ToolCallParseFallbackReason::TrailingContent:
        return "unexpected trailing content";
    case ninfer::ToolCallParseFallbackReason::None:
        break;
    }
    return "an unknown reason";
}

} // namespace

std::optional<ApiError> tool_call_demotion_signal(ninfer::ToolCallParseFallbackReason reason,
                                                  bool call_attempted) {
    if (!tool_call_demotion_signals(reason, call_attempted)) { return std::nullopt; }
    return tool_call_demotion_error(reason);
}

ApiError tool_call_demotion_error(ninfer::ToolCallParseFallbackReason reason) {
    ApiError error;
    error.status  = 409;
    // `tool_call_error` is deliberately not one of the client's recognized error codes: opencode
    // scans the error object's `type` as a code, and a recognized invalid-request code would
    // classify the in-band error as a non-retryable InvalidRequest (see the header contract).
    error.type    = "tool_call_error";
    error.code    = std::string(tool_call_demotion_code(reason));
    error.message = "the model's tool call was not delivered (" +
                    std::string(demotion_detail(reason)) + "); the request can be retried";
    return error;
}

} // namespace ninfer::serve
