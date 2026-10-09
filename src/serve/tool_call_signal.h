#pragma once

// ADR-0002 serve-side signaling contract for demoted tool-call regions: a residual demotion that
// lost a call (fallback_reason != None && call_attempted) becomes a client-visible, retryable
// error instead of markup delivered as assistant text. Prose-only demotions stay silent
// byte-exact round-trips. The class codes are string codes, and the error `type` is kept outside
// the client's recognized code sets: on the opencode client an in-band SSE error whose code and
// type are unrecognized classifies as UnknownProvider, which is retryable; the non-streaming
// response is HTTP 409 with `x-should-retry: true`, both of which the client classifies as
// retryable.

#include "serve/request.h"

#include <ninfer/types.h>

#include <optional>
#include <string_view>

namespace ninfer::serve {

// True when the demotion lost an attempted call and must be signaled.
[[nodiscard]] bool tool_call_demotion_signals(ninfer::ToolCallParseFallbackReason reason,
                                              bool call_attempted) noexcept;

// The wire code for one demotion class (`tool_call_malformed_structure`, ...); empty for `None`.
[[nodiscard]] std::string_view
tool_call_demotion_code(ninfer::ToolCallParseFallbackReason reason) noexcept;

// The error a signaled demotion renders: status 409, the class code, and a message that avoids
// every classification phrase that would disable the client's retry.
[[nodiscard]] ApiError tool_call_demotion_error(ninfer::ToolCallParseFallbackReason reason);

// The retryable error a demotion must signal, or std::nullopt when the demotion is silent and its
// text should be delivered as ordinary content. Implements the ADR-0002 rule (a demotion that lost
// an attempted call signals; a prose-only demotion is silent).
[[nodiscard]] std::optional<ApiError> tool_call_demotion_signal(
    ninfer::ToolCallParseFallbackReason reason, bool call_attempted);

} // namespace ninfer::serve
