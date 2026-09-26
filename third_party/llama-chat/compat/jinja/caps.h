#pragma once

// NInfer compat shim for llama.cpp's common/jinja/caps.h.
//
// The maintained jinja fork vendored at third_party/llama-jinja (baseline 76098465)
// predates upstream's capability-probing layer (caps.cpp at 95887577). The minimal
// chat-parsing port excludes probing by design; this header provides the type and
// entry points chat.h/chat.cpp expect. caps_get() returns upstream's struct defaults
// instead of executing the template, and the two apply functions set the same context
// variables upstream sets (they are cheap and semantics-preserving).

#include "jinja/runtime.h"

#include <map>
#include <string>

namespace jinja {

struct caps {
    bool supports_tools               = true;
    bool supports_tool_calls          = true;
    bool supports_system_role         = true;
    bool supports_parallel_tool_calls = true;

    // supports preserve reasoning trace in the full history, not just the last assistant message
    bool supports_preserve_reasoning = false;

    // supports reasoning effort levels
    bool supports_reasoning_effort = false;

    // one of the 2 content capabilities must be true
    bool supports_string_content = true;
    bool supports_typed_content  = false;

    bool supports_object_arguments = false;

    // for reporting to server /props
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program& prog);

void caps_apply_preserve_reasoning(jinja::context& ctx, bool enabled);
void caps_apply_reasoning_effort(jinja::context& ctx, const std::string& effort);

} // namespace jinja
