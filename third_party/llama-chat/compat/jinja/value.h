#pragma once

// FrInfer compat shim for the maintained jinja fork's jinja/value.h.
//
// Upstream's jinja/value.h declares jinja::global_from_json (a context seeder used by
// chat.cpp's render path); the fork vendored at third_party/llama-jinja does not have
// it yet. This wrapper re-exposes the fork's header and adds the declaration; the
// implementation is compat/compat.cpp.

#include "../../../llama-jinja/jinja/value.h"

#include "json.h"

namespace jinja {

void global_from_json(context& ctx, const common_json& json_obj, bool mark_input);

} // namespace jinja
