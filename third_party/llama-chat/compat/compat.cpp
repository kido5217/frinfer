// NInfer compat implementations for the two jinja-facing pieces the vendored chat
// layer expects but the maintained jinja fork (third_party/llama-jinja, baseline
// 76098465) does not provide: the capability struct and jinja::global_from_json.
// See compat/jinja/caps.h and compat/jinja/value.h for the rationale.

#include "jinja/caps.h"
#include "jinja/value.h"

#include "json.h"

#include <stdexcept>

namespace jinja {

std::map<std::string, bool> caps::to_map() const {
    return {
        {"supports_tools", supports_tools},
        {"supports_tool_calls", supports_tool_calls},
        {"supports_system_role", supports_system_role},
        {"supports_parallel_tool_calls", supports_parallel_tool_calls},
        {"supports_preserve_reasoning", supports_preserve_reasoning},
        {"supports_reasoning_effort", supports_reasoning_effort},
        {"supports_string_content", supports_string_content},
        {"supports_typed_content", supports_typed_content},
        {"supports_object_arguments", supports_object_arguments},
    };
}

std::string caps::to_string() const {
    std::string out;
    for (const auto& [key, value] : to_map()) {
        out += key + ": " + (value ? "true" : "false") + "\n";
    }
    return out;
}

caps caps_get(jinja::program& /*prog*/) {
    // The minimal port does not probe templates; return upstream's struct defaults.
    return caps();
}

void caps_apply_preserve_reasoning(jinja::context& ctx, bool enabled) {
    ctx.set_val("preserve_thinking", mk_val<value_bool>(enabled));
    ctx.set_val("clear_thinking", mk_val<value_bool>(!enabled));
    ctx.set_val("truncate_history_thinking", mk_val<value_bool>(!enabled));
    ctx.set_val("drop_thinking", mk_val<value_bool>(!enabled));
}

void caps_apply_reasoning_effort(jinja::context& ctx, const std::string& effort) {
    value var = mk_val<value_string>(effort); // bind to the same value for stats
    ctx.set_val("reasoning_effort", var);
    ctx.set_val("reasoning_strength", var);
}

static value from_json(const common_json& j, bool mark_input) {
    if (j.is_null()) {
        return mk_val<value_none>();
    } else if (j.is_boolean()) {
        return mk_val<value_bool>(j.get<bool>());
    } else if (j.is_number_integer()) {
        return mk_val<value_int>(j.get<std::int64_t>());
    } else if (j.is_number_float()) {
        return mk_val<value_float>(j.get<double>());
    } else if (j.is_string()) {
        // The fork marks request-supplied bytes through jinja::string's literal flag
        // (mirrors upstream's value_string_t::mark_input()).
        return mk_val<value_string>(string(j.get<std::string>(), 0, mark_input));
    } else if (j.is_array()) {
        auto arr = mk_val<value_array>();
        for (const auto& item : j) { arr->push_back(from_json(item, mark_input)); }
        return arr;
    } else if (j.is_object()) {
        auto obj = mk_val<value_object>();
        for (auto it = j.begin(); it != j.end(); ++it) {
            obj->insert(it.key(), from_json(it.value(), mark_input));
        }
        return obj;
    }
    throw std::runtime_error("global_from_json: unsupported JSON value type");
}

void global_from_json(context& ctx, const common_json& json_obj, bool mark_input) {
    if (json_obj.is_null() || !json_obj.is_object()) {
        throw std::runtime_error("global_from_json: input JSON value must be an object");
    }
    for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
        ctx.set_val(it.key(), from_json(it.value(), mark_input));
    }
}

} // namespace jinja
