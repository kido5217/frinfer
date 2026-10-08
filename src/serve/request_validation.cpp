#include "serve/request_validation.h"
#include "text/json_input.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

void validate_schema_number_input(const text::ParsedJsonNumbers& parsed) {
    if (parsed.inexact_numbers.empty() || !parsed.value.is_object()) return;
    const auto check = [&](const std::string& pointer, const std::string& param) {
        if (const auto error = text::inexact_schema_number(parsed, pointer))
            bad_request(
                "numeric schema value cannot be preserved by the JSON number representation",
                param + error->substr(pointer.size()), "unsupported_json_schema");
    };
    const auto& body = parsed.value;
    for (const auto& [path, param] :
         {std::pair{"/response_format/json_schema/schema", "response_format.json_schema.schema"},
          // A bare Chat Completions schema puts the document directly under `.json_schema`; the
          // wrapper form's `.schema` is a non-keyword member, so this pointer finds nothing there.
          std::pair{"/response_format/json_schema", "response_format.json_schema"},
          std::pair{"/text/format/schema", "text.format.schema"},
          std::pair{"/output_config/format/schema", "output_config.format.schema"}})
        check(path, param);
    const auto tools = [&](auto&& self, const RequestJson& list, const std::string& path) -> void {
        if (!list.is_array()) return;
        for (std::size_t i = 0; i < list.size(); ++i) {
            const auto& tool = list[i];
            if (!tool.is_object()) continue;
            const auto pointer = path + '/' + std::to_string(i);
            if (tool.contains("type") && tool["type"] == "namespace" && tool.contains("tools")) {
                self(self, tool["tools"], pointer + "/tools");
                continue;
            }
            const bool nested      = tool.contains("function") && tool["function"].is_object();
            const auto& definition = nested ? tool["function"] : tool;
            if (!definition.contains("strict") || definition["strict"] != true) continue;
            const std::string key =
                definition.contains("input_schema") ? "input_schema" : "parameters";
            const auto schema = pointer + (nested ? "/function/" : "/") + key;
            check(schema, schema.substr(1));
        }
    };
    if (body.contains("tools")) tools(tools, body["tools"], "/tools");
}

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

} // namespace ninfer::serve
