#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::text {
struct ParsedJsonNumbers;
}

namespace ninfer::serve {

// Upstream 2734a56e centralizes protocol output-format parsing here (parse_json_output_format /
// parse_structured_outputs). This fork instead keeps the protocol wire shapes in each route and
// moves the shared constraint logic — the NInfer `structured_outputs` extension, the schema
// contract, and the single-constraint / tools rules — into the protocol-neutral
// `ninfer::constraint` module (src/product/constraint/constraint_contract.*). Upstream's
// declarations are therefore intentionally not adopted; only the self-contained bounded-number
// check below is taken.

// Rejects schema numbers whose decimal spelling the JSON representation cannot preserve
// (upstream 81c8ce09, "support tuple schemas and bounded numbers"). Applied to every parsed
// request body by parse_json_body().
void validate_schema_number_input(const text::ParsedJsonNumbers& parsed);

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

} // namespace ninfer::serve
