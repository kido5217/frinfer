#include "product/constraint/constraint_contract.h"

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::constraint {
namespace {

using Json = nlohmann::ordered_json;

// Keywords the adopted XGrammar compiler (src/text/json_schema.*) genuinely enforces in at least
// one context; the context checks below pin down where. `allOf`/`oneOf`/`additionalItems` are
// deliberately absent (see docs/serving.md — rejected rather than approximated), and `format` is
// absent because the adopted compiler does not enforce it.
const std::unordered_set<std::string>& schema_keywords() {
    static const std::unordered_set<std::string> keywords = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "exclusiveMinimum",
        "maximum",
        "exclusiveMaximum",
        "const",
        "enum",
        "$ref",
        "$defs",
        "definitions",
        "anyOf", // annotations and metadata
        "title",
        "description",
        "default",
        "examples",
        "$comment",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$id",
        "$schema",
    };
    return keywords;
}

bool is_annotation(std::string_view key) {
    static const std::unordered_set<std::string> annotations = {
        "title",      "description", "default",   "examples", "$comment",
        "deprecated", "readOnly",    "writeOnly", "$id",      "$schema",
    };
    return annotations.contains(std::string(key));
}

[[noreturn]] void unsupported(std::string message, std::string param) {
    throw ConstraintError(std::move(message), std::move(param), "json_schema_unsupported");
}

[[noreturn]] void too_large(std::string message, std::string param) {
    throw ConstraintError(std::move(message), std::move(param), "constraint_too_large");
}

[[noreturn]] void invalid(std::string message, std::string param) {
    throw ConstraintError(std::move(message), std::move(param), "json_schema_invalid");
}

std::string join_path(const std::string& path, const std::string& key) {
    return path.empty() ? key : path + "/" + key;
}

bool type_is(const Json& schema, std::string_view name) {
    return schema.contains("type") && schema.at("type").is_string() &&
           schema.at("type").get<std::string>() == name;
}

bool has_any(const Json& schema, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (schema.contains(key)) { return true; }
    }
    return false;
}

// The converter dispatches on `$ref`, `anyOf`, a `type` union, or `const`/`enum` before it looks
// at any structural keyword, so those siblings are dropped. Only annotations (and the document
// containers) may accompany a dispatching keyword; `type` may constrain `const`/`enum` values.
bool is_dispatch(const Json& schema) {
    return schema.contains("$ref") || schema.contains("anyOf") || schema.contains("const") ||
           schema.contains("enum");
}

const char* dispatch_name(const Json& schema) {
    if (schema.contains("$ref")) { return "$ref"; }
    if (schema.contains("anyOf")) { return "anyOf"; }
    if (schema.contains("const")) { return "const"; }
    return "enum";
}

bool value_matches_type(const Json& value, std::string_view type) {
    if (type == "string") { return value.is_string(); }
    if (type == "integer") { return value.is_number_integer(); }
    if (type == "number") { return value.is_number(); }
    if (type == "boolean") { return value.is_boolean(); }
    if (type == "null") { return value.is_null(); }
    if (type == "object") { return value.is_object(); }
    if (type == "array") { return value.is_array(); }
    return true; // unknown type names fail in the converter
}

// `type` may accompany `const`/`enum`; every produced value must match it (or, for a type union,
// at least one member), or the emitted literals would be broader than the schema's language.
void validate_const_enum_values(const Json& schema, const std::string& path) {
    if (!schema.contains("type")) { return; }
    const Json& type = schema.at("type");
    std::vector<std::string> members;
    if (type.is_string()) {
        members.push_back(type.get<std::string>());
    } else if (type.is_array()) {
        for (const Json& entry : type) {
            if (!entry.is_string()) {
                invalid("JSON Schema type array entries must be strings", "type");
            }
            members.push_back(entry.get<std::string>());
        }
    } else {
        invalid("JSON Schema type must be a string or an array of strings", "type");
    }
    const auto matches = [&](const Json& value) {
        for (const std::string& member : members) {
            if (value_matches_type(value, member)) { return; }
        }
        invalid("JSON Schema const/enum value does not match type",
                path.empty() ? "response_format" : path);
    };
    if (schema.contains("const")) { matches(schema.at("const")); }
    if (schema.contains("enum")) {
        for (const Json& value : schema.at("enum")) { matches(value); }
    }
}

// How the converter's typed model resolves an untyped node: object when it declares properties or
// a constraining additionalProperties, then array (items/prefixItems), then string keywords, else
// any (json-schema.cpp dispatch order).
enum class UntypedKind { Any, Object, Array, String };

UntypedKind untyped_kind(const Json& schema) {
    if (schema.contains("properties") ||
        (schema.contains("additionalProperties") && schema.at("additionalProperties") != true)) {
        return UntypedKind::Object;
    }
    if (schema.contains("items") || schema.contains("prefixItems")) { return UntypedKind::Array; }
    if (has_any(schema, {"pattern", "minLength", "maxLength"})) {
        return UntypedKind::String;
    }
    return UntypedKind::Any;
}

void validate_schema_node(const Json& schema, int depth, const std::string& path);

void validate_schema_array(const Json& value, int depth, const std::string& path) {
    if (!value.is_array()) {
        invalid("JSON Schema keyword at '" + path + "' must hold an array", path);
    }
    for (std::size_t index = 0; index < value.size(); ++index) {
        validate_schema_node(value.at(index), depth + 1, path + "/" + std::to_string(index));
    }
}

void validate_schema_map(const Json& value, int depth, const std::string& path) {
    if (!value.is_object()) {
        invalid("JSON Schema keyword at '" + path + "' must hold an object", path);
    }
    for (const auto& [name, child] : value.items()) {
        validate_schema_node(child, depth + 1, path + "/" + name);
    }
}

void validate_schema_node(const Json& schema, int depth, const std::string& path) {
    if (depth > kConstraintNestingLimit) {
        too_large("JSON Schema nesting exceeds " + std::to_string(kConstraintNestingLimit) +
                      " levels",
                  path);
    }
    if (!schema.is_object()) {
        invalid("JSON Schema nodes must be objects; boolean schemas are not supported",
                path.empty() ? "response_format" : path);
    }
    for (const auto& [key, value] : schema.items()) {
        if (!schema_keywords().contains(key)) {
            unsupported("JSON Schema keyword '" + key + "' is not enforced by FrInfer", key);
        }
    }

    const bool typed          = schema.contains("type");
    const bool type_union     = typed && schema.at("type").is_array();
    const bool type_string    = typed && schema.at("type").is_string();
    const bool tuple_items    = schema.contains("items") && schema.at("items").is_array();
    const bool object_typed   = type_is(schema, "object");
    const bool array_typed    = type_is(schema, "array");
    const bool string_typed   = type_is(schema, "string");
    const bool integer_typed  = type_is(schema, "integer");
    const UntypedKind untyped = typed ? UntypedKind::Any : untyped_kind(schema);

    const auto union_contains = [&](std::string_view name) {
        if (!type_union) { return false; }
        for (const Json& entry : schema.at("type")) {
            if (entry.is_string() && entry.get<std::string>() == name) { return true; }
        }
        return false;
    };
    if (typed && !type_string && !type_union) {
        invalid("JSON Schema type must be a string or an array of strings", "type");
    }
    if (type_union) {
        for (const Json& entry : schema.at("type")) {
            if (!entry.is_string()) {
                invalid("JSON Schema type array entries must be strings", "type");
            }
        }
    }

    // A dispatching keyword drops every structural sibling in the converter; only annotations,
    // document containers, and (for const/enum) a matching `type` may accompany it.
    if (is_dispatch(schema)) {
        const std::string driver = dispatch_name(schema);
        for (const auto& [key, value] : schema.items()) {
            if (key == driver || key == "$defs" || key == "definitions" || is_annotation(key)) {
                continue;
            }
            if (key == "type" && (schema.contains("const") || schema.contains("enum"))) {
                continue;
            }
            unsupported("JSON Schema " + driver + " cannot be combined with '" + key + "'", key);
        }
        if (schema.contains("const") || schema.contains("enum")) {
            validate_const_enum_values(schema, path);
        }
        if (schema.contains("enum")) {
            const Json& values = schema.at("enum");
            if (!values.is_array() || values.empty()) {
                invalid("JSON Schema enum must be a non-empty array", "enum");
            }
        }
    } else {
        // Context gates: a keyword is admitted only where the converter's typed model enforces
        // it. A `type` union enforces the keyword in the matching alternative (correct JSON
        // Schema semantics); a single mismatched type or an unresolvable untyped node drops it.
        const bool array_context =
            array_typed || union_contains("array") || (!typed && untyped == UntypedKind::Array);
        const bool string_context =
            string_typed || union_contains("string") || (!typed && untyped == UntypedKind::String);
        const bool integer_context = integer_typed || union_contains("integer");
        const bool object_context =
            object_typed || union_contains("object") || (!typed && untyped == UntypedKind::Object);
        if (has_any(schema, {"minItems", "maxItems"}) && !array_context) {
            unsupported("JSON Schema item bounds are enforced only with type \"array\"",
                        "maxItems");
        }
        if (has_any(schema, {"minLength", "maxLength", "pattern"}) && !string_context) {
            unsupported("JSON Schema string keywords are enforced only with type \"string\"",
                        "maxLength");
        }
        if (schema.contains("pattern") && !schema.at("pattern").is_string()) {
            invalid("JSON Schema pattern must be a string", "pattern");
        }
        const bool number_context = type_is(schema, "number") || union_contains("number");
        const bool numeric_bounds =
            has_any(schema, {"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"});
        if (numeric_bounds && typed && !integer_context && !number_context) {
            unsupported("JSON Schema numeric bounds are enforced only with a numeric type",
                        "minimum");
        }
        if (has_any(schema, {"properties", "required", "additionalProperties"}) &&
            !object_context) {
            unsupported("JSON Schema object keywords are enforced only with type \"object\"",
                        "properties");
        }
        if (has_any(schema, {"items", "prefixItems"}) && typed && !array_context) {
            unsupported("JSON Schema items keywords are enforced only with type \"array\"",
                        "items");
        }
        if (tuple_items) {
            unsupported("JSON Schema tuple items arrays are not enforced; use prefixItems and a "
                        "tail items schema",
                        "items");
        }
        if (schema.contains("minimum") && schema.contains("exclusiveMinimum")) {
            unsupported("JSON Schema minimum and exclusiveMinimum cannot be combined", "minimum");
        }
        if (schema.contains("maximum") && schema.contains("exclusiveMaximum")) {
            unsupported("JSON Schema maximum and exclusiveMaximum cannot be combined", "maximum");
        }
        if (schema.contains("required")) {
            const Json& required = schema.at("required");
            if (!required.is_array()) {
                invalid("JSON Schema required must be an array", "required");
            }
            for (const Json& name : required) {
                if (!name.is_string() || name.get<std::string>().empty()) {
                    invalid("JSON Schema required entries must be non-empty strings", "required");
                }
            }
        }
        if (schema.contains("enum")) {
            const Json& values = schema.at("enum");
            if (!values.is_array() || values.empty()) {
                invalid("JSON Schema enum must be a non-empty array", "enum");
            }
        }
    }

    if (schema.contains("properties")) {
        validate_schema_map(schema.at("properties"), depth, join_path(path, "properties"));
    }
    if (schema.contains("additionalProperties") && schema.at("additionalProperties").is_object()) {
        validate_schema_node(schema.at("additionalProperties"), depth + 1,
                             join_path(path, "additionalProperties"));
    }
    if (schema.contains("items")) {
        const Json& items = schema.at("items");
        if (items.is_object()) {
            validate_schema_node(items, depth + 1, join_path(path, "items"));
        } else if (items.is_array()) {
            validate_schema_array(items, depth, join_path(path, "items"));
        } else {
            invalid("JSON Schema items must be a schema or an array of schemas", "items");
        }
    }
    for (const char* key : {"prefixItems", "anyOf"}) {
        if (schema.contains(key)) {
            validate_schema_array(schema.at(key), depth, join_path(path, key));
        }
    }
    for (const char* key : {"$defs", "definitions"}) {
        if (schema.contains(key)) {
            validate_schema_map(schema.at(key), depth, join_path(path, key));
        }
    }
}

} // namespace

std::string json_schema_constraint_source(const Json& schema) {
    const std::string serialized = schema.dump();
    if (serialized.size() > kConstraintPayloadLimit) {
        too_large("JSON Schema payload exceeds " + std::to_string(kConstraintPayloadLimit) +
                      " bytes",
                  "response_format");
    }
    validate_schema_node(schema, 0, "");
    return serialized;
}

} // namespace ninfer::constraint
