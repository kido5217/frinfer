#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    bool enforce_declared_names = false;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled);

} // namespace ninfer::models::qwen3_5::frontend
