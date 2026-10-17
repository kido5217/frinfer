#pragma once

#include "models/common/qwen/frontend/tool_contract.h"
#include "text/grammar.h"

namespace ninfer::models::qwen::frontend {
// The fork serve contract rejects an output constraint whenever tools are declared (the tool-call
// parser owns the turn), so the tool grammar never carries a body constraint.
[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const ToolCallOutputContract& contract,
                     std::string_view reasoning_close, std::string_view continuation);
} // namespace ninfer::models::qwen::frontend
