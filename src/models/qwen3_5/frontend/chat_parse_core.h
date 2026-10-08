#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Static per-family wire configuration. The parsing core never analyzes a chat template and never
// probes template capabilities: a family is supported by this explicit configuration, and a wire
// format that is not representable here is rejected upstream rather than guessed.
struct ChatParseWireFormat {
    std::string_view thinking_open;
    std::string_view thinking_close;
    std::string_view tool_call_open;
    std::string_view tool_call_close;
    std::string_view function_open;
    std::string_view function_close;
    std::string_view parameter_open;
    std::string_view parameter_close;

    // Qwen3.5 / froggeric XML tool calls (`tool_call_format` xml, the pinned mode):
    // `<think>`/`</think>` reasoning, `<tool_call>`/`</tool_call>` regions,
    // `<function=NAME>` and `<parameter=NAME>` blocks. The template's JSON mode is rejected at
    // configuration time; it is not this format and must never reach this parser.
    [[nodiscard]] static const ChatParseWireFormat& qwen3_5() noexcept;
};

struct ChatParseOptions {
    // Thinking enabled: the generation prompt already opened the reasoning block, so the turn
    // starts in the reasoning channel (R1).
    bool thinking_enabled = true;
    // Constrained/exact framing (ticket #247): the grammar owns the byte stream, including the
    // canonical reasoning close. The parser then drops exactly the close framing bytes and holds
    // back nothing else, so grammar-required leading whitespace in the answer survives.
    bool exact_framing = false;
    // The bytes after the thinking close marker that belong to the framing (the tail of the
    // canonical close serialization). Only meaningful when exact_framing is set.
    std::string_view close_framing = {};
    // Presentation bound on emitted function names (B3).
    std::size_t tool_name_max_length = 128;
};

// The effect of one round (or of the terminal flush) on the append-only channels. Deltas are
// published bytes to append; a terminal evaluation also carries the accepted tool calls and the
// terminal diagnostics.
struct ChatParseResult {
    std::string reasoning_delta;
    std::string content_delta;
    bool terminal = false;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

// Parses one assistant turn of the Qwen3.5/froggeric wire format from decoded text.
//
// The core is fed round by round with the bytes the model produced in that round (the generation
// prompt is not part of the feed). It keeps the reasoning/content channels append-only, withholds
// bytes whose interpretation is not yet decided, and publishes tool calls only at terminal.
//
// Preview purity: the preview protocol (begin_preview() / preview_feed() / preview_finish(),
// with commit() accepting a preview) evaluates from the last committed snapshot and never
// mutates committed state; a preview that is not committed has no effect.
class ChatParseCore {
public:
    ChatParseCore(std::shared_ptr<const ToolCallOutputContract> contract, ChatParseOptions options);
    ~ChatParseCore();
    ChatParseCore(ChatParseCore&&) noexcept;
    ChatParseCore& operator=(ChatParseCore&&) noexcept;

    ChatParseCore(const ChatParseCore&)            = delete;
    ChatParseCore& operator=(const ChatParseCore&) = delete;

    // The preview protocol: begin_preview() starts (or restarts) a preview from the committed
    // state, preview_feed() publishes the deltas of the bytes just fed, and preview_finish()
    // resolves the preview as a terminal flush. The session integration drives one feed per
    // decoded token so that stop policy, reasoning accounting and the thinking budget observe the
    // same channel boundary as the published deltas.
    void begin_preview();
    [[nodiscard]] ChatParseResult preview_feed(std::string_view text);
    [[nodiscard]] ChatParseResult preview_finish();
    // Commits the active preview; a preview that is never committed has no effect.
    void commit();
    // Channel phase of the active preview / of the committed state.
    [[nodiscard]] bool preview_in_reasoning() const noexcept;
    [[nodiscard]] bool in_reasoning() const noexcept;

    // Builds the process-wide region arena now instead of at the first terminal resolution.
    static void warm_up();

    // Cumulative published channels of the committed state.
    [[nodiscard]] std::string_view reasoning() const noexcept;
    [[nodiscard]] std::string_view content() const noexcept;
    // Bytes withheld from publication: marker prefixes or a pending tool-call region.
    [[nodiscard]] std::string_view held() const noexcept;
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] const std::vector<GeneratedToolCall>& tool_calls() const noexcept;
    // Moves the committed tool calls out; the committed vector is left empty.
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics diagnostics() const noexcept;

    // Feeds an assistant-continuation prefix into the committed state before any preview:
    // the bytes stream through the normal channels, so a partial tool region stays held
    // for the generated suffix to complete. Prompt bytes never publish as output deltas.
    void feed_prefix(std::string_view prefix);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Trailing partial tool region of an assistant continuation (from its last tool-call
// marker), or empty when the continuation carries no tool framing.
[[nodiscard]] std::string_view continuation_tool_region(std::string_view continuation) noexcept;

// Fail-closed continuation check: the trailing region must hold no completed call and, on
// a constrained contract, nothing malformed. Throws RequestError on violation.
void check_tool_continuation(const ToolCallOutputContract& contract, std::string_view region,
                             std::size_t max_name_length);

} // namespace ninfer::models::qwen3_5::frontend
