#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// Builds the full-stream "thinking wrapper" grammar around `gbnf`, the answer grammar whose
// `root` rule defines the answer language (the converted JSON Schema or the client's GBNF):
//
//   root  ::= body close ws answer | body
//
// `body` is a marker-avoiding DFA over the close marker's prefixes: a byte that does not continue
// the current prefix returns to the accepting body state, and the marker's first byte always
// (re)enters the prefix chain, so the chain never empties mid-prefix and the marker cannot be
// split across body elements. `close` is the literal `close_marker` (the chat wire format's
// thinking close, so the parser and the grammar share one source of truth), `ws` is one or more
// format whitespace bytes (what the parser's close rule R2 requires), and `answer` is `gbnf` with
// every `root` identifier renamed. EOG is admitted at completed body elements (the `| body`
// abort path, so the wrapper's reasoning rows stay full-vocabulary) and, once closed, only after
// a complete answer value.
//
// Wrapper rule names are collision-free by construction: they all share a generated prefix that
// does not occur anywhere in `gbnf`, and the answer root is renamed to a name with that same
// prefix. The close marker is treated byte-wise; the wire-format markers are ASCII, which is
// where byte-wise and codepoint-wise character classes agree.
//
// Returns the wrapper text, or nullopt with a diagnostic in `error` (when non-null): an empty
// close marker, a grammar with no `root` identifier, or no collision-free name prefix within the
// generation bound.
[[nodiscard]] std::optional<std::string>
wrap_thinking_constraint_grammar(std::string_view gbnf, std::string_view close_marker,
                                 std::string* error = nullptr);

} // namespace ninfer::models::qwen3_5::frontend
