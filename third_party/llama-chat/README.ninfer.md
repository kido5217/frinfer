# NInfer llama.cpp chat-parsing source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`95887577ab5fead779581a7030a83c7752ff3234`](https://github.com/ggml-org/llama.cpp/commit/95887577ab5fead779581a7030a83c7752ff3234)
(`refs/heads/master`, 2026-09-26), source paths under `common/`, under the MIT license; see
[LICENSE](LICENSE).

It is the **minimal parser-only port** decided by wayfinder map
[#25](https://github.com/kido5217/ninfer-yarn/issues/25) and design ticket
[#28](https://github.com/kido5217/ninfer-yarn/issues/28): the PEG engine, the chat-layer
mapper and builder, JSON-schema tool typing, and the specialized Qwen3-Coder parser that serves
the Qwen3.5/3.6/3.8 wire format, including the froggeric template. NInfer renders prompts with
its own maintained jinja fork (`third_party/llama-jinja`); this subtree is the parse side.

## File map

Every upstream file below is **byte-identical to the baseline** (`cmp` against a clean
checkout); all adaptation lives in `compat/`.

| Path | Upstream path | Role |
|---|---|---|
| `common/peg-parser.{h,cpp}` | `common/peg-parser.{h,cpp}` | String-based PEG engine, AST, arena (de)serialization, GBNF emission |
| `common/chat-peg-parser.{h,cpp}` | `common/chat-peg-parser.{h,cpp}` | AST→message mapper and typed chat parser builder |
| `common/chat.{h,cpp}` | `common/chat.{h,cpp}` | Public chat types, template apply, `common_chat_parse`/`common_chat_peg_parse`, msg diffs |
| `common/json.{h,cpp}` | `common/json.{h,cpp}` | `common_json`, the ordered-JSON wrapper over vendored nlohmann |
| `common/json-schema.{h,cpp}` | `common/json-schema.{h,cpp}` | Typed JSON-schema model used to type tool arguments |
| `common/json-schema-to-grammar.{h,cpp}` | `common/json-schema-to-grammar.{h,cpp}` | GBNF emission (see "Unused vendored paths") |
| `common/trie.{h,cpp}` | `common/trie.{h,cpp}` | Aho-Corasick automaton used by `ac()` rules |
| `common/unicode.{h,cpp}` | `common/unicode.{h,cpp}` | UTF-8 validation/decoding for AST sanitization |
| `common/chat-auto-parser.h` | `common/chat-auto-parser.h` | `autoparser::generation_params` (the inputs type) and the excluded analysis types |
| `common/chat-auto-parser-helpers.h` | `common/chat-auto-parser-helpers.h` | Marker-diff declarations (implementations excluded) |
| `common/parsers/parsers.{h,cpp}` | `common/parsers/parsers.{h,cpp}` | `foreach_function`/`foreach_parameter` helpers |
| `common/parsers/qwen3-coder.cpp` | `common/parsers/qwen3-coder.cpp` | The Qwen3-Coder / Qwen3.5 / froggeric parser handler |
| `LICENSE` | `LICENSE` | MIT, "Copyright (c) 2023-2026 The ggml authors" |

Include roots: `common/` first (vendored includes resolve to vendored files), then `compat/`
(NInfer shims), then `third_party/llama-jinja` and repo nlohmann via `ninfer_jinja` /
`ninfer::json`.

## Adaptations (all NInfer-owned, under `compat/`)

| Path | Adaptation |
|---|---|
| `compat/common.h` | Slice of `common/common.h`: grammar-trigger and reasoning-format enums, `llama_token`/`llama_tokens` and llama model/vocab/channel forward declarations, `string_starts_with`/`string_ends_with`, declarations of the llama runtime API the vendored layer calls. |
| `compat/common_helpers.cpp` | Definitions of `string_join`/`string_split`/`string_repeat`/`string_replace_all` (bodies from upstream `common/common.cpp`) and `trim_whitespace`/`trim_leading_whitespace`/`trim_trailing_whitespace`/`trim_trailing_newlines` (bodies from upstream `common/chat-auto-parser-helpers.cpp`), whose TUs are excluded. |
| `compat/log.h` | `LOG_DBG`/`LOG_INF` compile out; `LOG_WRN`/`LOG_ERR` print to stderr with a `llama-chat` prefix. |
| `compat/ggml.h` | Only `GGML_ASSERT` and `GGML_ABORT`; there is no ggml dependency in this subtree. |
| `compat/jinja/caps.h` | `jinja::caps` type and entry points expected by `chat.h`/`chat.cpp`. The maintained jinja fork predates upstream's caps layer; `caps_get()` returns the struct defaults instead of probing the template, and `caps_apply_preserve_reasoning`/`caps_apply_reasoning_effort` set the same context variables as upstream. |
| `compat/jinja/value.h` | Re-exposes the jinja fork's `value.h` (via a relative include) and declares `jinja::global_from_json`, which upstream's `chat.cpp` calls and the fork does not provide. |
| `compat/compat.cpp` | `caps` implementation and `jinja::global_from_json` against the fork's value API (`mark_input` maps to `jinja::string`'s literal flag). |
| `compat/excluded_stubs.cpp` | Throwing definitions for every symbol whose upstream implementation is excluded: the llama runtime API (`llama_model_chat_template`, `common_tokenize`, …), the 14 other specialized parsers, `is_lfm2_template`, `workaround::convert_tool_responses_gemma4`, the differential auto-parser (`analyze_template`, `peg_generator::generate_parser`, the three `build_parser` vtable anchors). Reaching one throws — never a silent no-op. |

## Unused vendored paths

`json-schema-to-grammar.{h,cpp}` is vendored for compile completeness only: `peg-parser.cpp`
and the Qwen3-Coder handler reference `build_grammar`/`gbnf_format_literal`, and cutting it
would mean editing vendored sources. NInfer does not consume `common_chat_params::grammar`
(constrained decoding stays rejected per design #28). Likewise, `chat-auto-parser.h` is needed
for `autoparser::generation_params`, but the analysis/generator implementations are excluded
and stubbed.

## Re-vendoring

The baseline above is the authority for what is vendored; the re-vendor procedure and its drift
alert live in
[`docs/maintainer/llama-chat-vendor.md`](../../docs/maintainer/llama-chat-vendor.md). Do not
hand-edit files under `common/`: reconcile upstream changes in the compat layer, and keep the
file map and adaptation table above current.

## Build and test

The library target is `ninfer_llama_chat` (wired in `cmake/Dependencies.cmake`); the smoke test
is `ninfer_llama_chat_parse_test` (`tests/text/test_llama_chat_parse.cpp`), which links only
this library — no NInfer core, no CUDA. It parses a canonical froggeric turn both through a
hand-built Qwen3-Coder-shaped arena and through `common_chat_params_init_qwen3_coder` (template
route, arena serialization, grammar emission), and asserts that an undeclared function name
fails while the same turn with a declared name parses.
