# NInfer llama.cpp grammar-runtime source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
source paths under `src/`, under the MIT license; see [LICENSE](LICENSE). The recorded baseline is
[`05af0d2b1398394cfa67e1918fee7feabccaa9bc`](https://github.com/ggml-org/llama.cpp/commit/05af0d2b1398394cfa67e1918fee7feabccaa9bc)
(`refs/heads/master`, 2026-09-30) — the same baseline NInfer vendors for the chat-parsing stack.

It is the GBNF runtime behind the constrained-decoding effort (wayfinder map
[#45](https://github.com/kido5217/ninfer-yarn/issues/45), design
[#47](https://github.com/kido5217/ninfer-yarn/issues/47)): grammar parsing and expansion,
pushdown-stack state, the per-token filter walk, and the per-stack rejection used to produce mask
rows. The NInfer module that drives it is
[`src/models/qwen3_5/frontend/grammar/`](../../src/models/qwen3_5/frontend/grammar) (`ninfer_grammar`).

## File map

Unlike the chat stack, this subtree is **not byte-identical to the baseline**: it carries a small,
documented patch set. Every maintained file below is reproduced exactly by applying
`patches/*.patch` in order to the pristine upstream file (see [Verification](#verification)).

| Path | Upstream path | sha256 of the maintained copy | Role |
|---|---|---|---|
| `llama-grammar.h` | `src/llama-grammar.h` | `b71d3d34a9afde788533e96472efe763dd3ca27392963129e70269ac69b16ceb` | Grammar element/stacks data model, parser and runtime entry points |
| `llama-grammar.cpp` | `src/llama-grammar.cpp` | `772cc4d5c9fab5285f4d6813e0335023cefd43ae71b4cb4b05a20e5abe389c2e` | Parser, expansion, stack advance/accept, rejection walk |
| `LICENSE` | `LICENSE` | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` | MIT, "Copyright (c) 2023-2026 The ggml authors" |

Include roots: this directory first (vendored includes resolve to the vendored files), then
`compat/` (NInfer shims). `llama-sampler.h` is included by the vendored `.cpp` but no symbol from
it is used; `compat/llama-sampler.h` exists so the include needs no patch.

## Patch set (the complete deviation set)

| Patch | File(s) | Deviation |
|---|---|---|
| `0001-repetition-bound.patch` | `llama-grammar.{h,cpp}` | Raises the repetition sanity bound from upstream's internal 2000 to the exported `LLAMA_GRAMMAR_MAX_REPETITION_BOUND = 65536` (declared in `llama-grammar.h`) and **removes the silent degradation**: upstream rewrites a finite `max_times > 2000` to "unbounded", so `maxLength: 2400` and similar JSON-schema bounds are not enforced; this port keeps the parsed bound and fails the parse above the exported limit instead. The bound is exported so the serve layer can fail closed before conversion. |
| `0002-parser-last-error.patch` | `llama-grammar.{h,cpp}` | Adds `llama_grammar_parser::last_error`, set by `parse()` on failure, so a caller can surface the diagnostic instead of scraping stderr. |
| `0003-quiet-parse-errors.patch` | `llama-grammar.cpp` | Removes the parse-failure `fprintf(stderr, ...)` that wrote the caller-supplied grammar text to stderr; the diagnostic stays available through `last_error` (patch 0002). NInfer rejects client grammars with a 400 and must not echo client bytes into server logs. |

No other deviation exists: everything else is upstream's code. In particular the engine's mask
semantics (empty/zero pieces rejected, `-INFINITY` write-back, EOG allowed only where a stack is
empty), the expansion rules and the initial-stack construction are untouched.

## Compat adaptations (all NInfer-owned, under `compat/`)

| Path | Adaptation |
|---|---|
| `compat/llama.h` | `llama_token` typedef and the `llama_token_data(_array)` shapes the filter writes into (mirrors `include/llama.h` at the baseline). There is no other llama.h dependency in this subtree. |
| `compat/llama-impl.h` | `LLAMA_LOG_ERROR`/`WARN` print to stderr with a prefix, `LLAMA_LOG_DEBUG` compiles out, `GGML_ASSERT`/`GGML_ABORT` abort (same bodies as `third_party/llama-chat/compat/ggml.h`). |
| `compat/llama-vocab.h` | `llama_vocab` as a callback table (`token_to_piece`, `is_eog`, `tokenize`) so the vendored logic stays untouched and the vocabulary is not tied to llama.cpp's runtime; the NInfer module backs it with its `GrammarVocabulary` view. |
| `compat/llama-sampler.h` | Empty: the upstream include is unused. |

## Verification

The patch set must reproduce the maintained copies byte-for-byte against a pristine checkout of
the baseline:

```bash
third_party/llama-grammar/verify-vendor.sh            # uses /tmp/opencode/llamacpp-chat-ref
LLAMA_CPP_REF=<clone> third_party/llama-grammar/verify-vendor.sh
```

The re-vendor procedure and its drift alert live in
[`docs/maintainer/llama-grammar-vendor.md`](../../docs/maintainer/llama-grammar-vendor.md). Do not
hand-edit `llama-grammar.{h,cpp}`: re-derive the copy from the pristine upstream file plus an
updated patch, and regenerate this README's hashes.

## Build and test

The library target is `ninfer_llama_grammar` (wired in `cmake/Dependencies.cmake`). Consumers link
`ninfer_grammar`, whose test is `ninfer_grammar_test`
(`tests/models/qwen3_5/test_grammar.cpp`): the ported upstream vectors, the differential mask
oracle, the repetition-bound checks and the row-cache behavior.
