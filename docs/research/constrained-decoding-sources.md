# Constrained decoding sources — llama.cpp GBNF, vLLM structured outputs, xgrammar

Research for the constrained-decoding effort in NInfer. Motivating field report:
[Neroued/ninfer#33](https://github.com/Neroued/ninfer/issues/33) (read-only) — a robot client needs
GBNF-style `grammar`, vLLM-style `structured_outputs`, and JSON Schema through
`response_format`. The practical acceptance surface recorded for this effort: GBNF literal
alternation, character classes with bounded repetition (`[^\n]{3,240}`), `key=value;` record
grammars, and a JSON-schema subset (object/string/array, `maxLength`, `maxItems`, `required`,
`additionalProperties:false`).

All claims are grounded in the pinned sources below; `path:line` citations refer to those exact
commits. Statements that go beyond what was read are marked **[inference]**; claims verified by a
command on this host are marked **[verified]**.

| | |
|---|---|
| **llama.cpp baseline** | `ggml-org/llama.cpp` commit `05af0d2b1398394cfa67e1918fee7feabccaa9bc` (`refs/heads/master` at read time, committed 2026-09-30 17:10:48 +0200) — the same baseline NInfer already vendors under `third_party/llama-chat/`. Read via `git ls-remote` + the existing blob-filtered clone `/tmp/opencode/llamacpp-chat-ref` (lazy blob checkout, `git rev-parse HEAD` [verified]). |
| **vLLM baseline** | `vllm-project/vllm` commit `cff08b461e086523cee8bba5bd3c66d302625134` (`refs/heads/main`, committed 2026-09-30 08:32:00 -0700). Read via `git ls-remote` + `git clone --depth 1 --filter=blob:none` under `/tmp/opencode/vllm-ref` [verified]. |
| **xgrammar baseline** | `mlc-ai/xgrammar` commit `8262b5c94161f4cd0e2c356ab3b62de16b919873` (`refs/heads/main`, committed 2026-09-29 22:05:31 -0400). Read via `git ls-remote` + depth-1 blob-filtered clone `/tmp/opencode/xgrammar-ref` [verified]. |
| **NInfer baseline** | `kido5217/ninfer-yarn` at `0dce9363` (master, 2026-09-30) for the "today" state. |
| **Licenses** [verified, LICENSE read in each checkout] | llama.cpp MIT ("Copyright (c) 2023-2026 The ggml authors"); vLLM Apache-2.0; xgrammar Apache-2.0 ("Copyright (c) 2024 by XGrammar Contributors"). |
| **Method** | Primary-source reading only (no builds, no live runs). Line references from the pinned checkouts; arithmetic (mask sizes) computed in-session. |

---

## 0. Answers in brief

1. **llama.cpp's GBNF engine is small and nearly standalone.** `src/llama-grammar.{h,cpp}` is
   194 + 1530 lines; it needs from the outside world only a token piece string per token id, an
   `is_eog` predicate, a tokenizer for `<token>` literals (the `<[id]>` form needs no vocab), plus
   logging/assert macros. There is no ggml/CUDA/runtime coupling.
2. **llama.cpp builds the mask on the CPU, against a host copy of the candidate logits**, and it
   force-disables device-side sampling when a grammar is active (`common/sampling.cpp:415-419`).
   The grammar is a *separate* sampler next to the chain, not a chain node; by default it
   validates the already-sampled token and only pays the full-vocabulary filter on rejection.
3. **In llama.cpp, speculative drafts are unconstrained.** The target verification samples each
   draft position with the grammar-aware sampler and accepts only grammar-consistent tokens; a
   partial acceptance restores a cloned sampler (grammar included). See §1.4.
4. **The JSON-schema→GBNF converter is already in NInfer, compiled, byte-identical to this
   baseline** (`third_party/llama-chat/common/json-schema-to-grammar.cpp`, `json-schema.cpp`,
   `trie.cpp`, `unicode.cpp` — `cmp` against the upstream checkout [verified]). What is missing
   for "adapt llama.cpp's engine" is the GBNF runtime (`llama-grammar.{h,cpp}`), a vocab adapter,
   and the sampler/device integration. Option A is therefore smaller than it first looks, but it
   inherits llama.cpp's CPU-shaped sampling model and its schema-degradation semantics (§1.6).
5. **The converter degrades silently by design**: an unanchored or unsupported regex `pattern`
   is replaced with "any string" with only a stderr warning
   (`common/json-schema-to-grammar.cpp:390-404`), and unknown JSON Schema keywords are ignored
   entirely (`common/json-schema.cpp` has no unknown-keyword error path). Constraints can be
   quietly weakened by request content.
6. **vLLM's model is "CPU fills a bitmask, GPU applies it".** Per scheduled request the engine
   fills one int32 bitmask row *per speculative position plus the bonus position*
   (`vllm/v1/structured_output/__init__.py:314-442`), moving the grammar through the scheduled
   draft tokens and then rolling it back; the worker applies the rows to the logits with a
   Triton kernel that writes `-inf` (`vllm/v1/worker/gpu/structured_outputs.py:124-162`) before
   the sampler/rejection sampler runs.
7. **vLLM applies masks outside any captured graph** (sampling is eager; only the model forward
   is graph-captured), so structured outputs need no graph-specific handling there. NInfer's
   decode sampling *is* inside captured graphs, so masks must be stable-address buffers (§4.2).
8. **xgrammar is the strongest mask machinery and the biggest option**: C++17, a 63-file ≈1.5 MB
   (bytes of source) core that precompiles per-state adaptive token masks, a batched multi-threaded
   matcher, draft-tree traversal (`TraverseDraftTree`), jump-forward strings, and per-rule
   temperature/budget rules (XGrammar-2, 2026-05). Mask generation is still CPU; the GPU kernels
   only *apply* bitmasks. Its C++ core has no CUDA/Torch dependency; the CUDA apply kernel in the
   repo is a Torch extension (`python/xgrammar/kernels/apply_token_bitmask_inplace_cuda.cu`).
9. **Request-surface semantics should follow vLLM regardless of engine choice**: exactly one
   constraint per request, `response_format` normalized into `structured_outputs`, per-request
   backend selection not supported, config-level `disable_any_whitespace`, reasoning gating.
   NInfer's current single rejection site (`src/serve/openai_chat_request.cpp:204-221`) is where
   this would land.
10. **Sizing for NInfer**: token domain 248077 (`include/ninfer/ops/speculative_round.h`
    registered profile) → a bitmask row is `ceil(248077/32) × 4 = 31 012` bytes. Ordinary decode
    needs one row per constrained request per step (≤8); MTP needs up to `K+1` rows per request
    (≤6 with `kMtpDecodeMaximumDrafts = 5`). Worst case ≈1.5 MB/round of host→device traffic
    [arithmetic; cost estimate is covered in §4.2].

---

## 1. llama.cpp — GBNF engine anatomy and exact copy surface

### 1.1 Data model and grammar language

`src/llama-grammar.h` defines everything: grammar elements (`LLAMA_GRETYPE_CHAR`, `_CHAR_NOT`,
`_CHAR_RNG_UPPER`, `_CHAR_ALT`, `_CHAR_ANY`, `_RULE_REF`, `_ALT`, `_END`, plus token terminals
`LLAMA_GRETYPE_TOKEN` / `TOKEN_NOT`, `llama-grammar.h:13-45`), a `llama_grammar_rule` as a flat
element vector, and a grammar as a set of pushdown **stacks** over rules plus a
`llama_partial_utf8` continuation state (`llama-grammar.h:52-69`, `126-151`). A stack is a
position in the rules; `llama_grammar_advance_stack` expands non-terminals at the top of a stack
(`llama-grammar.cpp:860-...`).

The parser (`llama_grammar_parser`, `llama-grammar.h:86-117`) accepts GBNF: rule definitions
`name ::= ...`, alternation `|`, sequences, string literals, character classes and negations
`[a-z]`/`[^...]`, ranges via `-`, `.` (any char), `*`/`+`/`?` and `{m,n}` repetition, comments,
and token terminals `<token-text>` or `<[token-id]>`
(`llama-grammar.cpp:186-235` for `parse_token`). Init rejects left-recursive rules
(`llama-grammar.cpp:1155-1175`, `1259-1270`), and `root` must exist
(`llama-grammar.cpp:1235`).

### 1.2 Per-step mask production and application

`llama_grammar_apply_impl(const llama_grammar&, llama_token_data_array* cur_p)`
(`llama-grammar.cpp:1359-1400`) is the whole mask:

1. If a lazy grammar is still awaiting its trigger, return without filtering.
2. `allow_eog` is true only when some stack is empty (the grammar can end here).
3. For every candidate token: get `vocab->token_to_piece(id)` from a prebuilt cache; reject EOG
   when `!allow_eog`; reject empty/control pieces (`piece.empty() || piece[0] == 0`); otherwise
   decode the piece's code points against the grammar's partial UTF-8 state and gather a
   `llama_grammar_candidate`.
4. `llama_grammar_reject_candidates` walks every grammar stack against every surviving
   candidate, then rejected candidates get `logit = -INFINITY`.

Cost is therefore `O(candidates × grammar stacks × token length)` per step on the CPU. In the
common server configuration `cur_p` starts as the **full vocabulary** (the candidate array is
loaded from the context, `common/sampling.cpp:130-162`), so each rejection costs a full-vocab
scan with a UTF-8 decode per token. llama.cpp itself has no published number for this:
`common_perf_print` still carries `// TODO: measure grammar performance`
(`common/sampling.cpp:542`).

State advances with tokens:

- `llama_grammar_accept_impl` (`llama-grammar.cpp:1402-1459`) handles lazy triggers: buffered
  tokens build `trigger_buffer`; on a trigger token or full-regex match the buffered pieces are
  replayed into the grammar from the match point (`1419-1446`).
- `llama_grammar_accept_token` (`llama-grammar.cpp:1476-1529`) advances token-type terminals or
  feeds each decoded code point through `llama_grammar_accept`; empty stacks afterwards throw.
- `llama_grammar_accept_str` exists for feeding raw text (used by trigger replay and tests).

The engine supports a *partial* UTF-8 sequence at a token boundary (`llama_partial_utf8`), which
matters for byte-level tokenizers whose tokens can split a codepoint.

### 1.3 Where the grammar meets sampling, and how the mask is shaped

The grammar sampler is a standalone `llama_sampler` node (`src/llama-sampler.cpp:2658-2860`:
`llama_sampler_init_grammar`, `..._lazy`, `..._lazy_patterns`), but in the common layer it is
kept **outside** the sampler chain as `common_sampler::grmr`, alongside the reasoning-budget
sampler (`common/sampling.cpp:427-435`, `445-450`).

`common_sampler_sample(gsmpl, ctx, idx, grammar_first)` (`common/sampling.cpp:594-676`):

- Default `grammar_first = false` (`common/sampling.h:65-68` — the comment says applying grammar
  before the samplers is "slower"): the chain samples first, then the sampled token is tested
  against the grammar as a one-element `llama_token_data_array`; only if it is rejected does the
  code re-fetch logits and apply the *full* grammar filter before re-sampling
  (`614-676`, the "grammar-based rejection sampling" comment at `646`).
- `grammar_first = true` applies the full filter before the chain. **[verified]** No caller in
  the tree passes `true` today (`git grep grammar_first` matches only `common/sampling.{h,cpp}`),
  so the fast rejection path is what actually runs.

Device ("backend") sampling and grammar are mutually exclusive: `common_sampler_init` flips
`params.backend_sampling` off with a warning when a grammar is present
(`common/sampling.cpp:415-419`); the sample path asserts it as well (`617-618`).

Lazy grammars additionally depend on the reasoning-budget sampler: `grammar_should_apply`
(`common/sampling.cpp:452-465`) suppresses the grammar while a reasoning budget is active, and on
budget completion the reasoning end-sequence tokens are replayed into the grammar
(`467-499`).

### 1.4 Speculative decoding × grammar in llama.cpp

- Draft proposals are **not** grammar-constrained. `common/speculative.cpp` contains no grammar
  reference at all [verified by grep], and synthesized draft acceptance explicitly does not
  advance grammar/reasoning state (`tools/server/server-context.cpp:55-96`, comment at `79`).
- Verification samples each draft position through the same grammar-aware sampler:
  `common_sampler_sample_and_accept_n` (`common/sampling.cpp:678-719`) calls
  `common_sampler_sample` per position and `common_sampler_accept`s the result; the loop stops at
  the first position whose sampled token differs from the draft. Because the mask is
  sample-then-validate, a grammar-invalid draft is simply not accepted.
- The server wraps spec verification in a sampler save/restore: a `common_sampler_clone` is kept
  and `common_sampler_copy` restores it when a partial acceptance forces a checkpoint restore
  (`tools/server/server-context.cpp:3890-3900`, `3926-3940`). Grammar state is part of that
  clone (`common/sampling.cpp:509-539`).

### 1.5 The vocab interface the engine needs (and its NInfer mapping)

Runtime coupling is exactly three calls:

| llama.cpp call | Where used | NInfer equivalent |
|---|---|---|
| `llama_vocab::token_to_piece(id)` (cached string, `llama-vocab.cpp:3052-3066`, built with `special=true` at `3373-3399`) | apply + accept | `Tokenizer::decoded_token_bytes_[id]` (`src/models/qwen3_5/frontend/tokenizer.h:156`); byte-level BPE tokens are already decoded to raw bytes at load (`tokenizer.cpp:826-830`, `decode_byte_level_token` at `428-443`); special/added tokens keep their literal text |
| `llama_vocab::is_eog(id)` (`llama-vocab.h:111`) | apply + accept | model EOG set + request stop ids (`default_stop_token_ids_`, `tokenizer.h:164`) |
| `llama_vocab::tokenize(text, parse_special=true)` | only for `<token>` literals, `llama-grammar.cpp:228` | NInfer's own `Tokenizer::encode`; `<[id]>` needs none |

`llama-grammar.h` includes `llama.h` only for the `llama_token` typedef; `llama-grammar.cpp` uses
`LLAMA_LOG_*` and `GGML_ASSERT` (17 sites) and otherwise only the C++ standard library. The
`#include "llama-sampler.h"` at the top is unused [verified by grep].

### 1.6 `common/json-schema-to-grammar` — supported features, self-containment, degradation

Public API (`common/json-schema-to-grammar.h:9-23`): `json_schema_to_grammar(common_json)`
(legacy) and `json_schema_to_grammar(common_chat_schema_document)`; `build_grammar(callback)`
used by the PEG builder (`add_rule`/`add_schema`). Under `LLAMA_USE_LLGUIDANCE` the `common_json`
entry returns a `%llguidance` Lark payload instead (`json-schema-to-grammar.cpp:993-1000`) — not
part of the default build.

The supported subset is fixed by the typed model in `common/json-schema.h` (comment at `12`,
kinds at `15-30`): any, same-document `$ref`, `anyOf`/`oneOf`/type-arrays (as unions), `allOf`,
`const`, `enum`, `null`, `boolean`, `number`, `integer` (inclusive bounds; exclusive folded),
`string` (pattern, formats `uuid`/`date`/`time`/`date-time`, `minLength`/`maxLength`), `array`
(`items`, `minItems`/`maxItems`), tuples (`prefixItems`), `object` (ordered properties,
`required`, `additionalProperties`). The full keyword list recognized by the front end is
`$ref`, `additionalProperties`, `allOf`, `anyOf`, `oneOf`, `const`, `enum`, `exclusiveMaximum`,
`exclusiveMinimum`, `format`, `items`, `maximum`, `maxItems`, `maxLength`, `minimum`, `minItems`,
`minLength`, `pattern`, `prefixItems`, `properties`, `required`, `type` [verified by extracting
all string literals from `common/json-schema.cpp`]. The converter's behavior is pinned by ~100
named cases in `tests/test-json-schema-to-grammar.cpp` (integer bounds, string lengths, regexp
corner cases, property ordering, `$ref`/`allOf`/`anyOf`, additional-property handling).

Degradation semantics (important for product decisions):

- Unanchored or unsupported `pattern` → warning `"pattern ... is not supported (...), accepting
  any string"` and an unconstrained string rule (`json-schema-to-grammar.cpp:390-404`; test
  "unanchored regexp" and "regexp with unsupported shorthand" at `tests/...:1450-1479`).
- Invalid regex (e.g. unbalanced parentheses) → hard error (`399-403`).
- Unknown schema keywords are ignored silently [verified by reading the keyword dispatch; there
  is no `fail()` for unknown keywords]. A client can believe `minProperties`/`not`/`uniqueItems`
  is enforced when it is not.
- Objects are closed unless `additionalProperties` is true/schema (`json-schema.cpp:155-169`);
  properties are emitted in schema order (`json-schema-to-grammar.cpp:699-794`); object values
  default to the primitive rule when no properties are declared (`166-169`, `897-901`).

Self-containment: the file includes `json-schema-to-grammar.h` (→ `json-schema.h` → `json.h`),
`common.h`, `trie.h`, `unicode.h` — all vendored in NInfer already. It does not need GBNF runtime
support to *emit* a grammar string; only the engine that consumes the string is missing.
**[verified]** `cmp` of the four vendored files against the upstream baseline: identical.

### 1.7 Server request surface (baseline)

- `json_schema`, `grammar`; using both is an error (`tools/server/server-common.cpp:1252-1256`).
- `response_format`: `json_object` (uses `schema` if present, else `{"type":"object"}`),
  `json_schema` (unwraps `.json_schema.schema`), `text`; anything else is a 400
  (`server-common.cpp:1258-1272`).
- Custom `grammar` and tools are mutually exclusive (`server-common.cpp:1349-1352`).
- The chat-template apply turns `inputs.json_schema` into a grammar in the legacy route
  (`common/chat.cpp:1416-1429`) and into a PEG **schema node** in the current route
  (`common/chat-auto-parser-generator.cpp:117-121`; the Qwen3-Coder handler used by
  Qwen3.5/froggeric builds `p.schema(p.json(), "response-format", inputs.json_schema)` at
  `common/parsers/qwen3-coder.cpp:85-86`). Tool-call grammars are lazy with triggers
  (`grammar_lazy`, `grammar_triggers`), and `grammar_type` distinguishes user / output-format /
  tool-call grammars (`common/common.h:188-222`).
- Output-format and tool-call grammars are **prefilled** with the generation-prompt tokens so
  the grammar starts where generation starts (`common/sampling.cpp:278-308`).
- The grammar string reaches the sampler through `common_params_sampling`; the slot initializes
  its sampler with `common_sampler_init(model_tgt, task.params.sampling)`
  (`tools/server/server-context.cpp:1764`).

### 1.8 Exact copy surface for option A

| Piece | Size | Would land as | Adaptation needed |
|---|---|---|---|
| `src/llama-grammar.{h,cpp}` | 194 + 1530 lines | new vendored subtree (e.g. `third_party/llama-grammar/`) | drop `llama.h`/`llama-sampler.h` includes; replace `llama_vocab*` with a 3-method adapter (piece/eog/tokenize); `LLAMA_LOG`/`GGML_ASSERT` → NInfer logging/asserts |
| `common/json-schema*`, `trie`, `unicode` | already in-tree | `third_party/llama-chat/` | none (already compiled into `ninfer_llama_chat`) |
| `json_schema_to_grammar` call | trivial | inside the chat/request layer | NInfer has the function but does not call it; the request layer must decide the conversion point |
| Sampler integration | — | NInfer-owned | NInfer has no host candidate array; see §4.2 |
| GBNF tests | `tests/test-llama-grammar.cpp`, `test-grammar-integration.cpp`, `test-json-schema-to-grammar.cpp` | port as corpus/reference vectors | tests are written against `llama_vocab`; the parameterized JSON-schema tests are mostly vocab-free |

The upstream file moves; `src/llama-grammar.cpp` churn would need its own drift note, like
`docs/maintainer/llama-chat-vendor.md`. Its test suites are the natural acceptance oracle.

---

## 2. vLLM — structured-output architecture (baseline `cff08b46`)

### 2.1 Layering

- **Request params**: `StructuredOutputsParams` with mutually exclusive constraint fields
  (`json`, `regex`, `choice`, `grammar`, `json_object`, `structural_tag`) plus options
  (`disable_any_whitespace`, `disable_additional_properties`, `whitespace_pattern`), exactly one
  constraint required (`vllm/sampling_params.py:88-154`).
- **Engine manager**: one `StructuredOutputManager` per engine; one backend instance for the
  engine, chosen from the request's validated `_backend` (`vllm/v1/structured_output/__init__.py:115-165`).
  Grammar compilation runs in a thread pool (async unless `external_launcher`), and failures are
  carried per request (`178-200`).
- **Backend interface**: `compile_grammar(type, spec, stop_token_ids) → StructuredOutputGrammar`
  and `allocate_token_bitmask(max_num_seqs)` (`backend_types.py:96-138`). The request-level
  interface is `accept_tokens`, `validate_tokens` (no advance; returns accepted prefix),
  `rollback`, `fill_bitmask(tensor, index)`, `is_terminated`, `reset` (`backend_types.py:31-94`).
- **Backends**: xgrammar (default), guidance (llguidance), outlines, lm-format-enforcer;
  `StructuredOutputOptions` enum covers JSON, JSON_OBJECT, REGEX, GRAMMAR, CHOICE,
  STRUCTURAL_TAG (`backend_types.py:19-25`).

**Backend selection** (`vllm/sampling_params.py:1140-1345`):

- One backend per engine; a request may carry `_backend` only if it was auto-assigned
  (`1156-1188`).
- `auto` tries xgrammar validation, then falls back to guidance (unless the tokenizer is a
  non-Tekken Mistral tokenizer or the schema uses `patternProperties`), else outlines
  (`1283-1341`). The special case: Tekken-Mistral tokenizers with a Lark-looking grammar route
  to guidance because xgrammar declares no special tokens for those tokenizers (`1272-1281`).
- Validation rejects: empty grammar/json/structural_tag, `json_object: false`, NUL in regex
  (`1189-1237`); diffusion models are rejected outright (`1150-1160`).
- Known xgrammar JSON-schema gaps are guarded before use: `multipleOf`; array
  `uniqueItems`/`contains`/`minContains`/`maxContains`; `format` outside a supported list
  (`email`, `date`, `time`, `date-time`, `duration`, `ipv4`, `ipv6`, `hostname`, `uuid`, `uri`,
  `uri-reference`, `uri-template`, `json-pointer`, `relative-json-pointer`); `pattern`/`format`
  combined with `minLength`/`maxLength` (xgrammar silently drops the length bounds);
  `propertyNames` conflicts; multiple/conflicting `patternProperties`
  (`backend_xgrammar.py:230-350`).

### 2.2 Mask lifecycle per step

The manager keeps one persistent CPU tensor of shape
`max_num_seqs × (1 + num_speculative_tokens)` int32 rows (`__init__.py:327-336`), and a
`_full_mask = torch.tensor(-1)` unmasked row (`58-59`).

`grammar_bitmask(requests, structured_output_request_ids, scheduled_spec_decode_tokens)`
(`__init__.py:314-442`):

- Small batches fill serially; >128 constrained requests with no spec decode fill in parallel via
  a thread pool (`346-373`).
- For each request, `constraint_start` decides where constraints begin (see reasoning gating
  below), then for each scheduled token position `i`:
  `fill_bitmask(row=cumulative_index)`; **advance** the grammar with the draft token via
  `accept_tokens`; increment. If the grammar rejects a draft token (shouldn't normally happen),
  later rows copy the last good row rather than unconstraining (`390-418`).
- After the loop, one more row (the bonus/non-spec token position) is filled, then the grammar is
  **rolled back** by the number of advancements (`426-433`).
- Rows for non-constrained positions are filled with the all-ones mask; rows beyond the batch are
  truncated; the tensor is returned as a numpy array for efficient serialization to the workers
  (`435-442`).
- Once the grammar is terminated, `fill_bitmask` rows read all-ones (nothing further constrained)
  (`202-213`).

**Reasoning gating**: constraints start only after the reasoning section ends (or immediately if
`enable_in_reasoning`), using the request's reasoning parser
(`__init__.py:220-292`, `enable_in_reasoning` at `35-42`). A per-request `reasoning_ended` flag is
latched.

**Stop tokens**: `compile_grammar` passes the request's EOS + user stop ids as
`override_stop_tokens`; the backend masks them until the grammar terminates (interface comment at
`backend_types.py:117-119`; xgrammar passes them to `GrammarMatcher(override_stop_tokens=...)` at
`backend_xgrammar.py:133-141`).

**GPU application** (V2 runner): after `compute_logits` and before sampling/rejection sampling,
the worker builds a bitmask→logits index mapping (accounting for spec-decode positions, and
resolving per-request logit offsets on device for adaptive verification), copies the bitmask
asynchronously, and launches a Triton kernel that writes `-inf` where a bit is 0
(`vllm/v1/worker/gpu/structured_outputs.py:12-162`). The kernel is "adapted from xgrammar's"
(`122-123`). The older V1 path reorders rows and calls `xgrammar.apply_token_bitmask_inplace`
(`vllm/v1/structured_output/utils.py:101-191`). vLLM also handles CPU logits explicitly
(`180-191`).

### 2.3 CUDA-graph compatibility

vLLM's GPU runner sampling is eager: the model forward (which may be graph-captured) produces
logits, and the bitmask is applied to those logits **outside** the graph, before the sampler
(`vllm/v1/worker/gpu/model_runner.py:1583-1590`). No structured-output-specific graph restriction
was found in the code [searched `cudagraph`+`structured`/`grammar` across `vllm/config` and the
worker; nothing]. `InputBatch` carries a `has_structured_output_reqs` flag
(`input_batch.py:100-101`), whose consumers I did not trace fully — **[inference]** it exists to
let batch sharding/padding paths know a bitmask is coming; it does not gate graph capture as far
as this read shows.

### 2.4 Speculative decoding policy

- The bitmask tensor includes K+1 rows per request, and the CPU fill loop simulates the draft
  prefix (advance per draft token, then rollback), so each speculative position gets the mask for
  its own grammar state (`__init__.py:314-442`).
- The rejection sampler consumes masked logits like any other sampler
  (`model_runner.py:1596-1610`, both the plain sampler and `rejection_sampler` branches receive
  the same post-bitmask `logits` tensor).
- `strip_speculative_padding` removes `-1` sentinel padding before any grammar sees token ids
  (`utils.py:48-59`; tests in `tests/v1/structured_output/test_scheduler_speculative_padding.py`).
- `validate_tokens` exists to truncate a spec-token block to its grammar-valid prefix without
  advancing (`backend_types.py:48-60`, xgrammar at `backend_xgrammar.py:191-211`); in this
  revision the manager-level wrapper is defined (`__init__.py:294-312`) and exercised by tests,
  but I found no in-tree caller outside the backend implementations
  [grep `validate_tokens` over `vllm/`; only definitions and backends].
- xgrammar's matcher is constructed with `max_rollback_tokens = num_speculative_tokens`
  (`backend_xgrammar.py:133-141`), but xgrammar v2 marks that parameter deprecated and unused
  (`include/xgrammar/matcher.h:76`).

Note: vLLM's own MTP/structured tests (`tests/v1/spec_decode/test_mtp_structured_output.py`)
pin "tokens after a terminating EOS do not reach the matcher" and "validate_tokens stops at
termination" behaviors.

### 2.5 Request surface → behavior

| Field | Mapping |
|---|---|
| `response_format: {"type":"text"}` | no constraint |
| `response_format: {"type":"json_object"}` | `structured_outputs.json_object = True` |
| `response_format: {"type":"json_schema", "json_schema": {...}}` | `structured_outputs.json = <schema>` (the inner schema object) |
| `response_format: {"type":"structural_tag", ...}` | vLLM extension → `structural_tag` |
| `structured_outputs: {json/regex/choice/grammar/structural_tag}` | extra body fields; merged with `response_format` overrides (`vllm/entrypoints/generate/base/protocol.py:161-191`, chat protocol `vllm/entrypoints/openai/chat_completion/protocol.py:231,378,655-659,724`) |

Request validation includes: `json_schema` requires the `json_schema` field
(`chat_completion/protocol.py:737-764`), and exactly one constraint kind must be set
(`sampling_params.py:106-127`). The legacy `guided_*` spellings were removed in v0.12.0
(`docs/features/structured_outputs.md:9-18`).

The repo at this SHA also carries a Rust tree (`rust/src/.../structured_outputs.rs`, etc.); I did
not read it, and treat it as out of scope.

---

## 3. xgrammar at API/algorithm level (baseline `8262b5c9`)

### 3.1 Compile path

`Grammar` constructors: `FromEBNF` (root default `root`), `FromJSONSchema` (`any_whitespace=true`,
`strict_mode=true` which forces unevaluated properties/items off, optional `indent`/`separators`/
`max_whitespace_cnt`/`any_order`), `FromRegex`, `FromLark` (root must be `start`), `FromStructuralTag`,
`BuiltinJSONGrammar`, `Union`/`Concat` (`include/xgrammar/grammar.h:88-205`). `GrammarCompiler`
preprocesses a grammar **against a tokenizer**: `CompileJSONSchema`, `CompileGrammar` (EBNF or
object), `CompileLark`, `CompileStructuralTag`, `CompileRegex`, `CompileBuiltinJSONGrammar`,
with a byte-limited cache (`include/xgrammar/compiler.h:60-114`). `CompiledGrammar` exposes its
memory size and JSON serialization (`compiler.h:31-49`) — **[inference]** useful for persisting
compiled grammars across restarts and for cache-sized journals.

`TokenizerInfo` is the vocabulary description the compiler needs (`include/xgrammar/tokenizer_info.h`;
vLLM builds it with `TokenizerInfo.from_huggingface` or a raw-vocab constructor for Tekken —
`backend_xgrammar.py:46-78`). The tokenizer must supply token byte strings and stop-token ids;
xgrammar handles byte-fallback vocabularies (`VocabType.BYTE_FALLBACK`).

### 3.2 Matcher and mask generation

`GrammarMatcher` (`include/xgrammar/matcher.h:67-215`) maintains an NPDA over the compiled
grammar: `AcceptToken`/`AcceptString`, `Rollback`, `IsTerminated`/`IsCompleted`, `Reset`,
`Fork`, `GetCaptures`, `GetStopTokenIds`, `GetTemperature` (per-rule budget feature), and
`FindJumpForwardString` (jump-forward decoding). `FillNextTokenBitmask(bitmask, index)` fills an
int32 bitmask whose bit is 1 when the token is allowed.

The efficiency core is a **compile-time adaptive token mask** per grammar state: for each state
reachable from the start state, the compiler classifies vocabulary tokens into
accepted / rejected / uncertain (bitset or indices), computed by walking the sorted decoded
vocabulary trie against the grammar (`cpp/grammar_matcher.cc` — `AdaptiveTokenMask`, store types
at `~556`; cache lookup at `1904-1948`). At runtime, `FillNextTokenBitmask`
(`cpp/grammar_matcher.cc:1684-1741`) unions the per-state masks for the current states, and
`FillBitmaskForStates` (`1896-...`; the common-prefix walk over consecutive vocabulary entries
is at `2069`) visits only each mask's "uncertain" tokens — the same common-prefix trick appears
in the character-budget variant (`1816-1851`). `BatchGrammarMatcher` multi-threads the fill over
many matchers (`matcher.h:226-296`) — this is what keeps JSON masking near the cost of a single
pass.

Speculative decoding: `TraverseDraftTree` (DFS over a draft **tree**, filling per-node bitmasks,
with an optional temperature output and a time threshold) is a v2 feature
(`matcher.h:118-145`); chain-style draft verification can be handled by sequential
accept/rollback like vLLM does today.

### 3.3 Kernels

Mask *generation* is CPU everywhere in this repo. Application kernels live under
`python/xgrammar/kernels/`: CUDA (`apply_token_bitmask_inplace_cuda.cu`, a Torch extension;
`LogitsBitmaskKernel` writes `-inf` where bitmask bit is 0, supports fp32/fp16/bf16, optional
row-index tensor), Triton (the default for CUDA tensors — `kernels/__init__.py` says the CUDA
one is "not used in the current implementation"), Torch (eager), torch-compile, MLX, and a CPU
path (`ApplyTokenBitmaskInplaceCPU`, `matcher.h:32-37`). The Python API documents the bitmask
format: int32, `ceil(vocab/32)`, bit=1 means allowed, application sets masked logits to `-inf`
(`python/xgrammar/matcher.py:27-96`).

### 3.4 JSON Schema coverage

The converter recognizes a much broader keyword set than llama.cpp's — including
`propertyNames`, `patternProperties`, `minProperties`/`maxProperties`, `multipleOf`,
`minContains`/`maxContains`, `unevaluatedProperties`/`unevaluatedItems`, exclusive bounds
[extracted from `cpp/json_schema_converter.cc`]. It too degrades rather than fails on some
combinations: unsupported keywords log a `WARNING` ("Keyword ... is not supported",
`json_schema_converter.cc:769`), `multipleOf` on numbers warns and is ignored (`1056`),
`pattern` combined with `minLength`/`maxLength` warns and drops the length bounds (`1162`).
vLLM's guard list in §2.1 encodes the empirically known gaps. Everything in the issue-#33
acceptance subset (object/string/array, `maxLength`, `maxItems`, `required`,
`additionalProperties:false`) is well inside both converters' supported sets.

### 3.5 Build footprint and licensing

- C++17; static library target `xgrammar` built from `cpp/*.cc` **excluding** `cpp/tvm_ffi/*`
  (Python bindings only); third-party deps are picojson + DLPack headers, optional cpptrace for
  backtraces (`CMakeLists.txt:30-31,77-87`, `cpp/CMakeLists.txt`). 63 core files, ≈1.49 MB of
  source bytes [verified via `git ls-tree --long`], an order of magnitude more than llama.cpp's
  grammar engine (1.7k lines) plus converter (1.0k lines).
- License Apache-2.0 (`LICENSE`), NOTICE "XGrammar".

---

## 4. Copyability assessment for NInfer

### 4.1 The three options

**Option A — adapt llama.cpp's engine (MIT).**

- *Capability*: full GBNF DSL; JSON Schema via the subset in §1.6; no regex entry point of its
  own (patterns only inside JSON Schema, with the degradation behavior); no jump-forward; no
  draft-tree traversal; V1-style tools grammars (lazy/triggers) come along for free if wanted.
- *Cost*: vendoring ≈1.7k lines + a vocab adapter + sampler integration. The JSON-schema
  converter is already in-tree and compiled, so the *schema* half of the request surface needs
  only a call site.
- *Fitness*: llama.cpp's mask producer assumes a host candidate array; NInfer must re-shape it
  into a "produce a bitmask/allowed-set per row/column" component. That is straightforward but is
  new code — the engine's rejection logic works on arbitrary candidate sets, so feeding it the
  full token domain per step (to build a dense mask) costs the same full-vocab scan llama.cpp
  pays; doing better requires xgrammar-style cached masks.
- *Risks*: upstream drift on a hot file; silent schema degradation (§1.6) is a product-semantics
  decision (fail-closed vs warn-and-loosen); no regex `\d`-style shorthands.

**Option B — vendor xgrammar (Apache-2.0).**

- *Capability*: strongest by far: regex, JSON Schema (broad), Lark, structural tags, jump-forward,
  per-rule temperature/budget (v2), draft-tree traversal, batched multi-threaded mask fill,
  compile cache, compiled-grammar serialization. This directly exceeds the issue-#33 surface and
  covers likely follow-on requests (`regex`, `choice`, `structural_tag`).
- *Cost*: vendoring ≈1.5 MB/63 files of C++17; a `TokenizerInfo` constructed from NInfer's
  tokenizer (token bytes + stop ids — the same data as in §1.5); integrating compile (thread
  pool) and per-request matcher state; and *writing* the apply kernel for NInfer's device logits
  (xgrammar's CUDA kernel is a Torch extension; the algorithm is ~100 lines and trivially
  portable, but it is not copy-pasteable into a Torch-free build).
- *Risks*: a large new vendored dependency with its own release cadence (XGrammar-2 is months
  old); C++17 vs NInfer's toolchain — fine (CUDA 13.1 host compiler); its failure modes
  (warnings + dropped constraints) mirror llama.cpp's in kind, so a product-level validation
  strategy is needed regardless.

**Option C — hybrid.** Use the already-in-tree llama.cpp JSON-schema converter and a
GBNF/regex-capable engine behind a vLLM-shaped request API. Concretely:
`common_json → grammar string` (A's converter, in-tree) + whichever engine. The request surface
(`response_format` + `structured_outputs` semantics, validation, auto selection, spec-decode
policy, reasoning gating) should follow vLLM (§2.5) independently of the engine choice, because
that is what clients (and the field report) expect. A third shape is "llama.cpp engine now,
xgrammar later behind the same seam" — the seam being "compiled constraint → per-step mask/row
states".

### 4.2 Integration points in this repo (concrete, [inference])

1. **Request layer**: replace the single rejection in `src/serve/openai_chat_request.cpp:204-221`
   (`validate_constrained_decoding_extensions`) and the Responses analogue
   (`src/serve/openai_responses_request.cpp:957-967`) with parsing into a constraint spec
   (vLLM-shaped fields). Compilation belongs where the serve layer has the tokenizer — the
   engine already resolves frontend resources per model; the constraint must travel with the
   request (`src/serve/translate.cpp:311-…` builds `ninfer::RequestOptions`; a
   `ConstrainedDecoding` option would extend `include/ninfer/types.h:263-267`).
2. **Machine**: NInfer's sampling is a device kernel, `ops::sample`
   (`include/ninfer/ops/sampling.h:73-81`) over BF16 logits with a device `SamplingConfig` per
   row, executed **inside captured graphs** (`src/models/qwen3_5/program/decode.cpp:26-79`).
   A mask can attach either as (a) an extra `const uint32_t*` in `SamplingConfig` (per-row
   pointer to a program-owned mask buffer; the kernel must exclude masked tokens in the greedy
   path *and* the top-20 candidate paths), or (b) a separate tiny "apply mask" op writing `-inf`
   into the logits buffer before `ops::sample` — (b) is simpler and adds one full-vocab pass;
   (a) adds no memory traffic beyond the mask read [design choice, cost not measured].
3. **Stable addresses**: graph capture bakes kernel pointers and scalars (NInfer rule: any
   device state a captured kernel reads must exist at a stable address before capture —
   the YaRN `inv_freq` table precedent). A mask buffer must therefore be a persistent
   `TensorRegion` in the program's per-round state, filled per round from the host via the
   existing ingress copy pattern (`OrdinaryDecodeIngress`/`MtpDecodeIngress`,
   `src/models/qwen3_5/program/round_buffers.h:38-70`), never allocated ad hoc. [inference from
   the capture rule; the buffers' pattern is read]
4. **Ordinary decode**: one mask row per row/request per step; host advances its grammar with the
   egress `sampled_tokens` (`round_buffers.h:48-50`) and uploads the next row. Round-trip is
   already host-synchronous (sampled tokens are copied to host every round).
5. **MTP decode**: the target acceptance is device-side and stochastic
   (`ops::speculative_accept_greedy_drafts` / `speculative_accept_sparse_drafts`,
   `include/ninfer/ops/speculative_round.h`). For grammar correctness, each verification column
   `j` needs the mask of "state after accepting drafts 0..j-1", which the host can precompute
   because drafts are known on host before the round (`MtpDecodeIngress.current_drafts`). That is
   exactly the scheme vLLM proves out (§2.4: accept per position, then roll back). After the
   round, the host advances the persistent grammar by the accepted prefix plus the
   correction/bonus token — the acceptance count is already on host (`accepted_drafts`); the
   correction token lives in `licensed_tokens[A]` on device, so one small extra copy (≤K+1 ints
   per row) is required per round [inference; the tokens' locations are read from the op
   contract].
6. **Draft proposals**: whether the MTP draft samplers themselves are constrained is optional;
   llama.cpp leaves drafts unconstrained, and vLLM's rejection sampler consumes masked target
   logits. Constraining drafts would raise acceptance but costs per-draft mask states; the
   registered MTP width is ≤6 columns, so per-column masks are affordable either way [design
   choice].
7. **Vocab adapter**: `decoded_token_bytes_` + `special_token_ids_` + the EOG set (§1.5) is all
   the pieces a llama.cpp-style engine needs; xgrammar's `TokenizerInfo` needs the same bytes
   plus stop ids. NInfer's tokenizer already carries both.
8. **Token-domain tail**: sampling only covers `[0, token_domain)` while the logits buffer may be
   padded (`ops::sample` contract); a mask built over the vocabulary must match `token_domain`
   (248077 here) so padded rows cannot be sampled, same as xgrammar's padding note
   (`python/xgrammar/matcher.py:83-95`).
9. **Host work per round**: grammar state advance + mask fill for constrained rows only. With
   per-state precomputation (xgrammar-style) the runtime fill is proportional to the "uncertain"
   token set, not the vocabulary; a naive port of llama.cpp's per-step apply is a full-vocab
   scan per constrained row/column, i.e. up to 8×(K+1) ≈ 48 scans/round at worst — the strongest
   argument for xgrammar's algorithm over a naive port [inference; no measurement exists in
   llama.cpp either, see §1.2].

### 4.3 Performance approach comparison

| | llama.cpp engine | xgrammar |
|---|---|---|
| Mask fill | full candidate scan per apply, UTF-8 decode per token | per-state precomputed masks; only uncertain tokens walked; batched multi-thread |
| Batch/concurrency | per-grammar; no batch API | `BatchGrammarMatcher` multi-threaded |
| Spec support | sample-then-validate; replay | `TraverseDraftTree`, rollback |
| Jump-forward | no | yes (`FindJumpForwardString`) |
| Schema coverage | subset, silent degrade | broad, warns on gaps |
| In-tree today | converter only | nothing |

Neither project publishes a per-step overhead number for our geometry; llama.cpp explicitly
carries a TODO for it. Any decision between "naive port" vs "precomputed masks" should be made
against a measured NInfer decode profile, not borrowed claims.

### 4.4 What I could not determine

- **Actual CPU cost of a per-step mask fill at 248 077 tokens** on this host, for either engine.
  (No source gives it; only a NInfer measurement can settle it.)
- **xgrammar v2's full JSON-schema feature matrix** — the converter source shows a wide keyword
  set but also warning-and-ignore paths; only vLLM's guard list and the source warnings were
  read, not the full test suite.
- **Whether xgrammar's C++ core builds cleanly under the nix devShell's CUDA 13.1 host compiler**
  — not attempted (no builds in scope).
- **vLLM's `has_structured_output_reqs` consumers** and the Rust tree — not traced.
- **How NInfer's stop-string scanning and output channels interact with a terminated grammar**
  (e.g. stopping exactly when the grammar completes, `maxLength` unbounded grammars).
- **MTP acceptance math under masking** — masking the target columns changes the rejection
  sampling distributions (`max(p−q,0)` residual); the correction-path math with masked `p` and
  unmasked proposal `q` is standard (masked `p` is zero where disallowed ⇒ rejection), but
  NInfer's kernels would need the mask plumbed into all three paths (greedy, sparse, bonus)
  for consistency; not prototyped.

---

## 5. Open questions for the design

1. **Engine choice (A/B/C)** against the acceptance surface: is GBNF + the llama.cpp schema
   subset enough, or are regex/choice/broader schema coverage in scope? (The in-tree converter
   makes the JSON half nearly free either way.)
2. **Where does compiled grammar state live** in NInfer's ownership model — per-request state in
   the Program's request slot, replicated per state image like KV/continuation state, and does a
   context-cache restore (checkpoint) need to restore grammar state too?
3. **Mask plumbing**: extend `SamplingConfig` (kernel-side exclusion in all sampling paths) or a
   separate apply-mask op? How do MTP's three acceptance paths consume per-column masks?
4. **Failure semantics**: fail-closed 400/500 on any schema feature we cannot honor
   (NInfer's current posture), or llama.cpp/vLLM-style warn-and-loosen? The field report's first
   ask was about *silent* ignoring; silent loosening is the same failure mode in a quieter form.
5. **Stop/EOS and termination semantics**: allow EOG only when the grammar can complete
   (llama.cpp's stack-empty rule) vs vLLM's stop-token override; what happens when a grammar
   never terminates (no `maxLength`) — token budget interaction?
6. **Thinking gating**: vLLM constrains only after reasoning ends unless explicitly enabled —
   does NInfer need the same for its preserved-thinking output, and if the grammar should apply
   during reasoning, which channel?
7. **Spec-decode policy**: mask target columns only (proven) vs also constraining MTP drafts;
   how the 16-candidate sparse path's proposal distributions interact with masked residual
   sampling.
8. **Lazy/triggered grammars**: needed now (tool-call reuse later) or defer until tool-calling
   itself is in scope?
9. **Venue for the converter/vendor**: extend the existing `third_party/llama-chat` vendor set
   (and its re-vendor runbook) with the grammar engine, or a separate vendor with its own drift
   note?
10. **Corpus**: which acceptance vectors ship with the feature — port
    `tests/test-json-schema-to-grammar.cpp` cases, add wire-level tests for the three grammar
    levels from issue #33, and pin spec-decode acceptance cases (vLLM's spec-output tests are a
    useful template).

---

## Sources

| Source | Commit | What was read |
|---|---|---|
| `ggml-org/llama.cpp` | `05af0d2b1398394cfa67e1918fee7feabccaa9bc` | `src/llama-grammar.{h,cpp}`, `src/llama-sampler.{h,cpp}` (grammar sampler section), `src/llama-vocab.{h,cpp}` (piece cache), `common/sampling.{h,cpp}`, `common/speculative.{h,cpp}` (grammar absence), `common/json-schema.{h,cpp}`, `common/json-schema-to-grammar.{h,cpp}`, `common/chat.{h,cpp}` (schema conversion sites), `common/common.h` (grammar-type enum), `common/chat-auto-parser-generator.cpp` (grep), `common/parsers/qwen3-coder.cpp` (grep, via the byte-identical NInfer copy), `tools/server/server-common.cpp`, `tools/server/server-context.cpp` (spec paths), `tests/test-json-schema-to-grammar.cpp` (case inventory + excerpts), `LICENSE`. (`tests/test-llama-grammar.cpp` and `tests/test-grammar-integration.cpp` are named as port sources in §1.8 but were not read.) |
| `vllm-project/vllm` | `cff08b461e086523cee8bba5bd3c66d302625134` | `vllm/v1/structured_output/{__init__,backend_types,backend_xgrammar,backend_guidance,backend_lm_format_enforcer,backend_outlines,request,utils}.py`, `vllm/config/structured_outputs.py`, `vllm/v1/worker/gpu/structured_outputs.py`, `vllm/v1/worker/gpu/model_runner.py` (sampling/bitmask sites), `vllm/v1/worker/gpu/sample/*` (logits processors, batch shard), `vllm/sampling_params.py`, `vllm/entrypoints/openai/chat_completion/{protocol,serving}.py`, `vllm/entrypoints/generate/base/protocol.py`, `docs/features/structured_outputs.md`, `tests/v1/structured_output/*`, `tests/v1/spec_decode/test_mtp_structured_output.py`, `LICENSE` |
| `mlc-ai/xgrammar` | `8262b5c94161f4cd0e2c356ab3b62de16b919873` | `README.md`, `include/xgrammar/{grammar,compiler,matcher}.h`, `cpp/grammar_matcher.cc`, `cpp/json_schema_converter.cc`, `python/xgrammar/{matcher,compiler}.py`, `python/xgrammar/kernels/*`, `CMakeLists.txt`, `cpp/CMakeLists.txt`, `LICENSE`, `NOTICE` |
| `kido5217/ninfer-yarn` | `0dce9363` (master) | `src/serve/openai_chat_request.cpp`, `src/serve/openai_responses_request.cpp`, `src/serve/translate.cpp`, `include/ninfer/{engine,types}.h`, `include/ninfer/ops/{sampling,speculative_round}.h`, `src/ops/kernel/sampling.cuh`, `src/ops/wrapper/sampling.cpp`, `src/models/qwen3_5/frontend/tokenizer.{h,cpp}`, `src/models/qwen3_5/program/{decode,round_buffers,context}.h|cpp`, `src/models/qwen3_5/program/speculative/*`, `third_party/llama-chat/README.ninfer.md` + byte-comparison of the vendored converter files |
