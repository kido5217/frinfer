# Mask engagement point vs the thinking boundary (ticket #92)

Research ticket: [kido5217/ninfer-yarn#92](https://github.com/kido5217/ninfer-yarn/issues/92)
("Mask engagement point vs the thinking boundary"), parent map #95.
All line references are to `master` @ `e83a9af5` unless stated otherwise.

## Question

For a serve request that starts in thinking mode (`starts_in_reasoning`) and carries a
grammar constraint (`response_format` → `GrammarConstraint`): where does the grammar's
mask actually engage relative to the thinking boundary?

- (a) every sampled token from the first one (reasoning tokens schema-valid), or
- (b) only when/after the `</think>` boundary is detected, or
- (c) something else.

## Method

- Static read of the decode path in a scratch worktree of `master` @ `e83a9af5`:
  serve option wiring (`src/serve/translate.cpp`), prompt render / `starts_in_reasoning`
  (`prompt_layout.cpp`, `chat_template.cpp`), the output session and its parse core
  (`output_session.{h,cpp}`, `chat_parse_core.{h,cpp}`), the engine mask-plan construction
  (`src/runtime/engine/engine_core.h`), the Program mask fill
  (`src/models/qwen3_5/program/decode.cpp`, `commit.cpp`), and the grammar runtime
  (`src/models/qwen3_5/frontend/grammar/grammar_runtime.{h,cpp}`, `grammar.cpp`).
- Read the shipped tests that pin the region semantics
  (`tests/models/qwen3_5/test_frontend.cpp` `test_reasoning_regions`,
  `tests/models/qwen3_5/test_grammar.cpp` `test_grammar_runtime`).
- Examined the existing thinking+grammar evidence from #89's re-probe:
  `/tmp/opencode/e2e/constrained-thinking-default/` (`SUMMARY.md`, `serve.log`,
  `results3.jsonl`, `raw3/G1..G4.json`).
- **No live token-level run.** `nvidia-smi --query-gpu=memory.used` reported `21984 MiB`
  (GPU busy, threshold is ~5 GiB), so per the ticket the GPU was not touched. The trace
  that would settle the residual question is specified below.

## Core answer (quotable)

**The reasoning region is never masked.** Every decode column whose post-feed parsing
phase is still Reasoning receives an all-ones mask row (nothing forbidden), and while all
columns of a round are reasoning the device mask stays deactivated entirely
(`grammar_runtime.cpp:50-52`, `decode.cpp:876-892`). The mask first engages at the
**transition column** — the column whose token carries the reasoning close's *deciding
byte* (the `</think>` marker completion plus a following format-whitespace byte), read
*after* that token's bytes are parsed. That column is treated as an answer column and is
masked with the grammar row of the **pre-boundary** (pre-answer) state; it and every later
answer column advance the grammar state, and their content may only leave through the
grammar (`output_session.h:70-81`, `grammar_runtime.h:19-33`,
`tests/models/qwen3_5/test_frontend.cpp:1880-1885`,
`tests/models/qwen3_5/test_grammar.cpp:1532-1538`). The switch itself is the ChatParseCore
phase transition Reasoning → Content, evaluated per round/column from the committed core
state plus the round's draft span; it is not a decoder-side latch or a token-id trigger.

For the ticket's options: **(a) is false** — reasoning tokens are unconstrained by
construction. **(b) is the right shape but imprecise**: the mask engages on the column
that *completes* the boundary, not on a column strictly after `</think>` is published, and
if the boundary is never completed the mask never engages at all (it stays all-ones for
the whole turn). **(c) clarifications**: `starts_in_reasoning` selects only the core's
*initial* phase — with thinking off the mask is applied from the very first sampled token
(prefill bonus sample included); with thinking on it is held off until the phase leaves
Reasoning. The serve default for a constraining request without an explicit thinking
request is thinking **off** (`translate.cpp:191-199`, issue #86 / PR #91), so the
boundary-gated case exists only for requests that explicitly enable thinking
(`enable_thinking: true` or a non-none `reasoning_effort`), as in the G arm of the #89
re-probe.

## Mechanism, step by step

Serve → engine:

1. `response_format`/`grammar` is resolved to GBNF by the serve contract
   (`src/serve/constraint_contract.cpp`), and a constraining request defaults thinking
   off unless explicitly enabled (`src/serve/translate.cpp:191-199`).
2. `to_request_options` fills `RequestOptions.constraint` =
   `GrammarConstraint{.gbnf = ...}` (`src/serve/translate.cpp:351`;
   `include/ninfer/types.h:263-275`).
3. `EngineCore::submit` compiles the grammar once and holds it on the request
   (`engine_core.h:183-196`); at materialization it attaches a per-request
   `GrammarRuntime` to the sequence (`engine_core.h:1554`), whose state starts at the
   grammar's initial state (`grammar_runtime.cpp:26-27`).
4. The prompt render decides `starts_in_reasoning` (final unclosed assistant prefix that
   opens `<think>` and has no `</think>`: `prompt_layout.cpp:99-111`), carried into
   `PromptSummary` (`frontend.cpp:672-677`) and into the OutputSession /
   ChatParseCore (`frontend.cpp:927-929`, `output_session.cpp:282-285`). The core's
   initial phase is Reasoning iff `thinking_enabled` (= `starts_in_reasoning`)
   (`chat_parse_core.cpp:722-727`).

Per round, mask planning:

5. Prefill sample: the engine asks the session for the region with an empty token span,
   so the single column's mask bit is the committed phase
   (`engine_core.h:1414-1419`; `output_session.cpp:447-462` reports the committed region
   for an empty span).
6. Decode round: the engine feeds the round's speculative draft span through the
   session's parse core preview and gets one post-feed flag per column plus the bonus
   tail flag; each flag becomes `MaskPlan.reasoning_mask` bit
   (`engine_core.h:1836-1851`). Flags are read *after* the column's bytes so the token
   that completes the close is itself flag 0 ("answer column"):
   `output_session.cpp:451-462` and its contract at `output_session.h:70-81`.
7. `refresh_mask_rows` turns regions into device rows: a column with region != 0 gets
   all-ones; a column with region 0 gets the grammar's row for the state reached by the
   answer columns before it; if no column is masked the device `mask_active` flag stays
   down and `apply_mask` does nothing (`decode.cpp:862-952`, `grammar_runtime.cpp:33-64`).
   The mask is applied before sampling/argmax in every path: ordinary decode
   (`decode.cpp:56-66`), MTP verify (`execution/text.cpp:772-780`), prefill bonus
   (`execution/text.cpp:1283-1290`), prefill step (`prefill.cpp:162-165`).
8. Commit: only region-0 columns are fed to the grammar (`commit.cpp:423-457`); a round
   whose plan is entirely reasoning advances nothing (`commit.cpp:430-435`). A rejected
   answer token is fail-closed (`commit.cpp:451-455`).

The switch itself:

9. The core's phase flips out of Reasoning in `feed_reasoning` when `</think>` is
   followed by a format-whitespace byte (R2); a marker without its deciding byte is held
   and stays reasoning (R4), a marker followed by any other byte is quoted protocol text
   and stays reasoning (R3) (`chat_parse_core.cpp:588-656`);
   `preview_in_reasoning()` is simply `phase == Reasoning`
   (`chat_parse_core.cpp:790-792`). The deciding byte therefore does not have to be the
   `</think>` token itself: the fixture pins `"nk>\n\nanswer"` (close completion + answer
   text in one token) as the transition column
   (`tests/models/qwen3_5/test_frontend.cpp:1869-1885`), and the grammar runtime pins the
   transition column as the first masked column using the pre-boundary state row
   (`tests/models/qwen3_5/test_grammar.cpp:1532-1538`).

## Evidence that the mask is off during reasoning

- `fill_round_rows`: `if (regions[column] != 0 || !reachable) { fill all-ones; continue; }`
  (`grammar_runtime.cpp:50-53`).
- `refresh_mask_rows`: when no column of the round is masked, the device active flag is
  lowered and the round returns without writing rows (`decode.cpp:876-892`); the prefill
  variant raises the mask only when the single region bit is 0 (`decode.cpp:954-982`).
- `advance_grammar_state` skips a request whose plan is entirely reasoning
  (`commit.cpp:430-435`).
- Tests: unconstrained columns are all-ones and their tokens never reach the grammar;
  the transition column is the first masked column
  (`test_grammar.cpp:1524-1538`); a fresh thinking session starts with flag 1
  (`test_frontend.cpp:1879-1885`).
- Consequence in the shipped evidence: the #89 G-arm raws (`enable_thinking: true` +
  `json_schema`) all stopped with `content` empty and the complete JSON answer inside
  `reasoning_content` (`reasoning_tokens` 2468–3181, `finish_reason` `stop`;
  `/tmp/opencode/e2e/constrained-thinking-default/raw3/G1..G4.json`). No constrained
  content was produced: the visible behavior is consistent with the core never leaving
  Reasoning (or leaving it with zero content bytes); `serve.log` contains 0 errors, so no
  fail-closed grammar divergence happened either.

## Residual uncertainty (not settled statically)

1. **The transition column's token vs the pre-boundary grammar row.** The implementation
   masks the deciding column with the pre-boundary answer state and then feeds the
   column's full token piece to the grammar on commit (`commit.cpp:443-455`,
   `grammar_runtime.cpp:66-77`). The vendored JSON-schema converter's object root starts
   with `{` and has no leading-whitespace alternative (`SPACE_RULE`,
   `third_party/llama-chat/common/json-schema-to-grammar.cpp:229,243`). So a
   whitespace-only deciding token (the usual `</think>` + `\n` tokenization) is not
   admissible at that row. Whether the close can still complete depends on whether the
   deciding byte happens to be sampled in a column whose region was still flag 1 from
   the round's draft span (the region gate is computed from the *drafts*, and a correction
   token can differ from them). No thinking+grammar test exists (all constrained tests are
   non-thinking), and the G raws never crossed the boundary, so this corner is untested
   end to end. A related contract sentence — "when it carries none the column contributes
   nothing to the grammar" (`grammar_runtime.h:26-29`) — has no explicit skip path in
   `GrammarRuntime::commit`/`advance_grammar_state`; the code feeds the token regardless.
2. **Draft-based planning transients.** Per-round regions come from the draft span, not
   from the eventual sampled tokens; within a speculative round the region gate can be
   temporarily misaligned with corrections. The next round recomputes from the committed
   state, so the boundary itself is not latched, but the exact column at which a real run
   first masks is a function of the MTP draft layout of that run.

### Trace that would settle it (GPU-gated, when the 5090 is free)

Run one constrained thinking request (same artifact/template as the #89 probe, G payload,
`--log-level trace`), with a small scratch instrumentation in a worktree logging per
round in `EngineCore::run_decode_round` / `ProgramImpl::refresh_mask_rows` /
`GrammarRuntime::fill_round_rows`:

- the round's draft token ids and `OutputSession::preview_reasoning_flags` output;
- `MaskPlan.reasoning_mask` and which columns were masked;
- at each masked column, the pre-mask argmax vs the eventual accepted token;
- the committed `ChatParseCore` phase per round, and every `</think>`/decider occurrence.

That distinguishes "the model never attempted the close" (no flag-0 column ever, mask
inactive throughout — what the G raws suggest) from "the close was attempted but the
transition column's mask rejected the deciding token" (a flag-0 column exists, its
pre-mask argmax is the decider/`</think>` token, the accepted token differs and the
committed phase stays Reasoning).
