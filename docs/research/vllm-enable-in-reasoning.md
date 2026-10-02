# vLLM `enable_in_reasoning`: what it actually constrains

- **Date:** 2026-10-02
- **Branch:** `research/vllm-enable-in-reasoning` (throwaway; no PR, no product change — research only)
- **Ticket:** [kido5217/ninfer-yarn#93](https://github.com/kido5217/ninfer-yarn/issues/93)
- **Pinned version:** vLLM **v0.30.0** (released 2026-09-22), tag commit
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (read from a depth-1 clone of the tag; all source
  line numbers below are at that commit).
- **History checked against:** full blobless clone of `vllm-project/vllm` (`/tmp/opencode/vllm-history`),
  plus raw-file probes of v0.11.0/v0.11.1/v0.11.2.

## 1. Answer (short form)

`enable_in_reasoning` is a **server-level `StructuredOutputsConfig` flag** (default `False`) that
decides **when the structured-output grammar (bitmask) starts being applied to a request's generated
tokens**. It is not a per-request field, and it does not change the API channel split.

- **Default (`False`), reasoning parser configured** (`--reasoning-parser <name>`): the grammar is
  **deferred**. While the model is in the reasoning interval, the per-step bitmask for that request
  is the *full* mask (every token allowed) and the grammar FSM does **not** advance. Once the
  reasoning-end marker is detected in the generated token ids, the grammar engages from the first
  token after the boundary; the answer region is grammar-constrained. This is the same semantics as
  NInfer v1 (answer-region constraint), implemented as a token-level gate rather than a text split.
- **Default, no reasoning parser configured:** there is nothing to detect the boundary with, so the
  grammar engages at **token 0** — the whole generation is constrained.
- **`enable_in_reasoning=True`:** the gate short-circuits and the grammar engages at **token 0 even
  when a reasoning parser is configured** — i.e. the reasoning interval itself must be
  grammar-valid ("structured output *in* reasoning", the flag's own docstring intent). The
  reasoning parser still decides which generated text is reported as `reasoning` vs `content`
  afterwards; the flag does not move that boundary.
- **Failure mode without the flag** (`False` + reasoning parser): **no error, no warning, no
  automatic fallback**. If the reasoning-end marker is never detected during the generation, the
  grammar never engages at all and the request silently produces unconstrained output. The vLLM
  docs describe this as "structured outputs might become disabled" (the Qwen3 Coder report in
  [PR #32441](https://github.com/vllm-project/vllm/pull/32441)).

So #86's phrasing — "structured outputs disabled during reasoning unless `enable_in_reasoning`" —
is **directionally right but needs two corrections**: (a) it only holds when a `--reasoning-parser`
is configured (otherwise the grammar applies from token 0); (b) it is a *deferral of engagement*
(reasoning tokens are mask-unconstrained, FSM frozen), not a disable, and it silently degrades to
"never engaged" if the boundary is never reached. Also, `enable_in_reasoning` does **not** turn on
answer constraints (they are already on); it extends the grammar backwards over the reasoning
interval.

## 2. The mechanism, from source

### 2.1 Config and entry point

- `vllm/config/structured_outputs.py:41-42` — the field:

  ```python
  enable_in_reasoning: bool = False
  """Whether to use structured input for reasoning."""
  ```

  It sits next to `reasoning_parser: str = ""` and `reasoning_parser_plugin: str = ""`
  (`vllm/config/structured_outputs.py:30-42`).
- `vllm/engine/arg_utils.py:1032-1046` — `--reasoning-parser` and `--reasoning-parser-plugin` are
  registered under the `StructuredOutputsConfig` argument group; `enable_in_reasoning` is reachable
  through vLLM's nested-config syntax `--structured-outputs-config.enable_in_reasoning=True`
  (documented on the Structured Outputs page, §"Reasoning Outputs").
- `vllm/v1/structured_output/__init__.py:96-98` — `StructuredOutputManager` reads the flag once at
  engine init (`self.enable_in_reasoning = ...structured_outputs_config.enable_in_reasoning`).

### 2.2 Per-step bitmask, and the constraint-start gate

The grammar is not a stream filter; it is a **logits mask applied every decode step, after the
forward pass and before sampling**:

- `vllm/v1/worker/gpu_model_runner.py:4596-4599` — `apply_grammar_bitmask(...)` is called on the
  logits right before `self._sample(...)`.
- `vllm/v1/structured_output/utils.py:100-189` — the manager's bitmask rows are reordered to the
  batch and applied with xgrammar: `xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, ...)`
  (`:175`).
- `vllm/v1/structured_output/backend_xgrammar.py:157-179` (`accept_tokens`, advances the FSM),
  `:181-201` (`validate_tokens`, non-mutating prefix check used for spec-decode drafts),
  `:208-209` (`fill_bitmask`, produces the next-token allowed set).

The gate that decides *whether a request's row is a real mask or the all-allow mask* is
`StructuredOutputManager._get_constraint_start(...)` (`vllm/v1/structured_output/__init__.py:220-297`):

- `:235-236` — `if self.enable_in_reasoning: return 0` (constrain from the first generated token).
- `:240-241` — if `structured_req.reasoning_ended` is latched `True`, return 0.
- `:246-254` — if the latch is still `None`, check whether reasoning already ended in the *prompt*
  (`reasoner.is_reasoning_end(request.prompt_token_ids)`); if not, latch `False`.
- `:257-292` — while reasoning has not ended, return `len(spec_tokens) + 1` (constrain nothing),
  using the reasoning parser's token-id methods (`find_reasoning_end_offset` on the parser-engine
  adapter when supported, else a backwards `is_reasoning_end_streaming` scan) to find the first
  token after the boundary.

Consumers of `constraint_start`:

- `_fill_bitmasks` (`:205-214`): when `apply_bitmask` is false or the grammar is terminated, the
  row is filled with `_full_mask` (`:59`, `torch.tensor(-1, ...)` = all bits set = every token
  allowed).
- `grammar_bitmask` (`:317-...`): parallel path at `:361` (`apply_bitmask = ... == 0`), serial path
  at `:385-406` (per scheduled token, `apply_bitmask = i >= constraint_start`), plus a rollback of
  speculative state advancements at `:426-427`.
- `accept_tokens` (`:438-459`): tokens before `constraint_start` are not fed to the FSM; the early
  return at `:452-455` keeps the FSM frozen during reasoning, otherwise `:457` latches
  `structured_req.reasoning_ended = True`.
- `validate_tokens` (`:298-303`): spec-decode drafts are trimmed to the prefix starting at
  `constraint_start`, so draft tokens in the reasoning interval are never grammar-validated.

`reasoning_ended` is per-request state on `StructuredOutputRequest`
(`vllm/v1/structured_output/request.py:20-31`, field `:27`), one of `None` / `False` / `True`.

### 2.3 Behavior matrix (v0.30.0)

| `--reasoning-parser` | `enable_in_reasoning` | Grammar engages | Reasoning interval mask | FSM advance |
|---|---|---|---|---|
| configured | `False` (default) | first token after the detected reasoning end | full mask (unconstrained) | frozen |
| configured | `True` | token 0 | grammar mask | active from token 0 |
| not configured | either | token 0 | grammar mask | active from token 0 |
| configured, thinking off for the request | `False` | token 0 | grammar mask | active from token 0 |

The last row follows from the reasoning parser being request-scoped and fed the request's
chat-template kwargs: `_get_reasoner` builds the parser with `structured_req.reasoning_parser_kwargs`
(`vllm/v1/structured_output/__init__.py:101-112`; kwargs propagated at `vllm/v1/request.py:92-94`,
`:261-262`). For Qwen3 the parser config is built with `initial_state=CONTENT`,
`wait_for_reasoning=False` when `enable_thinking` is false
(`vllm/parser/qwen3.py:90-103`), so `is_reasoning_end` returns `True` immediately
(`vllm/parser/engine/parser_engine.py:650-673`).

Note the empty-config case: if the server was started without `--reasoning-parser`,
`StructuredOutputManager.reasoner_cls` is `None` (`:75-95`) and `_get_reasoner` returns `None`,
which is why `_get_constraint_start` returns 0.

## 3. The channel model (reasoning parsers)

Two layers, both driven by the selected `--reasoning-parser`:

1. **API/output layer (text):** the parser splits generated text into the OpenAI `reasoning` and
   `content` fields, via `extract_reasoning` / `extract_reasoning_streaming`
   (`vllm/reasoning/abs_reasoning_parsers.py:62-190`; interface docs at
   `docs/features/reasoning_outputs.md`). For Qwen3 the markers are `<think>`/`</think>`, and
   `<tool_call>` also ends reasoning in the unified parser engine
   (`vllm/parser/qwen3.py:41-43`, `:205-241`; `docs/features/reasoning_outputs.md` model table).
2. **Gate layer (token ids):** the same parser class exposes token-id predicates used by the
   structured-output gate: `is_reasoning_end(input_ids)`,
   `is_reasoning_end_streaming(input_ids, delta_ids)`, and `find_reasoning_end_offset(token_ids)`
   (`vllm/reasoning/abs_reasoning_parsers.py:62-112`; `vllm/parser/abstract_parser.py:196-232`;
   `vllm/parser/engine/parser_engine.py:615-673`; `vllm/parser/engine/adapters.py:35-115`).

The gate is therefore **token-level and independent of the text-level split**: even if the frontend
parser would report text in `reasoning`, the grammar decision is made from token ids
(`_get_constraint_start` above). Boundary semantics: the grammar starts at the first token *after*
the reasoning-end marker; the marker token itself is not grammar-validated in the default mode.

## 4. Version history

- **When the flag appeared:** commit
  `7c572544e4292cb97e2f4f5a68c43f02406ad1d6` — "[GPT-OSS] Structure_Tag support for gpt-oss
  tool-call in cot (#25515)", authored 2025-10-18 — added `enable_in_reasoning: bool = False` to
  `StructuredOutputsConfig` and the gate short-circuits to the manager. First stable release
  containing it: **v0.11.1** (2025-11-18). Verified: the string is absent from
  `vllm/config/structured_outputs.py` at tag v0.11.0 (raw-file probe) and
  `git merge-base --is-ancestor 7c572544e... v0.11.1` is true, `... v0.11.0` is false.
  (`git log -S enable_in_reasoning` over a blobless clone returns exactly that commit.)
- **The deferral predates the flag:** v0.11.0's manager already skipped applying the xgrammar
  backend while reasoning had not ended (`vllm/v1/structured_output/__init__.py` at v0.11.0,
  `should_fill_bitmask` / `should_advance` reasoning checks near lines 260-290); the flag was added
  as the escape hatch. v0.11.1's version is the same logic with `enable_in_reasoning` guards
  (`should_fill_bitmask`: "if enable_in_reasoning is True ... we should always enable the bitmask
  filling"; `should_advance` likewise), in a much smaller file — the v0.30.0 code above is a
  later unification.
- **Docs note:** added by commit `138d891d7f42004c417561050a6813792316b13b`
  ("[Docs] Clarify structured outputs configuration for Qwen3 reasoning mode", PR #32441,
  merged 2026-03-04), which states the failing observation: "Qwen3 Coder output on vllm v0.11.0,
  v0.11.2, and v0.12.0 all fail to produce structured outputs with requests that have
  `response_format`. ... It sees reasoning has not ended and skips applying the xgrammar backend
  because enable_in_reasoning is not set." The `(v0.11.2+)` marker in the note is the docs' own
  annotation of the behavior window.

## 5. Failure mode without the flag (detail)

With a reasoning parser and `enable_in_reasoning=False`:

- Reasoning tokens are **unconstrained by design**: full mask, FSM frozen (see §2.2). No error or
  warning is emitted anywhere in the request path; the docs note calls the outcome "structured
  outputs might become disabled" (`docs/features/structured_outputs.md:213-217`).
- If the reasoning end is detected, the constraint engages and the answer region is constrained —
  which is why the combination mostly works.
- If the reasoning end is **never detected** (no `</think>`/`<tool_call>` in the generated ids, or a
  parser whose boundary detection fails for that model), the constraint never engages: the model's
  output is never grammar-validated and `response_format` has no effect. This is the Qwen3 Coder
  failure reported in PR #32441.
- `enable_in_reasoning=True` is the documented workaround: it forces the grammar on from token 0
  rather than waiting for a boundary — note this constrains the reasoning text itself, it does not
  merely "allow" the answer constraint to engage.

## 6. Delta vs NInfer v1 (`docs/serving.md` §"Constrained output", master `e83a9af5`)

NInfer v1 contract (`docs/serving.md:148-201`):

- "`grammar` and `response_format` constrain the answer region only (the reasoning region stays
  unconstrained)" (`:150-152`).
- Constraining requests default to non-thinking unless thinking is explicitly enabled
  (`:186-195`); an explicit enable still runs thinking with the answer region grammar-constrained.

Compared with vLLM v0.30.0:

| Aspect | NInfer v1 | vLLM v0.30.0 |
|---|---|---|
| Constraint target | answer region only | answer region by default (when a reasoning parser is configured); **whole generation (reasoning included) if `enable_in_reasoning=True`** |
| Boundary | chat parser's `</think>` split (`docs/serving.md:116-117` reasoning region; parser contract) | token-id reasoning-end detection via the selected reasoning parser |
| Engaging mechanism | answer-region constraint per v1 | per-step logits bitmask; engagement point chosen by `_get_constraint_start` |
| Thinking under constraint | defaults to non-thinking per request; explicit enable allowed | unaffected by the flag; thinking mode and constraint engagement are orthogonal |
| "Constrain inside reasoning" | not offered in v1 | exactly what `enable_in_reasoning=True` does (server-level) |
| Failure without the mode | n/a | silent non-application when the boundary is never detected |

Consequences relevant to seed #90 route (b) ("in-reasoning structured output — the vLLM route; the
answer may begin inside reasoning"):

- vLLM's flag does **not** capture an answer that starts inside reasoning into the `content` field.
  It applies the grammar from token 0; the text-level channel split still happens at the reasoning
  parser's `</think>` boundary afterwards. So with `enable_in_reasoning=True`, if the model emits
  grammar-valid JSON immediately, that JSON is the pre-boundary text, i.e. it is reported as
  `reasoning` unless the model also emits the boundary while the grammar still permits it.
- The flag is a blunt instrument: it turns the reasoning interval into constrained generation
  (useful for grammars that *describe* the reasoning/tool format, e.g. the GPT-OSS structure_tag use
  case it was introduced for), not a server-side "capture the answer from the reasoning stream"
  mechanism.
- For the graphiti/Qwen3.8 failure that motivated #86 (answer drafted in unconstrained reasoning,
  empty `content` under `response_format`), NInfer's #86/#91 default-thinking-off approach acts on a
  different lever (whether reasoning runs at all) and keeps the v1 answer-region contract; adopting
  vLLM's flag semantics would instead require the reasoning text itself to satisfy the grammar.

## 7. References (pinned)

Docs (v0.30.0):

- Structured Outputs — Reasoning Outputs note:
  https://docs.vllm.ai/en/v0.30.0/features/structured_outputs.html ; source
  `docs/features/structured_outputs.md:213-217` at `ced6857a`.
- Reasoning Outputs (parser model table, `reasoning` vs `content`):
  https://docs.vllm.ai/en/v0.30.0/features/reasoning_outputs.html .

Source at `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (GitHub permalink prefix
`https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/`):

- `vllm/config/structured_outputs.py:41-42` — the flag and docstring.
- `vllm/engine/arg_utils.py:1032-1046` — `--reasoning-parser` under `StructuredOutputsConfig`.
- `vllm/v1/structured_output/__init__.py:96-98`, `:205-214`, `:220-297`, `:298-303`, `:317-436`,
  `:438-459` — gate, mask fill, FSM advance.
- `vllm/v1/structured_output/utils.py:100-189` — mask application to logits (`:175`).
- `vllm/v1/structured_output/backend_xgrammar.py:157-209` — FSM accept/validate/fill/rollback.
- `vllm/v1/structured_output/request.py:20-31` — per-request `reasoning_ended`.
- `vllm/v1/worker/gpu_model_runner.py:4596-4599` — mask applied before sampling.
- `vllm/parser/engine/parser_engine.py:615-673` — reasoning-end token ids and scan.
- `vllm/parser/qwen3.py:41-43`, `:90-103`, `:205-241` — `<think>`/`</think>`, `<tool_call>`
  implicit reasoning end, thinking-dependent initial state.
- `vllm/parser/abstract_parser.py:196-232`; `vllm/reasoning/abs_reasoning_parsers.py:62-112`;
  `vllm/parser/engine/adapters.py:35-115` — reasoning-parser interfaces.
- `tests/v1/structured_output/test_structured_output_manager.py:365-420` (v0.30.0) —
  `test_initial_constraint_activation`, params `(use_reasoner, reasoning_ended,
  enable_in_reasoning)`, asserting the grammar engages for `enable_in_reasoning`, `reasoning_ended`,
  and no-reasoner cases, and that the request-local reasoner is not even constructed when the
  short-circuits apply (perf PR #52573).

History:

- `7c572544e4292cb97e2f4f5a68c43f02406ad1d6` — PR #25515, 2025-10-18, added `enable_in_reasoning`;
  first release v0.11.1 (2025-11-18).
- PR #32441 / commit `138d891d7f42004c417561050a6813792316b13b` — docs note, 2026-03-04, reports the
  Qwen3 Coder silent-failure mode.

NInfer side: `docs/serving.md:148-201` (Constrained output) at master `e83a9af5`
("feat(serve): default thinking off under constraining requests (#86) (#91)").

## 8. Residual uncertainty

- Findings are pinned to v0.30.0 (`ced6857a`); vLLM `main` moves fast. Open PRs at research time
  suggest boundary-token edge cases in the default mode — e.g. #44392 "[Bugfix][Core] Filter
  reasoning boundary tokens before structured-output FSM advance" and #50955 "[Bugfix] Preserve
  Qwen3 structured output content" — but they do not change the mechanism described here.
- The v0.11.1 code path was read via raw files (not a tag checkout) for the historical description;
  the v0.30.0 mechanism was read from a clean checkout of the tag.
- "Which channel the constrained text lands in" under `enable_in_reasoning=True` is a derived
  consequence of (grammar from token 0) + (text split at the parser boundary); it was not
  reproduced by running a server in this investigation.
