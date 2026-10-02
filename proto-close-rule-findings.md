# Prototype findings — constrained-mode close rule (`</think>` + first answer byte)

- **Ticket:** [kido5217/ninfer-yarn#97](https://github.com/kido5217/ninfer-yarn/issues/97) (map [#95](https://github.com/kido5217/ninfer-yarn/issues/95)).
- **Branch:** `proto/close-rule` (throwaway, never merged; this file and `proto/close-rule-e2e/` are the artifact).
- **Baseline:** `master` @ `e83a9af5` + one prototype patch (parser rule + engine seam). The live CR/RB/RU
  arms ran on the #97-only build; the composition arms additionally carry #96's env-gated
  `NINFER_PROTO_WRAPPER` engine patch (`proto/wrapper-thinking-boundary` @ `0ea1a20d`).
- **Foundation:** #94 `research/dodge-mechanics` @ `5f634b1c` (the two-rule deadlock), #92
  `research/mask-activation-boundary` @ `b7095db4` (mask engagement on the transition column), #96
  `proto/wrapper-thinking-boundary` (wrapper A works, wrapper C fails; M-vs-C answered as mask
  displacement).

## 1. Question

In constrained mode, does treating `</think>` followed immediately by the first
grammar-admissible answer byte (`{`) as the close - instead of requiring format whitespace (R2) -
break the deadlock: does the boundary column become an answer column masked with the pre-boundary
(schema-root) grammar row, and does generation proceed into a genuinely constrained answer region
with schema-valid non-empty `content`?

## 2. Implementation (prototype-sized)

- **Parser rule.** `ChatParseOptions.constrained_answer` (default `false`); when set, the R2
  deciding-byte test (`src/models/qwen3_5/frontend/chat_parse_core.cpp`, the `is_format_whitespace`
  check in `feed_reasoning`) also accepts `{`. Everything else - R3 quoted text, R4 hold, R5 terminal
  close, B1 whitespace framing - is untouched. The prototype byte set is `{` only, the observed
  object-root case; a grammar-derived first-byte set is the product route (see §6).
- **Engine seam.** `EngineCore::submit` passes `options.constraint.has_value()` through
  `Frontend::make_output_session` -> `OutputSession` -> `ChatParseOptions`. A request without a
  grammar keeps the v1 path byte-identical (the flag is never set).
- **No mask-side change was needed.** The parser phase flip moves the `{` column to region 0, so the
  existing planner masks it with the pre-boundary schema-root row (which admits `{`) and commits it
  to the grammar; every later answer column follows the schema. The rule and the mask meet on the
  same column by construction.
- **Corpus.** 4 new vectors appended (`tests/fixtures/chat_parsing/corpus.json`, driver reads
  `constrained`): `constrained-close-brace-single-round`, `constrained-close-brace-split-across-rounds`,
  `constrained-quoted-close-not-brace`, and the guard `reasoning-quoted-close-followed-by-brace`
  (same stream without the flag stays quoted protocol text). The 42 existing vectors are byte-for-byte
  unchanged (verified programmatically: 42 old ids, 0 changed, 4 added).

## 3. Results

### 3.1 Corpus and frontend tests

- `ninfer_qwen3_5_chat_parsing_corpus_test`: **46/46 vectors OK** (42 old + 4 new).
- `ninfer_qwen3_5_frontend_test`: corpus sessions **46 OK**; the binary's only failure -
  "a recompiled grammar reused a released grammar's row cache" - reproduces on the master-built
  binary and is pre-existing/unrelated.

### 3.2 Live (RTX 5090, #97-only build, deployed artifact and flags verbatim)

`proto/close-rule-e2e/launch-serve.sh` (deployed `ninfer-yarn-qwen38-27b` command line; artifact
`f0b43ad4.../qwen3_8_27b_nvfp4.ninfer`; `--spec mtp --draft-tokens 4 --lm-head-draft
--preserve-thinking --vision`; froggeric template). Payload = graphiti CombinedExtraction
(probe3's G arm): `enable_thinking: true` + plain `response_format` json_schema, temperature 1,
max_tokens 4096. **4 runs (criterion 1):**

| run | finish | reasoning tok | content chars | first key | schema |
|---|---|---|---|---|---|
| CR1 | stop | 2019 | 2462 | `extracted_entities` | valid (9 entities / 9 edges) |
| CR2 | stop | 1847 | 2249 | `extracted_entities` | valid (7 / 6) |
| CR3 | stop | 1732 | 2644 | `extracted_entities` | valid (8 / 8) |
| CR4 | stop | 2494 | 2435 | `extracted_entities` | valid (8 / 7) |

**Post-close behavior observed.** In all 4 runs `</think>` is consumed by the parser (it appears in
neither channel; content starts at the `{`), reasoning is ordinary prose ending in lines like
"Let me finalize my output.", and `content` is a **fresh, schema-shaped object** - schema-mandated
key order (`extracted_entities` first, the converter emits required properties first), full
jsonschema validation, and entities drawn from the episode (`Lake Baikal`, `Canyon Ultimate CF`,
`Angara River B&B`, ...). No run produced the model's own keys (`entities`/`facts`) or garbage, and
no serve-log/request-log error occurred (8 requests, all `stop_token`). Baseline comparator: the
same payload on the pre-prototype build gave 20/20 `</think>{` -> empty `content` with the complete
self-keyed JSON in `reasoning_content` (#94).

Because every accepted region-0 token after the boundary is masked and fed to the grammar, and the
grammar accepted the whole fed stream (no fail-closed), the answer region is genuinely constrained
from the `{` onward - the grammar's root row admitted `{`, and the schema walked the object to
completion. The first key/validation evidence is behavioral (no per-token mask trace; see §6).

**Regression arms.** RB1-2 (`enable_thinking: false` + json_schema): 2/2 schema-valid,
`reasoning_tokens=0`, 2.6-3.0 s. RU1-2 (no constraint, thinking on): 2/2 free-form; RU2 carried the
model's own `entities`/`facts` keys, i.e. the unconstrained path is unchanged.

### 3.3 Composition with #96 wrapper A

Build carrying both prototypes, `NINFER_PROTO_WRAPPER=1`, wrapper A as raw `grammar`,
`enable_thinking: true`, 4 runs: **4/4 schema-valid, marker consumed, no `{` decider ever appears**
(the wrapper forces the format-whitespace run after `</think>`, so the close still resolves through
R2 exactly as #96 measured). The close rule does not change wrapper A's behavior; wrapper A does not
need the close rule. Wrapper C is unaffected by definition (its answer is unconstrained by
construction).

## 4. Verdict per #97 criteria

1. **After a `</think>{` close the answer region is actually constrained - MET.** 4/4 explicit
   thinking-on constrained runs produced schema-valid non-empty `content` with the schema's keys and
   episode-derived values; the pre-prototype baseline was 0/4 (20/20 corpus-wide).
2. **Unconstrained and thinking-off behavior unchanged; the existing 42-vector corpus unchanged -
   MET.** 42/42 old vectors byte-identical (0 changed), corpus test 46/46; thinking-off 2/2
   schema-valid; unconstrained 2/2 free-form. The constrained rule fires only behind the flag, which
   the engine sets only for grammar-carrying requests.
3. **Composes with the wrapper variants - MET (wrapper A verified live; wrapper C unaffected).**
   See §3.3.
4. **What it does NOT fix - recorded in §6.**

## 5. M-vs-C, in light of this run

#96 already answered M-vs-C behaviorally: when the deciding whitespace is admissible (wrapper C),
the model closes with `</think>` + whitespace 4/4, so the old regime's `</think>{` is the schema-root
row displacing a whitespace preference (scenario M). This prototype is consistent with, and depends
on, that reading: the close rule is only useful if a boundary column was planned; the 4/4 success
(no fail-closed grammar feed) indicates the `{` column was planned region 0, masked with the root
row, and fed to the grammar - i.e. the mask met the parser on that column. The direct per-round
`MaskPlan` trace specified in #94 §7 was still not taken.

## 6. What this does NOT fix (criterion 4)

- **The `{` byte is fixed, not grammar-derived.** Object-root schemas only. A non-object root
  (`[`, string, number, boolean) still deadlocks: the parser waits for whitespace the mask can never
  admit. Deriving the set from the compiled grammar at submit was deliberately not attempted - row
  production belongs to the single serving worker and compiled grammars are shared across requests,
  so a `row_for` scan in `submit` would race; the product route needs a worker-safe first-byte query
  or a compile-time cache.
- **The corpus cannot represent the grammar.** The vectors pin the byte-level rule with a boolean
  flag; they cannot pin "the byte the actual grammar admits", non-object roots, or the mask meeting
  the parser.
- **Tokenization edges of "first grammar-admissible byte".** In all 8 live constrained runs
  `</think>` and `{` arrived as separate tokens (marker consumed, content starts at `{`). If a single
  token piece carried marker + `{`, the transition column would be masked with the root row and the
  full piece (`</think>{`) fed to the grammar on commit -> fail-closed. Not observed in this
  vocabulary/stream, not guaranteed in general.
- **Draft-derived planning vs corrections.** The mask plan and commit regions come from the round's
  MTP drafts; if the accepted `{` were a correction whose draft column was region 1, the parser would
  close but the `{` would be neither masked nor fed, and the next fed token would fail the grammar
  state (fail-closed). The 4/4 success implies this did not happen, but it is untested behavior and
  belongs with #96 §7(i).
- **Quoted `</think>{` inside reasoning in a constrained request now closes early** - a deliberate
  behavior change confined to constrained mode (the unconstrained guard vector pins that #2 holds).
  Real reasoning that literally quotes the wire format followed by `{` would be misread.
- **Not exercised:** thinking-budget early-close combined with the constrained close (inserted
  control tokens then `</think>{`); non-MTP backends; `assistant_prefill` continuation; per-grammar
  compile cost (no new mask path was added - the boundary column already existed in the plan).
- **Seam is prototype-sized.** The flag is derived automatically from `options.constraint` in
  `EngineCore::submit`; a product route needs a deliberate per-request mode/policy decision and a
  `docs/serving.md` contract sentence for the constrained close.

## 7. Residual uncertainty

- No per-token mask trace (the #94 §7 instrument) was taken, so "the mask engaged on the `{` column"
  rests on the absence of fail-closed feeds plus the schema-shaped output, not on a logged row.
- One payload/schema, temperature 1, MTP4, 4 runs per arm; CR entity/edge counts varied (7-9 / 6-9),
  all schema-valid.
- The composition arm shares the build with #96's env-gated patch; without `NINFER_PROTO_WRAPPER` the
  wrapper arm is not meaningful (as #96 established).

## 8. Artifacts

- Probe/launch: `proto/close-rule-e2e/probe_close_rule.py`, `probe_wrapper_compose.py`,
  `launch-serve.sh`, `wrapper-a.gbnf`.
- Live evidence: `proto/close-rule-e2e/results.jsonl`, `raw/CR*.json` (CR1-4, RB1-2, RU1-2),
  `compose/results.jsonl`, `compose/raw/WA*_compose.json`, `serve.log`, `serve-compose.log`,
  `request-log.jsonl`; scratch run dir `/tmp/opencode/proto-close-rule-e2e/`.
- Engine patch (this branch): `src/runtime/engine/engine_core.h` (submit seam + #96 env-gated
  wrapper patch for the composition arms); parser patch:
  `src/models/qwen3_5/frontend/chat_parse_core.{h,cpp}`, `output_session.{h,cpp}`, `frontend.{h,cpp}`.
