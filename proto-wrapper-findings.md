# Prototype findings — wrapper-grammar routes at the thinking boundary (ticket #96)

- **Ticket:** [kido5217/ninfer-yarn#96](https://github.com/kido5217/ninfer-yarn/issues/96) (map #95).
- **Branch:** `proto/wrapper-thinking-boundary` (throwaway, never merged; this file and
  `proto/wrapper-fill/` are the artifact).
- **Baseline:** `master` @ `e83a9af5` plus one env-gated prototype engine patch
  (`src/runtime/engine/engine_core.h`, `proto::wrapper_engage_from_zero`, enabled by
  `NINFER_PROTO_WRAPPER`; without the variable the v1 deferral path is byte-identical).
- **Evidence:** CPU walks `proto/wrapper-fill/cpu/`, live e2e `proto/wrapper-fill/e2e/`,
  scratch run dirs `/tmp/opencode/proto-wrapper*`.

## 1. Question and variants

Can a wrapper GBNF that carries the thinking stream and hands off to the answer grammar at
the close make an explicitly-thinking constrained request produce schema-valid content
(map #95 target (i)) without killing free-form reasoning?

Two variants on one harness, both with the mask engaged from token 0 (every accepted token
fed to the grammar):

- **A — full encoding (forced close).** Marker-avoiding body states (`B`…`B7`, the DFA for
  "text avoiding `</think`"), then the literal close, then a **required** format-whitespace
  run, then the answer schema. EOG is admitted at every completed body element (abort) and,
  once the close is taken, only after a complete schema value.
- **C — permissive prefix (admit-only).** `root ::= .* close ws schema | .*`: the body is a
  permissive star; the whitespace after the close is required by the answer branch but the
  permissive path is never killed by the close, so the producer's rows stay the union of a
  live all-accepting stack and the schema (no enforcement — measured, not assumed).

## 2. Method

### CPU (in-tree producer, real 248,077-token Qwen3.8 vocabulary)

`proto/wrapper-fill/wrapper_probe.cpp` (built by `build.sh`) converts the graphiti
`CombinedExtraction` schema with the vendored converter, renames its root to `schema`,
wraps it per variant, compiles through `CompiledGrammar::compile`, then walks: the
`raw3/G1` (and `G2`) `reasoning_content` prefix re-tokenized (**1903 / 1743 tokens**),
scripted `</think>` (1 token) + `"\n\n"` (1 token) + the `A1` thinking-off schema-valid
answer (518 tokens). Each step calls `row_for` (timed) and `accept`, and checks the fed
token against the row. Reference rows: `dot` (`root ::= .*`) and `plain` (bare schema).

### Live (RTX 5090, one-user local serve, deployed command line)

`build/apps/ninfer-yarn-serve` from this worktree with the deployed artifact and the
deployed flags verbatim (model `…/f0b43ad436b9fa8142c6ed6647c470a6fe409484/qwen3_8_27b_nvfp4.ninfer`,
`--kv-dtype nvfp4 --spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision`,
froggeric template `nixos-configs/users/kido/chat-template/chat_template.jinja`), launched
with `NINFER_PROTO_WRAPPER=1`; the only deviation is `--request-log-jsonl` redirected to the
scratch dir. `proto/wrapper-fill/e2e/probe_wrapper.py` sends probe3's G-arm payload
(CombinedExtraction messages/schema, temperature 1, max_tokens 4096) with
`enable_thinking: true` and `grammar` = the emitted wrapper: 4 runs per wrapper, plus
regression arms (thinking-off + `response_format` json_schema; unconstrained).

## 3. Results — CPU

G1 (G2 shapes are the same; 1 fill for 1743 steps):

| Variant | bytes | compile | reasoning | after close | after ws | answer (cold) |
|---|---|---|---|---|---|---|
| A (forced) | 2144 | 76–79 ms | 1903 steps, **1 fill / 1902 hits**, fill 7.5 ms, p50 1.4 µs | `can_end=0`, ws + space admitted, **`{` forbidden**, 247 652 unset (424 admitted) | schema root (426 admitted) | 518 steps, 175 fills / 57 rows, 1.8 s, p50 1.5 µs, max fill 136 ms |
| C (permissive) | 1900 | 75–79 ms | 1903 steps, **1 fill / 1902 hits**, fill 1.1 ms, p50 1.1 µs | `can_end=1`, ws + **`{` admitted**, row = permissive | row = permissive | 518 steps, 175 fills, 0.78 s, row = permissive |
| `dot` ref | 12 | 77 ms | — | row hash `7b36a807e8af53b0`, 165 unset | — | — |
| `plain` root | 1826 | 76 ms | — | `{` admitted, **ws forbidden**, 247 650 unset | — | — |

- The reasoning row for both wrappers is byte-identical to the `.*` reference row
  (same hash, same 165 unset ids). The 165 are vocabulary ids whose pieces are standalone
  invalid UTF-8 continuation bytes (e.g. `\xa1`) — an engine/producer property, not a
  wrapper restriction; 1 further id is unmaskable (empty/NUL-leading piece). "A single
  full-vocab row per step until the close" is confirmed exactly.
- `plain` reproduces the #94 root-row rejection of the deciding whitespace on this
  vocabulary (`ws_admitted=0`).
- Forcing corner (measured): if the reasoning text ends mid-marker-prefix (`…<`, `…</`),
  A's chain is mid-state and the root close cannot start there — the close token is
  rejected and the model must fall back one character before closing. Normal `<`/`</…`
  text inside the reasoning is fine (`cpu/html1.txt`, 3 fills / 14 steps). Live reasoning
  ended `…output.\n`, so the corner did not trigger.

## 4. Results — live

| Arm | result | content | reasoning | close handling | latency / tps |
|---|---|---|---|---|---|
| **WA1–4** (A, thinking on) | **4/4 schema-valid non-empty `content`** | 2519 / 2356 / 2545 / 2022 chars, keys `extracted_entities`+`edges` | 1601–2391 tokens, normal prose | marker consumed by the split | 10.6–14.2 s, mean 214 tok/s |
| **WC1–4** (C, thinking on) | 4/4 split but **schema-invalid content** | 2199–2585 chars, model's own keys `entities`+`facts` | 1878–2752 tokens | marker consumed by the split | 12.2–15.2 s, mean 219 tok/s |
| RB1–2 | 2/2 schema-valid | thinking-off + `response_format` | none | — | 3.0–4.2 s |
| U1–2 | normal unconstrained output | free-form (classifier's "bad" labels are it wrongly schema-checking unconstrained text) | 2055 / 2212 tokens | — | 12.3–12.8 s |

No serve-log errors; 12 `request_done`, all `stop_token`; `request-log.jsonl`, site log and
all raws are in `proto/wrapper-fill/e2e/`. Throughput is level with the old G arm
(210–217 tok/s).

## 5. Verdict per #96 criteria

1. **Schema-valid non-empty content with thinking explicitly enabled — MET by A (4/4).**
   **Not met by C**: it admits the deciding whitespace and lets the parser split, but its
   permissive stack survives the close, so the answer is never constrained (measured:
   permissive rows throughout the answer walk; model's own keys in `content`).
2. **Reasoning stays free-form — MET for both.** The reasoning rows are the exact
   permissive row (hash match); live reasoning is ordinary prose with no schema structure.
3. **Mask cost within the #90 bar — MET for the reasoning region.** One fill per
   1903-step reasoning walk (per-step p50 1.4 µs; ≪ 0.4 ms/step). The answer region uses
   the existing schema path: cold 175 fills / 57 rows / 1.8 s one-time (max single fill
   136 ms — counter-table builds on demand, the #79 disclosed gap), warm p50 1.5 µs. Live
   decode throughput matches the pre-prototype arm.
4. **No regression for thinking-off constrained / unconstrained — MET on the checked
   arms** (2/2 + 2/2). The engine change is env-gated; without `NINFER_PROTO_WRAPPER` the
   v1 deferral is untouched (not live-A/B'd on the same build).

## 6. M vs C (mask displacement vs model choice), observationally

- Old regime (production @ `e83a9af5`, #89/#94): 20/20 close-in-reasoning runs, byte after
  `</think>` always `{`, parser never split.
- New permissive C (whitespace admitted at the deciding column, sampling otherwise
  unconstrained): **4/4 runs split on `</think>` + whitespace** — the model's natural close
  is the marker plus whitespace when whitespace is allowed.
- The only systematic difference between the two regimes is the deciding column's row
  (pre-boundary schema root under the old plan vs permissive under C; the root forbids
  whitespace, and static reads #92/#94 already pinned the transition column's row). The
  old stream is therefore explained by the root row displacing a whitespace preference to
  `{` — **scenario M**, not a model preference for `{`. This is behavioral inference; the
  direct per-round `MaskPlan` trace specified in #94 §7 was not taken.

## 7. Residual uncertainty

- A's atomic marker-avoidance chain has the partial-prefix corner above; nested prefix
  runs also deepen the chain (producer stack cap 24) — neither triggered in these walks.
- C cannot enforce in principle: the "permissive prefix" only becomes a route if the
  permissive path is killed by the close, i.e. it collapses into A's marker avoidance.
- Cold answer fills are one-time per compiled grammar identity; they include counter-table
  builds on demand. No live per-step profile was taken; the claim is producer-side fills
  plus end-to-end throughput.
- One payload/schema, temperature 1, MTP on, 4 runs per wrapper.
- The engine patch is prototype-only: a product route needs a per-request mode (not an
  env var), the planner/commit contract for engage-from-0 (feed every accepted token), and
  the parser/mask change list in #94 §8. EOG policy chosen here: free during reasoning,
  only after a complete schema value once closed.

## 8. Artifacts

- Harness: `proto/wrapper-fill/wrapper_probe.cpp`, `proto/wrapper-fill/build.sh`,
  `proto/wrapper-fill/README.md`.
- CPU results: `proto/wrapper-fill/cpu/` (G1/G2 per variant, `dot`/`plain`, corner probes,
  emitted wrapper GBNFs).
- Live: `proto/wrapper-fill/e2e/probe_wrapper.py`, `results-wrapper.jsonl`, `raw/*.json`,
  `serve.log`, `launch-serve.sh`.
- Engine patch: `src/runtime/engine/engine_core.h` (this branch).
