# Live e2e: opencode ⇄ ninfer-yarn serve

Wayfinder map #139, ticket #143 — live e2e evidence of opencode compatibility
against the deployed ninfer-yarn unit. Ninfer-yarn alone (no baseline arms —
user decision 2026-10-02); evidence = a scripted HTTP probe suite replaying
opencode request shapes **and** real opencode agent sessions.

**Method and sources of record (all read/run this session, 2026-10-03)**

- Probe suite: `e2e/probe_suite.py` on this branch (stdlib-only Python; replays
  the request shapes enumerated in `docs/research/opencode-requirements.md`
  @ `research/opencode-requirements` and asserts the expectations documented in
  `docs/research/serve-capabilities.md` @ `research/serve-capabilities`).
  Three runs recorded: `e2e/results/probes-20261003-002025.jsonl` (run 1),
  `probes-20261003-002046.jsonl` (run 2), `probes-20261003-002106.jsonl`
  (run 3, final — 24/24 pass; file stamps are MSK, the final run is
  2026-10-02 21:21:06 UTC).
- Real sessions: `opencode run --standalone --model ai.kido.ws/Qwen3.8-27B#xhigh
  --auto --format json` (opencode v2.0.22, the installed binary), in scratch dirs
  `e2e/sessions/w{1,2,3}/` (task prompts + captured event streams + exit codes).
  Model variant `xhigh` mirrors the operator's live agent traffic
  (`requested_reasoning_effort: "xhigh"` in the request log).
- Request log: `/home/kido/trash/temp/ninfer-yarn-log.jsonl` (schema v22,
  append-only across restarts; `request_id` is per server instance).
  E2e window: 2026-10-02 21:21:30 → 21:28:20 UTC (= 2026-10-03 00:21:30 →
  00:28:20 MSK; timestamps 1790976090000–1790976500000 ms). The operator's own opencode session generated concurrent
  traffic in the same window; it is classified out by prompt size
  (>100,000 prompt tokens ≈ the operator's ~200K context; e2e sessions were
  ≤29K).
- Deployment: unit `ninfer-yarn-qwen38-27b` at 127.0.0.1:8827 (behind caddy at
  `ai.kido.ws/v1`), 446902 YaRN context, MTP4 + `--lm-head-draft`,
  `--kv-dtype nvfp4`, `--default-max-tokens 32768`, `--vision` on, froggeric
  v22.5 chat template, C=4 / P=64, unauthenticated — per the deployment table
  in `docs/research/serve-capabilities.md` §0 (verified live this session via
  `/v1/models` and the request log).

---

## 1. Probe suite (24 probes — final run 24/24 pass)

Each probe replays a shape opencode actually sends, or a documented failure
shape; expected behavior is from the two companion findings docs.

| Probe | Shape replayed | Key assertion | Result |
|---|---|---|---|
| P01 | `GET /v1/models` | id `Qwen3.8-27B`, `max_model_len` 446902 | pass |
| P02 | `GET /health` | 200 `{status: ok}` | pass |
| P03 | minimal agent-step body (system+user, both max fields, `reasoning_effort: xhigh`, `stream_options`, `store: false`) | SSE: start chunk, reasoning deltas, content deltas, `finish_reason` ∈ {stop,length,tool_calls} before `[DONE]`, no content after finish, usage chunk with `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens`, single choice per event | pass |
| P04 | `max_completion_tokens: 48` + `max_tokens: 8` (the repo test-pinned precedence case) | completion bounded by 48, not 8; `finish_reason: length` | pass (live confirmation of `max_completion_tokens` precedence) |
| P05 | no max field at all | 200 + full stream contract; log records `requested_output_tokens_source: server_default` | pass |
| P06 | agent step with one `strict: false` tool; model asked for a single call | one terminal `tool_calls` delta with complete `id` (`call_<16 hex>`) + `name` + valid-JSON arguments; `finish_reason: tool_calls` | pass |
| P07 | tool-history replay (assistant `content: null` + `tool_calls` + `reasoning_content`, then `tool` result) | 200 (no 400 on the replay fields), next tool call clean | pass |
| P08 | `tool_choice: "none"` with tools | 200, no tool calls emitted | pass |
| P09 | `tool_choice: "required"` | 400 `tool_choice_not_supported` (opencode never sends it) | pass |
| P10 | `parallel_tool_calls: false` + tools | 400 `parallel_tool_calls_not_supported` (opencode never sends it) | pass |
| P11 | tool with `strict: true` | 400 `strict_tools_not_supported` | pass |
| P12 | `store: true` | 400 `store_not_supported` | pass |
| P13 | wrong `model` id | **404** `model_not_found` (not 400) | pass |
| P14 | unknown top-level field | tolerated (ignored) | pass |
| P15 | `reasoning_effort: "none"` | 200, no `reasoning_content` deltas (thinking disabled) | pass |
| P16 | >446902-token prompt (~3.5 MB) | 400 `context_length_exceeded` at submit, message names `max_context 446902`, no SSE opened | pass |
| P17 | user message with `image_url` **data URL** + text (vision on) | 200 + full stream contract; log: `media_items: 1`, `raw_patches: 256`, `vision_tokens: 64` for the 1×1 PNG | pass (resolves the map's vision fog: `--vision` is live and the media path works end-to-end) |
| P18 | `stream: false` | 200 aggregate `chat.completion` with `message`, `finish_reason`, `usage` | pass |
| P19 | GBNF `grammar` (admits exactly 4 JSON strings) | output ∈ the 4 admitted strings (constraint enforced at the token level) | pass |
| P20 | `response_format: {type: json_object}` | 200, output valid JSON | pass |
| P21 | `json_schema` with a `pattern` keyword | 400 `json_schema_unsupported`, message names `pattern` | pass |
| P22 | `grammar` + constraining `response_format` | 400 `constrained_decoding_conflict` | pass |
| P23 | constraining `response_format` + `tools: []` | 400 `constrained_decoding_not_supported` | pass |
| P24 | P03's body via `https://ai.kido.ws/v1` (the caddy route opencode actually takes) | identical behavior: 200, full stream contract, `[DONE]` | pass |

**Run history (probe-side defects, serve-side none).** Run 1 failed 4 probes;
run 2 failed 1; run 3 is 24/24. All five failures were probe bugs, not serve
behavior: (1–2) the tool-call completeness check initialized its accumulator to
`false` (runs 1–2; the wire was correct — the captured delta carried complete
id + name + valid arguments); (3–4) the two constrained-output probes parsed
the SSE stream as aggregate JSON (run 1; fixed by requesting `stream: false`);
(5) the grammar probe's GBNF omitted the escaped `\"` quote characters, so the
serve correctly emitted unquoted JSON satisfying the (sloppy) grammar (run 2) —
which itself confirmed token-level enforcement. Every rejection probe (P09–P13,
P21–P23) passed on the first run with the exact documented codes.

**Not probed live (deliberate):** 429 capacity saturation (requires 68
in-flight requests; would disturb live traffic — the code path is cited in
`serve-capabilities.md` §5) and engine death mid-stream (not inducible without
breaking the unit — code-cited in §3c there; zero observed in the 71 h window
or the e2e window).

## 2. Real opencode sessions

Driven as `opencode run --standalone` (private server per run; `--auto` for
permissions; `--format json` for the event stream; model
`ai.kido.ws/Qwen3.8-27B#xhigh`). Captured per session: exit code, wall clock,
event stream, and the serve request-log records for the session's window.

### W1 — tool-heavy coding task (31 s, exit 0)

Task: create `stat.py` (CSV column stats CLI), `data.csv`, `test_stat.py`, run
the test, iterate until it passes, report the output verbatim.

- 4 agent steps, 5 tool calls: 1× `skill`, 3× `write` **batched in a single
  step** (parallel tool calls), 1× `shell`. All `completed`.
- Serve side: requests 627–630, all `stop_token`, `fallback_reason: none`,
  `duplicate_arguments_merged: 0`, `schema_mismatch: 0`. Request 628 (the
  batched step) served `structured_call_count: 3` in one response — parallel
  tool-call parsing clean.
- Outcome: test passed on first run; files created; final report verbatim
  (in `e2e/sessions/w1/`).

### W2 — long reply, no tools (228 s, exit 0)

Task: a ≥6000-word tutorial on TCP congestion control, no tool calls.

- Single request (633): `finish: output_limit`, `completion_tokens: 32768`
  (exactly the client-sourced bound), `prompt_tokens: 15600`, TTFT 1.44 s,
  decode 224.47 s → **≈146 tok/s** sustained over 32,768 tokens at the 446902
  window with MTP4 (19,954 of 51,252 drafted tokens accepted; 12,813 rounds;
  ≈2.56 tokens/round).
- **Finding F1 — zero-visible-content `length` truncation.** With the `xhigh`
  effort and no thinking budget on the unit (`--default-thinking-budget`
  unset), the model spent the entire client-set 32,768-token output bound on
  `reasoning_content` and was cut at the bound before emitting any visible
  text. The opencode v2 store for session
  `ses_f0180baaeffe2IOtQKxoOfHD5X` shows the assistant message = one reasoning
  part (98,642 chars), `finish: "length"`, `tokens: {input: 15600, output: 0,
  reasoning: 32768}`, and the session ended `outcome: succeeded`. The wire
  contract was fully honored (opencode's `length`-handling is a normal
  completion — no error, no retry); the operational consequence is a **silent
  empty reply** for a thinking-heavy turn under a finite output bound. The
  binding constraint is opencode's declared `limit.output: 32768`; serve
  enforced exactly what the client asked. Mitigations available on the serve
  side (`--default-thinking-budget`) and config side (larger `limit.output`,
  lower effort) are recorded for the verdict; none is applied on the deployment.
- **Observation O1 — client JSON-stream artifact.** `opencode run --standalone
  --format json` wrote only the initial `step_start` event (250 B) to stdout
  for this session, although the store proves the full 100 KB assistant
  message was received, persisted, and the step completed. This is an
  opencode-client-side display/streaming artifact of `run --format json`
  (the serve log shows a clean single request; nothing on the wire was lost).
  Recorded for completeness; not a serve-compatibility finding.
- Serve-log note: `result.model_thinking_tokens` is 0 for all e2e requests —
  the field tracks the *applied* thinking budget (none configured), not the
  thinking actually emitted; opencode-side counting (from `reasoning_content`
  deltas) is the observable figure.

### W3 — repeated tool calls with repeated parameters (10 s, exit 0)

Task: create six one-word files, then run `wc -w` on each **one at a time**
(six separate calls), then a summary table.

- 8 agent steps: step 1 batched the 6 file writes (`structured_call_count: 6`),
  steps 2–7 ran the six `wc -w` calls sequentially (identical parameter
  schema, differing values), step 8 emitted the table (no tool call).
- Serve side: requests 641–648, all `stop_token`, `fallback_reason: none`,
  `duplicate_arguments_merged: 0`, `schema_mismatch: 0`, queue waits 13–19 ms.
  The repeated-parameter path (2026-09-29 demotion class) was exercised on the
  wire by 7 calls sharing one parameter schema: zero demotions, zero merges
  needed (no byte-identical repeats within a single call).
- Outcome: correct markdown table (in `e2e/sessions/w3/`).

### E2e aggregate

| Metric | Value |
|---|---|
| e2e requests (W1: 4 + W2: 1 + W3: 8) | 13 |
| request errors / client disconnects / engine deaths | 0 / 0 / 0 |
| parser demotions (`fallback_reason ≠ none`) | 0 (vs 2 `malformed_structure` in the 71 h before the window: req 310 at 2026-10-02 09:57Z and req 533 at 2026-10-02 21:07:50Z, 13.5 min before the window — the serve-capabilities doc's 71 h snapshot predates req 533, which is why it recorded 1) |
| `schema_mismatch_arguments` / `duplicate_arguments_merged` | 0 / 0 |
| parallel tool-call batches served in one response | 3 (W1 req 628) and 6 (W3 req 641) |
| `output_limit` finishes | 1 (W2 — the F1 case) |
| queue wait | e2e requests: 12.4–18.9 ms; the 102 ms point in the window is on the operator's own concurrent 199K-prompt session request (req 634), not on an e2e request |
| TTFT (15–29K-token prompts) | 0.09–1.44 s (the ~0.09 s values are the warm-prefix W3 sequential steps 643–647; first-step TTFTs are 1.0–1.5 s) |

## 3. MTP transparency

MTP4 + `--lm-head-draft` is **fully transparent to opencode**: the wire shape
is identical (same SSE events, same `finish_reason` values, same usage fields;
no client-visible speculative field exists or is needed). The serve log's
`speculative` counters (per request) show the engine internals:
`accepted_per_position` decaying across the 4-position draft window (e.g.
W1 req 627: 282/199/141/101 of 407 rounds), `fallback_steps: 0` everywhere,
2.56–4.36 effective tokens per MTP round across the e2e requests (W2's
32,768-token turn: 32768/12813 = 2.56). Nothing in
opencode's stream contract is affected; no probe or session showed a
difference attributable to MTP.

## 4. Failure shapes: observed vs documented

| Failure shape | E2e evidence | Status |
|---|---|---|
| Parser demotion → silent stall | 0 observed in the e2e window (2 in the 71 h before the window: req 310, req 533) | **open risk** — still the only client-invisible failure mode; F1 (zero-visible `length`) is its benign cousin |
| `length` truncation | 1 (W2: the full 32768 bound consumed by reasoning) | handled per contract; operational gap recorded as F1 |
| Context overflow 400 | P16: exact code + message at submit, no GPU work | classified `context-overflow` (fatal step) by opencode; unreachable in practice (client fits to the same 446902 window) |
| Engine death mid-stream | 0 observed (not inducible safely) | code-cited: in-band SSE error, no `[DONE]` → opencode retryable dead-stream + continuation |
| 429 capacity | 0 observed (not inducibly without disturbing traffic) | code-cited: synchronous 429 `server_overloaded` at C+P=68 full → opencode retryable |
| Mid-stream client disconnect | 0 in e2e window (3 in the prior 71 h window) | normal; 499 recorded serve-side |

## 5. Inputs to the verdict (#144)

- **Conformance:** every opencode request shape probed or exercised is
  accepted; every response contract checked (stream framing, finish-reason
  discipline, tool-call identity, usage accounting, error envelopes) is met in
  the strongest form. No conformance gap found in the live layer.
- **Operational risks carried into the verdict:** (a) F1 — thinking-heavy
  turns can consume the whole client-set output bound leaving zero visible
  content; silent, `succeeded`-coded, no server signal beyond the JSONL;
  (b) the demotion class persists at low frequency with no client-visible
  signal; (c) model coverage (single architecture family, single model id) —
  structural, outside this ticket's evidence scope.
- **Performance envelope (this window, 446902 context, MTP4):** TTFT 0.09–1.44 s
  at 15–29K-token prompts (first steps 1.0–1.5 s, warm-prefix steps ~0.09 s);
  ≈146 tok/s sustained decode over a 32,768-token turn; e2e queue waits
  12.4–18.9 ms with a concurrent session in flight.

## Verification notes

- Probe runner and all three result files are on this branch
  (`e2e/probe_suite.py`, `e2e/results/`); re-running
  `python3 e2e/probe_suite.py --out e2e/results` against a live unit
  reproduces the final run (the suite is idempotent and short-running).
- Session event streams and exit codes are under `e2e/sessions/w{1,2,3}/`.
- Request-log mining is reproducible from the log file (schema v22;
  `request.request_id` keys; per-instance id counter — records with the same
  id from earlier server instances appear earlier in the file and fall
  outside the e2e window by timestamp).
- No completions were generated outside the probes and the three sessions;
  the unit was never stopped, restarted, or reconfigured during the run.
