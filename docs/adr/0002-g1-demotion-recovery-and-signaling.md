# Demotion recovery and signaling for tool-call parsing

Context: when the model emitted a tool-call region the parser could not accept ("demotion"), the
serve returned the region's markup verbatim as assistant content with `finish_reason`
`stop`/`length` and HTTP 200. Clients running an agent loop (opencode v2.0.22) saw a text-only
turn, ended the run, and idled until a manual "continue" — a silent stall (wayfinder map #145,
design ticket #148; five demotions observed 2026-10-02, three of them work-stopping). The
pipeline research (#146) and the llama.cpp port-surface survey (#147) established: the recent
live sweep was entirely `malformed_structure`; the work-stopping shapes are a real call whose
own argument value embeds inline marker fragments (the B4 balanced-nesting rule makes the value
unrepresentable) and a turn cut by the output bound mid-region; quoted markers in prose are
benign byte-exact round-trips. The design: parse-side recovery ported from the vendored
llama.cpp engine, plus a client-visible, retryable signal for residual call loss. Research:
`docs/research/g1-demotion-pipeline.md` @ `research/g1-demotion-pipeline` `56b08c3c`;
`docs/research/g1-llamacpp-delta.md` @ `research/g1-llamacpp-delta` `fcda549e`.

## Decision

**Recovery** (parse/recovery only; no serve-side generation-time constraint):

- Salvage structurally complete calls from a region that fails to consume to its end, reusing the
  vendored engine's retained parse nodes: the strict parse keeps every node it completed before
  the failure (a failed rule creates no node of its own), so a completed call node is exactly the
  "name + all parameters closed + `</function>`" criterion. Zero `third_party/` changes.
- Adopt upstream's raw-value rule as a **second-chance parse**: when the strict region grammar
  rejects a candidate, re-parse it with parameter values as raw bytes up to a newline-framed
  close. Strict-first keeps the balanced-nested representation.
- Salvage only **structurally complete calls** (name, all parameters closed, `</function>`);
  never emit truncated arguments and never mutate bytes. Salvage covers prose tails and bound
  cuts alike; when no candidate consumes to end, the furthest-consumed complete-call candidate
  is salvaged (ties → earliest), and it publishes byte-exact without a signal.
- Classify demotions by whether a call was attempted (some candidate formed a function node);
  only call loss signals. Prose demotions stay silent byte-exact round-trips.
- A turn cut mid-region that leaves no complete call is unsalvageable and signaled;
  `finish_reason` remains `"length"`.

**Signaling** (serve contract; the region is withheld):

- The engine stream surfaces a demotion event (region + class + call-attempted) so the serve can
  withhold the region on the streaming route; the Engine's aggregate result keeps the bytes for
  CLI/bench consumers.
- Streaming routes: an in-band SSE error event replaces the normal finish. On opencode an
  in-band error is retryable-eligible, and a non-numeric `code` classifies as `UnknownProvider`:
  no output started → the same request is retried; output started (reasoning counts) → the
  synthetic "previous response was interrupted" continuation. Both are bounded by the
  10-decision per-step budget and surface on exhaustion — never idle.
- Non-streaming requests: HTTP `409` with `x-should-retry: true`.
- Per-class string codes: `tool_call_malformed_structure`, `tool_call_undeclared_tool`,
  `tool_call_duplicate_parameter`, `tool_call_invalid_tool_name`; `trailing_content` remains
  unassigned and inherits the uniform treatment if ever assigned.
- All three advertised routes (OpenAI chat, OpenAI Responses, Anthropic Messages) implement the
  signal in their native error shapes, with schema tests and `docs/serving.md` updated together.
- Diagnostics: `fallback_reason` is demotion-only; add `call_attempted` and `salvaged_calls`
  (JSONL schema v23).

## Considered options

- **Vendor baseline advance / new vendored file / patch** — rejected: the measured delta
  `05af0d2b` → `bed0a85660` buys 8 lines of unrelated LLM-jp dialect registration; the machinery
  ports from the already-vendored engine without drift debt.
- **Full upstream value semantics** — rejected: its until-delimiter rule fails a value carrying
  a balanced newline-framed nested pair, a shape that works today; the second-chance variant
  recovers inline fragments while preserving it.
- **No value-rule change (signal only)** — rejected: the work-stopping call losses are exactly
  the inline-fragment shape, and they would stay lost.
- **Upstream's "flush once the name is known" / vLLM close-at-end** — rejected: both emit
  truncated argument bytes; a truncated `write` must never execute.
- **Signal every demotion** — rejected: prose reports quoting markers arrive intact today;
  signaling them would trade a benign round-trip for retries.
- **Pre-output HTTP 4xx (delay the first chunk)** — rejected: opencode treats an in-band stream
  error before any output identically (same-request retry), so the route restructure buys no
  client behavior.
- **Publish the region and append the signal** — rejected as default: tens of thousands of
  degenerate markup bytes as the assistant's message invite continuing the junk; withholding is
  reserved for call loss (benign demotions still publish).

## Consequences

- Corpus semantics change by decision: four vectors are rewritten
  (`tool-call-unmatched-nested-parameter`, `tool-call-standalone-parameter-close`, both
  `later-candidate-must-consume-the-end` variants) and four added (live 876 shape, live 907
  shape, complete-call-before-cut, later-call-not-swallowed); upstream's prefix-stability
  contract (every prefix of every corpus feed parses; incremental reconstruction) becomes part
  of the gate.
- The advertised behavior changes across all three routes: a call-loss demotion is no longer
  `200` + markup + normal finish; clients see a retryable error and either recover or surface.
- A deterministic call-loss demotion consumes up to the client's retry budget before surfacing —
  the accepted cost of never idling.
- The serve's streaming path gains a demotion event and a withhold branch; the Engine's
  aggregate `content` keeps the demoted bytes, so CLI/bench behavior is unchanged.
- Deployment (nixos-configs pin bump + unit restart) remains the human's step; a post-deployment
  reaction e2e is optional.

Implementation note (2026-10-03, #149): the LENIENT `need_more_input` vs `fail` typing was dropped
as a vehicle: the retained-completed-node scan above gives the structurally-complete-call
criterion directly, and no consumer branches on bound-cut-vs-malformed. Taking a LENIENT pass would
also have contradicted the B3 optional-close tolerance, because a `</tool_call>` cut mid-literal
would stop reading as a complete call.
