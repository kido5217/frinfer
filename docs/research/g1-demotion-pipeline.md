# G1: the demotion pipeline and opencode's reaction map

Wayfinder map #145 (G1 — tool-call demotion → silent stall), ticket #146.
Part A: the current pipeline on master. Part B: opencode v2.0.22's reaction
to candidate post-fix wire shapes.

Sources read this session (2026-10-03): `src/models/qwen3_5/frontend/chat_parse_core.{h,cpp}`,
`tool_call_parser.cpp`, `output_session.cpp`, `include/ninfer/types.h`,
`src/serve/{openai_chat_response.cpp,request_log.cpp,operational_log.cpp}`,
`tests/fixtures/chat_parsing/corpus.json` (42 vectors), the request log
`/home/kido/trash/temp/ninfer-yarn-log.jsonl` (3,943 `request_done` records,
schema v22), the opencode store `~/.local/share/opencode/opencode.db`, the
opencode v2.0.22 source (`527f0b931`), and
`docs/research/opencode-requirements.md` @ `research/opencode-requirements`
8cde0058 (§4 re-verified against the source). Every citation below is a
file:line in these artifacts.

## 1. The five demotion classes

`ToolCallParseFallbackReason` (`include/ninfer/types.h`; the name mapping at
`:330-342`). The policy comment above the enum: normalization failures do
**not** validate the call — only a structure/identity failure can return a
complete marker region to ordinary content.

| Class | Exact condition | Layer |
|---|---|---|
| `malformed_structure` | (1) the region PEG fails to parse a candidate, or the parse does not consume to end-of-input (`chat_parse_core.cpp:164-172`); (2) the parse succeeds but no `kRuleToolCall` node exists (`:170`); (3) every candidate failed — the first recorded failure is reported (`:467,528-539`) | structural PEG (B2 candidate loop, `:473-532`) |
| `invalid_tool_name` | a call name fails `valid_function_name`: empty, > 128 bytes (`ChatParseOptions::tool_name_max_length`), or a byte outside `[A-Za-z0-9_-]` (`:186-192,481-484`) | adapter check 1 |
| `undeclared_tool` | a call name is not in the request's tool contract — `enforce_declared_names` is unconditionally true whenever tools are enabled (`tool_call_parser.cpp:141-152`, set at `:145`) | adapter check 2 (`:486-493`) |
| `duplicate_parameter` | a repeated parameter whose raw value bytes differ from the first occurrence's (`:499-516`); byte-identical repeats merge instead (`:507-518`, the #42 fix) | adapter check 3 (B5, `:494-520`) |
| `trailing_content` | **no production assignment** — the enum value appears only in `types.h` and two test helpers (`test_frontend.cpp:2059`, `test_chat_parsing_corpus.cpp:109`). "Prose after the last call" is the PEG root's `end()` requirement and lands as `malformed_structure` (corpus `tool-call-later-candidate-must-consume-the-end`) | — |

Order (B5): structure → declared-name policy → parameters. B6 normalization
runs on accepted calls only and never demotes: schema mismatches are served
with the raw value stringified; counters `schema_mismatch_arguments` /
`empty_arguments_omitted` (`chat_parse_core.cpp:403-434`, enum at `:357-361`).

## 2. The demotion → wire path (code)

1. Feed: a `tool_call_open` marker in either phase moves the session to
   `Phase::ToolRegion`, withholding the marker plus the format-whitespace run
   before it (B1: `:570-580` in content, `:624-634` in reasoning); a partial
   marker suffix stays held via `longest_suffix_prefix` (`:582-585`). In
   `ToolRegion`, every byte is held and nothing streams (`:667-669`).
2. Terminal flush (`terminalize`, `:673-699`): `ToolRegion` + contract →
   `resolve_tool_region` (B2): try every marker occurrence as a candidate
   start; accept the first candidate that parses to end-of-input **and**
   passes all adapter checks (`:473-532`). No candidate qualifies → the whole
   held region is published to the content channel and
   `diagnostics.fallback_reason = first_failure` (`:535-539`).
3. Session (`output_session.cpp:354-382`, `flush_terminal(merge_held=true)` —
   the limit **and** terminal paths): a demotion appends the region's bytes
   as the final content delta (`:378-381`); an accepted region instead clears
   the withheld framing whitespace (`:375-377`).
4. Serve (`src/serve/openai_chat_response.cpp:118-129`): `OutputLimit` /
   `ContextCapacity` → `finish_reason "length"`; `StopToken` / `StopString` /
   `None` / `Cancelled` → `"stop"`.

**Wire shape of a demotion:** the pre-region reasoning/content deltas were
already streamed; the demoted region arrives as final `content` deltas;
HTTP 200; `finish_reason` `"stop"` on a normal turn, `"length"` when the turn
was cut by the output bound; `[DONE]` follows. JSONL diagnostics:
`result.tool_call_parse` = `{marker_seen, structured_call_count,
fallback_reason, duplicate_arguments_merged, empty_arguments_omitted,
schema_mismatch_arguments}` (`src/serve/request_log.cpp:88-94`); the
operational log pretty-prints the demotion (`operational_log.cpp:296-304`).

## 3. Recovery coverage: salvaged vs fall-through

Salvaged by design (no demotion):
- **B2 candidate search**: a leading malformed or quoted region does not
  poison a later well-formed call, provided the later call's own subregion
  parses to end-of-input and passes the adapter checks (`:473-532`).
- **B1 framing**: the whitespace before a marker is withheld and re-emitted
  only on demotion; no byte is lost (corpus `tool-call-marker-split-across-rounds`).
- **R2–R5** (the reasoning boundary; `:588-656` in `feed_reasoning`, R5's
  terminal implicit-close at `:677-688`): first `</think>` followed by
  format whitespace closes reasoning; a close followed by other bytes is
  quoted protocol text and stays reasoning; a trailing partial marker is held;
  at terminal an implicit close is consumed.
- **B3**: a missing `</tool_call>` before the next call is tolerated
  (grammar comment `:69-78`, `call` rule `:104-105`).
- **B6**: schema mismatches serve (stringified).
- **Byte-identical duplicate parameters** merge (#42).

Falls through to demotion (the G1 surface):
- A region whose **tail** does not parse to end-of-input: prose after the
  last call, a truncated turn, an unbalanced nested `parameter` open/close
  inside a value (B4, grammar comment `:74-76`), a standalone close, or any
  PEG failure — `malformed_structure` for the whole held region, **including
  any real call inside it**.
- Name charset/length (`invalid_tool_name`), undeclared name with tools
  present (`undeclared_tool`), differing-value duplicate (`duplicate_parameter`).
- **Quoted markers in the model's own prose**: the hold starts at the first
  `tool_call_open` byte sequence anywhere — including a literal marker the
  model is *discussing*. Unlike the reasoning channel, where R3 gives
  `</think>` a "followed by non-whitespace ⇒ quoted" rule (`:612-614`), the
  tool marker has no quote/context discrimination.

**Truncated-region shape (the G2 interaction), answered:** a turn cut off
mid-`<parameter>` by the output bound ends in `Phase::ToolRegion` with the
partial region held; `flush_terminal(merge_held=true)` resolves it through
the same `resolve_tool_region`; the partial region fails the PEG `end()`
requirement → `malformed_structure` demotion on the **limit** path,
`finish_reason "length"`, HTTP 200. Confirmed live (payload 907 below): a
turn cut at the 32,768 output bound mid-region returned its 32,768-token
markup as content with a length finish. No corpus vector pins this shape.

## 4. Ground truth: the five live payloads

Log sweep (this session): **5 demotions, all `malformed_structure`**;
`marker_seen: true`, `structured_call_count: 0` in every one. The demoted
bytes are not in the JSONL (diagnostics only) — recovered from the opencode
store (assistant messages).

| Req | When (UTC) | Prompt / completion | Client session | Trigger | Client-visible result |
|---|---|---|---|---|---|
| 310 | 10-02 09:57:47 | 90,823 / 10,073 | a map-#86 ("thinking default under constrained requests") research subagent (`explore`) | **(a) quoted marker in prose** — its report "Current-code facts for issue #86" quotes the literal `tool_call_open` marker in prose | benign round-trip: the tail re-emitted as content; the report text arrived (#86 closed ~38 min later) |
| 533 | 10-02 21:07:50 | 222,012 / 7,070 | map #139's adversarial review subagent (`general`) | **(a)** same — its verification checklist cites the marker | benign round-trip: the tail re-emitted; the review report arrived |
| 876 | 10-02 22:08:15 | 123,262 / 354 | the #146 research subagent itself (`ses_f0159c808…`) | **(b) marker inside the call's own arguments** — its own `shell` call carried an inline `<parameter=` fragment in the query string | **call lost**: the demoted call text became its final output (the garbled "result" the parent read); that subagent idled at 22:08:15 |
| 878 | 10-02 22:08:40 | 179,434 / 2,613 | **this session** (map #145 charting; the retry dispatch to the #146 subagent) | **(b)** again — the retry brief quoted marker fragments | **call lost**; session went `idle` at 22:08:40; the user's "continue" at 22:08:57 is the manual recovery |
| 907 | 10-02 22:17:33 | 233,186 / **32,768** (`output_limit`) | same (the findings-doc write) | **(c) degenerate repetition + limit-path truncation** — the model collapsed into marker repetition; the turn hit the output cap mid-region | 32,768 tokens of degenerate markup returned as content, `finish_reason "length"`; the write never executed; the user switched models |

Trigger families:
- **(a) quoted markers in prose** — a benign round-trip (bytes re-emitted as
  content); the corpus pins recovery of a real call that follows a quoted
  marker (`tool-call-quoted-marker-before-real-call`,
  `pr309-quoted-marker-before-real-call`), but every occurrence is still a
  recorded demotion.
- **(b) markers inside a call's own arguments** — the argument text carries
  `parameter` open/close fragments; unbalanced (B4) or tail-breaking content
  demotes the real call. **The call is lost and the client stalls.**
- **(c) degenerate repetition** — the model collapses into marker loops (seen
  live as a 32,768-token run), the turn is cut by the bound, and the region
  demotes on the limit path.

**Content-triggered demotion — part of the bug.** All five payloads come from
sessions whose *content discusses the protocol markup* (a map-#86 research
subagent and map #139's review subagent writing about the wire; the #146
research subagent and this charting session working on the G1 fix itself).
The mechanism is a deterministic property of the pipeline, not model
randomness: the hold starts at the first marker byte sequence in any
generated content — prose, quoted examples, or a call's own argument values
— and B2 then tries every marker occurrence as a candidate. A quoted marker
in prose is usually recovered or round-trips benignly (payloads 310/533).
The class becomes work-stopping when the call *itself* embeds marker
fragments — an unbalanced nested marker in a value fails B4, so no candidate
(the call's own subregion included) consumes to end and the real call is
demoted (payloads 876/878) — or when a markup loop is cut by the output
bound (payload 907). A conversation about the protocol thereby degrades its
own tool calls; this trigger class is a first-class robustness target (§7),
not an incident footnote.

## 5. The corpus vectors that pin demotion (and the gaps)

Demotion-class vectors in `tests/fixtures/chat_parsing/corpus.json`:
`tool-call-duplicate-parameter`, `tool-call-duplicate-parameter-conflict-wins`,
`tool-call-unmatched-nested-parameter`, `tool-call-standalone-parameter-close`,
`tool-call-later-candidate-must-consume-the-end`,
`tool-call-marker-quoted-without-call`, `tool-call-undeclared-tool-name`,
`tool-call-name-invalid-characters`,
`pr309-later-candidate-must-consume-the-end`.

Already pinned: quoted-marker-then-real-call recovery
(`tool-call-quoted-marker-before-real-call`,
`pr309-quoted-marker-before-real-call`) and the family-(b) demotion shape
(`tool-call-unmatched-nested-parameter`). Gaps (no vector today): the
**truncated-region** shape (payload 907) and a **degenerate-repetition**
region.

## 6. opencode's reaction map (v2.0.22)

From the reviewed requirements doc §3.7/§4 (source re-verified this session):

- **Retryable 4xx** (RateLimit 429; ProviderInternal — 408/409/any 5xx or
  server-error codes without a 4xx status; UnknownProvider): retried with
  exponential backoff from 2 s, per-gap cap 10 s, **at most 10 retries**
  (≈84 s total), jittered; `Retry-After` is a floor, capped at 15 min
  (`packages/core/src/session/runner/retry.ts:31-40,42-62`). The
  `x-should-retry: true|false` response header **overrides** retryability
  (`provider-error.ts:75-78`).
- **Non-retryable 4xx** (InvalidRequest: context-overflow, payload-too-large,
  auth, quota, content-policy, `model_not_found`, any other 4xx): the step
  fails; with `compaction.auto: false` there is no automatic recovery; the
  error surfaces in the session.
- **Dead stream** (`incomplete-stream` — the stream ended without a
  `finish_reason`, `openai-chat.ts:1224-1229`; in-band SSE error event,
  `openai-chat.ts:280-286,1048-1057`; mid-stream transport read failure,
  `step.ts:281-285`): if **no output
  had started**, the whole step is retried with the same request
  (`step.ts:192-197`); if output had started, the partial assistant message
  persists, a synthetic user message *"The previous response was interrupted.
  Continue from where you left off without repeating completed content."* is
  appended, and the loop continues with a fresh step (`step.ts:253-261`,
  `llm.ts:35-37,278-286`). On exhaustion the error surfaces.
- **200 + markup + `finish_reason "stop"|"length"`** (today's demotion shape):
  a **normal completion** — the step ends, the agent loop has no pending tool
  calls, and it waits for the next input (the silent stall; observed).
  `finish_reason "length"` is likewise normal (`step.ts:163-172` only escalates
  `"unknown"`).

Design levers this yields: a pre-output 4xx + `x-should-retry: true` (or a
5xx-like code) is retried up to 10× with the same request, then surfaces an
error — no silent idle; an in-band error event after output started routes
through the continuation path — the loop resumes without user action.

## 7. What this implies for the fix (input to the design, #148)

Robustness (parse-side; the llama.cpp-port ranking is #147's job):
1. **Value semantics** (family (b), the work-stopping shape): a value rule
   that admits marker fragments as text — cf. #147's M2 delimiter-terminated
   values, a #148 decision against the corpus-pinned rows 21/22 (the
   unbalanced nested `parameter` rule is what kills the real call today; a
   "quoted marker stays text" rule à la R3 would be the belt-and-braces
   prose form).
2. **Truncated-region handling** — a region cut by the bound (payload 907):
   close-at-end à la vLLM vs. demote-as-markup vs. a distinct signal.
3. **Degenerate repetition** — the 32,768-token loop: is a repetition
   collapse detectable earlier (the output bound is the only stop today)?

Signaling (serve contract): two viable shapes — (A) a pre-output 4xx for
demotions that would leave nothing to continue from, with
`x-should-retry: true` → bounded retry then a surfaced error; (B) an in-band
SSE error event when output has already started → the continuation path
(synthetic "continue"). Open questions for #148: the "no output started"
boundary against reasoning/content deltas already streamed; the per-class
code vocabulary; route scope (OpenAI chat route only, or all three); the
retry-loop's behavior under repeated demotion.

**Operational consequence (map-level):** because the trigger is content
(§4), sessions doing this map's work are the most exposed — they should run
on a model not served by ninfer-yarn until the fix lands, or accept manual
"continue" recoveries (five payloads, three of them work-stopping). Once
the fix lands, the corpus gate must cover the true gaps (§5): the
truncated-region shape and a degenerate-repetition region; and if the M2
value semantics are adopted (#147), the family-(b) vectors rows 21/22 are
rewritten alongside.

## 8. Incident note

The five payloads were observed live, not reconstructed: a map-#86 research
subagent (310 — quoted markers in its report prose), map #139's adversarial
review subagent (533 — same), the #146 research subagent's own demoted
`shell` call (876 — the inline fragment in its query string ended its
investigation), and this charting session (878 — the retry dispatch to that
subagent was lost, the session idled at 22:08:40, and the user's "continue"
at 22:08:57 recovered it; 907 — the findings-doc write collapsed into a
markup loop and demoted on the 32,768 output bound). Session evidence:
`ses_f03f5ebeeffeshC0Dwc8EU7Yai` (map #86), `ses_f0197ab5effeMUmnDwneXjf8gQ`
(map #139 review), `ses_f0159c808ffe734rS9NWVeOe0C` (#146 research),
`ses_f01bf532dffezDA16aYsYdnVPi` (charting).
