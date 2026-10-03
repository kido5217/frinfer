# #165 — probing the undeclared-name emission axes

Wayfinder map #163, ticket #165; source issue #156 (open questions Q2/Q3/Q4).

**Question.** What makes the served model emit an undeclared tool name — and which
prompt axes produce a *genuine* (non-copying) undeclared-tool attempt that the
chat parser demotes as `undeclared_tool`?

**Setup (all probes).** Scratch serve on `127.0.0.1:8899` (never the deployed unit),
artifact `qwen3_8_27b_nvfp4.ninfer`, froggeric v22.5 template, `--preserve-thinking`,
`--spec mtp --draft-tokens 4 --lm-head-draft`, `--max-context 32768`,
`--max-concurrency 1`. Two sampler regimes:

* **greedy** — `--greedy` forces temperature 0 (exact argmax), the same serve for
  every axis probe. Deterministic: every repeated probe returned byte-identical
  parse fields (14 repeat generations, all matched).
* **sampled** — the same serve restarted without `--greedy`, requests carrying the
  deployed unit's sampler (`temperature 1.0`, `top_k 20`, `top_p 0.95`), used to
  test whether the #151 "framing 1 refuses" anchor is sampler-dependent.

Per-probe evidence is the HTTP/SSE response (content, reasoning, finish reason,
in-band error) joined to the serve's `request_done` record (`marker_seen`,
`call_attempted`, `structured_call_count`, `fallback_reason`, `finish_reason`).
The join was validated by matching `tool_count` to each request; zero mismatches.

**Scale.** 69 distinct probe definitions, 91 generation calls: 67 greedy + 24
sampled (38 of those explicit repeats). 400-rejected probes (D4/D5, below) do not
emit `request_done` and were excluded from the join.

---

## The one genuinely-emitting axis

Under **greedy**, the only route to a parsed `undeclared_tool` demotion that is
*not copying* is:

> the visible history contains an assistant tool call to name **X**, **and** the
> final instruction explicitly names **X**, **and** **X** is absent from the
> `tools` list (a non-empty list is required for the contract to be active).

This is the #156 "misremembered / renamed declared tool" case. The model's own
reasoning shows a deliberate decision, not a copy — e.g. the #151 anchor:

> "…the previous turn, I called a 'frobnicate' tool and it worked… I think the
> safest approach is to attempt the call as the user requested, since it worked
> in the previous turn."

Demotions confirmed for X = `frobnicate` (F1a/A1b/R1/R2/F16/F23/R3), `read_file`
(F9/F13/F25), `get_weather`, `send_email`, `search_web` (F26). Refusals for
X = `web_search` (F6/F22/F24), `frobnicate_v9` (F26) under greedy — but that
refusal is **not stable** (see sampler dependence).

**Copying is not genuine.** The #151 verbatim-print framing (A2/S1–S4) demotes
100% of the time, in both regimes, but the model is only transcribing supplied
bytes. It remains a valid *signal-path* forcing (deterministic), not a truthful
*model-behaviour* probe.

**Everything else resolves to a declared name or refuses.** No other axis
produced a parsed undeclared call:

* semantic need with wrong declared tools → **substitute** the declared tool
  (`allowed`) or **refuse** in text; never a hallucinated name (F2a/F2b/F8);
* renamed tool when the instruction does **not** name the old name → model
  adapts to the declared name (F1b/F1c/F20);
* casing / namespacing mismatch → model emits the **exact declared** name
  (C6/F4 `Frobnicate`; F11 `mcp__srv__read`);
* long filler → flips substitution to refusal, not to an undeclared attempt
  (E2/E3);
* prose-only tool descriptions never produce a parsed call (B2/B3/B5).

---

## Matrix

`CALL` = accepted structured call (name in parens); `DEMOTED` = parsed
`undeclared_tool` (or other class); `TEXT` = no marker, ordinary text completion;
`REJECTED(code)` = HTTP 400. `[g]` greedy, `[s]` sampled.

| Probe | Axis | Tool declared / instruction | Greedy | Sampled |
|---|---|---|---|---|
| A1a use_frob_simple | A | `[allowed]`, "use frobnicate" | CALL (allowed) | — |
| A1b use_frob_history | A | `[allowed]`, history frob, "same frobnicate again" | **DEMOTED undeclared_tool** ×3 | **6/8 DEMOTED, 2/8 TEXT refusal** |
| A2 copy_frob_literal | A | `[allowed]`, print literal frob block | **DEMOTED undeclared_tool** (copy) | 4/4 DEMOTED |
| A3 use_allowed | A | `[allowed]`, "use allowed" | CALL (allowed) | — |
| A4 copy_allowed_literal | A | `[allowed]`, print literal allowed block | CALL (allowed) | — |
| B1 tools_field_only | B | `[allowed]`, "use frobnicate" | CALL (allowed) | — |
| B2 system_only_empty_tools | B | `[]`, prose describes frobnicate | TEXT (JSON-style block) | — |
| B3 system_and_tools_field | B | `[allowed]`, prose + field | CALL (allowed) | — |
| B4 system_contradicts | B | `[allowed]`, prose says only allowed | TEXT refusal | — |
| B5 no_tools_key_system_prose | B | no `tools` key, prose | TEXT (JSON-style block) | — |
| C1 matching_declared | C | `[frobnicate]` | CALL (frobnicate) | — |
| C2 nonmatching_single | C | `[allowed]` | CALL (allowed) | — |
| C3 empty_list | C | `[]` | TEXT (refusal / JSON) | 2/2 TEXT |
| C4 two_tools_match_second | C | `[allowed, frobnicate]` | CALL (frobnicate) | — |
| C5 matching_param_mismatch | C | `[frobnicate{other}]` | CALL (frobnicate, maps note→other) | — |
| C6 declared_capitalized | C | `[Frobnicate]`, lowercase instruction | CALL (Frobnicate) | — |
| D1 choice_absent | D | `[allowed]` | CALL (allowed) | — |
| D2 choice_auto | D | `[allowed]`, auto | CALL (allowed) | — |
| D3 choice_none | D | `[allowed]`, none | TEXT (JSON, contract off) | — |
| D4 choice_required | D | required | REJECTED(tool_choice_not_supported) | — |
| D5 choice_named_allowed | D | function allowed | REJECTED(tool_choice_not_supported) | — |
| D6 choice_allowed_tools_auto | D | allowed_tools auto | CALL (allowed) | — |
| E1 no_filler_control | E | `[allowed]` | CALL (allowed) | — |
| E2 long_user_filler | E | ~6k-token filler | TEXT refusal | 2/2 TEXT |
| E3 filler_then_system | E | filler + system | TEXT refusal | — |
| F1a history_frob_declared_allowed | F | `[allowed]`, anchor | **DEMOTED undeclared_tool** | (A1b) |
| F1b renamed_v1_history_v2_declared | F | `[send_email_v2]`, no name | CALL (send_email_v2) | — |
| F1c renamed_v1_again | F | `[send_email_v2]`, "same tool again" | CALL (send_email_v2) | — |
| F1d renamed_both_declared | F | `[send_email_v1, v2]` | CALL (send_email_v1) | — |
| F2a semantic_weather_no_tool | F | `[calculator, allowed]` | CALL (allowed) | — |
| F2b semantic_websearch_no_tool | F | `[calculator, allowed]` | TEXT refusal | 2/2 TEXT |
| F4 template_case_mismatch | F | `[Frobnicate]` | CALL (Frobnicate) | — |
| F5 declared_nonmatching_plus_weather | F | `[get_weather]`, "weather tool" | CALL (get_weather) | — |
| F6 history_undeclared_then_ask_again_named | F | `[calculator]`, history web_search, named | TEXT refusal ×3 | 4/4 DEMOTED |
| F7 history_undeclared_continue | F | `[calculator]`, "do the same" | TEXT refusal | — |
| F8 truncation_lookalike | F | `[get_temperature,…]`, "get_weather" | TEXT refusal | — |
| F9 renamed_named_old | F | `[read_file_v2]`, history read_file, named | **DEMOTED undeclared_tool** | 3/4 DEMOTED, 1/4 CALL (read_file_v2) |
| F10 renamed_named_old_v1_declared | F | `[read_file, read_file_v2]` | CALL (read_file) | — |
| F11 mcp_bare_suffix | F | `[mcp__srv__read]`, "read" | CALL (mcp__srv__read) | — |
| F12 truncation_control_declared | F | `[get_temperature,…]`, declared name | CALL (get_temperature) | — |
| F13 history_read_declared_web | F | `[web_search]`, history read_file, named | **DEMOTED undeclared_tool** | — |
| F16 history_frob_declared_frob_v2 | F | `[frobnicate_v2]`, history frobnicate, named | **DEMOTED undeclared_tool** | — |
| F18 no_history_read_v2_declared | F | `[read_file_v2]`, "read_file" | CALL (read_file_v2) | 4/4 CALL (read_file_v2) |
| F19 no_history_read_v2_old_name_noted | F | `[read_file_v2]`, "old name" note | CALL (read_file_v2) | — |
| F20 history_no_name | F | `[read_file_v2]`, no tool named | CALL (read_file_v2) | — |
| F21 history_declared_name_control | F | `[read_file_v2]`, names declared | CALL (read_file_v2) | — |
| F22 history_websearch_named_in_user | F | `[calculator]`, history web_search, named | TEXT refusal | 4/4 DEMOTED |
| F23 history_frob_named_declared_calc | F | `[calculator]`, history frobnicate, named | **DEMOTED undeclared_tool** ×2 | — |
| F24 history_web_unnamed_declared_read | F | `[read_file_v2]`, history web_search, named | TEXT refusal | — |
| F25 history_read_unnamed_user | F | `[read_file_v2]`, history read_file, named | **DEMOTED undeclared_tool** | — |
| F26 get_weather / send_email / search_web | F | `[calculator]`, history + named | 3/3 **DEMOTED undeclared_tool** | — |
| F26 frobnicate_v9 | F | `[calculator]`, history + named | TEXT refusal | — |

---

## Per-axis verdicts

* **A — Framing.** The two #151 framings are not symmetric framings of one
  behaviour. Use-framing **without** history substitutes the declared tool
  (`CALL allowed`); use-framing **with** history that established the name
  attempts the undeclared name; copy-framing always emits it. The anchor's
  refusal is a sampled minority, not the deterministic response (next section).
* **B — Placement.** When the `tools` field is non-empty the model always
  resolves to a declared name (substitute or refuse); prose descriptions never
  win and never yield a parsed call. With `tools` absent or empty the contract is
  off (`tools_enabled=false`, `frontend.cpp:746`) and the model emits a
  **JSON-style** `<tool_call>{"toolName": …}` block that the parser does not
  recognize — `marker_seen=false`, no demotion.
* **C — Relation.** Non-match → substitute; match → call; empty list → no
  contract. Param-schema mismatch is served (the model maps `note`→`other`),
  matching the B6 "mismatches serve" policy.
* **D — `tool_choice`.** Absent == `auto`. Only `none` disables the contract.
  `required` and a named function are rejected with `tool_choice_not_supported`
  (`tool_choice` is a two-value enum on this engine). `allowed_tools mode=auto`
  filters and behaves as `auto`. No `tool_choice` value produced an undeclared
  attempt.
* **E — Context distance.** ~6k tokens of filler flipped substitution into a
  text refusal (`"I don't have a \"frobnicate\" tool available…"`). It did not
  produce a demotion — but it does demonstrate the substitute/refuse boundary is
  fragile to context.
* **F — Wild causes.** The rename/mismatch hypotheses are answered in the next
  section. Only history + explicit re-name emits; every other wild hypothesis
  either substitutes, refuses, or resolves to the declared name.

---

## Answers to #156's specific hypotheses

1. **Misremembered / renamed tool.** Genuine, but only when the instruction
   *explicitly re-names the old tool*. If the instruction omits the name
   ("send a follow-up email", "call the same tool again", "read the file now"),
   the model adapts to the declared renamed tool — F1b/F1c/F20 all called the
   declared `*_v2` and were accepted. So a rename is a demotion source only for
   clients that echo the stale name into the prompt.
2. **Tool described only in prose.** Never yields a parsed undeclared call.
   With `tools=[]`/absent the marker is not parsed at all; with a non-empty
   `tools` field the structured list wins (B3 called `allowed`, not the
   prose-described `frobnicate`).
3. **Apparent tool-list truncation.** No undeclared attempt. A lookalike absent
   name ("get_weather" with "get_temperature" declared) produced a refusal and a
   suggestion (F8); a bare MCP suffix ("read" vs declared `mcp__srv__read`)
   resolved to the **declared** namespaced name (F11).
4. **Template / name mismatch.** No undeclared attempt. A case mismatch
   (declared `Frobnicate`, instruction `frobnicate`) resolved to the declared
   `Frobnicate` (C6/F4).

**Q3 (client-config diagnostic vs model hallucination).** The observed
`undeclared_tool` class is not free-form hallucination. In every case the
undeclared name text was **present in the prompt** — either supplied for copying
(A2) or carried by the conversation history and re-named by the instruction
(F1a/F9/F13/…). The model validated every other instruction against the declared
list (substitute or refuse), including semantic needs it could not satisfy. The
class is therefore a **client/model tool-list mismatch diagnostic**, matching
the issue's hypothesis.

**Q4 (#151 e2e forcing fidelity).** The copy framing manufactures the signal by
transcription; the higher-fidelity forcing for the same class is the
history + explicit-re-name prompt (A1b/F1a), which reproduces `undeclared_tool`
through a genuine tool-use decision. Its drawback is determinism: 6/8 demote and
2/8 refuse under the deployed sampler, so an e2e built on it needs either greedy
serving or a retry budget. That trade-off belongs to the map owner.

---

## Sampler dependence (the load-bearing uncertainty)

The #151 record — "tool-use framing refuses" — is a single sampled draw. Under
the deployed sampler (`temperature 1.0`, `top_k 20`, `top_p 0.95`) the anchor
payload A1b:

| Outcome | Sampled (n=8) | Greedy (n=3) |
|---|---|---|
| `DEMOTED undeclared_tool` | 6 (S2–S7) | 3 |
| TEXT refusal (no marker) | 2 (S1, S8) | 0 |

The refusal text is stable when it occurs:

> S1: "I don't have access to a \"frobnicate\" tool. The only tool available to
> me is \"allowed\". I can't call a tool that isn't in my available toolset…"
>
> S8: "I don't have a `frobnicate` tool available in my current tool set. My
> only available tool is `allowed`. I'm unable to make that call."

Greedy ≠ typical sample in both directions: F22 (`web_search`) **refused** under
greedy but **demoted 4/4** sampled; F9 (`read_file`) demotes greedily but accepted
`read_file_v2` in 1/4 sampled draws. Greedy guarantees reproducibility, not the
modal behaviour.

Identifier surface also matters and is not predictable: `frobnicate` and
`read_file` emit, `frobnicate_v9` refuses; `search_web` emits, `web_search`
refuses under greedy. Every probe is therefore reported with its exact payload;
no name-class rule is claimed.

---

## What was not exercised, and why

* **Only one model/artifact.** Qwen3.8-27B nvfp4 is the sole local NInfer
  artifact; no other-model campaign (map #163 scoped this out).
* **Only the OpenAI chat route.** The Anthropic messages route shares the parser
  and the same signal contract; its wire translation was not re-probed.
* **`tool_choice='required'`/named could not be probed for emission** — the
  engine rejects them at the protocol layer (`tool_choice_not_supported`), so
  their effect on the model is unobservable through this serve.
* **The deployed unit's exact sampler is only emulated** (temperature/top_k/top_p
  copied from its request log); the anchor's 2/8 refusal rate is a sample
  estimate (n=8), not a precise probability.
* **A mask/grammar forcing was not used** — `grammar`/`response_format` with
  `tools` is rejected by design (`constrained_decoding_not_supported`, #52).

---

## Raw evidence

All artifacts are ephemeral on this host, produced this session:

* `/tmp/opencode/research-156-emission-axes-evidence/probes.json` — 35-probe
  matrix (axis A–F).
* `…/probes2.json` – `probes6.json` — repeats, discriminator and sampled probes.
* `…/probes.jsonl` — greedy serve `request_done` log (schema v24).
* `…/probes-sampled.jsonl` — sampled serve log.
* `…/out/` — greedy run 1 + `results.joined.json` (log-joined) + `full_dump.txt`
  + `consolidated.json`.
* `…/out2/` `out3/` `out4/` `out5/` — greedy repeats/discriminators;
  `…/out6/` — sampled repeats. Each has `results.json`, `raw/<id>.sse`,
  `raw/<id>.req.json`.
* `…/start-serve.sh` (greedy), `…/start-serve-sampled.sh` (sampled) — exact serve
  invocations; `…/gen_probes*.py`, `…/probe.py`, `…/consolidate.py`,
  `…/summarize.py`, `…/extract.py` — drivers.
* `…/serve.log`, `…/serve-sampled.log` — serve stderr.

Scratch serve shut down; port 8899 released; deployed unit untouched.
