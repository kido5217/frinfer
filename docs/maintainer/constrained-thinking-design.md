# Constrained thinking design: the wrapper-grammar route

Status: **superseded design context** (ADR
[0001](../adr/0001-wrapper-grammar-for-constrained-reasoning.md) amendment, ticket #245). The
wrapper semantics this document specifies are implemented today by upstream's
`text::GrammarCompiler` reasoning-prefix composition, not by the retired fork hand-written wrapper
(`src/models/qwen3_5/frontend/grammar/thinking_wrapper.cpp`,
`Frontend::compile_grammar(..., ConstraintScope::Thinking)`). The active constraint authority is
[Constrained decoding](constrained-decoding.md). This document is retained for its design rationale,
wrapper construction, rejected alternatives and verification plan, not as a description of the
shipped implementation.

Originally **design, handed off for execution** (wayfinder map #95; seed #90). Selected on prototype
evidence: wrapper route `proto/wrapper-thinking-boundary` @ `0ea1a20d` (ticket #96); close-rule
prototype `proto/close-rule` @ `6c1b1b8b` (ticket #97, rejected alternative); research
`research/mask-activation-boundary` @ `b7095db4`, `research/vllm-enable-in-reasoning` @
`e4050edd`, `research/dodge-mechanics` @ `5f634b1c` (tickets #92–#94).

## Problem

`grammar` / `response_format` constrain the **answer region only**. Under a constraining request
with thinking explicitly enabled, Qwen3.8-27B drafts the answer inside the unconstrained
reasoning stream and the answer region comes back empty (`finish_reason: stop`, `content: ""`).
The measured mechanism (tickets #94 and #96): the mask plan derives from the parser's phase, and
the column following the close marker is planned as the first answer column — masked with the
converted schema's **root row**, which admits `{`-starting tokens and forbids whitespace. That row
is exactly what displaces the model's natural whitespace close (scenario M, established by
flipping the deciding column's row in ticket #96): the model emits the close marker `</think>`
**immediately followed by `{`**, which the parser's close rule (R2: `</think>` plus format
whitespace) does not accept. The parser therefore stays in Reasoning, no answer column beyond the
displaced decider is planned, the grammar state stops advancing, and natural EOG ends the turn.
The deadlock is two-rule: R2 wants whitespace at the deciding column, the schema root forbids it.

The #86 default routes around this for the default case (a constraining request with no explicit
thinking field resolves thinking off). This design fixes the **explicit-enable path**.

## Decision

Select the **full-encoding wrapper grammar** (ADR [0001](../adr/0001-wrapper-grammar-for-constrained-reasoning.md)):
the grammar carries the thinking stream and hands off to the answer grammar at the close, so a
thinking-on constrained request produces free-form reasoning in `reasoning_content` and a
schema-valid non-empty answer in `content`, with the `</think>` split contract unchanged. The
wrapper applies **automatically** (no new request field or server flag) to a request that carries
a constraining `response_format`/`grammar` and resolves thinking enabled; the previous v1 behavior
for those requests is replaced.

## Contract

- Request surface: unchanged. `grammar`, `response_format` (`json_object` / `json_schema`),
  thinking enablement (`enable_thinking`, `reasoning_effort`), and the #86 default are unchanged.
- Selection: wrapper iff the request carries a constraint **and** resolves thinking enabled
  (`enable_thinking == true` or a non-`none` `reasoning_effort`) **and** the rendered prompt starts
  in reasoning. Otherwise the existing paths are untouched: thinking-off constrained requests use
  the converted schema grammar as today; unconstrained requests are unchanged.
- Streaming and channels: unchanged. Reasoning streams as `reasoning_content`, the answer as
  `content`; the split rule stays R2 (`</think>` followed by format whitespace).
- Error codes: unchanged (the existing constrained-decoding fail-closed set).

## Design

### Wrapper construction

At constraint compilation, the converted schema grammar is wrapped (the vendor converter and its
context-accurate allowlist are unchanged):

```
root   ::= body close ws answer
body   ::= marker-avoiding DFA (states are prefixes of the close marker; a byte that does not
           continue the current prefix returns to the accepting body state, and a `<` always
           (re)enters the prefix chain — the chain never empties mid-prefix, so the marker cannot
           be split across body elements; the prototype's `B`-state grammar)
close  ::= the wire-format close marker's literal bytes
ws     ::= [ \t\r\n]+
answer ::= the converted schema grammar's root (rule names kept collision-free)
```

- The close marker text is taken from the chat wire-format configuration (`ChatParseWireFormat`),
  not hard-coded — one source of truth with the parser.
- `ws` after the close is what satisfies R2: the parser's close rule needs format whitespace, and
  the wrapper forces exactly that byte at the deciding column, so the parser closes and the mask
  engages by construction.
- `body` forbids the literal close sequence inside reasoning: the sequence closes the reasoning
  phase wherever it appears, so a "quoted" `</think>` in reasoning is not representable (the model
  rewrites such text). Once a prefix is begun it must resolve — the marker completes, or the chain
  restarts at the next `<` / returns to the body state. The prototype measured the continuation set
  with corner probes and saw no live stall; the implementation verifies the mid-prefix states.
- EOG: the prototype's wrapper admits EOG at completed body elements (an abort path,
  `root ::= body close ws answer | body`), which keeps the reasoning rows byte-identical to the
  full-vocabulary row — that measured variant is the policy adopted here, and no live run aborted
  (the model always closed and answered). Barring EOG (forcing a complete answer) is an
  implementation-time option that must be re-measured: it clears the EOG bit in the body rows and
  its behavior is unproven.

### Mask lifecycle

With the wrapper the grammar covers the whole stream, so the mask engages from **token 0** and the
phase-dependent deferral (reasoning columns unmasked until the answer boundary) no longer applies
to constrained requests. The reasoning region's rows are byte-identical to the full-vocabulary
row — the prototype measured **1 fill / 1,902 hits over 1,903 steps** (p50 1.4 µs) — so the
per-step cost is unchanged. The served answer region reuses the existing schema path (cold
counter-table fills are the known, one-time #79 gap). With selection automatic, the deferral
becomes unreachable for constrained requests; the implementation simplifies accordingly.

### Parser interplay

R2/R3 and the quoted-protocol handling are untouched (the close rule was the rejected alternative
precisely to avoid changing parser semantics). The thinking-budget early-close inserts control
tokens at the cap; the implementation verifies the inserted close path is admissible under the
wrapper (close + forced whitespace + answer).

## Rejected alternatives

- **Constrained-mode close rule** (ticket #97, `proto/close-rule` @ `6c1b1b8b`): close on
  `</think>` + first grammar-admissible byte. Works and composes, but is pinned to a fixed `{`
  byte — non-object-root schemas still deadlock — a `</think>{` arriving as a single token piece
  would fail closed, and it changes parser close semantics in constrained mode (including
  early-close on a quoted `</think>{`). Recorded as the minimal fallback shape.
- **Permissive-prefix wrapper** (prototype C): fails by construction — the permissive path never
  yields to the answer grammar.
- **Fail-closed rejection of explicit thinking + constraint** (design-space route (d)): not
  selected; remains the emergency fallback if the wrapper's residuals prove unacceptable.
- **vLLM comparison** (ticket #93): vLLM's `enable_in_reasoning=True` engages the converted
  structured-output grammar from token 0, so the reasoning text itself must satisfy the grammar —
  whole-generation constraint, not a reasoning-carrying wrapper. Its default mode (with a reasoning
  parser configured) defers the grammar until the reasoning end, with a silent unconstrained
  fallback when the boundary is missed — the behavior this design replaces.

## Verification plan (for the execution effort)

- **Grammar/producer**: wrapper compiles; DFA walk admits free-form reasoning bytes and transitions
  at the close; corner probes for mid-prefix states; wrapper size/rules bounded.
- **Serve/schema**: selection matrix — thinking-on + constraint selects the wrapper; thinking-off
  constrained and unconstrained requests are byte-identical to today; budget cap interplay.
- **Behavior (5090)**: the `add_episode` CombinedExtraction payload with `enable_thinking: true` +
  `response_format`: schema-valid non-empty `content`, free-form `reasoning_content` (baseline: all
  20 close-in-reasoning samples across the `55c67615` and `e83a9af5` probe campaigns came back
  empty).
- **Performance**: the constrained-decoding record — the mask-production budget
  (`docs/maintainer/grammar-mask-production.md`) and the measured constrained-path numbers
  (map #45 / ticket #54: ordinary Δ within ±0.24 ms/step at ~13.7 ms/step; cold first-request
  grammar setup ~0.7–0.8 s).
- **Docs**: the Constrained output section of `docs/serving.md` documents the wrapper behavior and
  its edge policies.

## Hand-off

Execution is a fresh wayfinder map (charter: execution-carrying) with this document and ADR 0001 as
its specification input. The prototypes and research branches above remain the primary evidence.
