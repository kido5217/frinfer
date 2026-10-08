# Wrapper grammar for constrained reasoning

Context: thinking-on + constraining (`grammar`/`response_format`) requests returned an empty
answer region — the model drafted the answer inside reasoning; the schema-root row on the deciding
column displaced the whitespace close the parser's R2 rule requires, the parser stayed in
reasoning, and no answer column beyond the displaced decider was planned (tickets #94/#96). Two
working prototypes settled the fix: a full-encoding wrapper grammar (ticket #96: 4/4 schema-valid
answers, free-form reasoning, reasoning-region mask cost ~1 fill per ~1,900 reasoning steps) and a
constrained-mode parser close rule (ticket #97: 4/4, but object-root schemas only). We decided the
**wrapper grammar**, applied automatically to constrained requests that explicitly enable
thinking, because it covers every schema shape, satisfies R2 by construction, and leaves parser
close semantics unchanged. See `docs/maintainer/constrained-thinking-design.md`.

## Considered options

- **Wrapper grammar (chosen)** — whole-stream grammar: marker-avoiding reasoning body, forced close
  and whitespace, then the converted schema. Costs a mask-lifecycle simplification (engagement from
  token 0) and per-request wrapper construction.
- **Parser close rule** — accept `</think>` + the first grammar-admissible byte. Smaller diff, but
  pinned to a fixed `{` byte (non-object-root schemas still deadlock) and changes the parser's close
  semantics in constrained mode.
- **Both** — composed in the prototypes but added a contract surface for no extra capability.
- **Fail-closed rejection** of explicit thinking + constraint — kept only as the emergency fallback.

## Consequences

- Constrained requests with thinking enabled no longer dodge via a displaced close: the model's
  whitespace close is admitted at the deciding column, and the literal close sequence is not
  representable inside reasoning. The wrapper's body still admits EOG (an abort path); no live run
  aborted, and barring EOG is an unmeasured implementation-time option.
- The phase-dependent mask deferral becomes unreachable for constrained requests, removing a
  special case from the mask lifecycle.
- Every constrained thinking-enabled request constructs a wrapper around its converted schema; the
  wrapper's rule names and size must stay bounded (measured within the existing budget).

## Amendment (ticket #245): the XGrammar composition is the wrapper authority

Ticket #239 (verdict C) adopts upstream's vendored XGrammar constrained-decoding stack and rebuilds
this ADR's semantics on it. Upstream already implements the whole-stream wrapper this ADR chose:
`text::GrammarCompiler::compile(source, reasoning_close, continuation)` concatenates a
`reasoning_prefix` DFA — a marker-avoiding body that admits any bytes through the first complete
delimiter — before the answer grammar whenever `reasoning_close` is non-empty
(`src/text/grammar.cpp`), and `Frontend::make_output_session` supplies the canonical close
serialization exactly when the rendered prompt starts in reasoning. That construction satisfies R2
by construction and is covered by upstream's own framing test in `tests/test_grammar.cpp`.

Therefore **upstream's `GrammarCompiler` reasoning-prefix composition is the winning authority**.
The fork's hand-written wrapper (`src/models/qwen3_5/frontend/grammar/thinking_wrapper.cpp`,
`Frontend::compile_grammar(..., ConstraintScope::Thinking)`) is superseded and is retired with the
in-tree trie stack; the fork's separate `thinking_enabled` gate is subsumed by
`starts_in_reasoning`, which is the exact condition under which upstream inserts the prefix. The
`reasoning_end` control is unaffected: it is a fork-only model-round control
(`OutputSession::request_reasoning_close` / `Request::reasoning_end`, consumed at the engine commit
boundary) that upstream does not have, and it stays on the request-owned matcher. ADR-0002
(tool-call demotion signaling) is orthogonal to the constraint stack and is unaffected.

Status: this amendment is the authority for the #245 port; the fork wrapper and trie producer are
removed when that port lands. The upstream grammar core is already in the tree
(`src/text/grammar.*`, target `ninfer_text_grammar`) and passes its test, but no consumer has
switched off the fork stack yet.
