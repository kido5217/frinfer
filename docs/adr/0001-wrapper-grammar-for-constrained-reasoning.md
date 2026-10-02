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
