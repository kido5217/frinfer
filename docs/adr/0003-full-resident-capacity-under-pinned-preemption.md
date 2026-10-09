# Full worst-case Main KV capacity under pinned preemption

> **Superseded by [ADR 0004](0004-match-upstream-scheduling-and-capacity.md).**

Context: the fork pins active-request preemption off (`pause_resident` returns false; #238/#244).
The adopted upstream runtime still carries its fallback for the impossible case at
`src/runtime/engine/engine_core.h:2066-2072`: when the oldest resident cannot obtain its legal unit
and `pause_reclaim_victim` fails, it throws `std::logic_error("oldest resident cannot obtain its
legal unit")`. The pin makes that fallback reachable, so an undersized Main KV pool fails a resident
mid-serve instead of degrading (#246 reproduced capacity 512 with two ~448-token requests). The pool
is shared across lanes; each lane's Main KV frontier can reach `max_context`, and its MTP/DFlash
backend frontier is bounded by `max_context` too, with the backend pool sized on top of the Main
pool (the per-lane draft window extra in `persistent_layout`). The startup validator previously
accepted any explicit capacity whose page count fell in
`[max(ceil(max_context/64), C), C × ceil(max_context/64)]`, whose lower end is exactly a pool that
violates the pin's expectation.

## Decision

The Main KV pool must cover `max_concurrency` lanes at full `max_context`:

- **Explicit**: `validate_target_options` requires the page-rounded pool to equal the curve upper
  bound `C × ceil(max_context / 64)` and rejects a smaller one with `std::invalid_argument` naming
  the required pages/tokens and the supplied capacity.
- **Automatic**: the resolved page count is only known after the memory budget is applied, so
  `construct_model` rejects an `auto` resolution below `curve.maximum_main_page_groups` with the
  same contract message.
- The runtime's error branch becomes unreachable by construction. `engine_core.h` keeps upstream's
  bytes; only a NON-CALLABLE comment marks the pinned-off fallback (no behavior change).

## Considered options

- **Port the deleted `admission_policy` complete-resource reservation** — rejected: it rebuilds code
  upstream deliberately deleted (`b9114396`), hooks the runtime at the site every future sync
  re-merges, and re-derives a policy the fork no longer carries.
- **Accept the throw and surface a clean serve-layer rejection** — rejected: divergence at the
  pressure site every sync re-merges, and an impossible state still exists at runtime.
- **Keep the range's lower end and treat the throw as intended failure** — rejected: it fails a
  resident after acceptance, which the no-preemption contract forbids.

## Consequences

- The pool is always the design's worst-case ceiling, so the accepted range's lower end collapses to
  the upper end. The usable pool for inactive cache/checkpoints shrinks to the worst-case reserve —
  the accepted memory tax; `maximum_pages64` was already the range's upper bound.
- `auto` can no longer undersize; on a memory-starved machine it rejects at startup instead of
  degrading at runtime.
- An omitted `--kv-capacity` follows `--max-concurrency` lanes at full `--max-context` in the
  serve, so the product's own default invocation for a supported configuration starts without an
  operator-supplied capacity; the CLI keeps its single-lane `--max-context` default.
- Low-level Program tests that build deliberately tight pools keep doing so through the
  planner/finalize path, but must now declare a compliant `kv_capacity`. `test_engine_no_preemption_real`
  pins the accepted concurrent run, a cached-context restore under the full compliant pool, and both
  ends of the capacity rejection message.
