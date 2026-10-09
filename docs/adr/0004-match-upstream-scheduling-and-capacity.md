# Match upstream scheduling and Main-KV capacity

Context: the fork pinned active-request preemption off and required the Main KV pool to cover every
resident lane at full `max_context` (ADR [0003](0003-full-resident-capacity-under-pinned-preemption.md),
#238/#244/#249). That divergence made a sub-full pool unstartable (#282) and removed the upstream
runtime's pause/reclaim/replay behaviour. The fork was otherwise current with upstream
`Neroued/ninfer` (`81c8ce09`).

## Decision

FrInfer matches upstream on scheduling and Main-KV capacity:

- **Preemption**: the `PreemptionPolicy` gate is removed and `pause_resident` is upstream's live
  pause body, so a resident may be paused/reclaimed to admit a younger request; the paused queue and
  the Snapshot/Replay recovery routes are reachable.
- **Capacity**: `validate_target_options` accepts upstream's range
  `[max(page_count(max_context), C), C × page_count(max_context)]`; `require_full_resident_capacity`
  is removed so `auto` may resolve down to the minimum; an omitted `--kv-capacity` follows
  `--max-context` (one lane).
- The fork's `test_engine_no_preemption_real` and `test_preemption_policy` are retired, and upstream's
  `test_engine_preemption_real` is restored and registered.

This supersedes ADR 0003 and reverses the #238/#244/#249 decisions.

## Considered options

- **Keep the pin and restore the pre-r8 complete-resource admission reservation** — rejected: it
  rebuilds code upstream deleted and diverges at the pressure site every future sync re-merges.
- **Keep the pin and only relax the validator** — rejected: a sub-full pool then fails a resident at
  runtime instead of at boot.

## Consequences

- A sub-full Main KV pool is a supported configuration: the runtime pauses/reclaims a resident when
  the pool cannot hold every active lane at full context, as upstream does.
- The accepted pool's lower end is reachable again; the pool need not equal the worst-case ceiling.
- The fork's remaining divergences are feature patches (streaming logprobs, reasoning-close,
  constrained admission), unaffected by this change.
