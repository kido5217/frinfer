# Wayfinder map #236 / ticket #237 — fork-vs-upstream delta map for the 81c8ce09 sync

Sync target: upstream `Neroued/ninfer` `master` at `81c8ce09`; last synced tip and merge base
`d44ab584` (`docs/maintainer/upstream-sync.md`). Upstream delta: **44 commits**, 452 files
(+80310/−35101). Fork delta since the base: **94 commits**, 344 files. This document is an index
that later sessions zoom; it does not paste diffs.

## Provenance and method (read-only)

Every claim below comes from a command run in this session against the local clone. No file in the
product tree was modified; the only artifact is this file.

| Fact | Command |
|---|---|
| merge base / tips / counts | `git merge-base master upstream/master`; `git rev-parse --short upstream/master`; `git rev-list --count master..upstream/master` |
| per-commit stats | `git log --reverse --shortstat --format='@@%h|%s' d44ab584..upstream/master` |
| commit bodies | `git log -1 --format='%s%n%n%b' <sha>` |
| overlap set | `comm -12 <(git diff --name-only $BASE..master \| sort) <(git diff --name-only $BASE..upstream/master \| sort)` |
| per-file theme/patch | `git log --format=%h $BASE..<side> -- <file>` mapped to the group tables below |
| added/deleted classes | `git diff --name-only --diff-filter={A,D} $BASE..<side>` |
| upstream policy text | `git show 81c8ce09:AGENTS.md`, `git show 81c8ce09:docs/maintainer/constrained-decoding.md` |
| downstream tickets | `gh issue list --repo kido5217/frinfer` (#238–#242 titles quoted below) |

Fork patch areas are identified by the fork commits named in the ticket; the 121-row table in
[Overlap file set](#overlap-file-set-121-files) attributes every overlapping path to its upstream
theme(s) and fork patch area(s). Fork-side commit groups not named in the ticket (`rename`,
`chat-parse`, `demotion-signaling`, `version`, `show-prompt`, `reasoning-token-schema`,
`docs`, `justfile`, `devshell-cache`, ancestry merges) are labelled literally.
`ancestry(merge)` = a fork merge commit that touched the file; it is not a patch area.

The overlap-file table marks 121 of the 452 upstream-touched files. **Zero add/add collisions; one
modify/delete collision** (`src/models/qwen3_5/program/storage/context.cpp`, upstream deleted,
fork modified 4×). The historical conflict set (`AGENTS.md`, `bench/README.md`,
`src/models/qwen3_5/execution/text.cpp`, `src/models/qwen3_5/program/planning/startup.cpp`) has
**grown to 121 files** — all four historical paths are still present.

## The 44 upstream commits — grouped inventory

Adoptability is the merge ticket's first decision per group: **clean** (fork never touched the
files), **mechanical** (same path, edits are reconcilable/additive without semantic choice), or
**semantic** (both sides implement competing behaviour and a design decision is required).

### Group (a) — runtime / scheduling rupture (5 commits, 183 files, 72 overlap)

| commit | subject | files | adoptability | fork patch area it collides with |
|---|---|---|---|---|
| `b9114396` | feat(runtime): replace context cache and add preemptive scheduling | 176 | **semantic** (rupture) | grammar lifecycle (`context.cpp` deleted under it, `transactions/commit.cpp`, `program_impl`), logprob transport (`decode.cpp`/`prefill.cpp`/`round_buffers.h`), YaRN planning (`startup.cpp`/`request_plan.cpp`/`text.cpp`) |
| `911d34db` | fix(runtime): preserve cache sources across admission waits | 25 | **semantic** (builds on the new runtime) | logprob transport, request_log/serve metrics, `types.h`, `request_plan.cpp` |
| `a8e212ac` | perf(engine): reduce preparation and context management overhead | 25 | **semantic** | `engine_core.h`, `paged-kv-cache.md` (fork rename), `test_frontend.cpp` (fork grammar/reasoning_end) |
| `064965c7` | refactor(kv): require explicit publication streams | 7 | **mechanical→semantic** (API: `cudaStream_t` → explicit stream on KV publication) | `program_impl.cpp`, `storage/context.cpp` (later deleted) |
| `75a89050` | fix(runtime): wait for compute before request cleanup | 1 | **mechanical** (6 lines) once (a) lands | `transactions/commit.cpp` (fork grammar commit boundary) |

`b9114396` adds `src/runtime/engine/scheduler.h` (comment: *"Membership and fairness only"*),
`src/runtime/engine/context_cache/`, `src/models/qwen3_5/program/storage/{checkpoints,sequence}.cpp`,
`transactions/{context_transaction,pause,reclaim,replay,binding}.cpp`, and
`tests/models/qwen3_5/test_engine_preemption_real.cpp`; it deletes the admission-policy, pressure
planner, materialization and old context-cache planners (19 paths). It is a wholesale replacement of
the fork's `src/models/qwen3_5/program/` + `src/runtime/engine/` core, i.e. of almost every file the
fork's grammar and logprob patches copy edited.

**Invariant at stake.** The fork's product line, stated in `AGENTS.md` (root): *"one GPU, one
resident model, startup-fixed concurrency of one to eight requests, bounded FIFO ingress, **no
active-request preemption**, and one compact decode batch per round"*, and repeated in
`docs/maintainer/engine-architecture.md` line 45 (active-request preemption listed as out of scope).
Upstream's replacement explicitly adds preemptive scheduling and reframes the product line as *"one
to eight resident execution lanes fixed at startup"* (`git show 81c8ce09:AGENTS.md`; the words
"no active-request preemption" and "bounded FIFO" are gone upstream). The fork's per-round grammar
mask lifecycle (`docs/adr/0001`, `docs/adr/0002`) and its logprob-topk transport assume the current
round/commit boundary; preemption/replay recovery is a new axis through both.

**Downstream question (→ #238).** Does the fork adopt the new runtime wholesale (accepting preemption
as a capability and updating the product line), adopt only its cache/perf parts while keeping
FIFO + no-preemption, or hold group (a) and stay on the old context cache for this sync?

### Group (b) — constrained-decoding stack + vendored xgrammar (7 commits, 206 files, 70 overlap)

| commit | subject | files | adoptability | fork patch area it collides with |
|---|---|---|---|---|
| `7e2973ee` | build: vendor xgrammar cpu core | 79 | **clean** (new `third_party/xgrammar/`, no fork path) | none at file level; semantically duplicates `third_party/llama-grammar` |
| `08902f73` | feat: add gbnf constrained decoding across generation backends | 86 | **semantic** (competing stack) | whole fork grammar stack (`frontend/grammar/*`, `src/product/constraint/*`, `constraint_selection.h`, `constraint_state.*`) plus `decode.cpp`/`prefill.cpp`/`program_impl`/`round_buffers.h`, `draft.cpp`, `startup.cpp`, CLI, serve, bench |
| `eb6696ae` | feat: add json mode and schema constrained decoding | 62 | **semantic** | fork JSON-schema channel (`src/product/constraint/constraint_contract.*`, `third_party/llama-chat/common/json-schema*`), serve request/response schemas |
| `41e50d0d` | feat: add constrained tool calling | 48 | **semantic** | fork tool-call parser + chat-parse, `tool_call_parser.*`, request/response schemas |
| `8ffa3131` | feat: add choice and regex constrained decoding | 33 | **semantic** | fork grammar channel, `tests/README.md`, bench grammar attachment |
| `2734a56e` | feat: extend constrained decoding composition and diagnostics | 69 | **semantic** | `src/product/constraint*` vs upstream `src/product/constraint_observation.h`, schema composition, bench README |
| `81c8ce09` | feat: support tuple schemas and bounded numbers | 26 | **semantic** (tip) | JSON-schema conversion, `third_party/xgrammar/cpp/json_number.*` |

Upstream vendors XGrammar and builds `src/text/{grammar,json_schema,schema_composition,json_input}.*`,
`src/models/qwen3_5/{frontend/tool_grammar.*,program/grammar_masks.cpp}`, request-owned matchers, and
`docs/maintainer/constrained-decoding.md`. The fork independently built an in-tree token-trie mask
producer over a vendored llama.cpp grammar (`third_party/llama-grammar`, ticket #79) plus
`src/models/qwen3_5/frontend/grammar/*`, `src/product/constraint/*`,
`src/models/qwen3_5/program/constraint_state.*`, `src/runtime/engine/constraint_selection.h`, and two
ADRs. Both sides modify the same execution seams (`decode.cpp`, `prefill.cpp`, `round_buffers.h`,
`program_impl.*`, `output_session.*`, `draft.cpp`) and both add a `*grammar*`/`*constraint*` module
namespace. `7e2973ee` is a pure vendor drop and is conflict-free at file level, but it forks the same
domain.

**Invariant at stake.** Op/semantic qualification (`docs/maintainer/op-development.md`, and the
"each floating-point Op uses a naive FP32/FP64 oracle" rule in `AGENTS.md`) applies to the new
constrained-sampling Op either side adds; the fork's ADR `docs/adr/0001-wrapper-grammar-for-constrained-reasoning.md`
(binding thinking-on constrained requests to a wrapper grammar) and
`docs/adr/0002-g1-demotion-recovery-and-signaling.md` are fork design decisions upstream does not
share; the advertised OpenAI/Anthropic protocol schema tests are an external contract
(`AGENTS.md` "Change policy"). Upstream's own `docs/maintainer/constrained-decoding.md` states
thinking is handled by the model's outer rules and that constraint state advances on the same
commit boundary — i.e. a second, incompatible answer to the same problem.

**Downstream question (→ #239).** Keep the fork's in-tree grammar/trie stack, adopt upstream's
XGrammar stack, or port upstream's stack and re-implement the fork's wrapper-grammar/reasoning-region
semantics on it?

### Group (c) — SM-derived launch refactors (3 commits, 58 files, 18 overlap)

| commit | subject | files | adoptability | fork patch area it collides with |
|---|---|---|---|---|
| `a667efdd` | refactor(rope): derive launch capacity from device sm count | 12 | **semantic on the API seam** | YaRN (`include/ninfer/ops/rope.h`, `src/ops/launcher/rope.*`, `src/ops/wrapper/rope.cpp`, `execution/attention.*`, `startup.cpp`, `tests/ops/test_rope.cpp`) |
| `417eb3d6` | refactor(ops): derive norm and moe launches from device sm count | 16 | **mechanical→semantic** | NVFP4 MoE (`include/ninfer/ops/sparse_moe.h`, `src/ops/sparse_moe/prefill/*`, `src/ops/wrapper/sparse_moe.cpp`) |
| `e621c7d6` | refactor(attention): derive launch plans from device sm count | 33 | **mechanical** | `attention.cpp/.h` (fork YaRN-adjacent), `startup.cpp`, `program.cpp`, `docs/maintainer/engine-architecture.md` |

`a667efdd` changes the rope signature from `cudaStream_t stream` to `DeviceExecutionView execution`
(`include/ninfer/ops/rope.h`); the fork's YaRN patch adds `YarnScale`/`YarnTable` overloads and
rewrites the same doc block on `cudaStream_t`. That is a direct same-hunk conflict: adopt
`DeviceExecutionView` and re-apply the YaRN overloads on it. `e621c7d6`'s body claims *"Preserve
launch decisions at 170 SM"* and *"full numerical qualification while improving FP64 oracle
locality"*.

**Invariant at stake.** The YaRN `s <= 1` byte-identity invariant (runbook `Verification standard`;
documented in `include/ninfer/ops/rope.h` as *"at extension factor s == 1 the table degenerates to
the power law and mscale == 1, so the path is structurally inert"*) must survive the launch/geometry
refactor; the op-development oracle rule governs the new MoE/attention launch variants.

**Downstream question (→ #240).** Re-apply the YaRN overloads and NVFP4 MoE launches onto the
`DeviceExecutionView`/SM-count launch API, and confirm the YaRN `s <= 1` byte-identity check still
passes?

### Group (d) — serving / bench / prometheus / frontend (5 commits, 54 files, 33 overlap)

| commit | subject | files | adoptability | fork patch area it collides with |
|---|---|---|---|---|
| `abb7f14f` | feat(serve): expose prometheus runtime and request metrics | 18 | **semantic** (duplicate feature) | fork Prometheus #229 (`src/serve/prometheus_metrics.*` vs upstream `src/serve/metrics.*`), `http_server.*`, `serve_options.cpp`, `ProductTests.cmake`, `docs/serving.md`, `README.md` |
| `f854788b` | bench: expand ttft workloads and recovery diagnostics | 30 | **mechanical** | fork-renamed `tools/bench/*` + `tools/ninfer_serve/client.py`; `http_server.*`, `request_log.*`, `engine_core.h` (also group a) |
| `c772812b` | fix: align serving benchmarks and docs with runtime contracts | 9 | **mechanical** | `tools/bench/*`, `tests/README.md`, `docs/serving.md`, model-cards |
| `68c54356` | fix(frontend): preserve cache boundaries through template trimming | 8 | **mechanical** | `tests/models/qwen3_5/test_frontend.cpp`; new upstream path `src/models/qwen3_5/frontend/prompt_layout.cpp` is fork-untouched |
| `624dcdd4` | fix(bench): share cold timing quantiles | 1 | **clean** (`bench/ops/ninfer_bench_common.h` not in overlap) | none |

Upstream adds `src/serve/metrics.{cpp,h}` + `tests/test_metrics.cpp`; the fork already shipped
`src/serve/prometheus_metrics.{cpp,h}` + `tests/test_prometheus_metrics.cpp` (#229). Both wire the
same `http_server.*`, `serve_options.*`, `CMakeLists.txt`, `ProductTests.cmake`, `docs/serving.md`,
`README.md` paths. Two parallel implementations of one advertised endpoint cannot both land.

**Invariant at stake.** The advertised HTTP surface (Prometheus `/metrics` opt-in) and the serving
schema tests are an external contract (`AGENTS.md`); `docs/serving.md` is the product guide that must
name the chosen endpoint. Group (d) also carries the preemption-shaped TTFT/recovery diagnostics
(agent continuation after preemption, `test_engine_preemption_real.cpp`) that only exist once (a)
lands.

**Downstream question (→ #241).** Keep the fork's Prometheus implementation or adopt upstream's
`metrics.*`; and does the TTFT/recovery benchmark suite follow group (a)'s preemption model or stay
on the fork's no-preemption model?

### Group (e) — `feat(ops): support and tune … linear …` (23 commits, 49 files, 1 overlap)

Commits: `35e9b5c8 04ded76f 8f0aa859 58ab3f22 471924b0 cd521ab7 f58e32e0 714c5149 bcdcbd45
8beb4cdd 16bdb491 2553e26e cf91c818 07e1f8c3 d0f91cd9 070fa61a 7971ff18 c02a07b4 dd73c88e 9879d052
a64b5eea b21780f3 15cba227` (23, not ~22). The only overlapping path is `tests/README.md`; every
`src/ops/linear/**`, `tests/ops/linear/**` and `bench/ops/linear_bench.cu` change is fork-untouched
(verified: `git diff --name-only d44ab584..master -- src/ops/linear` is empty).

**Adoptability: clean** (mechanical only for `tests/README.md`, which the fork also renames/edits).
The tuning commits legitimately change accumulation order; the tests are FP64-oracle based
(`tests/ops/linear/test_bf16_a16.cpp` defines `oracle_all_rows` returning `double`), matching the
op-development oracle rule.

**Invariant at stake.** The op-development oracle rule (each floating-point Op validated against a
naive FP32/FP64 oracle at the relevant shapes) — adopted kernels must keep the oracle suite green
even though bit-identity across the sync is not expected (runbook `Verification standard`).

**Downstream question (→ #240).** Are upstream's retuned bf16/q8 linear shapes (incl. full-vocabulary
`248320x2560`, `cf91c818`) accepted as the new tuned baseline, and do all oracle tests pass unchanged?

### Group (f) — docs (1 commit, 4 files, 3 overlap)

| commit | subject | files | adoptability | fork patch area it collides with |
|---|---|---|---|---|
| `a643abcd` | docs: streamline agent guidance and align references | 4 | **mechanical → semantic** | `AGENTS.md` (fork product rules + flake env), `docs/maintainer/op-development.md` (fork rename-only), `tests/README.md` |

`a643abcd` rewrites `AGENTS.md` from 198 to 70 lines, **deleting** the fork's expanded Objective/
Change-consistency/Verification/Reporting sections and the *"no active-request preemption"* wording,
and replacing the local-environment section with a Python-3.11 conda path (the fork uses the nix
flake + Python 3.13). `docs/maintainer/linear-tuning.md` (new upstream path) is not in the overlap.
`docs/maintainer/op-development.md` is a 2-line rename on the fork side.

**Invariant at stake.** The fork's `AGENTS.md` is the fork's own operating authority; the flake
devshell (`nix develop`, CUDA 13.1, Python 3.13) is a named fork invariant, and the no-preemption
product line lives in this file. Upstream's streamlined doc has no knowledge of either.

**Downstream question (→ merge #242).** Adopt upstream's streamlined structure, then re-apply the
fork's product constraints, fork section, flake environment, and bug-reporting policy?

## Overlap file set (121 files)

Computed exactly as the runbook step 4:

```bash
BASE=$(git merge-base master upstream/master)   # d44ab584
comm -12 <(git diff --name-only $BASE..master | sort) \
         <(git diff --name-only $BASE..upstream/master | sort)
```

Upstream-theme legend: `a` runtime/scheduling · `b` constrained decoding + xgrammar · `c` SM launch ·
`d` serving/bench/prometheus · `e` linear ops · `f` docs.

| file | upstream theme(s) | fork patch area(s) |
|---|---|---|
| `AGENTS.md` | af | justfile, rename, docs, ancestry(merge), flake |
| `apps/cli/main.cpp` | ab | cli-flags, model-acquire, show-prompt, version, grammar(cli/adr), docs, reasoning-token-schema |
| `apps/cli/options.cpp` | b | cli-flags, model-acquire, show-prompt, version, grammar(cli/adr), yarn |
| `apps/cli/options.h` | b | cli-flags, model-acquire, show-prompt, version, grammar(cli/adr) |
| `bench/inference/benchmarks.cmake` | b | version |
| `bench/inference/ninfer_bench.cpp` | ab | version, grammar |
| `bench/inference/ninfer_bench_support.cpp` | b | version, grammar |
| `bench/inference/ninfer_bench_support.h` | b | version, grammar |
| `bench/README.md` | ab | nvfp4-moe, version, justfile, rename, grammar, ancestry(merge) |
| `cmake/Dependencies.cmake` | b | rename, grammar, chat-parse |
| `docs/cli.md` | ab | tokenize, cli-flags, model-acquire, show-prompt, version, grammar(cli/adr), rename, yarn |
| `docs/maintainer/engine-architecture.md` | abc | grammar(cli/adr), rename |
| `docs/maintainer/logging.md` | a | rename |
| `docs/maintainer/op-development.md` | f | rename, ancestry(merge) |
| `docs/maintainer/paged-kv-cache.md` | a | rename |
| `docs/maintainer/qwen3_5-model.md` | a | rename |
| `docs/maintainer/resource-scheduling-and-context-cache.md` | a | rename |
| `docs/performance/methodology.md` | d | rename, ancestry(merge) |
| `docs/README.md` | ab | rename, grammar, docs |
| `docs/serving.md` | abd | prometheus, reasoning_end, model-acquire, version, logprob_topk, grammar(anthropic), rename, reasoning-token-schema, demotion-signaling, grammar, chat-parse, yarn |
| `include/ninfer/ops/rope.h` | c | yarn |
| `include/ninfer/ops/softmax_attention.h` | c | ancestry(merge), yarn |
| `include/ninfer/ops/sparse_moe.h` | c | nvfp4-moe |
| `include/ninfer/types.h` | abd | tokenize, logprob_topk, reasoning-token-schema, demotion-signaling, grammar, chat-parse |
| `model-cards/Qwen3.6-27B-NInfer/README.md` | a | rename |
| `model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md` | a | rename |
| `model-cards/Qwen3.6-35B-A3B-NInfer/README.md` | a | rename |
| `model-cards/Qwen3.8-27B-NInfer/README.md` | ad | rename |
| `model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md` | ad | rename, ancestry(merge) |
| `README.md` | a | docs, tokenize, prometheus, reasoning_end, model-acquire, gguf, nvfp4-moe, show-prompt, integrity-manifest, version, logprob_topk, grammar(anthropic), grammar(cli/adr), rename, ancestry(merge), yarn |
| `src/models/CMakeLists.txt` | b | rename, grammar, grammar(trie), chat-parse |
| `src/models/qwen3_5/execution/attention.cpp` | c | yarn |
| `src/models/qwen3_5/execution/attention.h` | c | yarn |
| `src/models/qwen3_5/execution/draft.cpp` | abc | logprob_topk |
| `src/models/qwen3_5/execution/text.cpp` | ac | logprob_topk, grammar, ancestry(merge), yarn |
| `src/models/qwen3_5/execution/text.h` | a | yarn |
| `src/models/qwen3_5/frontend/frontend.cpp` | ab | tokenize, show-prompt, grammar, chat-parse |
| `src/models/qwen3_5/frontend/frontend.h` | b | tokenize, show-prompt, grammar |
| `src/models/qwen3_5/frontend/output_session.cpp` | b | reasoning_end, logprob_topk, reasoning-token-schema, demotion-signaling, grammar, chat-parse |
| `src/models/qwen3_5/frontend/output_session.h` | b | reasoning_end, logprob_topk, grammar |
| `src/models/qwen3_5/frontend/prepared_prompt.h` | ab | show-prompt |
| `src/models/qwen3_5/frontend/processor.cpp` | ab | show-prompt |
| `src/models/qwen3_5/frontend/processor.h` | b | show-prompt |
| `src/models/qwen3_5/frontend/tool_call_parser.cpp` | b | chat-parse |
| `src/models/qwen3_5/frontend/tool_call_parser.h` | b | chat-parse |
| `src/models/qwen3_5/program/decode.cpp` | ab | logprob_topk, grammar |
| `src/models/qwen3_5/program/planning/request_plan.cpp` | a | logprob_topk |
| `src/models/qwen3_5/program/planning/startup.cpp` | abc | grammar, ancestry(merge), yarn |
| `src/models/qwen3_5/program/prefill.cpp` | ab | logprob_topk, grammar |
| `src/models/qwen3_5/program/program.cpp` | abc | grammar |
| `src/models/qwen3_5/program/program.h` | abd | logprob_topk, grammar |
| `src/models/qwen3_5/program/program_impl.cpp` | ab | grammar, ancestry(merge), yarn |
| `src/models/qwen3_5/program/program_impl.h` | ab | logprob_topk, grammar |
| `src/models/qwen3_5/program/round_buffers.h` | b | logprob_topk, grammar, ancestry(merge), yarn |
| `src/models/qwen3_5/program/runtime_types.h` | a | grammar |
| `src/models/qwen3_5/program_sources.cmake` | ab | grammar |
| `src/models/qwen3_5/program/speculative/mtp.cpp` | ab | logprob_topk |
| `src/models/qwen3_5/program/speculative/target_verification.cpp` | ab | logprob_topk |
| `src/models/qwen3_5/program/storage/context.cpp` | a | grammar |
| `src/models/qwen3_5/program/transactions/commit.cpp` | abd | logprob_topk, grammar |
| `src/ops/launcher/rope.cu` | c | yarn |
| `src/ops/launcher/rope.h` | c | yarn |
| `src/ops/sparse_moe/prefill/sparse_moe_prefill.h` | c | nvfp4-moe |
| `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu` | c | nvfp4-moe |
| `src/ops/wrapper/rope.cpp` | c | yarn |
| `src/ops/wrapper/sparse_moe.cpp` | c | nvfp4-moe |
| `src/runtime/contract/execution.h` | b | logprob_topk |
| `src/runtime/contract/request.h` | ab | logprob_topk, grammar |
| `src/runtime/engine/engine_core.h` | abd | reasoning_end, logprob_topk, grammar |
| `src/runtime/engine/engine.cpp` | bd | tokenize, reasoning_end, show-prompt, logprob_topk, grammar |
| `src/runtime/engine/model_instance.cpp` | ab | rename |
| `src/runtime/engine/request_record.h` | abd | reasoning_end, logprob_topk, grammar |
| `src/serve/anthropic_messages_http.cpp` | abd | logprob_topk, demotion-signaling |
| `src/serve/anthropic_messages_request.cpp` | b | grammar(anthropic), rename |
| `src/serve/anthropic_messages_response.cpp` | b | grammar(anthropic) |
| `src/serve/CMakeLists.txt` | bd | prometheus, reasoning_end, grammar(cli/adr), demotion-signaling, grammar |
| `src/serve/generation_service.cpp` | abd | logprob_topk, demotion-signaling, grammar |
| `src/serve/generation_service.h` | ab | logprob_topk, demotion-signaling, grammar |
| `src/serve/http_server.cpp` | ad | prometheus, reasoning_end |
| `src/serve/http_server.h` | d | prometheus, reasoning_end |
| `src/serve/openai_chat_http.cpp` | ad | reasoning_end, logprob_topk, demotion-signaling |
| `src/serve/openai_chat_request.cpp` | b | reasoning_end, logprob_topk, grammar(cli/adr), rename, grammar |
| `src/serve/openai_chat_response.cpp` | b | logprob_topk |
| `src/serve/openai_responses.h` | ab | logprob_topk |
| `src/serve/openai_responses_http.cpp` | abd | logprob_topk, rename, demotion-signaling |
| `src/serve/openai_responses_request.cpp` | ab | logprob_topk, rename |
| `src/serve/openai_responses_response.cpp` | b | logprob_topk |
| `src/serve/operational_log.cpp` | a | reasoning_end, reasoning-token-schema |
| `src/serve/request.h` | b | logprob_topk, grammar |
| `src/serve/request_log.cpp` | abd | reasoning-token-schema, demotion-signaling, chat-parse |
| `src/serve/request_log.h` | ad | reasoning-token-schema, demotion-signaling, chat-parse |
| `src/serve/serve_options.cpp` | a | prometheus, model-acquire, version |
| `src/serve/translate.cpp` | b | logprob_topk, grammar |
| `tests/cmake/CoreTests.cmake` | a | chat-parse |
| `tests/cmake/ProductTests.cmake` | d | tokenize, prometheus, reasoning_end, model-acquire, version, grammar(cli/adr), grammar |
| `tests/cmake/RuntimeTests.cmake` | ab | grammar |
| `tests/models/qwen3_5/test_frontend.cpp` | abd | reasoning_end, show-prompt, reasoning-token-schema, demotion-signaling, grammar, chat-parse |
| `tests/models/qwen3_5/test_runtime_mechanisms.cpp` | a | grammar |
| `tests/models/qwen3_5/tests.cmake` | ab | grammar, grammar(trie), ancestry(merge), chat-parse |
| `tests/ops/test_rope.cpp` | c | yarn |
| `tests/README.md` | abcdef | tokenize, justfile, rename, devshell-cache, ancestry(merge), chat-parse, docs |
| `tests/test_anthropic_schema.cpp` | b | grammar(anthropic), demotion-signaling |
| `tests/test_cli_options.cpp` | b | cli-flags, model-acquire, show-prompt, version, grammar(cli/adr) |
| `tests/test_ninfer_bench_support.cpp` | b | version, grammar |
| `tests/test_openai_responses.cpp` | ab | logprob_topk, demotion-signaling |
| `tests/test_openai_schema.cpp` | ab | reasoning_end, logprob_topk, grammar(cli/adr), demotion-signaling, grammar |
| `tests/test_request_log.cpp` | ad | reasoning_end, rename, reasoning-token-schema, demotion-signaling, chat-parse |
| `tests/test_serve_corpus.py` | d | reasoning-token-schema, demotion-signaling, ancestry(merge), chat-parse |
| `tests/test_serve_options.cpp` | a | model-acquire, version, rename, grammar |
| `tests/test_tool_call_parser.cpp` | b | chat-parse |
| `tools/bench/README.md` | bd | rename, reasoning-token-schema, demotion-signaling, ancestry(merge), chat-parse |
| `tools/bench/run_ninfer_bench_matrix.py` | b | rename, grammar |
| `tools/bench/run_serve_concurrency.py` | d | rename, ancestry(merge) |
| `tools/bench/run_serve_corpus.py` | d | rename, reasoning-token-schema, demotion-signaling, ancestry(merge), chat-parse |
| `tools/bench/run_serve_ttft_campaign.py` | ad | rename |
| `tools/bench/run_serve_ttft.py` | a | rename |
| `tools/bench/ttft/cases.py` | ad | rename |
| `tools/bench/ttft/execution.py` | ad | rename |
| `tools/bench/ttft/README.md` | ad | rename |
| `tools/bench/ttft/render.py` | ad | rename |
| `tools/ninfer_serve/client.py` | ad | rename |

### Overlap summary by fork patch area

Counts are files carrying that patch label (a file may carry several).

| fork patch area | overlapping files | upstream themes |
|---|---|---|
| grammar (incl. cli/adr, anthropic, trie) | 55 | `a` `b` `c` `d` |
| rename (frinfer rebrand of tracked paths) | 39 | `a` `b` `c` `d` `e` `f` |
| logprob_topk (op + transport + serve schemas) | 35 | `a` `b` `d` |
| chat-parse / demotion-signaling | 19 each | `a` `b` `d` |
| YaRN | 17 | `a` `c` |
| reasoning_end control endpoint | 17 | `a` `b` `d` |
| version / show-prompt / reasoning-token-schema | 16 / 13 / 12 | `a` `b` `d` |
| model-acquire / tokenize / prometheus / cli-flags | 10 / 8 / 7 / 5 | `a` `b` `d` |
| nvfp4-moe | 6 | `c` |
| gguf / integrity-manifest / devshell-cache | 1 each | `b`/`a` |

### Collision classes (recomputed)

| class | count | files |
|---|---|---|
| add/add (both created same path) | 0 | — |
| modify/delete (upstream deleted, fork edited) | 1 | `src/models/qwen3_5/program/storage/context.cpp` (fork grammar commits `afde8bff`, `23d7ec8e`, `701214e8`, `6d5a0a3f`) |
| delete/modify (fork deleted, upstream edited) | 0 | — |
| modify/modify (both edited) | 120 | the rest of the table |

### Historical conflict set — still holds, and has grown

The runbook's historical four (`AGENTS.md`, `bench/README.md`,
`src/models/qwen3_5/execution/text.cpp`, `src/models/qwen3_5/program/planning/startup.cpp`) are all
still in the overlap set. Notable **new** collisions beyond them:

- `src/models/qwen3_5/program/storage/context.cpp` — the only modify/delete; the fork's grammar
  lifecycle edits must be re-homed into upstream's `transactions/*` replacements.
- `include/ninfer/ops/rope.h` — same-hunk API collision (`cudaStream_t` vs `DeviceExecutionView`).
- `src/serve/http_server.{cpp,h}` + `src/serve/serve_options.*` — duplicate Prometheus wiring.
- `src/runtime/engine/engine_core.h` + `src/runtime/engine/request_record.h` — both sides widen the
  central request/engine record (fork logprob + reasoning_end vs upstream preemption/metrics).
- `include/ninfer/types.h` + `src/runtime/contract/request.h` — both sides add request fields
  (fork constraint/logprob vs upstream constraint/metrics).
- `apps/cli/options.{cpp,h}` — fork flags vs upstream grammar flags on the same parser.
- The whole `tools/bench/ttft/*` + `tools/ninfer_serve/client.py` set — fork-renamed, upstream-rewritten.

### Fork-only files that are collided indirectly

Upstream never touches these (so they are not in the overlap set), but they are the fork's landing
points for behaviour that upstream now also implements, and the merge ticket must keep them in view:
`include/ninfer/ops/logprob_topk.h`, `src/ops/{launcher,wrapper}/logprob_topk.*`,
`bench/ops/logprob_topk_bench.cu`, `tests/ops/test_logprob_topk.cpp`,
`src/models/qwen3_5/frontend/grammar/*`, `src/product/constraint/*`,
`src/runtime/engine/constraint_selection.h`, `third_party/llama-grammar/*`,
`docs/adr/000{1,2}-*.md`, `src/product/model_acquire/*`,
`src/ops/sparse_moe/nvfp4/*`, `tools/convert/sources/*` (GGUF), `justfile`, `flake.nix`.

## Per-group downstream question (the one line each group leaves open)

- **(a) → #238** "Decide: preemptive scheduling and context-cache replacement — adopt, adapt, or
  hold": adopt the new runtime (and drop no-preemption), port only its cache/perf parts, or hold?
- **(b) → #239** "Decide: constrained-decoding and xgrammar reconciliation": fork in-tree stack,
  upstream XGrammar stack, or port upstream and rebuild the fork's wrapper-grammar semantics on it?
- **(c) → #240** "Qualify: SM-derived launch refactors…": re-apply YaRN + NVFP4 MoE onto
  `DeviceExecutionView`/SM-count launches and re-prove the `s <= 1` byte-identity invariant?
- **(e) → #240** (same ticket): accept the retuned bf16/q8 linear shapes as the new baseline under
  the oracle suite?
- **(d) → #241** "Reconcile: serving, bench, and prometheus overlaps plus the frontend
  cache-boundary fix": one Prometheus implementation, and does the TTFT/recovery suite follow
  preemption or no-preemption?
- **(f)/AGENTS.md → merge #242**: adopt upstream's streamlined guidance, then re-apply the fork's
  product line, fork section, flake environment, and bug-reporting policy?

## Verification implications (for the execute ticket #242)

The runbook's verification standard is unchanged and must be run on the merged tree: `nix develop -c
cmake --build build -j`; `ctest --test-dir build`; `nix develop -c python3.13 -m pytest tests/convert
-q`; and, with a free GPU, the `qwen3_8_27b_nvfp4.ninfer` smoke at `--max-context 262144` and a YaRN
boundary run at `262145`. Upstream's numerical/perf commits legitimately change accumulation order;
cross-side bit-identity is not expected, but the YaRN `s <= 1` byte-identity invariant, the renamed
binaries, the flake, and the advertised serving schema tests must stay exact. The merge must land as
a **true merge commit** (not squash) to preserve upstream ancestry (runbook step 6).
