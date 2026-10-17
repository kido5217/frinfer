# Upstream sync 4 — seam absorption dispositions

Research ticket #308. Range: `81c8ce09..64d10fbf` (77 upstream commits). Merge base verified as
`81c8ce093b2c1646a87566a8e59d807fcf0ec95c` (`git merge-base master upstream/master`). This is the
per-seam briefing for the merge ticket; artifact-format seams are covered by #307
(`docs/research/upstream-sync4-artifact-formats.md`, branch `research/upstream-sync4-artifact-formats`).

Method: the step-4 `comm -12` overlap recipe was recomputed at sync-doc current text (127 files,
list reproduced by the recipe run this session), then every seam was inspected on both sides with
`git log --oneline 81c8ce09..64d10fbf -- <path>` and `git diff` per file. Two corrections to the
raw overlap set matter (see "Overlap set corrections" below): the set **under-reports** the frontend
and sparse-MoE conflict clusters because upstream rename detection moves paths the fork edited, and
it **understates** the engine-seam risk because upstream edits adjacent lines without touching the
fork's exact lines.

## Headline answers to the ticket's four deep questions

1. **Grammar.** Upstream's `0a8530d4 fix(grammar): optimize length-bounded json string masks` is
   entirely inside `third_party/xgrammar/**` plus `tests/test_grammar.cpp`,
   `tests/text/test_json_schema.cpp` and 2 added lines to `docs/maintainer/constrained-decoding.md`.
   The fork's XGrammar adoption (README table, "Constrained decoding" row; `third_party/xgrammar`
   already present at `81c8ce09`) means this fix is **adopt-upstream with zero fork re-application**
   — and it is a direct speed win for the fork's own `structured_outputs`/`response_format` routes.
   The fork's retired llama-grammar stack: `git diff --name-only 81c8ce09..64d10fbf | grep -iE
   'llama|gbnf|peg'` → **none**; upstream never references `third_party/llama-chat`, so the fork's
   vendored PEG chat-parsing subset (`third_party/llama-chat/`, fork-only) and its drift runbooks
   (`docs/maintainer/{llama-chat-vendor,llama-grammar-vendor}.md`, fork-only new files) are untouched.
   `src/models/qwen3_5/program/grammar_masks.cpp` was changed in-range only by `6a42de35`, a one-line
   mechanical `device.stream()` adaptation → adopt, no fork work.

2. **Context-cache rework vs scheduling/capacity glue.** The rework (`48d8d185`, `e9a7de36`,
   `ee5a5ffa`, `4423e0ac`) does **not** absorb any fork patch — the fork's scheduling and Main-KV
   capacity are already upstream code (ADR 0004 revert; README rows "Scheduling tracks upstream" /
   "Main KV capacity tracks upstream's range"). `main_page_groups` survives in upstream's
   `startup.cpp` (9 hits at `64d10fbf`) and `startup.h` (2 hits). What the rework **does** do is
   rewrite the entire host of every fork hunk in `src/models/qwen3_5/program/planning/startup.cpp`:
   `validate_target_options` re-signs to `ModelProfile`, `persistent_layout`/`build_workspace_plan`
   are restructured, and `build_sequence_candidate` is replaced by
   `make_sequence_planner_impl(profile, …, moe)`. The fork's remaining startup patch is pure YaRN
   (`plan_yarn_extension` + `RoundStateSpec{.yarn_inv_freq,.yarn_mscale}` + `validate_yarn_extension`
   + the deletion of upstream's native-capacity reject); it must be re-carried on top. Upstream
   additionally changed `prefill_chunk == 0` from reject to auto-1024 and restored the draft-bound
   check shape — keep upstream's forms, port the fork's three YaRN hunks, and re-run
   `test_engine_preemption_real.cpp` (changed in-range by `ee5a5ffa`, `c7c36e88`) plus the
   `--max-context 262145` YaRN boundary smoke.

3. **Engine/runtime seams.** The fork's streaming-logprobs/control/observation surface is **not**
   natively covered: `git grep -ln 'reasoning_end' 64d10fbf -- src include` → **none**, and
   upstream's serve still *rejects* the flag — `bad_request("logprobs=true requires per-token log
   probabilities in the response, which …")` at `src/serve/openai_chat_request.cpp:109-114`.
   Upstream's `target_logprobs` op predates the fork work (exists at `81c8ce09`,
   `include/ninfer/ops/target_logprobs.h`; in-range touched only by the `8cc45088`/`6a42de35` file
   moves) and stays scoring-only. Disposition for all four named engine files: **hybrid /
   re-apply-fork** on upstream's new structure (details in the table).

4. **Docs consolidation.** Upstream's `f1ace85e`/`64d10fbf` deletes 9 whole upstream-authored docs
   (`docs/maintainer/linear-benchmark.md` + 8 `qwen3_8-flash-next-*.md` files; rename map shows the
   only in-range `D` under `docs/` is `linear-benchmark.md`) and rewrites 16 more. **No fork-only
   doc is touched** (`upstream-sync.md`, `llama-chat-vendor.md`, `llama-grammar-vendor.md`,
   `constrained-thinking-design.md`, `grammar-mask-production.md` — fork-only additions, not in the
   overlap set). Every shared doc where both sides edited is re-apply-fork (table below); the
   mechanical trap is upstream's *new* text, which says `NInfer`/`ninfer-serve` — re-brand
   (README "End of the fork section" marker, `Last sync` stamp at line 8, drift line at line 52).

## Disposition table

| # | Seam (files) | Upstream evidence | Disposition | Notes for the merge |
|---|---|---|---|---|
| 1 | XGrammar mask engine — `third_party/xgrammar/cpp/**` (9 files), `tests/test_grammar.cpp`, `tests/text/test_json_schema.cpp` | `0a8530d4` (+387/−249) | **adopt-upstream** | Fork never edited vendored xgrammar (not in overlap set). Internal rework of counted-repetition mask classes; public `xgrammar::` API surface unchanged for fork consumers (`src/text/*` — fork-only, not in range). Direct perf win for fork structured-output routes. |
| 2 | `docs/maintainer/constrained-decoding.md` | `9cfd5646`, `0a8530d4` (+2) | **adopt-upstream + re-apply-fork** | Fork rewrote 168 lines (XGrammar route doc). Keep upstream's mask-optimization note; keep the fork's adopted-route framing; check no `NInfer` branding slips into upstream's added lines. |
| 3 | `src/models/qwen3_5/program/grammar_masks.cpp` | `6a42de35` 1-line `device.stream()` → `device.stream()` accessor | **adopt-upstream** | Fork never touched it; no conflict. |
| 4 | Vendored llama.cpp chat-parse stack — `third_party/llama-chat/**`, `src/models/qwen3_5/frontend/chat_parse_core.{cpp,h}` | none in range | **re-apply-fork (trivial)** | Upstream never references the stack; fork-only files land untouched. |
| 5 | `src/models/qwen3_5/program/planning/startup.cpp` | `48d8d185`, `9cfd5646`, `19367582`, `17bea0bc`, `545a171a`, `c7c36e88`, `8cf26bb6`, `6a42de35`, `610f79e9`, `f2620157`, `f7d3090b` | **hybrid** | See deep answer 2. Adopt `ModelProfile`/planner restructure; re-apply the YaRN hunks (`plan_yarn_extension` call, `RoundStateSpec` yarn fields, `validate_yarn_extension` + deleted native-capacity reject + masked-backend draft bound). |
| 6 | Round-state layout — `src/models/qwen3_5/program/round_buffers.{h,cpp}`, `execution_context.h`, `internal.h`, `program.{h,cpp}`, `program_impl.{h,cpp}`, `decode.cpp`, `prefill.cpp`, `speculative/mtp.cpp`, `transactions/{binding,commit}.cpp`, `program_sources.cmake` | `48d8d185`, `e9a7de36`, `9cfd5646` + MOE rework | **hybrid** | Fork adds `RoundStateSpec{.yarn_inv_freq,.yarn_mscale}`, logprob round buffers (`logprob_round.h`, `logprob_assembly.h` — new fork-only files, adopt-then-register), and observation/logprob wiring. Upstream rewrites the same struct and all consumers. `RoundStateSpec` at `64d10fbf` has no yarn fields (7 fields, `round_buffers.h:20-27`) → re-add both. |
| 7 | YaRN planning glue — `src/models/qwen3_5/program/planning/yarn_policy.{cpp,h}`, `src/models/load_options.h` | none | **re-apply-fork** | New fork files; `is_masked_draft_backend` already covers `DFlash2` (same enum at both tips → no upstream addition to handle). Register `yarn_policy.cpp` in `program_sources.cmake`. |
| 8 | RoPE op family — `include/ninfer/ops/rope.h`, `src/ops/launcher/rope.cu`, `src/ops/wrapper/rope.cpp`, `src/ops/kernel/rope.cuh`, `tests/ops/test_rope.cpp` | `6733be72` (reorg: `launcher/rope.cu` **deleted**, `src/ops/rope/{apply,kernels,rmsnorm}` family), `6a42de35` (`CudaExecutionView` exec-param) | **hybrid** | Fork adds `YarnScale`/`YarnTable`/`compute_rope_yarn_table` (+75/+136/+111/+361 lines at old paths); upstream moves+re-families the same code and re-signs `rope(...)` to take `CudaExecutionView`. Port the fork YaRN path into `src/ops/rope/` layout, keep both fork overloads' yarn parameter + upstream's execution-view parameter in the public header. Fork golden tests must pass unchanged against the port (`s <= 1` byte-identity invariant is fork-patched). |
| 9 | Logprob ops — `src/ops/{kernel,launcher,wrapper}/logprob_topk.*` | none (fork-new files) | **re-apply-fork (registration only)** | Upstream's `target_logprobs` op is older and scoring-only (deep answer 3). Add fork sources to the new `src/ops/.../sources.cmake` family layout; no call-site change. |
| 10 | Ops API port — `src/ops/weight_input.cpp`, `include/ninfer/ops/weight_input.h`, `src/models/qwen3_5/execution/parameters.cpp` | `6a42de35`/`610f79e9` remove `prepare_{attn,gdn}_input_proj_weights` + `prepare_sparse_moe_weights` for `prepare_projection_weights(std::span<const WeightInput>)` (`weight_input.h:28`) | **adopt-upstream + port fork call sites** | All three fork call sites are in `execution/parameters.cpp` (lines 79/93/101/231). Rewrite to the new span API; the attn/gdn sites map directly, the `prepare_sparse_moe_weights` site must compose `SparseMoeWeights` from `prepare_projection_weights` + `prepare_linear_{swiglu,}weight` parts (upstream's `operands.cpp` keeps the bank/`QType::NVFP4` admission, so the fork's NVFP4 validation in `weight_input.cpp` (+73 lines, fork-only hunk) survives by re-application. Keep YaRN `s<=1` invariant at the RoPE boundary (unaffected here but regression-gate it). |
| 11 | NVFP4 35B-A3B MoE route — `src/ops/sparse_moe/**` (decode/prefill/small_t + fork's `nvfp4/` kernels), `include/ninfer/ops/sparse_moe.h` | `610f79e9` "rebuild moe around routed expert interfaces", `7e691b0b` moves `sparse_moe/**` → `src/ops/moe/**` (`git ls-tree 64d10fbf src/ops/sparse_moe` → empty) | **hybrid** | Fork's codec-admission and the fork-only `nvfp4/sparse_moe_nvfp4_{gate_up,down}.{cu,h}` kernels must move under the new `moe/` family (its `routed_swiglu`/`topk_softmax_route` dispatches are the insertion points; `operands.cpp:189-201` already validates NVFP4 A4 divisors and `kNvfp4RoutedExperts = 256` is a fork constant). This is the fork's *in-progress* README row 50 route — port the kernels, keep the fork's A16 T<20 and W4A4 T>=20 dispatch, re-run the FP64-oracle suites. |
| 12 | Model-execution call sites — `src/models/qwen3_5/execution/{text,attention,draft}.{cpp,h}`, `src/models/qwen3_5/execution/parameters.cpp` | `9cfd5646` (move shared parts to `common/`), `610f79e9`, `4eb2c042`, `6a42de35`, `19367582`, `c7c36e88`, `48d8d185` | **hybrid** | Fork edits are YaRN (`text_rope(..., &yarn_, ...)`), logprob gather, `execution_view` threading. Upstream rewrites the same functions (rmsnorm/rope signatures take `CudaExecutionView`; MOE split). Adopt upstream, re-apply the 3 fork hunk families; expect conflict in `mtp_forward_*`, `prefill_impl`, `attn_mix`. |
| 13 | Frontend move — whole `src/models/qwen3_5/frontend/*` | `9cfd5646` renames to `src/models/common/qwen/frontend/*` (R053–R099 per file) and rewrites `frontend.cpp` (93 lines) + adds `common/context/work.h`, `common/bindings.cpp` | **hybrid** | The overlap set *hides* this seam: fork-side paths differ from upstream-side paths, so `comm -12` misses them. The fork's chat-parse stack (`chat_parse_core.{cpp,h}`, fork-only), `tool_call_parser` (G1 salvage), `tool_contract`/`tool_grammar` (constrained tool-calling), `processor`/`output_session` (control + logprob lanes), `prepared_prompt.h` (`rendered_text()` for `--show-prompt`), and `frontend_sources.cmake` must all be re-applied **at the new `common/qwen/frontend/` paths**, and `include/ninfer/engine.h`-visible surfaces (`PreparedPrompt::rendered_text`, `Engine::detokenize`, `Engine::token_piece`) kept. Upstream also moved `tokenizer.cpp` → `detokenize`/`token_piece` call `active->frontend.*` in engine.cpp (fork-added methods ride upstream's moved frontend). |
| 14 | `include/ninfer/types.h` | `ee5a5ffa`, `19367582`, `c996baa8`, `c7c36e88` | **hybrid** | Fork adds `TokenPiece`, `ExecutionOptions::logprobs`, XGrammar-cache comment, constraint/tool-choice re-doc. Upstream adds MOE/offload fields. Textual overlap is moderate; keep both, re-ordering as needed. |
| 15 | `src/runtime/engine/engine.cpp` | `9cfd5646`, `985121d4` | **re-apply-fork** | Fork surface: `logprobs` resolve, `PreparedPrompt::rendered_text`, `reasoning_end_flag` Concept hook + `control()`, `GenerationControl::end_reasoning`, `detokenize`/`token_piece`, submit-signature reflow. Upstream changes the same file's `submit` region (consumer/observation plumbing). Merge region-by-region; keep both. |
| 16 | `src/runtime/engine/request_record.h` | `ee5a5ffa` | **hybrid** | Upstream restructures `RequestHostTiming` (field alignment, new migration counters) and keeps `constraint_draft_wait_exposed_ns` (lines 38/65/88 — see retirement item 1); changes `context_transfers` from scalar to per-pool array. Fork adds `content_logprobs` + `reasoning_end` shared_ptr and deletes the constraint_draft lines. Adopt upstream's timing/transfer shapes, re-apply the fork's field deletions and additions. |
| 17 | `src/runtime/engine/engine_metrics.inl` | `ee5a5ffa` | **hybrid** | Upstream adds migration/pool counters and still accumulates `constraint_draft_wait_ns` there; the fork deletes that one accumulation line. Adopt upstream's counters, re-apply the deletion — and keep it consistent with seams 16/19/#R1 below, because the field lives in six places at `64d10fbf`. |
| 18 | `src/runtime/engine/model_instance.cpp` | 8 upstream commits (`9cfd5646`, `19367582`, `17bea0bc`, `545a171a`, `c7c36e88`, `c18d72db`, `6a42de35`) | **re-apply-fork (branding only)** | Fork change is the string `"FrInfer accepts only .ninfer artifacts"` vs upstream `"NInfer …"`; adopt upstream's rewrite, re-brand the one line. |
| 19 | Serve request-log surface — `src/serve/request_log.{h,cpp}`, `operational_log.cpp`, `metrics.cpp`, `http_server.cpp` | `ee5a5ffa` (v29), `19367582` (v28), `c7c36e88` (v27) | **hybrid** | Upstream bumps `kRequestLogSchemaVersion` 29 with per-pool transfer fields; fork is at 25 (v24 `reasoning_tokens`). Merged version must be ≥ upstream's 29 **and** keep the fork's `reasoning_tokens`/logprob/observation fields (`request_observation.h` is fork-new; `reasoning_control.{cpp,h}`, `tool_call_signal.{cpp,h}` fork-new → re-apply-fork). Update `docs/serving.md` schema line together (fork rewrote 393 lines of it). |
| 20 | Constrained surface in serve — `openai_chat_request.cpp`, `openai_responses_request.cpp`, `request_validation.cpp` | `3124dffe` | **hybrid** | Upstream `3124dffe` still *rejects* `logprobs=true` (deep answer 3) and reshapes the request-validation path; fork implements the real surface. Keep fork's routes, adapt to upstream's reshaped validator; keep the fork's `x-should-retry`/409/ADR-0002 codes. |
| 21 | `apps/cli/{main,options}.{cpp,h}` | `985121d4`, `19367582`, `c7c36e88` | **hybrid** | Fork adds YaRN/`--show-prompt`/model-acquisition/tokenizer/`--prompt-file|stdin`/build-id options; upstream adds qwen4exp/offload/startup-calibration options. Merge option tables; keep the `frinfer` family (fork rename row in README; `options.*` conflicts are line-level, resolve by keeping fork usage text and upstream new flags). |
| 22 | `bench/inference/ninfer_bench*.{cpp,h}` + `bench/README.md` | `3124dffe`, `69c098bf`, `7e691b0b`, `df4ff5fe`, `1b50c72d`, `5ac4d9f0` | **hybrid** | Fork: `--grammar` mode, report fields, branding. Upstream: rewrites bench support for new ops + README sections. Keep both; upstream's new doc text re-brand (`NInfer`→`FrInfer`) per README marker rule. |
| 23 | Docs surface — `AGENTS.md`, `README.md`, `docs/README.md`, `docs/{cli,serving,performance,perplexity,weight-conversion}.md`, `docs/maintainer/*` (all overlap docs files) | `f1ace85e` (+5704-line deletions of *upstream's own* plan docs; `linear-benchmark.md` deleted), `64d10fbf`, `93fb1874` (.agent-tmp section), qwen4exp doc commits | **re-apply-fork + adopt upstream structure** | See deep answer 4. AGENTS.md: fork governance wins; re-add upstream's `.agent-tmp` paragraph (it doesn't collide with the fork's `out/` convention; fork AGENTS.md has no `.agent-tmp` yet → adopt). README: keep "This fork" + "End of the fork section" markers; refresh `Last sync` stamp + drift line. `docs/README.md` fork map wins where upstream deleted entries. `docs/performance/methodology.md` + `docs/performance/qwen3.6-35b-a3b.md` (fork-only perf docs) → re-apply-fork, no conflict (not in overlap set). |
| 24 | `cmake/Dependencies.cmake`, `src/models/CMakeLists.txt`, `src/runtime/CMakeLists.txt`, `tests/cmake/CoreTests.cmake`, `tests/{ops,artifact,models/qwen3_5}/tests.cmake`, `bench/ops/benchmarks.cmake`, `Dockerfile` | `8cea711b`, `74671a4e`, `9cfd5646`, ops-family cmake moves | **hybrid** | Fork adds `third_party/llama-chat` (README.frinfer.md), `xgrammar` link, YaRN/logprob test registration; upstream adds cpu-execution deps and rewrites per-family cmake. Merge source lists per file; verify the new `moe/` family cmake includes the fork's `nvfp4/` files (seam 11). |
| 25 | `tools/convert/__main__.py` (GGUF `_open_source` dispatch) | `9cdbf221`, `7c4c5e3e`, `d8c0fa6f` | **hybrid** (per #307) | Fork dispatches `.gguf` in `_open_source`; upstream adds Qwen4Exp branch + manifest provenance in the same functions. Adopt upstream's structure, keep the fork's GGUF opener as the suffix branch. |
| 26 | `tools/convert/official_recipes.py` | same 3 commits | **hybrid** (per #307) | Fork's `qwen3_6_35b_a3b_nvfp4` recipe + registry vs upstream import rewrite + `qwen3_8_flash_next`. Keep both recipes. |
| 27 | `tools/artifact/{formats,layouts}.py` branding near new registry entries | `7c4c5e3e` | **re-apply-fork** (per #307) | 4 branding lines; upstream's new nvfp4-bank entries adopt cleanly; re-brand after. |
| 28 | `src/artifact/reader.cpp` error wording ("FrInfer v3 entry magic") | `8cea711b` | **re-apply-fork** | One string; upstream's IO rework wins structurally. |
| 29 | `tests/` suite list (real-engine + unit) | `ee5a5ffa`, `9cfd5646`, `19367582`, `64d10fbf` | **hybrid** | `test_engine_{dflash,prefix,score}_real.cpp`, `test_native_transactions.cpp`, `test_runtime_mechanisms.cpp`, `test_{metrics,request_log,cli_options,serve_options,http_transport,ninfer_bench_support}.cpp`, `tests/README.md`: keep both field sets; upstream's own `context_transfers` array assertions already updated there. |
| 30 | `eval/*`, `tools/bench/*`, `tools/README.md`, `tests/bench/ttft/*`, `.gitignore`, `CONTRIBUTING.md` | qwen4exp/ple/moe commits, `f1ace85e` | **hybrid, low risk** | Branding + the `eval/run_qwen3_8_27b_nvfp4_reasoning.sh` → `run_qwen3_8_27b_reasoning.sh` rename (R074) which the fork also edited; keep fork edits at the new path. |

## Overlap set corrections (merge-ticket briefing)

- The step-4 `comm -12` output **misses** the frontend seam (#13) and the sparse_moe seam (#11):
  upstream renamed the paths (`qwen3_5/frontend/*` → `common/qwen/frontend/*`, `sparse_moe/*` →
  `moe/*`), so fork-side and upstream-side names never intersect. With rename detection
  (`git diff -M --name-status`) they do, and they are the two largest hybrid clusters in this sync.
  Run the overlap with `-M` (or the name-status table) for the merge, not the plain `comm -12`.
- `src/ops/sparse_moe/**` is **deleted upstream** (`git ls-tree 64d10fbf src/ops/sparse_moe` →
  empty); the overlap list shows the fork-side files because upstream still reports *modified*
  lines under rename pairing. Expect delete/modify conflicts on every fork `sparse_moe` file.
- Fork-only file families with **zero** upstream contact in range (no conflicts, re-apply-fork
  trivially): `third_party/llama-chat/**`, `src/serve/{reasoning_control,tool_call_signal,request_observation}.{cpp,h}`,
  `src/models/qwen3_5/program/logprob_*.h`, `src/text/{model_grammar,model_grammar_internal}.cpp`,
  `src/models/qwen3_5/program/planning/yarn_policy.*`, `docs/research/**`.

## Where the fork can RETIRE code

**No fork feature patch is natively covered by this range** — there is no "adopt upstream, retire
fork code" headline in functionality terms. What upstream absorbs is *structure* (the ops `moe`
family, the `common/qwen/frontend` tree, the context-cache state model), which retires file
locations, not features. Two specifics:

1. **`constraint_draft_wait_*` is NOT retired upstream — the reverse.** At `64d10fbf` the field set
   is fully alive: `include/ninfer/types.h` (2 hits), `src/runtime/contract/timing.h` (3),
   `engine_metrics.inl:127`, `request_record.h:38/65/88-89`, `src/serve/request_log.cpp:416/478-479/780`
   (`constraint_draft_wait_exposed_seconds` / `constraint_draft_wait_seconds` JSONL fields),
   `bench/inference/ninfer_bench_support.{h,cpp}`. The fork deleted it everywhere when #239-C retired
   the GBNF constraint-draft path. Disposition: **re-apply the fork's deletion across all six
   upstream files** (they are in the overlap set or receive new upstream lines in-range), or the
   public API, the request-log schema, and the bench report schema silently regrow the dead field.
   This is the sync's clearest example of a fork removal the merge can lose without a conflict —
   adopt-upstream on a neighboring hunk re-introduces the line.
2. **`grammar_masks.cpp`** — fork's in-tree token-trie producer was already retired by the #239-C
   XGrammar adoption; nothing left to retire; this sync only makes it faster (#1, #3).

## Silent-break candidates (fork contracts that can break without a merge conflict)

- **YaRN `s <= 1` byte-identity** (seams 5/8/12): upstream rewrites the whole RoPE family (#8) and
  the startup host (#5); both merge cleanly *per line* but silently bypass the fork's power-law
  gate. Gate: YaRN boundary smoke (`262145`) + `tests/ops/test_rope.cpp` goldens.
- **Request-log schema version** (#19): adopt-upstream on `request_log.h` alone silently renumbers
  the fork's v24 `reasoning_tokens` contract to upstream's v27–29 field set; the fork's schema
  tests (`tests/test_request_log.cpp`) catch it only if the merged file actually compiles both —
  verify the version constant is a single number ≥29 with fork fields present.
- **`constraint_draft_wait` regrowth** (retirement item 1): every adopt-upstream resolution in
  `types.h`, `timing.h`, `engine_metrics.inl`, `request_record.h`, `request_log.cpp`,
  `ninfer_bench_support.*` re-introduces the field the fork deleted — no conflict marks the spot
  when the fork side is a pure deletion adjacent to upstream's new lines.
- **Renamed binaries** (#21/#22/#30): upstream's new text says `ninfer*`; re-brand sweep per README
  rule, plus `--version` build identity across all five binaries (fork row "Build identity").
- **`frinfer_`-namespaced metrics** (#19): upstream rewrites `metrics.cpp` (`ee5a5ffa`) with its
  own histogram set; keep the fork's `--metrics` opt-in gate and fork counters or `/metrics`
  contract tests pass while the scrape contract regresses.
- **NVFP4 35B route** (#10/#11): the new `moe` family already validates NVFP4 (`operands.cpp:189`),
  so the port can build and the oracle suites can pass *without* the fork's T<20 A16 path being
  reachable — re-run the FP64-oracle coverage at `T=1` and the small-T boundaries explicitly.
- **`main_page_groups` naming** (#5): survived this range, but the rework keeps renaming planning
  internals (`build_sequence_candidate` → `make_sequence_planner_impl`); any fork patch that keyed
  on the old planner API must re-key on the new one — the merge must *not* re-add
  `require_full_resident_capacity` (ADR 0004) just because a fork-side hunk's context moved.

## Verification gates for the merge ticket (beyond sync-doc step 5)

1. `nix develop -c cmake --build build -j` then `ctest --test-dir build` (upstream's own suites
   included: `tests/ops/moe/test_nvfp4.cpp`, `test_state_store.cpp`, `test_decode_graph.cpp`).
2. YaRN boundary run `--max-context 262145` + native `262144` on `qwen3_8_27b_nvfp4.ninfer`
   (byte-identity of the native path vs pre-sync).
3. `pytest tests/convert` (GGUF + 35b + flash-next recipes) and `pytest tests/artifact`.
4. Serve contract: `tests/test_request_log.cpp`, `test_serve_corpus.py`, Anthropic/OpenAI schema
   tests, plus the `/metrics` opt-in gate behavior (404 naming `--metrics`).
5. Grammar: `tests/test_grammar.cpp` + `tests/text/test_json_schema.cpp` after the 0a8530d4
   adoption (upstream's own new assertions), then the fork's structured-output corpus
   (`tests/text/test_chat_templates.py`).
6. NVFP4 35B FP64-oracle coverage at `T>=20` (W4A4 route) and `T=1`/small-T boundaries (A16 route).
