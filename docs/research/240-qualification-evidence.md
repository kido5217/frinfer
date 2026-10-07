# #240 qualification evidence — SM launch refactors + linear tuning flood

Durable, reproducible record. All test results below were produced from the **committed** branch
state (clean worktrees), not a scratch tree.

## Exact revisions

| ref | SHA | what |
|---|---|---|
| upstream tip | `81c8ce09` (`81c8ce093b2c1646a87566a8e59d807fcf0ec95c`) | Neroued/ninfer master |
| fork base | `f66e419f` | `master` (merge base with upstream `d44ab584`) |
| `tmp/240-qual` code tip | `f57d4368` | fork master + upstream-retuned bf16/q8 linear paths |
| `tmp/240-qual` HEAD | `8c4757c7` | merge recording upstream `81c8ce09` ancestry (`-s ours`) |
| `tmp/240-qual-c` cherry-pick tips | `4ed5750c` (rope), `9169d808` (norm+moe), `ba3c9963` (attention) | upstream commits applied |
| `tmp/240-qual-c` reconciliation | `96701fde` | fork YaRN/NVFP4 re-applied onto `DeviceExecutionView` |
| `tmp/240-qual-c` ancestry merge | `66e12d3e` | merge recording upstream `81c8ce09` ancestry (`-s ours`); branch tip is the commit that introduces *this* evidence file |

`git merge-base --is-ancestor` holds for `a667efdd`, `417eb3d6`, `e621c7d6` on `tmp/240-qual-c`,
and for `cf91c818` on `tmp/240-qual`. `git diff upstream/master -- src/ops/linear` is empty on
`tmp/240-qual`. Both worktrees are clean.

The ancestry is recorded with `git merge --no-ff -s ours 81c8ce09`: the tree keeps the fork plus the
targeted commits (the other upstream groups are deferred), while the upstream commit objects become
real ancestors. This is why the upstream SHAs verify with `--is-ancestor` even though the applied
changes were produced by cherry-pick.

Branch tips at push: `tmp/240-qual` = `8c4757c7` (exact, no later commits); `tmp/240-qual-c` = the
commit containing this file (the merged ancestry tip `66e12d3e` plus this doc commit) —
`git rev-parse origin/tmp/240-qual-c`.

Environment: RTX 5090, `sm_count=170`, cc 12.0 (`sm_120a`); driver 595.91.07, CUDA 13.1 devShell.
SM count probe (`cudaGetDeviceProperties`): `name=NVIDIA GeForce RTX 5090 sm_count=170 cc=12.0`.
CUDA needs `/run/opengl-driver/lib` first on `LD_LIBRARY_PATH` inside `nix develop`.

## Raw CTest summaries (from the committed state)

Group (c), `tmp/240-qual-c`:

```
1/9 Test  #83: ninfer_rmsnorm_test ....................   Passed    4.68 sec
2/9 Test  #85: ninfer_gated_rmsnorm_test ..............   Passed    1.52 sec
3/9 Test  #95: ninfer_rope_test .......................   Passed    2.14 sec
4/9 Test #108: ninfer_softmax_attention_test ..........   Passed  508.53 sec
5/9 Test #111: ninfer_rmsnorm_rope_test ...............   Passed    2.56 sec
6/9 Test #116: ninfer_sparse_moe_test .................   Passed    2.57 sec
7/9 Test #118: ninfer_sparse_moe_nvfp4_gate_up_test ...   Passed   11.29 sec
8/9 Test #119: ninfer_sparse_moe_nvfp4_prefill_test ...   Passed    6.63 sec
9/9 Test #120: ninfer_sparse_moe_nvfp4_decode_test ....   Passed    3.81 sec
100% tests passed, 0 tests failed out of 9
```

Group (e), `tmp/240-qual`:

```
1/2 Test #135: ninfer_linear_q8_a16_test ........   Passed   13.35 sec
2/2 Test #140: ninfer_linear_bf16_a16_test ......   Passed  149.85 sec
100% tests passed, 0 tests failed out of 2
```

Command (identical shape for both, from inside the worktree):

```bash
nix develop -c bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH; \
  ctest --test-dir build --output-on-failure -R "<the suites above>"'
```

The YaRN `s <= 1` byte-identity invariant is enforced by `run_yarn_structural_inert` inside
`ninfer_rope_test` (s=1.0 YaRN output compared byte-for-byte with the unextended power-law output);
the built binary contains the labels `yarn table: s=1.0 ...`, `yarn 27b text inert`,
`yarn structural-inert`.

## Group (c) — per-commit verdicts

| commit | verdict | collision / action |
|---|---|---|
| `a667efdd` rope launch capacity from device sm count | **take** | same-hunk conflict with fork YaRN (`include/ninfer/ops/rope.h`, `src/ops/launcher/rope.{h,cu}`, `src/ops/wrapper/rope.cpp`, `execution/attention.{h,cpp}`, `execution/text.cpp`, `tests/ops/test_rope.cpp`). Re-applied the YaRN overloads/launcher on `DeviceExecutionView` (reconciliation `96701fde`). |
| `417eb3d6` norm + moe launches from device sm count | **take** | collides with fork NVFP4 MoE (`src/ops/sparse_moe/prefill/sparse_moe_prefill_{h,kernels.cu}`, `src/ops/wrapper/sparse_moe.cpp`, `include/ninfer/ops/sparse_moe.h`); upstream hunks and fork NVFP4 hunks are disjoint, auto-merged cleanly; two fork-only NVFP4 test call sites updated. |
| `e621c7d6` attention launch plans from device sm count | **take** | fork-touched `planning/startup.{cpp,h}` and `execution/text.cpp` auto-merged; `src/ops/softmax_attention/**` and the attention oracle are fork-untouched. |

### Static identity at 170 SM (why the refactor is launch-neutral here)

- rope: old `kLargeBlockWaveCapacity = 1020`; new `execution.multiprocessor_count *
  kLargeBlockResidentCtasPerSm = 170 * 6 = 1020`.
- moe prefill: old `kPrefillMaxBlocks = kPrefillMaxBlocksPerSm * kRtx5090SmCount = 32 * 170 = 5440`;
  new `170 * kPrefillQueuedCtasPerSm(32) = 5440`. The fork NVFP4 prefill/down launches keep their
  own fork-owned caps `kNvfp4MaxBlocks`/`kNvfp4DownMaxBlocks = 32 * 170` (upstream does not touch
  them; equal to the same 5440 at 170 SM).
- attention: `causal_query_tiles_underfill_sms(it,170)` reproduces `((170/it)*it) < 170*9/10`;
  `causal_partition_target(x,it) = clamp(x/it, 1, CausalKvPartition::kMaxSplits=256)` matches the
  prior `clamp(...,1,256)`; `mxfp8_tiled_partition` `ceil(ctas/170)` unchanged.

### Measured before/after (fork `master` build vs `tmp/240-qual-c` build, same GPU)

- `ninfer_rope_bench`: 23/23 cases, identical `route=` labels, median delta 0.000 %, mean +0.29 %,
  max +5.3 % on a 4.38 µs case (timer granularity); 0.0 % on vision 4096/49152.
- `ninfer_causal_softmax_attention_bench` (both entries, all geometries/dtypes, B=1,8):
  100/100 cases, **0 workspace/peak differences**, median delta 0.000 %, mean +0.19 %; the only
  >5 % deltas are 20–43 µs decode cases where the 2 µs timer step is 5–10 %.
- `ninfer_sparse_moe_bench` (q4-q5, T=1/33/65/97): identical workspace; medians
  16.384/251.904/319.488/382.944 → 16.384/253.952/319.488/380.928 µs (≤0.8 %).
- `ninfer_rmsnorm_bench --kind gated27`: 2.816→3.360 µs (T=1) … 18.848→18.144 µs (T=2048), noise.

## Group (e) — the 23 `feat(ops): support and tune … linear` commits

All **take**. `src/ops/linear/**`, `tests/ops/linear/**`, `bench/ops/linear_bench.cu`,
`include/ninfer/ops/linear.h` and `tests/ops/quantized_weight.h` are fork-untouched, so the merged
result is exactly the upstream tip state; only `tests/README.md` (docs) overlaps.

| shape(s) | commits | verdict | covering suite |
|---|---|---|---|
| bf16 geometries (18) | `471924b0 f58e32e0 714c5149 bcdcbd45 8beb4cdd 16bdb491 2553e26e cf91c818 07e1f8c3 d0f91cd9 070fa61a 7971ff18 c02a07b4 dd73c88e 9879d052 a64b5eea b21780f3 15cba227` | take | `ninfer_linear_bf16_a16_test` Passed |
| q8 geometries (5) | `35e9b5c8 04ded76f 8f0aa859 58ab3f22 cd521ab7` | take | `ninfer_linear_q8_a16_test` Passed |

Both suites use the FP64 `dot_fp64`/`oracle_all_rows` oracle. The bf16 suite also re-checks the two
pre-existing geometries (14336×5120, 5120×6144) and the selector (256×5120), so the five commits
that edit shared schedule/MMA headers (`bcdcbd45 324×10240`, `16bdb491 10240×320`,
`c02a07b4 3456×1152`, `a64b5eea 1152×4304`, `04ded76f q8 12288×2560`) are regression-checked on
existing shapes too. Full-vocabulary `cf91c818` (248320×2560) is in the passing bf16 suite.

## Regression / fog item

**No measurable regression.** The SM refactors are launch-neutral at `sm_count == 170`, so the map's
re-tuning fog item is **not** graduated. Residual portability note (not a regression): the fork NVFP4
MoE caps and the fp8 linear schedule headers still hardcode `170`; correct on the 5090 and the NVFP4
oracle suites pass, but they are not SM-neutral if a non-170-SM target is ever intended.

## Not verified

- End-to-end `.ninfer` inference (op/runtime-library level only; `ninfer_model_runtime` compiles).
- Non-170-SM devices — the refactors' portability purpose is not exercised.
- Upstream groups (a)/(b)/(d)/(f); this record covers (c) and (e) only.
- Served grammar/logprob behavior in the reconciled `text.cpp` (compiles; op suites only).
- Aggregate end-to-end perf attribution (op microbenchmarks only).
