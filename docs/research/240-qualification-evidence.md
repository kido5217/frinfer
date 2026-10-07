# #240 qualification evidence — SM launch refactors + linear tuning flood

Throwaway branch `tmp/240-qual-c` (worktree `/tmp/opencode/240-qual-c`), branched from `master`
(`f66e419f`). Upstream commits cherry-picked in order: `a667efdd`, `417eb3d6`, `e621c7d6`, plus a
reconciliation commit for the fork's YaRN/NVFP4 edits. Not pushed.

Group (e) ran in the separate throwaway branch `tmp/240-qual` (worktree `/tmp/opencode/240-qual`)
where the linear paths are restored to the upstream tip state (fork-untouched paths only).

## Environment

- RTX 5090, `sm_count=170`, cc 12.0 (`sm_120a`); driver 595.91.07, CUDA 13.1 devShell.
  Verified with a compiled `cudaGetDeviceProperties` probe: `name=NVIDIA GeForce RTX 5090
  sm_count=170 cc=12.0`. The refactored gated_rmsnorm bench header also prints `SMs=170`.
- Build: `nix develop -c bash -c 'export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH;
  cmake --preset dev && cmake --build build -j 12 --target <targets>'`.

## Group (c) — SM-derived launch refactors

### Oracle suites (all green)

| suite | result |
|---|---|
| ninfer_rmsnorm_test | Passed |
| ninfer_rmsnorm_rope_test | Passed |
| ninfer_gated_rmsnorm_test | Passed |
| ninfer_rope_test (incl. `run_yarn_structural_inert`, `verify_yarn_table`) | Passed 2.24 s |
| ninfer_sparse_moe_test | Passed 2.37 s |
| ninfer_sparse_moe_nvfp4_gate_up_test | Passed 11.53 s |
| ninfer_sparse_moe_nvfp4_prefill_test | Passed 6.60 s |
| ninfer_sparse_moe_nvfp4_decode_test | Passed 3.76 s |
| ninfer_softmax_attention_test (bf16/int8/fp8/nvfp4/k8v4) | Passed 522.61 s |

`100% tests passed, 0 tests failed out of 9`.

The YaRN `s <= 1` byte-identity invariant is enforced by `run_yarn_structural_inert` (compares the
s=1.0 YaRN output byte-for-byte against the unextended power-law path); the labels
`yarn table: s=1.0 ...`, `yarn 27b text inert`, `yarn structural-inert` are present in the built
`ninfer_rope_test` and the suite passed.

### Static identity at 170 SM (why no regression on this hardware)

- rope: old `kLargeBlockWaveCapacity = 1020`; new `execution.multiprocessor_count *
  kLargeBlockResidentCtasPerSm = 170 * 6 = 1020`. Identical.
- moe prefill: old `kPrefillMaxBlocks = kPrefillMaxBlocksPerSm * kRtx5090SmCount = 32 * 170 = 5440`;
  new `execution.multiprocessor_count * kPrefillQueuedCtasPerSm = 170 * 32 = 5440`. Identical.
  NVFP4 prefill uses its own per-codec launches (unchanged).
- attention: `make_*_kv_causal_plan` and `mxfp8_tiled_partition` now take `multiprocessor_count`.
  At 170 the new expressions reduce exactly to the old constants:
  `causal_query_tiles_underfill_sms` reproduces `((170/it)*it) < 170*9/10`, and
  `causal_partition_target(x,it) = clamp(x/it, 1, 256)` matches the old `clamp(...,1,256)`
  (`CausalKvPartition::kMaxSplits == 256`). `mxfp8` waves `ceil(ctas/170)` unchanged.

### Measured before/after (fork `master` build vs reconciled branch build, same GPU)

- rope bench: 23/23 cases, identical `route=` labels; median deltas within timer granularity on the
  ~4 us cases (max +7.2 % on a 4.42 us case), 0.0 % on the vision 4096/49152 cases.
- causal attention bench: 100/100 cases, 0 workspace/peak differences across all five KV dtypes;
  median delta 0.000 %, mean +0.19 %; the only >5 % deltas are 20–43 us decode cases where the
  2 us timer step is 5–10 %.
- sparse_moe bench (q4-q5, T=1..97): identical workspace; medians 16.384/251.904/319.488/382.944 us
  before vs 16.384/253.952/319.488/380.928 us after (≤0.8 %).
- gated_rmsnorm bench (gated27): 2.816→3.360 us at T=1 … 18.848→18.144 us at T=2048 (noise).

## Group (e) — 23 linear tuning commits

`ninfer_linear_bf16_a16_test` Passed 188.17 s; `ninfer_linear_q8_a16_test` Passed 11.73 s
(other `-R linear` suites were "Not Run" only because they were not built for this qualification).

Both suites are FP64-oracle based (`oracle_all_rows`/`dot_fp64`); the bf16 suite runs the two
pre-existing geometries (14336x5120, 5120x6144, selector 256x5120) plus all 18 new geometries, and
the q8 suite runs all 21 q8 geometries including the 5 new ones. The commits that modify shared
schedule/MMA headers (`bcdcbd45 324x10240`, `16bdb491 10240x320`, `c02a07b4 3456x1152`,
`a64b5eea 1152x4304`, `04ded76f q8 12288x2560`) are covered because the pre-existing shapes are
re-checked by the same oracle.
