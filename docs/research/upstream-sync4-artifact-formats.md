# Upstream sync 4 — do our `.ninfer` artifacts need re-conversion?

Research ticket: kido5217/frinfer#307. Upstream range: `81c8ce09..64d10fbf` (77 commits;
merge-base `81c8ce09`, verified with `git merge-base master upstream/master`). Fork baseline:
`baab27b6`. All claims below are from `git diff`/`git show` on local refs of this repository.

> Review gate (#307): verdict unchanged; this revision fixes reviewer findings — commit
> attribution of the QType 9 / QuantLayout 4 appends (`d8c0fa6f`, not `7c4c5e3e`), the
> rank-3-rejection test citation (`test_reader.cpp`, not `test_materialization.cpp`), artifact
> paths (`models/`, not `out/`, for the 27B pair) and the 27B stored-identity enumeration (now
> byte-verified), and the new-format count in the seam list (two, not one).

## Verdict

**Keep both artifacts. No re-conversion required.** Every change in the range to the artifact
container, the format/layout registry, the loader validation path, and the existing-recipe packing
is additive: new registered identities that old records never use, internal refactors with identical
byte semantics for those identities, and relaxed (not tightened) validation. The pre-agreed rule on
the map — additive → keep, tightened validation or changed packing/semantics → re-convert — lands
on "keep". The merged-tree load smoke (sync procedure step 5) remains the final gate; this report
explains why it is expected to pass.

Stored identities below were confirmed by direct inspection of each file's v3 directory (review
correction pass), not only by recipe inference.

| Artifact | Route | Stored identities | Verdict |
|---|---|---|---|
| `models/qwen3_6_27b.ninfer` (deployed) | `qwen3_6_27b` / `qwen3_6_27b_nvfp4` recipe | pre-existing identities only — `q4_g64_fp16` (183), `q5` (246), `q6` (3), `q8` (7), `bf16` (582), `fp32` (96), `int32` (1), with `row_split_k128_v1` / `contiguous_le_v1` layouts | keep |
| `models/qwen3_8_27b_nvfp4.ninfer` (smoke; named pathless in `docs/maintainer/upstream-sync.md:61`) | `qwen3_8_27b_nvfp4` recipe (`tools/convert/official_recipes.py:151`) | 112 rank-2 `nvfp4` + `block_scale_k16_m128x4_v1`, 146 `fp8_e4m3fn_row_bf16` + `row_scale_v1`, remainder pre-existing (`contiguous_le_v1` etc.) | keep |
| `out/qwen3_6_35b_a3b_nvfp4.ninfer` (on-disk sibling) | `qwen3_6_35b_a3b_nvfp4` | same class: 30,720 rank-2 `nvfp4` expert parents, `q8_g32_fp16`/`q6_g64_fp16` (`row_split_k128_v1`), `bf16`, `fp32` (conversion report `/formats`) | keep |

## Evidence

### 1. The v3 container itself is untouched

`src/artifact/framing.h`, `src/artifact/schema.cpp` (directory parsing/required fields), and
`src/artifact/layouts.cpp` (the C++ geometry computation for every registered layout) have **zero
diffs** across `81c8ce09..upstream/master` (`git diff --stat` over the full range lists neither).
No container version identifier changed: grepping the range diff of `src/artifact` and `src/core`
for `version` returns nothing. `kEntryMagic` still ends at v3 and the v2-rejection branch in
`reader.cpp` is preserved semantically (the fork only reworded the error to "FrInfer v3"). The
directory record set stays: same binding parts, no new required field, and the new converter-era
conveniences (`AuxiliaryReference`, `placement`, `source_origin` in `tools/convert/recipe.py`) are
producer-side recipe plumbing for future conversions, not loader requirements.

### 2. New format/layout identities are pure appends

`d8c0fa6f` adds `QType::NF4_G16_E4M3FN = 9` and `QuantLayout::RowPagedG16_4K = 4`; `7c4c5e3e` adds
`QType::FP8_E4M3FN_BLOCK128X128_BF16 = 10` and `QuantLayout::BatchedBlockScaleK16M128x4 = 5`,
`BlockScaleM128K128 = 6` (`src/core/weight.h`). The pre-existing numeric values 0–8 / 0–3 are unchanged, so old record
fields decode identically. The same additive pattern is on the Python side
(`tools/artifact/formats.py`, `tools/artifact/layouts.py`): new dataclasses and registry entries;
the nvfp4 / row-scale / row-split / contiguous entries keep their old definitions. The rewritten
docs (`docs/maintainer/storage-layouts.md` §4.1, §5.1) confirm: `batched_block_scale_k16_m128x4_v1`
stores rank-3 `[E,N,K]` "with the same inner-matrix code packing and scale swizzle as Section 4" —
Section 4 (`block_scale_k16_m128x4_v1`) itself has no diff hunks. The registry line for the old
layout is unchanged, including its rank-2-only constraint; upstream's new test even asserts
`weight_geometry(QType::NVFP4, BlockScaleK16M128x4, [E,N,K])` **rejects** a bank
(`tests/artifact/test_reader.cpp:166` at tip, added in `7c4c5e3e`: "single-matrix NVFP4 layout
accepted a bank"). So the old identity's domain did not move.

### 3. Rank-2 NVFP4 bytes and divisor semantics are bit-identical

The "nvfp4 banks" feature generalizes geometry from one matrix to E: Python
`block_scale_geometry` now multiplies plane sizes by `matrices` (= 1 for rank-2), and
`payload_bytes = weight_divisor_offset + 4 * matrices` reduces to the old `+ 4`
(`tools/artifact/codecs/nvfp4.py` `_positive_fp32_word` → `_divisor_words(value, count)`; for
count 1 the check and encoding are the same). The swizzle permutation for rank-2 is the identical
expression (`source.reshape(1 * n // 128, 4, 32, k_tiles, 4)`). On the C++ side
`src/artifact/layouts.cpp` is unchanged, and the new `WeightGeometry` carries
`divisor_offset`/`divisor_count` with `divisor_count = 1` for `block_scale_k16_m128x4_v1`
(`src/core/weight_view.cpp`). The doc phrase change "one positive FP32 weight divisor" → "one
positive FP32 divisor **per matrix**" is a no-op for rank-2 (one matrix = the whole object), which
is exactly how every nvfp4 record in our artifacts is stored — per-expert rank-2 parents, as the
fork's own note in `docs/maintainer/tensor-formats.md` ("The Qwen3.6-35B-A3B MoE routed banks…")
and the conversion report attest.

### 4. The loader path changed by refactor, not by new requirements

- `src/artifact/reader.cpp` (`3124dffe` + `6a42de35`): `InputFile` → `io::ReadOnlyFile`, optional
  direct (`O_DIRECT`) mode, and re-checks on open. Every check has a base-era predecessor: the
  entry-file length check `bytes == entry_payload_start + payload_bytes` existed verbatim in the
  base `Impl` constructor ("entry length differs from directory"), continuation part-magic/index/id
  and length checks are unchanged, and the new `header != entry_header` comparison just re-reads
  the bytes the directory was parsed from. An artifact that loads on the base loads on the merged
  tree. Exception translation (`std::system_error` → `ArtifactError`) preserves the contract.
- `src/artifact/binder.cpp`: **drops** the `Residency` parameter and the eager
  `require_device`/`host_object` demand recording from `binding()` — strictly more permissive at
  bind time; divisor/format/element checks are the same (`exact_format` validation line unchanged).
- `src/artifact/materializer.cpp` (`7c4c5e3e`, `fddba033`, `c18d72db`): placement/backing
  reorganization (new `placement.cpp`/`backing.cpp`). The divisor validation moved from
  `read_divisor` (finite positive FP32) to `validate_divisors` over the host-borrowed span — same
  predicate ("invalid weight divisor"). `fddba033` itself changes only host page-mapping load
  mechanics (`materializer.cpp`, `placement.cpp`, `docs/maintainer/expert-weight-cache.md`) and
  touches no record interpretation.
- `src/core/weight_view.cpp` `native_weight`: the "requires a complete FP8/NVFP4 parent" restriction
  was replaced by tile-aligned view support (128-row domains, per-matrix divisor selection) — a
  relaxation; complete-parent rank-2 views (what qwen3_5 builds today) pass unchanged, and the
  finite-positive-divisor requirement persists.
- `src/ops/weight_input.cpp`: hard-coded projection-shape/format allowlists were removed in favor
  of generic bank regions — fewer rejections on existing paths.

### 5. The converter and model changes are new routes, not old ones

`tools/convert/` changes (`__main__.py` architecture dispatch, `qwen4_exp.py` 648 new lines,
`official_recipes.py` +218, `methods.py`, `sources/fp8_block.py`, `sources/modelopt_nvfp4.py`,
`stored_scale*`) add the `Qwen4ExpForCausalLM` route and its producer-side codecs. The
`qwen3_5` model refactor (`9cfd5646`) moves shared code under `src/models/common/qwen` but keeps
the logical parameter names identical (`p + "attention/query"`… lines are relocated verbatim), so
bindings in our artifacts resolve unchanged. The MTP block-fp8 import in `7c4c5e3e` writes new
`block_scale_m128_k128_v1` records that our artifacts simply don't contain.

## Fork-seam collision notes for conflict resolution

Overlap set for this area (both sides changed; recompute at merge time):

- `src/artifact/reader.cpp` — fork changes only the magic error string (`NInfer` → `FrInfer`);
  upstream rewrote the surrounding IO code → textual conflict at `Impl::Impl`/`open_file`. Adopt
  upstream's structure, re-apply the FrInfer wording.
- `tools/artifact/formats.py`, `layouts.py` — fork changed one docstring line each; upstream added
  registry entries nearby. Trivial; keep branding + upstream registrations.
- `tools/convert/__main__.py` — fork added the GGUF dispatch (`_open_source`, `SourceInputs`,
  `--model` base selection); upstream added the Qwen4Exp architecture branch and
  `source-manifest.json` provenance in the same functions. Real conflict: re-apply `_open_source`
  so `--model`/`--source` still accept `.gguf` on upstream's flow.
- `tools/convert/official_recipes.py` — fork inserted `qwen3_6_35b_a3b_nvfp4` right after
  `qwen3_8_27b_nvfp4` and added registry entries; upstream edited the import block (new
  `methods`/`sources` symbols) and added `qwen3_8_flash_next` + registry lines. Conflicts in the
  imports and the recipe registry; keep both sides' functions.
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, `tensor-formats.md`,
  `docs/weight-conversion.md` — upstream rewrote these; the fork renamed branding, adjusted the
  format count ("nine" — now stale; the range registers two new numeric formats, so adopt upstream's
  rewritten count verbatim rather than computing nine+N), added the
  per-expert-parent paragraph (tensor-formats §near row_scale) and the GGUF/35b conversion
  sections (weight-conversion.md). Adopt upstream's text, then re-apply: branding, the 35b
  per-expert-parent note (still accurate: banks are optional), and the fork's GGUF route docs.
- `tests/artifact/tests.cmake` — fork added `test_inspect_hash.py`; upstream added
  `test_expert_banks.py` (via `7c4c5e3e`). Adjacent-list conflict.
- Adjacent non-artifact seams the sync doc already tracks: `src/models/qwen3_5/execution/text.cpp`
  and the ops overlap (`src/ops/weight_input.cpp`, `src/ops/wrapper/sparse_moe.cpp`) — upstream
  removed `prepare_attn_input_proj_weights`/`prepare_gdn_input_proj_weights`/
  `prepare_sparse_moe_weights` in favor of `prepare_projection_weights`; fork call sites in the
  qwen3_5 execution files must be ported while preserving the YaRN `s <= 1` byte-identity
  invariant.

## Residual risk

The one thing this analysis cannot replace is the merged-tree gate from
`docs/maintainer/upstream-sync.md` step 5: build, `ctest`, `pytest tests/convert`, and the
`qwen3_8_27b_nvfp4.ninfer` smoke (native at `--max-context 262144`, YaRN boundary at 262145).
Everything above predicts those pass for artifact loading; if the smoke fails, the failure mode to
look for first is the qwen3_5 shared-foundations refactor changing which parameters/roles the
frontend requests — not the stored representation.
