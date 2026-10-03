# llama.cpp inference performance candidates — 5090 lens

Survey for wayfinder map #167 ("Wayfinder map: llama.cpp port candidates"), ticket #173.
Scope: kernel/algorithm techniques in llama.cpp that are port candidates for NInfer's
single-GPU (RTX 5090, sm_120a) inference path: attention, KV cache, MoE, speculative
decoding, decode batching, memory layout, CUDA graphs, published Blackwell gains.

- **llama.cpp state verified at**: `1537a0a8b2f8711d840878b0a0677ab2213c882c`
  (master, committed 2026-10-03 15:19:13 UTC, "server : fix laya abort by limiting
  n_batch to n_ubatch (#29903)"). Verified via `gh api repos/ggml-org/llama.cpp/commits/master`
  on 2026-10-03; unchanged from the sibling survey's checkout. All path citations are at
  this commit (local mirror: `/tmp/opencode/llama-1537a0a8/`, extracted from the GitHub
  tarball of the verified commit).
- **NInfer master ref**: `3e0ff840b9f3f3efc6ee82305b540198f6e6bd81`
  (2026-10-03, `feat(cli): report realized reasoning tokens … (#162)`).
- **Method**: primary-source code reads at the verified commit; `gh api` for PR/issue
  state and benchmark claims (hardware recorded per number); NInfer state verified
  against `docs/` and `src/` of the primary repo (read-only). No GPU runs, no builds.
  Verdicts rank by (a) contract/protocol gap > (b) measurable 5090 headroom > (c) QoL;
  (b) is the primary axis for this family. A performance "port" names the evidence
  required before implementation (graduates to a follow-on ticket).

NInfer baseline facts used here (from `docs/performance.md`, `docs/performance/qwen3.8-27b.md`
at the ref above, campaign on one RTX 5090, driver 617.14, CUDA 13.4, NInfer rev `7f6aafed`):
Qwen3.8-27B `nvfp4` prefill 12,819 tok/s @ 7,680 ctx and 4,016 tok/s @ 260K ctx;
MTP0 single-request decode 74.1 tok/s @ 7,680 ctx (53.4 @ 260K); MTP3 decode 123.4–231.7
tok/s (36.6–90.6% acceptance); DFlash2 K=7 decode 115.0–377.3 tok/s (15.9–77.3%
acceptance) — full single-request ranges at both quantizations. Decode-batch baseline
~13.7 ms/step (grammar A/B at
`b84e4fb4`). KV profiles in production: FP8 E4M3 row-256 (Qwen3.8), INT8 group-64 (Qwen3.6);
paged KV P=64, CUDA-graph-captured decode per exact-`B` topology with page IDs as stable
input data (`docs/maintainer/paged-kv-cache.md`, `docs/maintainer/engine-architecture.md`).
Model geometry (27B dense / 35B-A3B MoE): 64 / 40 layers, 16 / 10 full-attention +
48 / 30 GDN, head dim 256, GQA 24:4 / 16:2, MoE 256 experts top-8 + shared expert
(`docs/maintainer/qwen3_5-model.md`).

---

## C1 — NVFP4 MoE expert weights on Blackwell (W4A4 tensor-core GEMM)

**llama.cpp state** (commit `1537a0a8`): merged native NVFP4 path for sm_120:
- `ggml/src/ggml-cuda/mmq.cu` — `mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>` selected
  under `blackwell_mma_available(cc)`; `use_native_fp4` feeds `src1_scale` to the FP4
  SRAM-layout MMQ kernel (`mmq-config-blackwell.cuh` `CASE(GGML_TYPE_NVFP4, 256, 1, 128, …,
  GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, …)`).
- MoE GEMM (`GGML_OP_MUL_MAT_ID`) routes through the same MMQ family with
  `expert_bounds` scatter scheduling and `dedup_bcast` (duplicate expert-id collapse)
  (`mmq.cu:235-303`, `mmq.cuh:1387`).
- Closed-UNMERGED PR #28572 "ggml-cuda: pipeline NVFP4 MMQ tile loads with cp.async and
  TMA on Blackwell (+14% pp)" (closed 2026-09-17; NOT in master at 1537a0a8 — a grep for
  TMA/cp.async/tensor_map across `ggml-cuda/` at the verified commit finds no such
  kernel): the proposed mechanism was overlap of tile loads with MMA + TMA-engine loads +
  spill-free accumulate, NVFP4/Blackwell only.
- In-flight (open, NOT in master): #26159 "compact Blackwell NVFP4 MoE work scheduling
  (+10% to +15% prefill, NVFP4-only)", #29857 "Optimize accumulation in mmq for NVFP4
  type", #26704 "experimental SM120 CUTLASS MoE prefill for MXFP4 and NVFP4", #28898
  "quant scales for fp8 and nvfp4".

**Mechanism + published gains**: NVFP4 = 4-bit E2M1 elements with 16-element FP8(E4M3)
block scales (two-level scaling + optional RHT in "true" NVFP4); on sm_120a the GEMM runs
native FP4 tensor-core MMA. Measured numbers found this session:
- #28572 (closed **unmerged** — its table reports the master baseline `dbeb37548` vs the
  PR build; a PR-build number, not merged-master state): RTX 5090 @ 600 W, CUDA 13.3,
  Qwen3.8-27B NVFP4, `llama-bench -ub 1024 -p 16384 -n 0 -r 3`: pp16384 6,146.5 ± 9.8 →
  7,009.7 ± 9.1 tok/s (+14.0%);
  wikitext-2 perplexity 7.1927 → 7.1957 (no measurable quality change); non-NVFP4 model
  unchanged (Q5_K pp4096 3,199.5 → 3,199.9, tg64 71.1 → 70.8 tok/s).
- #23572 (**open**, unmerged, claim only): "Blackwell PP +40%" for a CUDA-native NVFP4
  quantization; mean KLD 0.0894 vs Q4_0 0.0486 (between Q3_K and Q4_0); second scale level
  alone costs −8% without a fused kernel.
- For scale, NInfer's own dense-27B data shows the NVFP4 vs int gap on this GPU:
  prefill 12,819 vs 3,332 tok/s @ 7,680 ctx (~3.8×, compute-bound prefill).

**NInfer applicability**: NInfer's dense 27B already ships and is published as `nvfp4`
(tensor-core NVFP4 linears; `src/ops/linear`). The MoE 35B-A3B does **not**:
`include/ninfer/ops/sparse_moe.h` admits only `Q4+Q5, Q4+Q6, Q8+Q8` for the two routed banks (shared
banks Q8), and `docs/performance.md` publishes 35B-A3B only under `groupwise-int`. This is
an (a)-axis contract gap (artifact codec profile + converter recipe + `sparse_moe` codec
support) with a clear (b) payoff on prefill; decode is bandwidth-bound and the byte
savings of NVFP4 (4.5 bpw effective) over Q4-class banks are modest — the decode gain must
be measured, not assumed.

**Port cost**: high. Converter recipe for routed-expert NVFP4 (two-level scales) + loader
binding + `sparse_moe` NVFP4 routed-bank profile (new Op codec, oracle-qualified per
`docs/maintainer/op-development.md`) + graph/envelope checks; artifact regeneration for
35B-A3B.

**Verdict: port** (axis a+b). Highest-ranked candidate: the only one with an open artifact
contract gap *and* a demonstrated ~4× prefill lever on this exact GPU/model class.
**Required evidence before implementation** (follow-on ticket):
1. Op-level: NVFP4 routed-bank GEMM qualified against the independent oracle at the
   registered geometry (256 experts, top-8, shared expert, T=1..8 decode and prefill T),
   packed inputs decoded with stored scales.
2. Kernel: decode-step and prefill-phase measurements for 35B-A3B `nvfp4` vs the
   published `groupwise-int` baseline at the claimed scope (decode ms/step at C=1..8;
   prefill phase at 8K..256K).
3. End-to-end: serving decode phase + corpus makespan through the public route, and
   perplexity parity vs `groupwise-int` (`docs/perplexity.md` contract).

## C2 — NVFP4 GEMM load pipelining (cp.async + TMA) for NInfer's dense NVFP4 path

**llama.cpp state**: #28572 (closed unmerged, above) proposed: the NVFP4 MMQ kernel
overlapping tile loads with math, moving loads to the TMA engine, removing register
spills in the accumulate step. Not in master at the verified commit.
Explicitly NVFP4/Blackwell-only (`blackwell_mma_available` gate).

**Mechanism + gains**: see C1 (#28572, +14% pp16384 on RTX 5090, Qwen3.8-27B NVFP4;
perplexity unchanged). The PR's author states it originated from profiling the prefill gap
between NInfer and llama.cpp, and points at a further ~25% from "Qwen3.5 gated delta net
prefill, GLU→MMQ fusion" as separate follow-ups (the GLU one is open as #28702; GDN
prefill work appears as open #26001 "GDN chunked kernel for prefill").

**NInfer applicability**: technique-level port into NInfer's own dense NVFP4 linear
kernels (Ops boundary, `src/ops/linear`). NInfer already publishes nvfp4 27B prefill at
12.8K tok/s @ 7.7K ctx vs the llama.cpp master baseline of 6,146.5 tok/s @ 16K (per
#28572's table; its ~7,010 tok/s PR build is closed-unmerged) — NInfer's dense NVFP4
prefill is not currently the loser, so headroom is uncertain and must be measured, not
inferred from llama.cpp's numbers (different kernel structure, ubatch, context).

**Port cost**: medium (kernel rewrite of the NVFP4 linear load/accumulate pipeline;
oracle qualification per Op contract).

**Verdict: defer** (axis b, unproven headroom). Rank below C1. **Evidence required**:
prefill-phase A/B (8K..256K ctx, both weight profiles) at the claimed scope before any
implementation commitment; if NInfer's dense NVFP4 shows no gap to the TMA-pipelined
reference, close without porting.

## C3 — Attention kernel tuning family on Blackwell (MMA tiling, GQA-packed decode, stream-K)

**llama.cpp state** (commit `1537a0a8`), all in `ggml/src/ggml-cuda/`:
- Dispatcher `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:~560-720`): selects among
  `VEC` (small batch, quantized K/V native dots), `TILE` (f16/bf16, large batch),
  `MMA_F16` (tensor-core f16). For decode (Q rows=1, batch=1) on sm_120 (Ada+) it picks
  VEC **except** when GQA ratio > 4 and (D ≥ 256 or KV ≥ 8192) — exactly NInfer's 27B
  geometry (D=256, GQA 6) — which then takes `MMA_F16` with the in-kernel **sparse GQA
  gather**: a (DKQ, DV, ncols1=1, ncols2=8) configuration packing the GQA group's query
  heads into one MMA tile (`fattn-mma-f16.cuh`,
  `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse`).
- MMA kernel carries per-arch tuned parameters (`nbatch_fa`, `nbatch_K2/V2`, `nstages`,
  swizzle, Q-in-reg, ncols tiling) via `ggml_cuda_fattn_mma_get_*`; **no Blackwell
  config in master** — PR #24565 "fattn-tune: Add Blackwell MMA config" is open.
- **Stream-K**: `should_use_stream_k` splits the KV loop across extra blocks when the
  output tiles under-fill the SMs (< 75% efficiency), with uniform/general fixup kernels
  (`fattn-common.cuh:1141-1292`, `flash_attn_stream_k_fixup_{uniform,general}`). Grid
  sizing uses `max_blocks_per_sm * nsm` (issue #27444 shows the failure mode when `nsm`
  is misread).
- Pre-filter kernels for masked attention: `flash_attn_mask_to_sparse_indices` /
  `ggml_cuda_flash_attn_ext_compact_mask` build per-query-tile visible-KV index lists
  (union over the tile's queries) — the substrate for sliding-window/sink-style sparsity
  in the FA2 path (which also carries a `sinks` tensor, `dst->src[4]`).

**Mechanism + published gains**: the Blackwell-specific tuning is still open (#24565), so
master's published numbers are the NVFP4/GEMM ones (C1/C2); no merged Blackwell-attention
pp/tg benchmark number was found this session. Issue #27444 (RTX 5090, Qwen3.8-27B IQ4_XS):
decode 85.3 tok/s @ 2k → 41.0 tok/s @ 17k ctx (vec→MMA kernel switch near KV 8192;
resolved as a build-environment bug, not a kernel defect — "the heuristic itself is
correct"), and "the same model on vLLM and NInfer looks ok on the same generation length".
That last line is a directional (unverified-here) signal that NInfer's long-context decode
does not exhibit llama.cpp's 8k-ctx attention cliff.

**NInfer applicability**: NInfer's full-attention decode (16 layers @ 27B, D=256, GQA 6/8,
B=1..8, up to 524,288 visible keys incl. YaRN) is Ops-owned (`softmax_attention` reads
paged KV directly, including quantized profiles — it does *not* need the vec/quantized-KV
family). The transferable ideas are technique-level: GQA-group-packed MMA tiles for
decode; stream-K partitioning for long-KV/small-batch; per-arch stage/swizzle tuning.
None of them changes the public Op contract.

**Port cost**: medium–high (NInfer kernels are bespoke; this is re-implementation of
techniques with per-geometry tuning, not a file copy).

**Verdict: port** (axis b) — the only remaining decode-path lever with a plausible 5090
headroom at long context, where NInfer's published decode drops 74.1 → 53.4 tok/s
(7.7K → 260K ctx, nvfp4 MTP0). Ranked below C1 (no contract gap, smaller proven delta).
**Required evidence** (follow-on ticket): attention microbenchmark grid over
(B ∈ 1..8 × ctx ∈ {7,680, 64K, 130K, 260K} × 16 full-attn layers, D=256 GQA-6/8)
comparing the current kernel vs (i) GQA-packed MMA tile and (ii) stream-K variant;
oracle qualification per `docs/maintainer/op-development.md` (naive FP64 SDPA); then
decode-phase A/B through the public serving route at the claimed scope.

## C4 — Quantized KV attention (vec family, quantized K *and* V)

**llama.cpp state**: `fattn-vec` instantiates head sizes {64,128,256} × K,V types
{f16,bf16,q8_0,q5_1,q5_0,q4_1,q4_0} (`template-instances/fattn-vec-instance-*.cu`,
generated by `generate_cu_files.py`); CLI exposes `--cache-type-k` / `--cache-type-v`
over {f32,f16,bf16,q8_0,q4_0,q4_1,IQ4_NL,q5_0,q5_1} (`common/arg.cpp:304-314`). The
MMA/TILE (prefill) path dequantizes K/V to f16 first (`need_f16_K/V`,
`fattn-common.cuh:1031-1093`) — quantization only helps the small-batch decode path.

**NInfer applicability**: NInfer is ahead. Its attention reads quantized cache rows
directly in all routes (BF16, INT8-G64, FP8-E4M3FN row-256, NVFP4-G16, K8V4 with Hadamard
rotation; `include/ninfer/ops/softmax_attention.h`, `kv_cache_append`), prefill included.
llama.cpp's 4–8-bit integer K/V block quants are coarser than NInfer's FP8/NVFP4+rotation
profiles.

**Port cost**: n/a. **Verdict: no** — NInfer already implements the superset; nothing
measurable to port on the 5090.

## C5 — Attention sinks

**llama.cpp state**: a `sinks` tensor (`LLM_TENSOR_ATTN_SINKS`, `blk.%d.attn_sinks`,
`src/llama-arch.cpp:697`) consumed by the FA2 path as `dst->src[4]`; StreamingLLM-style
always-visible first tokens for models trained with sink tokens (Gemma2-class).

**NInfer applicability**: Qwen3.5/3.6/3.8 configs carry no sink tensor; adding one would
change the model's attention mathematics (Models boundary) for a capability the served
models do not use.

**Verdict: no** — not in the served models' math; a port would be a product/model change,
not a performance technique.

## C6 — Sliding-window / ISWA / DSA / MSA KV-cache variants

**llama.cpp state**: `src/llama-kv-cache-iswa.cpp` (dual-cache: non-SWA + SWA layers),
`llama-kv-cache-dsa{,-iswa}.cpp` (DeepSeek sparse attention: lightning-indexer +
top-k token selection; `ggml/src/ggml-cuda/lightning-indexer.cu`), `llama-kv-cache-msa.cpp`
(indexer-cache KV twin with per-ubatch pos→cell maps), `llama-kv-cache-dsv4.cpp`, plus
`llama_swa_type` {NONE, STANDARD, CHUNKED, SYMMETRIC} (`src/llama-hparams.h:25-28`) and the
`llama-memory-hybrid*` compositions.

**NInfer applicability**: Qwen3.5's full-attention layers are full-context GQA (no window
in the math); GDN layers are linear attention with their own state (NInfer implements
GDN + ReplaySSM). Windowed attention in NInfer exists where the math requires it (DFlash
draft local layers, S=2048/4096). The DSA/MSA/indexer family is a *different model
mathematics* (DeepSeek/MiniMax-class sparse attention), not a kernel technique.

**Verdict: no** for the windowed variants (not in the served math); **flag** the
DSA/MSA/indexer family as a new mathematical architecture (see "Flagged for the map").

## C7 — Context eviction / cache reuse (server KV management)

**llama.cpp state**: `--cache-reuse` (prefix KV retention, default 256 tokens in presets,
`common/arg.cpp:3582,4600`), server-side context shift / sequence-removal-based eviction
in `tools/server`, `--swa-full` for windowed layers. All product-level KV *policy*, not
kernel performance.

**NInfer applicability**: NInfer's cache/retention/pressure policy is a different,
documented system (ResourceManager/pressure planner, checkpoints, prefix reuse —
`docs/maintainer/resource-scheduling-and-context-cache.md`). Policy porting is out of
family (QoL/product, not 5090 kernel performance).

**Verdict: no (perf family)**; flag as QoL/product for the map (see below).

## C8 — ngram speculative family (simple / map-k / map-k4v / mod / cache)

**llama.cpp state** (`common/speculative.cpp`, commit `1537a0a8`): CPU-side drafters with
no draft model — `ngram-simple` (last-position match over the prompt), `ngram-map-k{,4v}`
(in-memory Markov map, key-only or k:4v), `ngram-mod` (4 MiB match container,
`n_match`≥16 recommended, low-acceptance-round damping — cites PR #19164),
`ngram-cache` (static/dynamic on-disk n-gram caches, fixed n_draft=8). All run on the CPU
and feed the same verify/accept loop as draft models.

**NInfer applicability**: NInfer's spec backends are fixed Program internals (MTP /
DFlash / DFlash2, `docs/maintainer/engine-architecture.md` §1); ngram drafting would be a
CPU-side mechanism with no artifact support, and its acceptance ceiling on these models is
far below the published MTP3 (36.6–90.6% acceptance) / DFlash2 (15.9–77.3%) numbers.
Adding it buys nothing measurable for the served workloads.

**Verdict: defer** (axis c at best). Rationale: no draft weights needed is the only
advantage; acceptance quality is strictly worse than the trained draft backends NInfer
already ships.

## C9 — Acceptance strategies (rejection sampling, grammar-aware)

**llama.cpp state**: `common_sampler_sample_and_accept_n_rejection`
(`common/sampling.cpp:732`): accept draft token x iff `q_x > 0` and
(`p_x ≥ q_x` or `u < p_x/q_x`); on rejection sample the correction token from the
normalized residual `max(p − q, 0)` and stop; grammar-masked tokens carry p=0 with
renormalized p; `grammar_first` option. Used by the server when `temp > 0` **and** the
draft type supplies candidate probabilities (`spec_draft_q` — populated by
`draft-simple` and `draft-mtp` only; `tools/server/server-context.cpp:4124-4135`);
greedy requests use sample-and-match. DFlash/EAGLE3/DSpark drafts (top-k=10, no
probabilities) fall back to token-match acceptance.

**NInfer state (verified)**: `include/ninfer/ops/speculative_round.h` —
`speculative_accept_sparse_drafts` is "the variable-K, 16-candidate form of speculative
rejection sampling": accept with probability `min(1, p(d)/q(d))`, first rejection samples
normalized `max(p−q, 0)`, terminal token from target column P; `speculative_accept_greedy_drafts`
does the same with a one-hot proposal distribution (MTP proposals are greedy argmax —
`TextContext::mtp_propose_batch` → `proposal_argmax`, `src/models/qwen3_5/execution/text.cpp:842` —
so one-hot q is exact for NInfer's MTP).

**Verdict: no** — the acceptance machinery is already equivalent (NInfer implements the
identical rejection/residual math as an Op; DFlash2's 16-candidate lattice corresponds to
llama.cpp's candidate-probability path). No gap found.

## C10 — DSpark: Markov head + confidence gating on the DFlash draft

**llama.cpp state**: `src/models/dflash.cpp` (commit `1537a0a8`): a DFlash draft model
carries optional `markov_w1 [rank, n_vocab]` / `markov_w2 [rank, n_vocab_draft]` tensors;
the Markov head adds a low-rank, *previous-draft-token-conditioned* bias to each position's
draft logits (`build_dspark_markov_head`, `w2 @ w1_prev`), with anchor-first sampling
(`sample_from_anchor`, full `block_size` drafts) and an optional confidence head
`conf = sigmoid(conf_proj·[hidden; markov_w1[prev]] + b)` used with `--spec-draft-p-min`
to gate drafting. Auto-detected from GGUF metadata (`common_speculative_types_from_gguf`).
The draft model file is `LLM_ARCH_DFLASH` (`src/llama-arch.cpp:147`).

**NInfer applicability**: NInfer's DFlash2 already implements the analogous
"top-16 conditional path, rank 256" selector (docs/maintainer/dflash.md) — DSpark is a
different *parameterization* of the same idea (low-rank Markov over the draft vocab +
confidence gating + anchor-first layout). Supporting it means a new draft-backend
geometry (new tensors, new proposal math) — an explicit product/architecture decision,
not a perf technique.

**Verdict: flag** (no verdict in this map — draft-family expansion, axis a of a different
family). See "Flagged for the map".

## C11 — EAGLE3 draft backend

**llama.cpp state**: `common_speculative_impl_draft_eagle3`
(`common/speculative.cpp:514-982`): a separate draft *model* whose decoder layers each
consume (token, g_embd) pairs — target features extracted from `target_layer_ids` layers
of the target (`llama_set_embeddings_layer_inp`), three stacked draft layers per the
EAGLE-3 design; same verify/accept loop, top-k=10 draft sampler, optional backend
sampling offload.

**NInfer applicability**: a new draft model architecture (EAGLE-3 draft weights are a
different mathematical family from MTP/DFlash/DFlash2). NInfer's Program backends are the
fixed set MTP/DFlash/DFlash2 (`docs/maintainer/engine-architecture.md` §1).

**Verdict: flag** — new draft architecture requires an explicit product change; outside
this ticket's perf scope.

## C12 — Multi-device draft placement / expert parallelism / tensor split

**llama.cpp state**: the draft context takes its own `devices` list
(`common_params_speculative_draft`, `common/common.h:328`; used as `params.devices` at
`src/speculative.cpp:246/2537`); MoE/weight placement uses `--tensor-split` /
`--n-cpu-moe` (multi-GPU split, CPU expert offload).

**NInfer applicability**: NInfer's product is one GPU, one resident model
(`AGENTS.md`). Any multi-device variant crosses the stated boundary.

**Verdict: flag only** (boundary-crossing; no verdict). See "Flagged for the map".

## C13 — GDN recurrent-state snapshot fusion (rollback slots)

**llama.cpp state**: `ggml/src/ggml-cuda/gated_delta_net.cu/.cuh` + graph-level fusion
`ggml_cuda_try_gdn_cache_fusion` (`ggml-cuda.cu:2809`): the GDN kernel writes K
recurrent-state snapshots (rollback slots, newest = slot 0) directly into the cache,
skipping the trailing `CPY` node — the speculative-decoding support for GDN state.

**NInfer applicability**: NInfer solves the same problem differently and documents why:
ReplaySSM records the raw state-transition inputs during verify and replays the accepted
prefix with a bit-exact finite-precision fold (`docs/maintainer/replayssm-gdn.md`,
`include/ninfer/ops/gdn_replay.h`). llama.cpp's K-snapshot approach costs K × full state
images (K × 144 MiB for the 27B geometry) vs NInfer's small record streams. The only
residual is a possible micro-fusion (state write fused into NInfer's GDN kernel epilogue) —
small, Op-internal, and NInfer's design already avoids the copy it targets.

**Verdict: defer** (axis b, low expected delta). **Evidence if pursued**: GDN op
microbenchmark (snapshot write vs fused write) at the registered geometry; oracle
qualification unchanged.

## C14 — CUDA graph in-place update + per-graph buffer pool

**llama.cpp state**: `ggml/src/ggml-cuda/common.cuh:1258-1293` (`ggml_cuda_graph`) and
`ggml-cuda.cu:4515-4600` (`ggml_backend_cuda_graph_compute`): capture on demand per graph
key (first tensor identity), 2-call warmup protocol, **in-place** kernel-node parameter
updates when node properties change (`cudaGraphExecKernelNodeSetParams` via
`ggml_cuda_graph_update_required`), fallback to direct execution while properties churn;
`ggml_cuda_pool` bump-allocator keeps tensor addresses stable per graph.

**NInfer applicability**: NInfer captures decode graphs once at program construction per
exact-`B` topology with stable device addresses, and treats request identity / page IDs
as stable *input data* rather than graph keys (`docs/maintainer/engine-architecture.md`
§8; device state a captured kernel reads must sit at a stable address). The
in-place-update machinery exists in llama.cpp because its pool re-lays-out tensors per
shape; NInfer's fixed-layout capture makes it unnecessary. No launch-overhead gap
measurable: NInfer already replays captured graphs.

**Verdict: no** — equivalent-or-better by design; porting the update path would add a
second graph-management mode without a measurable gain.

## C15 — Memory layout / allocator

**llama.cpp state**: CUDA backend allocates per-buffer `cudaMallocManaged` with
`cudaMalloc` fallback (`ggml-cuda.cu:145-166`); no stream-ordered allocator
(`cudaMallocAsync`/memory pools) anywhere in `ggml/src/ggml-cuda/`; KV cache is
contiguous per layer (no paging), graph stability via the C14 pool.

**NInfer applicability**: NInfer uses Program-planned fixed arenas + paged KV pools
(`src/core` arenas, `docs/maintainer/paged-kv-cache.md`). Nothing in llama.cpp's
allocator approach is a performance technique.

**Verdict: no**.

## C16 — MoE graph fusions (topk-router, GLU, shared expert, spec-path)

**llama.cpp state**: graph-level pattern fusions in `ggml-cuda.cu`:
`ggml_cuda_topk_moe_fusion` (sigmoid/softmax/sqrt-softplus → reshape → [bias] → argsort →
MUL_MAT_ID collapsed into `topk-moe.cu` + `moe-weighted-reduction.cu`), merged #27621
"extend MOE fusion to specdec … earlier MOE glu fusion and topk-router fusion were
restricted to 1 token" (2026-08), merged #29184 "CUDA: fuse shared experts into MMVQ"
(2026-10): Qwen3.5-35B-A3B-Q8_0 on **RTX Pro 6000**, tg +0.42..+5.06% at
microbatch 1..8 (e.g. 1,017.8 → 1,031.97 t/s at 7).

**NInfer applicability**: NInfer's `sparse_moe` is a single closed Op that already owns
"router projection and selection, selected routed and shared SwiGLU projections, down
projections, their merge, and the AddResidual epilogue" (`include/ninfer/ops/sparse_moe.h`) — the
llama.cpp fusion family is the same idea done at graph level because ggml has no closed
MoE op.

**Verdict: no** — already covered by NInfer's closed-op design; the +0.4–5.1% numbers are
against a baseline that lacked the fusion.

## C17 — Decode batching internals (ubatch splitting)

**llama.cpp state**: `llama_batch_allocr::split_{simple,equal,seq}`
(`src/llama-batch.cpp:518-747`) bound memory by grouping tokens into ubatches of
`n_ubatch` (default = `n_batch`); the top-of-master commit #29903 (the verified commit)
is a *server robustness* fix for this interaction (laya model abort when a batch's
question count exceeds `n_ubatch`) — not a performance technique.

**NInfer applicability**: NInfer's compact decode round (all decode-ready requests, exact
`B`, no padding) and staged 1,024-token prefill chunks are the product execution model
(`docs/maintainer/engine-architecture.md` §5).

**Verdict: no** — ubatch splitting is a memory-management device for llama.cpp's
contiguous KV; NInfer's paged KV + compact round is the superset at the product level.

## C18 — Programmatic dependent launch (PDL)

**llama.cpp state**: `ggml_cuda_pdl_sync()` prologues across the kernel surface
(e.g. `fattn.cu:15`). **NInfer state**: `src/core/pdl.cuh` sets
`cudaLaunchAttributeProgrammaticStreamSerialization` — NInfer already uses PDL.

**Verdict: no** — no gap.

---

## Summary table

| # | Candidate | Verdict | Axis | One-line rationale |
|---|---|---|---|---|
| C1 | NVFP4 MoE experts + Blackwell W4A4 GEMM (35B-A3B) | **port** | a+b | Open artifact codec gap; ~4× dense-NVFP4 prefill precedent on this GPU; decode gain unproven, must be measured |
| C2 | NVFP4 GEMM TMA/cp.async pipelining (dense path) | defer | b | NInfer's dense NVFP4 prefill already leads llama.cpp's; headroom unproven — measure first |
| C3 | Attention tuning family: GQA-packed MMA tile, stream-K, Blackwell stage tuning | **port** | b | Only remaining decode-path lever at long context (74→53 tok/s ctx growth); techniques, not code, transfer |
| C4 | Quantized KV attention (vec family, quantized K/V) | no | — | NInfer reads FP8/NVFP4/K8V4 rows directly in all routes; superset already shipped |
| C5 | Attention sinks | no | — | Not in Qwen3.5 math; would change the model, not perf |
| C6 | SWA/ISWA/DSA/MSA KV variants | no (DSA/MSA: flag) | — | Different model families; DSA/MSA = new mathematical architecture |
| C7 | Context eviction / cache reuse | no (flag QoL) | c | Product-level KV policy, out of perf family |
| C8 | ngram spec family | defer | c | No draft weights needed, but acceptance ceiling far below shipped MTP/DFlash backends |
| C9 | Rejection-sampling acceptance | no | — | Verified equivalent: NInfer's accept ops are the identical rejection/residual math |
| C10 | DSpark Markov head + confidence gating | flag | a (draft family) | New draft parameterization of NInfer's selector idea; product decision |
| C11 | EAGLE3 draft backend | flag | a | New draft model architecture; explicit product change required |
| C12 | Multi-device draft / expert parallelism / tensor split | flag | — | Boundary-crossing: one-GPU product |
| C13 | GDN snapshot fusion (rollback slots) | defer | b | ReplaySSM already avoids the copy; micro-fusion, small expected delta |
| C14 | CUDA graph in-place update + pool | no | — | NInfer's fixed-layout capture is equivalent-or-better by design |
| C15 | Allocator / memory layout | no | — | Per-buffer cudaMalloc; nothing to port |
| C16 | MoE graph fusions (router/GLU/shared/spec-path) | no | — | NInfer's closed `sparse_moe` already fuses all stages |
| C17 | ubatch splitting / decode batching | no | — | Memory device for contiguous KV; NInfer compact round is the product model |
| C18 | PDL | no | — | NInfer already uses PDL (`src/core/pdl.cuh`) |

## Flagged for the map

Boundary-crossing (multi-GPU / new architecture — no verdict, out of this map's scope):

1. **Multi-device speculative draft** (C12): llama.cpp's draft context takes an
   independent `devices` list — draft model on a second GPU. Crosses NInfer's
   one-GPU/one-resident-model boundary.
2. **Expert parallelism / tensor split / CPU expert offload** (C12): `--tensor-split`,
   `--n-cpu-moe` — multi-device and CPU-offload placement.
3. **Sparse-attention model family** (C6): DSA (lightning indexer + top-k KV selection),
   MSA, dsv4 KV caches and their kernels — a new mathematical architecture
   (DeepSeek/MiniMax-class), which per `AGENTS.md` requires an explicit product change.
4. **Draft-family expansion** (C10/C11): DSpark (Markov head, confidence gating,
   anchor-first block) and EAGLE3 (3-hidden-state draft model). Both are new draft
   backends beyond the fixed MTP/DFlash/DFlash2 set; if official Qwen artifacts for
   either land, they become a product decision with an artifact-contract (axis-a)
   component.
4a. **Continuous batching / preemptive batching** (ticket scope): llama.cpp does not
   implement preemptive continuous batching either — its slot model (bounded per-slot
   concurrency with ubatch splitting inside a request, C17) is the closest analogue to
   NInfer's bounded-FIFO compact rounds; there is nothing to port. Recorded to close the
   ticket's boundary-flag scope.
4b. **MoE load balancing** (ticket scope): meaningful only under expert parallelism
   (flag 2, boundary-crossing); N/A for NInfer's single-GPU closed `sparse_moe` op, where
   expert selection is a deterministic top-k inside the Op.

Adjacent / not-performance observations the coordinator may want to route:

5. **Speculative prefill** (upstream open PR #27692 "Speculative prefill"): TTFT-oriented
   (draft during prompt processing) — a different axis (TTFT, not decode) and still open
   upstream; NInfer's prefill (12.8K tok/s nvfp4) is not the reported bottleneck.
6. **Upstream in-flight Blackwell work** to watch, not port, until merged: #26159
   (NVFP4 MoE work scheduling +10–15% pp), #29857 (NVFP4 MMQ accumulation), #26704
   (CUTLASS SM120 MoE prefill), #28702 (GLU→MMQ fusion), #24565 (fattn Blackwell MMA
   config), #26001 (GDN chunked prefill). If C1/C3 graduate, these are the upstream
   references to re-check at their merge state.
7. **Server robustness (non-perf)**: top-of-master #29903 ("fix laya abort by limiting
   n_batch to n_ubatch") is a batch-sizing bug fix in the server path; no NInfer
   analogue identified (NInfer's prefill chunking is fixed at 1,024 tokens).
8. **ngram-cache static/dynamic files** (C8): on-disk draft artifacts — a CPU/tooling
   concern, not a GPU performance item.

## Claims not verifiable this session (disclosed)

- #28572's +14% pp16384 is a **closed-unmerged** PR-build number (its table: master
  baseline `dbeb37548` 6,146.5 → PR build 7,009.7 tok/s, RTX 5090). The pipelined NVFP4
  MMQ kernel is not in master at the verified commit, so no merged-master prefill number
  exists to compare against; C1's port verdict rests on NInfer's own verified 12,819 vs
  3,332 tok/s nvfp4-vs-int prefill gap instead.
- #23572's "+40% Blackwell PP" is an **open, unmerged** PR claim (no hardware in the claim;
  body discusses Blackwell generally) — excluded from any verdict basis.
- #27444's "same model on vLLM and NInfer looks ok" is a reporter assertion inside the
  issue (used only as a directional signal about NInfer's long-ctx decode; NInfer's own
  published numbers were read directly instead).
- llama.cpp's decode attention has no merged Blackwell-specific benchmark number at the
  verified commit; C3's headroom argument rests on NInfer's published ctx-growth decode
  drop plus the mechanism analysis, and the follow-on ticket must measure it.
