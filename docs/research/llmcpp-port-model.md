# llama.cpp model & artifact capability candidates (quants, GGUF, adapters, tasks)

Wayfinder map #167 ("Wayfinder map: llama.cpp port candidates"), ticket #168.
Sibling ticket #173 (closed) covers the performance candidates:
`docs/research/llmcpp-port-perf.md` @ `research/llmcpp-perf` 0cafbf67, verdict
**C1: NVFP4 MoE experts + Blackwell W4A4 GEMM for 35B-A3B = port**. C1 is not
re-derived here; the one upstream NVFP4 GEMM fact that intersects it is cited
in §C-C for the record.

## Verified references

- llama.cpp master: **1537a0a8b2f8711d840878b0a0677ab2213c882c**
  (2026-10-03T15:19:13Z), re-verified this session against the GitHub commits
  API (tip of `ggml-org/llama.cpp` `master`). All llama.cpp citations are
  `path:line` in that commit, read from the full tree
  `/tmp/opencode/llama-1537a0a8` (tarball of the same commit; the blobless
  clone `/tmp/opencode/llamacpp-serve-ref` has the identical tip).
- NInfer master: **3e0ff840b9f3f3efc6ee82305b540198f6e6bd81**
  (2026-10-03 17:51:47 +0300). NInfer citations are `path:line` at that ref.
- Method: code-level reads of the primary sources above. No web search was
  needed to settle any llama.cpp claim (the live commit equals the local
  tree); no GPU work, builds, or conversions were performed.

Scope: model/artifact capabilities only — quantization formats, GGUF
loading/conversion, LoRA/adapters, and the embeddings/pooling/rerank task
surface. Verdict axes (user-fixed): (a) exposed contract/protocol gap >
(b) measurable 5090 performance headroom > (c) QoL/ecosystem parity.

## NInfer state today (verified)

- **Artifact formats.** Exactly nine registered persistent numeric formats
  (`docs/maintainer/tensor-formats.md` §1): `bf16`, `fp32`, `int32`,
  `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` (signed codes +
  one FP16 scale per 32/64-element group), `nvfp4` (E2M1 4-bit codes, K16
  groups, one E4M3FN block scale/group, one positive FP32 weight divisor),
  `fp8_e4m3fn_row_bf16`. "Additional formats need an explicit numeric
  definition and codec implementation."
- **Converter sources** (`tools/convert/sources/`,
  `docs/weight-conversion.md` "Read another source"): HF-layout **Safetensors**
  (model dir or indexed shards; `safetensors.py`), the built-in
  **compressed-tensors** reader for per-row FP8 and NVFP4 code/scale
  conventions (`compressed_tensors.py`), and a custom **`LogicalSource`**
  extension point (`logical.py`: flat C-order `read_values`, optional
  `read_encoded` + `weight_divisor`/`input_divisor` for encoded import). No
  GGUF reader exists.
- **NVFP4 today.** Imported via `import_encoded` from compressed-tensors
  checkpoints; no built-in FP→NVFP4 quantizer
  (`docs/weight-conversion.md` formats table,
  `docs/maintainer/tensor-formats.md` §2.4). Runtime NVFP4 GEMMs are A16
  (`src/ops/linear/nvfp4/nvfp4_a16_simt.cuh` plus shape-specialized CUDA
  kernels in `src/ops/linear/nvfp4/shapes/`). W4A4 (A4 activations) is C1's
  subject.
- **Engine/serve surface.** `EnginePurpose` = {Generation, CausalScoring}
  (`include/ninfer/types.h:38-41`); `Engine` exposes
  prepare/submit/generate/`score_tokens` (`include/ninfer/engine.h`). Serve
  routes: `/health`, `/v1/models`, `/v1/chat/completions`,
  `/v1/responses` (+`input_tokens`, `compact`), `/v1/messages/count_tokens`,
  `/v1/messages` (`src/serve/http_server.cpp:428-475`). No embeddings, rerank,
  or adapter routes; no adapter concept anywhere in the public types.
- **Not an embedding task.** `src/ops/wrapper/embedding.cpp:197-238` is the
  token-embedding table gather op (BF16 / Q6_G64 / Q8_G32 / FP8 tables); it is
  part of the causal-LM forward path, not an embeddings output.
- **Boundary rule** (repo `AGENTS.md`): a new mathematical architecture or
  execution platform is an explicit product change; a quantization format or
  GGUF-source support is artifact/converter/codec work, not a product change.

## C-A. GGUF as a converter source

### llama.cpp state (1537a0a8b2)

- GGUF v3 container: magic `"GGUF"`, version, tensor count, KV count, typed KV
  metadata, per-tensor (name, dims, `ggml_type`, data offset), then the tensor
  data blob (`ggml/include/gguf.h:1-46`; `GGUF_VERSION 3` at `:42`). Bounded
  reads suffice for ingestion (the C API even takes a reader callback,
  `gguf.h:79-80`).
- Tensor content = any active `ggml_type` (registry `ggml/include/ggml.h:390-432`);
  26 public row-dequantize references define the exact codec semantics
  (`ggml/src/ggml-quants.h:46-76`).
- Community GGUFs for NInfer's target family exist: `LLM_ARCH_QWEN35` /
  `LLM_ARCH_QWEN35MOE` ("qwen35"/"qwen35moe", `src/llama-arch.cpp:41-42`,
  model classes at `src/llama-model.cpp:326-328`); GGUF tensors are already
  split per projection (attn q/k/v/o separate), matching NInfer's logical
  projection names better than fused HF tensors.
- Base conversion (`convert_hf_to_gguf.py:60`) writes `f32/f16/bf16/q8_0/tq1_0/tq2_0/auto`;
  fine-grained ftypes (Q2_K…Q6_K, IQ*, TQ, MXFP4, NVFP4 — `include/llama.h:117-162`)
  come from the C++ `llama-quantize` tool.
- **NVFP4 GGUFs are repacks of already-NVFP4 checkpoints**, not requants:
  `conversion/base.py:849` `_repack_nvfp4` writes `GGML_TYPE_NVFP4` blocks plus
  sidecar tensors `<name>.weight_scale_2` (global scale) and
  `<name>.input_scale` (`base.py:858,873-874,892-893`); the Qwen path transforms
  weights/scales at `conversion/qwen.py:494,581-583`.
- KV metadata also carries tokenizer data (`LLM_KV_TOKENIZER_MODEL`,
  `src/llama-arch.h:366`), chat template (`:390`), and rope/YaRN scalars — not
  an HF `tokenizer.json`/config.

### NInfer state

No GGUF source reader (verified above). The `LogicalSource` extension point is
designed exactly for "another file format"
(`docs/weight-conversion.md`): a reader supplies logical values, and for
encoded import supplies `read_encoded` + the format's divisor accessors.

### What a port means for NInfer's boundaries

Pure converter work — a new `tools/convert/sources/gguf.py`
(`LogicalSource` subclass: parse header/KV/tensor-info, bounded data-blob
reads, dequantize to logical values with the `ggml-quants.c` reference
semantics as the exact oracle, or pass encoded rows straight through for
NVFP4) plus a recipe-side GGUF-name → NInfer-logical-name mapping table
(`tools/convert/qwen3_5.py` owns the mapping machinery). No new persistent
format, no new Op, no artifact change: the converter already requantizes
logical values into the nine registered formats. Tokenizer/chat-template
resources must be supplied separately (GGUF does not ship HF tokenizer JSON) —
`--resource` overrides already cover that. **Not a product change.**

### Port shape + cost

- Minimum viable: BF16/F16/F32 + `q8_0` + `q4_K`/`q5_K`/`q6_K` + NVFP4
  (byte-identical block layout: 4 UE4M3 sub-block scales + 32 packed E2M1
  bytes per 64 weights, `ggml/src/ggml-common.h:221-227` — the same words as
  NInfer's `nvfp4` format, so encoded import is a reframe of words plus the
  `.weight_scale_2` → weight-divisor and `.input_scale` → activation-divisor
  mapping) + F32 scale sidecars. That is every type a Qwen-family GGUF
  plausibly contains.
- Effort: low–medium — a few days for reader + codec + golden-tensor
  qualification against the ggml reference dequant; the full 26-type surface
  (IQ scale arrays, TQ) adds ~a week and is not needed for the Qwen family.

### Verdict

**Port** — axis (c): GGUF is the de facto distribution format for the
community's quantized Qwen artifacts (incl. NVFP4 MoE-expert files whose
sidecars map 1:1 onto NInfer's `import_encoded`); today NInfer cannot ingest
them at all, and this candidate is also the vehicle that makes every C-B
quant format consumable at conversion time without a new persistent format.

## C-B. Additional quantization formats as NInfer persistent formats

### llama.cpp state (1537a0a8b2)

Registry `ggml/include/ggml.h:390-432`; families NInfer does not cover:

| Family | Types | Codec notes |
|---|---|---|
| Legacy blocks | `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1`, `Q1_0/Q2_0` | plain 32-block, FP32 scale (`Q8_0` = int8 codes) |
| K-quants | `Q2_K`…`Q8_K` | 256-elem super-block, shared FP16/FP32 scales, d_min; Q3_K/Q5_K/Q6_K carry sub-block fields (Q4_K/Q6_K-derived); each has a dequant reference (`ggml-quants.h:58-63`) |
| IQ-quants | `IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS` | 2.06–4.25 bits/weight; scale arrays stored outside blocks (`qscales`/`scales`), LUT-based codes with permute masks |
| Tersey / MX | `TQ1_0/TQ2_0` (256-block, 54/67 B), `MXFP4` (32-block, E8M0 scale) | `gguf-py/gguf/constants.py:6105-6108` |

CUDA support is broad (mmvq/mmq template instances per type; NVFP4/MXFP4
Blackwell paths at `ggml/src/ggml-cuda/mmq.cu:128`), i.e. upstream runs these
on-device.

### NInfer state

Nine formats (verified above). The grouped-int family (g64/g32, FP16 scales)
covers 4/5/6/8 bits; `q8_g32_fp16` is *not* Q8_0 (FP16 scale per 32 vs FP32
per 32); no K-quants, IQ, TQ, MXFP4.

### What a port means for NInfer's boundaries

Each new format is artifact/codec work per `docs/maintainer/tensor-formats.md`
§1 and AGENTS.md ownership: explicit numeric definition + format registry
entry (`tools/artifact/formats.py`, `src/artifact/formats.cpp`) + layout
(`src/artifact/layouts.*`) + CUDA dequant/GEMM Ops + qualification against an
independent oracle. Not a new mathematical architecture — **not a product
change** — but it is the most expensive candidate class here (kernel
families on sm_120a, per-format).

Note the alternative route: with C-A, any of these formats is consumable
*as a GGUF source* (dequantize at conversion, store as an existing NInfer
format). A new persistent format is only needed to *store* the ecosystem
codec natively in a `.ninfer`.

### Port shape + cost (per family)

- **Q8_0**: cheapest — 32-block FP32-scale int8; codec + one dequant/GEMM
  variant near the existing `q8_g32` family; low (≤2 days + qualification).
- **Q*_K**: one new kernel family per bit-width (Q4_K/Q5_K/Q6_K) + super-block
  codec; medium each (~3–5 days), high for all three.
- **IQ1–IQ4**: scale-array + LUT code families, nine variants; highest codec
  complexity; high per format (week-scale for the family).
- **TQ/MXFP4/Q1_0/Q2_0**: exotic; same order as K-quants for the kernels,
  lower priority.

### Verdict

- **IQ-quants (i1–i4): no** — no (a) gap; no (b) case on a 32 GB 5090 where
  27B dense runs in NVFP4 and MoE in groupwise-int (the 2.06–4.25-bit quality
  tradeoff is a CPU-VRAM-scarcity device); the highest-cost family; the (c)
  ingest value is fully served by C-A.
- **Q*_K family: defer** — (b) is weak on sm_120a today (K-quants are the
  community's CPU-side default, not a 5090 headroom play); (c) value is
  served by C-A; revisit only if a low-bit-dense product decision lands.
- **Q8_0: defer** — same axis (c) as Q*_K but a low-cost codec/kel; worth a
  one-liner revisit if high-precision GGUF artifacts (Q8_0 is the common
  "near-lossless" community tier) become a source the user actually carries.

## C-C. NVFP4 upstream status (note, not a candidate)

NInfer already registers the same scheme (`nvfp4`: E2M1, K16, E4M3FN/group,
FP32 weight divisor) as upstream's `GGML_TYPE_NVFP4`
(`ggml/include/ggml.h:430`; block `ggml/src/ggml-common.h:221-227`; ftype
`MOSTLY_NVFP4 = 39`, `include/llama.h:157`). Upstream additions that matter:

- **CUDA GEMMs for NVFP4**: A16 (`mmvq.cu:81-86`) and a **W4A4 Blackwell
  instance** (`ggml/src/ggml-cuda/mmq-instance-nvfp4.cu:3-5`
  `DECL_MMQ_CASE_W4A4(GGML_TYPE_NVFP4)`, gated on `blackwell_mma_available`,
  `mmq.cu:128`) — this is the upstream reference for C1 (perf ticket #173,
  already verdicted "port"); recorded here only for the map's intersection.
- **NVFP4 in GGUF**: repack-from-checkpoint flow (§C-A) with
  `.weight_scale_2`/`.input_scale` sidecars — a second wire convention for
  the same words NInfer's compressed-tensors reader already imports.

**Verdict: no (already covered)** — NInfer's `nvfp4` format needs no change;
the open delta (W4A4) lives in C1, and the source delta lives in C-A.

## C-D. LoRA / adapters

### llama.cpp state (1537a0a8b2)

- **Format**: a *separate GGUF file* with `general.type == "adapter"`,
  `adapter.type == "lora"`, `adapter.lora.alpha` metadata, optional
  ALiBi invocation-token array; tensors named `<base tensor>.lora_a` /
  `<base tensor>.lora_b` (`src/llama-adapter.cpp:200-246,269-295`). Adapter
  A/B tensors may themselves be stored quantized — the converter offers
  `--outtype f32/f16/bf16/q8_0/auto` (`convert_lora_to_gguf.py:287`) and the
  loader applies no type filter.
- **API**: `llama_adapter_lora_init` loads against a base model
  (`include/llama.h:688`); `llama_set_adapters_lora` applies a set of
  adapters with per-adapter scales per request (`include/llama.h:726-729`);
  cvec context adapters exist too (`include/llama.h:738`).
- **Math**: standard low-rank delta on every adapted linear and on the token
  embedding — `res += scale · (B·(A·x))`, `scale = (lora_scale·alpha)/rank`
  (`src/llama-graph.cpp:1514-1598`, embedding at `:2409-2415`); adapter
  weights are uploaded per backend (`src/llama-adapter.cpp:247-300`).
- **Serve**: `--lora`/`--lora-scaled` startup flags (`tools/server/README.md:94-95`),
  a per-request `lora: [{id, scale}]` field (`README.md:603`), and
  `GET/POST /lora-adapters` hot-swap (`tools/server/server.cpp:292-294`,
  `--lora-init-without-apply` `README.md:243`).

### NInfer state

Nothing: no adapter in the public types, engine, artifact, or serve surface
(verified above). Product shape: one resident model, startup-fixed.

### What a port means for NInfer's boundaries

The *math* is a fixed function over the existing Qwen3.5 architecture
(rank-r delta on linear projections) — not a new mathematical architecture.
The *capability* needs: (1) artifact component for adapter A/B matrices
(+alpha) — new optional component framing/binding; (2) a rank-delta Op
(fused `B·(A·x)` GEMV/GEMM + scale, or two mul_mat + add) wired into the
model's projections; (3) a product decision on selection semantics:
**startup-resident adapter** fits the existing contract (one resident model,
fixed at load) and is pure artifact+Op work; **per-request/per-client
adapter switching** would add a new exposed serve contract (no current client
— opencode included — expects it) and is a protocol decision.

### Port shape + cost

Medium–high: adapter component (framing + loader + validation) + delta Op
with independent-oracle qualification + e2e on a real Qwen LoRA; plus the
selection-semantics decision. A week-scale effort for the startup-resident
variant.

### Verdict

**Defer** — axis (c) only: ecosystem parity (community Qwen LoRAs exist) with
no (a) client contract gap and no (b) headroom; revisit on demonstrated
demand, with startup-resident selection as the boundary-safe shape.

## C-E. Embeddings / pooling / rerank (boundary flag, no verdict)

### llama.cpp state (1537a0a8b2)

- **Pooling types** `NONE/MEAN/CLS/LAST/RANK` (`include/llama.h:176-183`;
  RANK "used by reranking models to attach the classification head to the
  graph"), defaulted from GGUF KVs `pooling_type` /
  `classifier_pooling_type` (`src/llama-model.cpp:1327-1328`) or the
  `--pooling` flag (`tools/server/README.md:175`).
- **Computation is graph-level**: the model graph emits a pooled
  `t_embd` — MEAN = masked row-mean, CLS/LAST = `get_rows`, RANK =
  mean/cls then a classification head (`cls`/`cls_b`, gelu/tanh activation,
  optional `cls_norm`, optional direct `cls_out` projection) producing
  `n_cls_out` scores per sequence (`src/llama-graph.cpp:3752-3805`); the
  context extracts per-token or per-sequence results
  (`src/llama-context.cpp:1957-2010`). APIs: `llama_get_embeddings` /
  `_ith` / `_seq` (`include/llama.h:1151-1164`).
- **Serve**: `POST /v1/embeddings` (OpenAI-compatible; marker/`[q|k]`
  dot-product extensions, `tools/server/server-task.h:182-197`) and
  `POST /rerank` + `/v1/rerank` + `/reranking` + `/v1/reranking`
  (`tools/server/server.cpp:277-281`), prompt format
  `[BOS]query[EOS][SEP]doc[EOS]` (`tools/server/server-common.h:569-570`),
  gated by `--embedding` / `--rerank` startup flags
  (`tools/server/README.md:210-211`). Real embedding/rerank models are a
  different architecture family (BERT-like, or encoder-LM variants with
  classifier heads) than Qwen3.5 causal LM.

### NInfer state

`EnginePurpose` is {Generation, CausalScoring} only; serve has no
embeddings/rerank routes (verified above). NInfer implements the Qwen3.5
dense/MoE causal-LM mathematics.

### Boundary flag

Adding embeddings/pooling/rerank is a **new task surface** (a different
output contract than next-token generation/scoring) and, for real
embedding/rerank models, a **new mathematical architecture** — per AGENTS.md
that is an explicit product change, not artifact/converter work. Recorded to
the extent needed to justify the flag; **no verdict** (routed to the map's
Out of scope).

## Summary table

| Candidate | Verdict / flag | Axis | One-line rationale |
|---|---|---|---|
| C-A GGUF converter source | **port** | (c) | Only way to ingest the community's quantized Qwen GGUFs (NVFP4 sidecars map 1:1 to `import_encoded`); pure converter work, low–medium cost |
| C-B IQ1–IQ4 as persistent formats | **no** | (b)/(c) fail | Highest-cost codec family for a 5090 whose models already fit in NVFP4; ingest value served by C-A |
| C-B Q*_K as persistent formats | **defer** | (c) via C-A | K-quant GEMM kernels are a CPU-side story; revisit only with a low-bit-dense product decision |
| C-B Q8_0 as persistent format | **defer** | (c) | Cheapest of the family; revisit if high-precision GGUF artifacts become a real source |
| C-C NVFP4 (upstream state) | **no (covered)** | — | NInfer's `nvfp4` is the same scheme; W4A4 delta = C1 (perf map), GGUF delta = C-A |
| C-D LoRA/adapters | **defer** | (c) | Ecosystem parity with no client contract gap or perf headroom; needs a selection-semantics product decision (startup-resident = boundary-safe) |
| C-E embeddings/pooling/rerank | **flag** (boundary-crossing) | — | New task + new architecture family = explicit product change per AGENTS.md; no verdict |

## Flagged for the map

1. **NVFP4 W4A4 GEMM (upstream reference)** — `mmq-instance-nvfp4.cu`
   (C-C): intersects C1 from the perf ticket (already verdicted "port"); keep
   the upstream instance as the reference implementation when C1 lands.
2. **GGUF export** (writing `.ninfer` → GGUF, the inverse of C-A): a new
   artifact-direction product question outside this ticket's model/artifact
   ingest scope — route to the map's Out of scope / product-change lane.
3. **MXFP4 / TQ1_0 / TQ2_0 / Q1_0 / Q2_0** (ggml types 34/35/39/41/42):
   same family as C-B, same verdict disposition (no/defer); listed so the
   map can see the full registry diff, not just the IQ/K/Q8_0 rows.
4. **Per-request adapter switching + `/lora-adapters` hot-swap** (llama.cpp
   serve contract): if C-D is ever ported, the upstream per-request
   `lora: [{id, scale}]` field is a *new* exposed protocol NInfer clients do
   not expect; the boundary-safe port is startup-resident selection only.
5. **cvec context adapters** (`llama_set_adapter_cvec`,
   `include/llama.h:738`): smaller sibling of C-D, no Qwen-ecosystem demand
   observed; subsumed by the C-D disposition.
6. **ALiBi invocation tokens** inside LoRA files
   (`src/llama-adapter.cpp:227-246`): sub-feature of C-D, noted so a future
   C-D design does not miss it.
