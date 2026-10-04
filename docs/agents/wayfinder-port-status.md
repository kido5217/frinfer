# Wayfinder port status

Status of [map #175 — llama.cpp port implementation](https://github.com/kido5217/frinfer/issues/175) as of 2026-10-05.
Temporary working note: remove this file when the map closes.

## Destination

All 20 tickets closed: the 15 port items landed (one PR per ticket through the pipeline) plus the three
evidence gates resolved. The NVFP4 MoE item was split into three ordered tickets (converter + codec +
binding → W4A4 GEMM → e2e A/B + perplexity); the original ticket is closed as superseded.

## Landed

| Ticket | Outcome | PR |
|---|---|---|
| Evidence: logprobs spec-shape oracle + Engine gather design | port | research branch `research/logprobs-evidence` @ `36386a6e` |
| Evidence: NVFP4 MoE W4A4 kernel A/B + Op oracle + e2e plan | port | research branch `research/nvfp4-moe-evidence` @ `cf3bef1c` |
| Evidence: attention tuning headroom | **NO-PORT** | research branch `research/attention-tuning-evidence` @ `08171824` |
| Implement: attention tuning | closed no-port (per the gate) | — |
| Implement: CLI constrained decoding (`--grammar` / `--json-schema`) | landed | #200 → `3ddf354f` |
| Implement: Anthropic `output_config.format` | landed | #202 → `b89fc98c` |
| Implement: logprobs / `top_logprobs` | landed | #205 → `dc40c531` |
| Implement: NVFP4 35B-A3B converter + expert codec + binding | landed | #209 → `cc2c0d1c` |
| Implement: NVFP4 MoE experts + W4A4 GEMM (original) | closed superseded, split into three | — |

Non-map fixes landed this session: #203 → `854ef166` (`cuda_profiler_api` in the devShell, restored the
bench build) and #204 → `43419e0d` (`just` task runner + `justfile`).

## Open frontier

Wave 3–4 tickets, all open and unblocked except as noted: jinja for-loop scope fix, GGUF converter
source, model-acquisition flags, `reasoning_end` control endpoint, Prometheus `/metrics`, `--version`,
CLI `--prompt-file`/stdin, CLI `--show-prompt`, tokenizer CLI, artifact integrity hashing.
The W4A4 `sparse_moe` GEMM (#207) is unblocked; the 35B-A3B NVFP4 e2e A/B + perplexity (#208) is
blocked by it.

## Resources on this machine

- `models/qwen3_6_27b.ninfer` — 17.5 GB, sha256-verified; ordinary/MTP serve and real tests.
- `models/qwen3_8_27b_nvfp4.ninfer` — 23.7 GB; carries the `dflash2` component the 27B lacks.
- `out/qwen3_6_35b_a3b_nvfp4.ninfer` — 20,647,342,340 bytes; the produced NVFP4 artifact (30,720 routed
  nvfp4 parents). Loads; execution awaits #207.
- Local source: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` (compressed-tensors `nvfp4-pack-quantized`) in the
  HF cache; the non-`Fast` variant has FP8 experts in layers 32–39, so `Fast` is required.

## Next

1. #207 — the W4A4 `sparse_moe` GEMM. The gate measured at the fused `N=1024` gate/up shape; the landed
   representation binds two `N=512` GEMMs, so re-measure at `N=512` before reusing the gate's number.
2. #208 — e2e A/B against the published `groupwise-int` baseline + perplexity.

## Backlog

Open non-map bugs: #199 (silent stall — a served turn ends with no tool call and no demotion) and #201
(closed). Open backlog is #199 only.
