# Wayfinder port status

Status of [map #175 — llama.cpp port implementation](https://github.com/kido5217/frinfer/issues/175) as of 2026-10-06.
Temporary working note: remove this file when the map closes.

## Destination

All 20 tickets closed: the 15 port items landed (one PR per ticket through the pipeline) plus the three
evidence gates resolved. The NVFP4 MoE item was split twice: converter + codec + binding (#206) →
routed prefill gate/up (#211) → routed prefill down + merge (#212), with the `T < 20` routed path
(#213) landed in the same chain; the e2e A/B + perplexity check (#208) is now unblocked. The original
ticket (#182) is closed as superseded.

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
| Implement: NVFP4 routed gate/up W4A4 prefill stage | landed | #218 → `612c25a8` |
| Implement: NVFP4 grouped W4A4 down + prefill routed merge | landed | #219 → `9d91068b` |
| Implement: NVFP4 routed expert path for `T<20` (decode + small-T) | landed | #221 → `6501fa3e` |
| Implement: jinja for-loop scope fix (llama-jinja fork) | landed | #214 → `26e4688d` |
| Implement: `--version` on all four binaries | landed | #215 → `36f4f558` |
| Implement: artifact integrity hashing | landed | #216 → `1ecd85aa` |
| Implement: CLI `--show-prompt` | landed | #217 → `1b225087` |
| Implement: NVFP4 MoE experts + W4A4 GEMM (original) | closed superseded, split twice | — |

## Open frontier

NVFP4 stream: **#208** (35B-A3B NVFP4 e2e A/B + perplexity) is now unblocked — its last blocker,
**#213** (routed `T<20` decode + small-T), landed in #221.

Wave 3–4 tickets, all open, unblocked and unclaimed: GGUF converter source (#185), model-acquisition
flags (#186), `reasoning_end` control endpoint (#187), Prometheus `/metrics` (#188), CLI
`--prompt-file`/stdin (#190), tokenizer CLI (#192). Claim order is wave order (#174's locked
decision): the NVFP4 stream (#213) precedes the wave-4 items.

## Resources on this machine

- `models/qwen3_6_27b.ninfer` — 17.5 GB, sha256-verified; ordinary/MTP serve and real tests.
- `models/qwen3_8_27b_nvfp4.ninfer` — 23.7 GB; carries the `dflash2` component the 27B lacks.
- `out/qwen3_6_35b_a3b_nvfp4.ninfer` — 20,647,342,340 bytes; the produced NVFP4 artifact (30,720 routed
  nvfp4 parents). Loads; the routed path executes end to end at every `T` (`T>=20` prefill, `T<20`
  decode/small-T).
- Local source: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` (compressed-tensors `nvfp4-pack-quantized`) in the
  HF cache; the non-`Fast` variant has FP8 experts in layers 32–39, so `Fast` is required.

## Next

1. #208 — e2e A/B against the published `groupwise-int` baseline + perplexity (now unblocked).

## Backlog

Open non-map bugs: #199 (silent stall — a served turn ends with no tool call and no demotion). Open
backlog is #199 only.
