# FrInfer

> It’s alive. It just OOMs.

## This fork: `kido5217/frinfer` stands for `Frankenstein + ninfer`

This repository is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer), kept current
through periodic true-merge syncs (last sync: `d44ab584`, 2026-09-30). It adds YaRN context
extension, a llama.cpp-derived chat/tool-call parsing stack, and constrained decoding, and ships
the `frinfer` product binaries. The rest of this README describes the combined product
(fork + upstream).

### Fork constraints

- **Single NVIDIA GeForce RTX 5090.** The fork is developed, tested, and published on one 5090;
  the build rejects CUDA architectures other than `sm_120a`. There is no multi-GPU, offload, or
  distributed path.
- **NVFP4 weights.** The fork's features are developed and validated against NVFP4 weight
  artifacts. The upstream `groupwise-int` artifacts remain loadable but are not a fork-validated
  configuration.
- **Built for opencode.** The primary intended client is the opencode agent harness: the serve is
  the opencode backend (compatibility study: [wayfinder map #139](https://github.com/kido5217/frinfer/issues/139)).
  One-shot CLI use also works.

### Changes from upstream

Maintained: add or update a row whenever a fork feature changes or its status changes.
Last updated 2026-10-05 (`port/version-flag`, `ba76d4fe`).

| What | Why | How | Status | Source |
|---|---|---|---|---|
| YaRN context extension — `--max-context` beyond the model's native `max_position_embeddings` (β_fast=32, β_slow=1, ext_factor=1) | Long-context agent sessions (opencode compaction, 200K+ contexts) exceed the model's native capacity on 32 GB VRAM | YaRN correction in the RoPE op, golden-vector tested; unsupported by the DFlash draft backend | shipped | FrInfer (implementation); spec cross-checked against the YaRN original, llama.cpp, HF transformers, and vLLM |
| llama.cpp-derived chat/tool-call parsing stack for Qwen3.5 turns (reasoning / content / tool-call regions) | Agent tool-call parsing must survive quoted markers, truncation, and degenerate loops without silent demotion | Vendored llama.cpp subset (chat params, PEG grammar runtime) under `third_party/`, re-vendor runbook with drift alert, 48-vector parse corpus + e2e gate | shipped (vendored baseline `05af0d2b`) | llama.cpp (vendored subset) |
| Tool-call demotion recovery + client signaling (G1) | A malformed or truncated tool call demoted to plain text, stalling the agent silently | Partial-AST salvage + second-chance raw-value parse (llama.cpp-derived, in-core); lost calls signal a retryable in-band error (SSE error frame / 409 + `x-should-retry`) with per-class codes (ADR-0002) | shipped | Salvage: llama.cpp-derived; signaling: FrInfer (verified against the opencode client's error classifier; llama.cpp-flush and vLLM close-at-end alternatives rejected) |
| Constrained decoding — per-request GBNF grammar + OpenAI `response_format` `json_object`/`json_schema` + Anthropic `output_config.format` `json_schema` | Agents depend on guaranteed structured output (tool arguments, JSON) | Engine GBNF mask via an in-tree token-trie producer (differential-verified); serve enforcement with a context-accurate schema allowlist and fail-closed codes shared by the OpenAI `response_format` and Anthropic `output_config.format` routes; thinking-aware wrapper grammar; CLI exposure (`--grammar`/`--grammar-file`/`--json-schema`/`--json-schema-file`) sharing the serve contract; grammar A/B perf-neutral (±0.24 ms/step) | shipped | GBNF language, PEG runtime, and JSON-schema converter: llama.cpp (vendored + 3 fork patches); mask producer and serve contract: FrInfer |
| Token log probabilities — OpenAI `logprobs`/`top_logprobs` on the chat + Responses routes | Clients and evaluators need the model's per-token confidence (OpenAI spec shape, not llama.cpp's self-declared non-compatible one) | New bounded `logprob_topk` split-CTA Op gathers the full-vocabulary log-softmax of the post-mask, penalty/temperature-scaled sampler logits before truncation; Engine per-committed-token channel aligned with `preview_model`; chat `{content, refusal}` + Responses aggregate/streaming shapes; Anthropic route unaffected | shipped | OpenAI OpenAPI spec (shape); evidence gate `research/logprobs-evidence` (distribution, oracle, cost) |
| Realized reasoning tokens in the request log (schema v24) + CLI summary | Clients must distinguish realized thinking tokens from the thinking budget | `reasoning_tokens` in `request_done.result`, `model_thinking_tokens` → `budget_thinking_tokens`, realized tokens in the CLI generation summary | shipped (r5) | FrInfer (opencode-deployment driven) |
| Benchmark extensions | Measure fork features at their claimed scope | `ninfer_bench --grammar` (constrained-decoding mode); Qwen3.8 GPQA context-window comparison | shipped | GPQA-Diamond (public benchmark); runners: FrInfer |
| NVFP4 35B-A3B MoE experts — converter recipe + `.ninfer` codec + loader binding | The MoE line had no native FP4 expert route; prefill is compute-bound on the Q4/Q5 banks | `qwen3_6_35b_a3b_nvfp4` imports the compressed-tensors per-expert gate/up/down tensors as one `nvfp4` parent per expert projection (the source selects a per-expert FP32 divisor and gate/up are separate tensors; gate/up share the source divisor within an expert). `SparseMoeWeights` carries the expert banks and the Op validation/route-plan profiles admit the codec. The W4A4 GEMM and e2e follow in #207/#208 | in progress — converts, round-trips and loads; execution pending | FrInfer; evidence gate `research/nvfp4-moe-evidence` (#177) |
| Upstream sync process | Keep the fork current while preserving upstream ancestry | True merge commits per the sync runbook; drift alert on the vendored subset; last sync `d44ab584` (2026-09-30) | ongoing | Neroued/ninfer (upstream) |
| Product executables renamed to the `frinfer` family | Distinguish the fork's binaries from upstream's | CMake target rename (`frinfer`, `frinfer-serve`, `frinfer-perplexity`) | shipped | FrInfer |
| Nix flake dev environment | Reproducible build environment | `flake.nix` at the repo root: CUDA 13.1, C++20, CMake/Ninja, FFmpeg, libcurl, Python 3.13 + torch; builds run in `nix develop` | shipped | nixpkgs (CUDA 13.1 `cudaPackages_13_1`) |
| Agent/governance documentation | The fork is developed with LLM agents in the loop | `AGENTS.md` (product boundaries, verification, reporting), issue tracker + triage-label conventions, branch→commit→push→PR→merge→rebase landing pipeline; the LLM is permitted to open and merge PRs | ongoing | FrInfer |
| Build identity — `--version` on all four product binaries | Scripts and humans probe tools with `--version`; its absence was the survey's basic ecosystem-parity gap | Configure-time version (`git describe --tags`) + short git sha + build config via `configure_file`, exposed as the `ninfer_build_info` interface target; `--version` on `frinfer`, `frinfer-serve`, `frinfer-perplexity`, and `ninfer_bench` | shipped | llama.cpp (the `--version` pattern, adopted) |

### Bug reporting

Bugs in this fork are reported in [this repository's issue tracker](https://github.com/kido5217/frinfer/issues).
Do not report bugs to the upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) project unless the
bug is verified to reproduce on the original NInfer — an upstream master build with none of this fork's
changes applied (no YaRN, no chat parsing stack, no constrained decoding, and no other row in the table
above). If a repro exists only in this fork, it is a fork bug — report it here. The verification burden
is on the reporter: the upstream tracker is for upstream bugs only, and unverified reports there poison
the upstream project.

### Disclaimer

This fork is provided **as is**, without warranty of any kind, either express or implied,
including but not limited to the warranties of merchantability, fitness for a particular purpose,
and non-infringement. The fork's author assumes no responsibility or liability for any loss or
damage of any kind — including data loss, model output quality, or system damage — arising from
the use of this fork or its artifacts. This fork is an independent derivative project; it is not
endorsed by, sponsored by, or affiliated with the upstream [Neroued/ninfer](https://github.com/Neroued/ninfer)
project or its maintainer. Use at your own risk.

> **End of the fork section.** Everything below documents the product: this content is synced from
> the upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) README and describes the
> combined product (fork + upstream). Fork-specific changes, constraints, and the bug-reporting
> policy are in the fork section above.

---

> Selected checkpoints. Maximum single-GPU inference performance.

FrInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

FrInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/kido5217/frinfer.git
cd frinfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run FrInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/frinfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results. Qwen3.8 uses FP8 E4M3 row-256 KV;
Qwen3.6 uses INT8 group-64 KV.

### Concurrent MTP3 decode

Saturated decode used CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### Single-request serving

The serial serving corpus used CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

## Evaluation

Capability scores were measured through FrInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  frinfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; FrInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. When it exceeds the model's native position
capacity (`max_position_embeddings`), FrInfer activates the YaRN context extension (spec defaults
β_fast=32, β_slow=1, ext_factor=1), which the DFlash draft backend does not support. `--kv-capacity`
sizes the shared Main Text KV pool used by active requests and retained prefixes; `auto` resolves
the largest legal capacity at startup from the memory remaining after weights while keeping 1 GiB
of sizing headroom. Explicit capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

FrInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

FrInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
