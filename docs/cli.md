# FrInfer CLI

`build/apps/frinfer` runs one request against one v3 `.ninfer` artifact. Build FrInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Tokenizer

`build/apps/frinfer-tokenize` answers "how does this text tokenize?" on the artifact's own tokenizer:
it encodes text to token ids, decodes ids back to text, and reports each id's spelling. This is the
debugging primitive for prompt framing, chat-template drift, and token-budget questions.

```bash
# How many tokens is this, and where do the boundaries fall?
./build/apps/frinfer-tokenize models/qwen3_8_27b_nvfp4.ninfer \
  --file prompt.md --pieces --count
```

```
727 -> 'def'
281 -> ' f'
2007 -> '(x'
1590 -> '):'
198 -> '\n'
```

`--text` takes a string, `--stdin` reads a pipe, `--ids-only` makes stdout a bare id list, and
`--count` prints the token count to stderr. Pieces are quoted and control bytes escaped (`\n`,
`\t`, `\xNN`), so one token is always one line and a boundary is visible in the output rather than
invisible in the terminal.

Decoding runs the other way:

```bash
./build/apps/frinfer-tokenize models/qwen3_8_27b_nvfp4.ninfer \
  --decode-text --ids-file ids.txt
```

Ids may be separated by whitespace, commas, or newlines, so the output of `--ids-only` is valid
input here. `--pieces` output is not: its lines carry the spelling after the id.

### Special tokens

Encoding applies no chat template and adds no implicit control token, so a count here counts the text
you passed, not a framed prompt — a chat turn's count is larger, by the template's own tokens. That
difference is not a constant: the tokenizer merges across the boundary between the template's last
token and your first character, so the overhead depends on your text.

A control token you name explicitly is still recognised, as one id flagged special, and `--pieces`
shows its spelling so prompt framing can be audited verbatim. Generated content is published without
the terminal marker, so to drop it from a decode pass `--skip-special-tokens`; the flag applies to
both directions and without it the decode is faithful, ending with the marker's spelling.

Reproducing a *request's published text* from its ids goes further than that, and only holds under
the default stop policy: the publish path also rolls back a stop token's contribution and cuts at a
stop string, which a plain decode does not. For a request stopped by a non-special token id or by a
stop string, the decoded text carries content the client never received.

Encoding normalises its input to NFC, so decoding reproduces the normalised form rather than the
exact bytes when the input was not already normalised — a decomposed `café` comes back composed.

An id outside the checkpoint vocabulary is rejected by name (`token is outside the checkpoint
vocabulary`), and a malformed id list is rejected before the model loads.

The tool loads the artifact through the same public Engine route as `frinfer` and
`frinfer-perplexity`, so it pays for a weight load. There is no tokenizer-only load path.

## Text input

```bash
./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one prompt source is required: `--prompt <text>`, `--prompt-file <file>`, `--prompt-stdin`, or
`--messages <messages.json>`. Naming two is an error that reports both flags, rather than one
silently winning:

```bash
# prompt.md holds the text verbatim, so it needs no quoting at all:
./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt-file prompt.md --max-context 32768 --max-new 8192

# Anything on stdin works, including a file, a pipe, or another command:
git show HEAD:README.md | ./build/apps/frinfer \
  models/qwen3_8_27b_nvfp4.ninfer --prompt-stdin --max-context 32768 --max-new 8192
```

`--prompt-file` and `--prompt-stdin` read the plain-text body verbatim: no escape processing, no
trailing-newline trimming, no re-encoding. The same bytes produce the same prompt and the same token
count as `--prompt` with those bytes, which is the property that makes a file or a pipe usable for a
long, awkward prompt (code, tabs, newlines, CJK) without shell quoting. An empty file or empty stdin
is still a supplied prompt; the request fails on an empty context rather than silently becoming a
bare completion. `--messages` remains the route for structured multi-turn input and applies the
chat template; the two text sources do not.

The CLI normally omits `--kv-capacity`, so the shared Main Text KV pool follows the example's
32,768-token `--max-context`.

Answer content is streamed to stdout. Human-readable startup milestones and runtime errors are
written to stderr without service timestamps. Reasoning and the CLI result report (timings,
throughput, GPU memory, token IDs when requested, and speculative-decoding statistics) also use
stderr as unprefixed product output, so stdout can be redirected independently. On a terminal,
weight materialization is one transient progress line followed by a compact Engine-ready summary.
Redirected stderr contains persistent readable progress for long loads and no carriage returns or
ANSI escapes. `--log-level debug` exposes every startup phase. Option and local prompt/message input
failures remain direct command diagnostics:

```bash
./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting FrInfer:

```bash
./build/apps/frinfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort. `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Constrained decoding

`--grammar`/`--grammar-file` and `--json-schema`/`--json-schema-file` constrain the answer to a
formal language. Exactly one of the four flags may be given, and a constrained request cannot be
combined with the DFlash/DFlash2 speculative backends (the ordinary and MTP backends carry a
constraint).

- `--grammar GBNF` passes llama.cpp-compatible GBNF text (root symbol `root`) directly;
  `--grammar-file FILE` reads the same text from a file. The Engine compiles and validates the
  grammar at submission. Text it cannot compile, a grammar whose initial mask admits nothing, or
  an empty grammar fails closed with `grammar_invalid` and a non-zero exit.
- `--json-schema JSON` passes a JSON Schema document as text; `--json-schema-file FILE` reads it
  from a file. The document is admitted and converted to GBNF through the same context-accurate
  allowlist the serve route uses, and rejected with the same fail-closed codes:
  `json_schema_invalid` for a malformed document, `json_schema_unsupported` for a keyword FrInfer
  cannot enforce, and `constraint_too_large` beyond the 64 KiB payload limit.

```bash
./build/apps/frinfer models/qwen3_6_27b.ninfer \
  --prompt "Return a JSON object with one numeric field named answer." \
  --json-schema '{"type":"object","additionalProperties":false,"required":["answer"],"properties":{"answer":{"type":"number"}}}' \
  --max-new 64 \
  --greedy
```

A constrained request resolves thinking off unless `--reasoning-effort` explicitly enables it
(`--grammar ... --reasoning-effort medium`). With thinking on and a reasoning-starting prompt, the
Engine wraps the grammar so the constraint carries the reasoning stream and hands off to the
answer grammar at the wire-format close. Grammar completion ends generation; a `--max-new`
truncation can cut a constrained answer mid-document.

## Model acquisition

The CLI loads a local `.ninfer` file by default, or downloads one from Hugging Face
into a local cache on first use. Exactly one source is accepted: the positional
`<model.ninfer>`, `--hf-repo OWNER/REPO` with `--hf-file FILE`, or `--model-url
https://.../model.ninfer`:

```bash
./build/apps/frinfer \
  --hf-repo neroued/Qwen3.8-27B-nvfp4-NInfer \
  --hf-file qwen3_8_27b_nvfp4.ninfer \
  --prompt "Reply with one short sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8
```

Downloads stream to `--cache-dir` (default `~/.cache/frinfer`, or `$FRINFER_CACHE_DIR`)
with resume across interrupted runs, and cached files are reused without network access.
`--hf-revision` (default `main`) pins the Hub revision; `--hf-token` supplies Hub
authentication, defaulting to `$HF_TOKEN` / `$HUGGING_FACE_HUB_TOKEN`. `--offline`
fails closed when the file is not cached. `--cache-list` prints cached `.ninfer`
artifacts and exits without running a prompt. A missing Hub file fails with
`hf_not_found`; an interrupted download resumes on the next run.

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only
  the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI uses root-only context mode, so it does not reserve an extra Device
  checkpoint StateImage or capture a continuation that no later request could consume.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
35B-A3B DFlash or Qwen3.8-27B DFlash2 backend with one to fifteen. Both masked-draft backends
may be combined with `--vision`.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/frinfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/frinfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`.
DFlash2 accepts every draft count from 1 through 15; seven is the checkpoint recommendation.
Both `groupwise-int` and `nvfp4` artifacts use the same Engine route, including CUDA Graph,
concurrent requests, sampling penalties, and prefix reuse. An artifact without the companion
weights reports a missing DFlash2 component when selected. Vision, MTP and DFlash follow the same
rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per Engine. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts one to fifteen draft tokens; seven forms the measured
block length eight, while fifteen uses the maximum supported block length sixteen.

## Common options

The table lists executable defaults. The examples above select FP8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling; above the native `max_position_embeddings` it activates the YaRN context extension (DFlash rejects it) | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash/DFlash2 `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--show-prompt` | dump the exact rendered prompt to stderr before generation | off |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--hf-repo OWNER/REPO` + `--hf-file FILE` | Hugging Face artifact source (cached) | unset |
| `--hf-revision REV` | Hub revision for `--hf-repo` | `main` |
| `--hf-token TOKEN` | Hub bearer token (env fallback) | unset |
| `--model-url URL` | direct `https://` artifact source (cached) | unset |
| `--cache-dir DIR` | download cache directory | `~/.cache/frinfer` |
| `--offline` | reuse the cache, no network | off |
| `--cache-list` | list cached artifacts and exit | off |
| `--grammar GBNF` / `--grammar-file FILE` | GBNF answer grammar, compiled by the Engine | unset |
| `--json-schema JSON` / `--json-schema-file FILE` | JSON Schema answer grammar, converted through the serve contract | unset |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics. `--show-prompt` writes the exact rendered prompt (the
template output with media placeholders already expanded) to stderr before generation, so stdout
stays the answer channel; it is the direct tool for debugging chat-template rendering.

Run `./build/apps/frinfer --help` for the exact option contract, and
`./build/apps/frinfer --version` for the version, git sha, and build config.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/frinfer models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The official artifacts have a native context limit of 262,144 tokens. The practical allocation
on one RTX 5090 depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations;
`--kv-dtype` independently selects runtime KV storage. The prepared prompt must fit
`--max-context`; a value above the model's native position capacity activates the YaRN context
extension (spec defaults β_fast=32, β_slow=1, ext_factor=1), and the DFlash draft backend rejects
it. Generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup FrInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
