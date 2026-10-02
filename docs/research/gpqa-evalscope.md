# GPQA-Diamond under EvalScope 1.10.0: dataset, prompt rendering, and rule scoring

Scope: what the pinned evaluation stack pulls for `dataset: gpqa_diamond`, how the
0-shot prompt is rendered, how `judge_strategy: rule` extracts and scores answers, where
data is cached, and how `eval/ninfer_eval` forwards the name. Prepared to support the
staged two-arm GPQA-Diamond comparison (Qwen3.8-27B nvfp4, `--max-context` 262144 vs
446902).

Evidence base (all read 2026-10-02):

- EvalScope source: `https://github.com/modelscope/evalscope` tag `v1.10.0`, commit
  `9d052ca0240ebf8b603c053fa44727b863ff3933`; PyPI `evalscope==1.10.0` published
  2026-08-04 (the version pinned in `eval/requirements.txt`). Local clone:
  `/tmp/opencode/evalscope-1.10.0`.
- ModelScope dataset `AI-ModelScope/gpqa_diamond`: hub page
  `https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond`, plus a git clone of
  `https://www.modelscope.cn/datasets/AI-ModelScope/gpqa_diamond.git`
  (master `25a6db1609cf37628cfc7f3952ba94c6d1012cdf`, last commit 2025-01-30).
- `modelscope` v1.40.1 and `modelscope_hub` sources on GitHub (cache layout; the
  evaluation install resolves `modelscope[datasets]>=1.34`).
- This repository at `151d6194`: `eval/configs/qwen3_8_27b_nvfp4_reasoning.yaml`,
  `eval/ninfer_eval/`, `eval/README.md`.

## 1. Dataset identity and snapshot

| Property | Value |
|---|---|
| EvalScope benchmark name | `gpqa_diamond` |
| ModelScope dataset id | `AI-ModelScope/gpqa_diamond` |
| Split / subset | `train` / `default` (the only split and subset) |
| Revision pin | none — EvalScope passes no revision, so the hub's default branch `master` is resolved at download time |
| Master commit (2026-10-02) | `25a6db1609cf37628cfc7f3952ba94c6d1012cdf` (2025-01-30, "Create file dataset_infos.json in gpqa_diamond project") |
| Items | 198 |
| Payload | `train.jsonl`, 1,859,862 B, 198 lines, 78 columns per record |

Sources and details:

- `evalscope/benchmarks/gpqa/gpqa_adapter.py:17-57` registers
  `BenchmarkMeta(name='gpqa_diamond', dataset_id='AI-ModelScope/gpqa_diamond',
  metric_list=['acc'], few_shot_num=0, train_split=None, eval_split='train',
  prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT)`. There is no `revision`.
- Load path: `evalscope/api/benchmark/adapters/default_data_adapter.py:233-264` →
  `evalscope/api/dataset/loader.py:83-119` (`RemoteDataLoader`) →
  `evalscope/api/dataset/hub.py:69-104` (`DatasetHub.load`) →
  `MsDataset.load(dataset_name='AI-ModelScope/gpqa_diamond', split='train',
  subset_name=None, trust_remote_code=True)`. `subset_name` is `None` because the
  subset is the default one (`hub.py:93`); the hub page's SDK example shows the
  equivalent `MsDataset.load(..., subset_name='default', split='train')`.
- Repo file listing (from the clone): `.gitattributes` (3,818 B), `README.md`
  (2,731 B), `dataset_infos.json` (3,953 B), `train.jsonl` (1,859,862 B). `train.jsonl`
  is stored via git-LFS in the hub, but the download manager fetches the real bytes.
- Count: direct line count = 198 records with 198 unique `Record ID`s. This matches
  EvalScope's bundled statistics (`evalscope/benchmarks/_meta/gpqa_diamond.json`:
  `total_samples: 198`, subset `default`) and this repo's recorded run (179/198,
  `eval/README.md`).
- Fields: the adapter consumes `Question`, `Correct Answer`, `Incorrect Answer 1..3`
  and an optional `subset` column (`gpqa_adapter.py:69-84`); `dataset_infos.json`
  declares all 78 columns as `Value` features and the `train` split. This dataset has no
  `subset` column, so `subset_key` stays `''`.
- Choice construction: `_process_input` (`gpqa_adapter.py:91-111`) puts
  `[Incorrect 1, Incorrect 2, Incorrect 3, Correct]` through `random.shuffle` and sets
  the target to the letter of the correct position (`A`..`D`).
- Hub page metadata: 1.87 MB, updated Jan 30 2025. Front matter says "Apache License
  2.0" while the card body cites CC BY 4.0 (upstream GPQA's license) — a hub-side
  inconsistency worth knowing before redistribution.

## 2. Rendered 0-shot prompt

Template `SINGLE_ANSWER_COT` (`evalscope/utils/multi_choices.py:21-27`), verbatim:

```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

The template is `.strip()`ed, then filled by `prompt()` (`multi_choices.py:115-133`):

- `{letters}` → `A,B,C,D` (`format_letter_choices`).
- `{choices}` → `A) <choice1>` newline `B) <choice2>` … (`answer_options`), choices in
  shuffled order, `A)`-style labels.
- Resulting message: a single user message; the full text is
  `<instruction>\n\n<question>\n\nA) …\nB) …\nC) …\nD) …`.
  `evalscope/benchmarks/_meta/gpqa_diamond.json`'s `sample_example.input` shows exactly
  this shape and a single input element (no system message).
- Few-shot: `few_shot_num: 0` is set explicitly by both
  `eval/configs/qwen3_8_27b_nvfp4_reasoning.yaml` and the budgets config, forwarded into
  `dataset_args` by `eval/ninfer_eval/backends/evalscope.py:290-291`. The adapter's
  default is 0; any non-0/non-5 value is coerced to 5 with a warning
  (`gpqa_adapter.py:63-67`), and 5-shot would prepend `FEW_SHOT_TEMPLATE` + 5
  demonstrations. 0-shot is the active setting.
- System prompt: none at runtime. The registered `BenchmarkMeta` passes no
  `system_prompt` (dataclass default `None`, `evalscope/api/benchmark/meta.py:53`), and
  `DefaultDataAdapter.process_sample_str_input` only inserts a system message when it is
  not `None` (`default_data_adapter.py:143-147`). The generated docs file
  `_meta/gpqa_diamond.json` records `"system_prompt": ""`, but the runtime registry uses
  the in-code meta (`evalscope/api/registry.py:89-127`), so the request carries the user
  message only.
- Determinism: `run_single_task` calls `seed_everything(task_cfg.seed)`
  (`evalscope/run.py:28-32`); ninfer-eval sends `seed: 42` (from `generation.seed`).
  The per-record choice shuffle is therefore deterministic for a fixed raw dataset and
  EvalScope version.
- Prompt size (`_meta/gpqa_diamond.json` statistics): 340–5,845 chars, mean 841; the
  longest prompt is far below either context arm.

## 3. `judge_strategy: rule`: extraction and scoring

- `rule` is `JudgeStrategy.RULE = 'rule'` (`evalscope/constants.py:110-114`).
  `LLMJudgeMixin.use_llm_judge` returns `False` for it, so no judge model is
  initialized or called (`evalscope/api/mixin/llm_judge_mixin.py`), and
  `calculate_metrics` goes straight to the rule path
  (`default_data_adapter.py:668-729`).
- Extraction: `MultiChoiceAdapter.extract_answer` → `parse_answers(..., multiple_correct=False)`
  (`evalscope/api/benchmark/adapters/multi_choice_adapter.py:77-88`), whose result is
  joined/sorted into one string; the Chinese extractor is not used for this template.
- `parse_answers` (`evalscope/utils/multi_choices.py:163-231`) tries, in order:
  1. Strict: `re.search(r'(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)', completion,
     re.MULTILINE)` — a line that starts with `ANSWER:` (keyword case-insensitive).
  2. Loose: `re.search(r'(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)')` — anywhere
     in the completion.
  3. Fallback: `_fallback_parse_answer` returns the last `isupper()` character of the
     whole completion — any uppercase letter, **not** restricted to the option set and
     not checked against it.
  4. Else the empty set.
  Single-answer mode then requires the stripped, trailing-period-stripped capture to be
  exactly one member of the case-sensitive allowed set `{A,B,C,D}`; otherwise empty.
- Metric: `acc` is `Accuracy(ExactMatch)` —
  `normalize_text(prediction) == normalize_text(reference)` with
  `normalize_text = text.strip().lower()` (`evalscope/metrics/nlp/metrics.py:15-51`,
  `evalscope/metrics/utils/functions.py:12-14`), aggregated as the mean over all 198
  samples.
- Unparseable answers are scored incorrect, not skipped: an empty extraction never
  equals `A`–`D`, so the sample contributes 0.0 to the mean. There is no retry, no
  error status and no `ignore_errors` interaction at the metric level.
- Outcomes verified by executing the exact v1.10.0 regex/extraction logic locally:

| Completion | Extracted | Effect |
|---|---|---|
| `blah blah\nANSWER: A` | `A` | correct if target is A |
| `blah\nANSWER: A\nbecause …` | `A` | newline terminates the capture |
| `ANSWER: D. Because of X` | `D` | period terminates the capture |
| `**ANSWER: A**` | `A` | loose match, non-word char terminates |
| `Final: ANSWER: C` | `C` | `ANSWER:` need not start a line (loose match) |
| `ANSWER: D because of X` | *(empty)* | same-line prose is captured and rejected |
| `ANSWER: A, B` | *(empty)* | single-answer mode rejects multi-letter captures |
| `ANSWER: b` / `answer: c` | *(empty)* | keyword is case-insensitive, the letter check is not |
| `ANSWER: (D)` | *(empty)* | loose pattern matches with an empty capture, so the fallback never runs |
| `no marker` | last uppercase letter (fallback) | `I think C … the correct one is A.` → `A` |

Practical consequence: an uppercase `A`–`D` after `ANSWER:` (terminated by newline,
period, or a non-word character) is required; lowercase letters, multi-letter captures,
and same-line prose after the letter all score as incorrect. Without any `ANSWER:`, the
last uppercase letter in the text is used unchecked.

## 4. Cache paths and offline pre-download

Two caches matter for an offline run:

1. ModelScope hub snapshot (the dataset payload). Cache root = `MODELSCOPE_CACHE` when
   set, otherwise the SDK default; the layout depends on the installed `modelscope`:
   - `modelscope < 1.38` (legacy): root default `~/.cache/modelscope/hub`; dataset at
     `<root>/datasets/AI-ModelScope/gpqa_diamond/` with `train.jsonl` etc. directly
     inside.
   - `modelscope >= 1.38` (current): root default `~/.cache/modelscope`; snapshot at
     `<root>/datasets/AI-ModelScope--gpqa_diamond/snapshots/master/`
     (`modelscope_hub/src/modelscope_hub/_download.py::_repo_cache_dir_path` +
     `download_repo`, effective revision defaults to `master`; base default from
     `HubConfig.cache_dir`, `modelscope_hub/.../config.py`; `DEFAULT_CACHE_DIR_NAME =
     "modelscope"`).
   - `modelscope >= 1.38` detects a populated legacy directory and reuses it instead of
     re-downloading (`modelscope/hub/snapshot_download.py` capability probe +
     `DownloadManager._find_legacy_repo_dir`).
   - Version context: `modelscope 1.38.0` was released 2026-07-01; a fresh install today
     resolves 1.40.1 (`evalscope` requires `modelscope[datasets]>=1.34`;
     `modelscope v1.40.1` requires `modelscope-hub>=0.4.3`). Check the actual venv with
     `eval/.venv/bin/python -c "import modelscope; print(modelscope.__version__)"`.
2. EvalScope's converted-dataset cache: `~/.cache/evalscope/datasets/<safe>-<hash>`
   (override with `EVALSCOPE_CACHE`); the name is `safe_filename(path)` +
   `gen_hash(f'{path}{split}{subset}{version}{kwargs}')` where `gen_hash` is the first
   32 hex chars of the MD5 (`evalscope/api/dataset/loader.py:96-101`,
   `evalscope/utils/io_utils.py:536`). For this exact call the derived name is
   `AI-ModelScope_gpqa_diamond-0458611f9123d946f7dd3cd2dc90d3bb`. If that directory
   exists, EvalScope loads from it and never touches ModelScope.

Pre-download command (the pattern already used in `eval/README.md` for other datasets),
run once with the eval venv and network access:

```python
from modelscope import dataset_snapshot_download

print(dataset_snapshot_download('AI-ModelScope/gpqa_diamond'))
```

The return value is the snapshot root — use it to confirm where the files landed. Since
EvalScope passes no revision, the download captures whatever `master` points to at that
moment; today's master is `25a6db1609cf37628cfc7f3952ba94c6d1012cdf`.

Not verified here: the exact behavior of `MsDataset.load` with a warm cache but no
network (no local eval venv or cache exists in this checkout; no offline run was
performed).

## 5. `eval/ninfer_eval` mapping of `gpqa_diamond`

`gpqa_diamond` is passed straight through; there is no dataset-name translation in the
coordinator:

- `eval/ninfer_eval/config.py:143-175` — `JobConfig.dataset` is validated only as a
  non-empty string; no allowlist, no renaming.
- `eval/ninfer_eval/backends/evalscope.py:283-339` (`_task_dict`) builds
  `datasets: [job.dataset]` → `['gpqa_diamond']`, `dataset_args: {'gpqa_diamond': {...}}`,
  and forwards `judge_strategy: 'rule'`, `dataset_hub: 'modelscope'`, and
  `few_shot_num: 0` (inserted into `dataset_args`).
- `eval/ninfer_eval/backends/evalscope.py:20-60` keeps
  `_DATASET_COUNTS['gpqa_diamond'] = {'default': 198}` — used only by `plan()` for
  progress totals; scoring uses EvalScope's report.
- `eval/ninfer_eval/backends/registry.py` registers the `evalscope` backend by that
  name; `validate()` checks the pinned package version (`evalscope==1.10.0`).
- EvalScope side: `evalscope/benchmarks/__init__.py` auto-imports every
  `*_adapter.py`, so `GPQAAdapter` registers `gpqa_diamond` in `BENCHMARK_REGISTRY`;
  `get_benchmark('gpqa_diamond', config)` resolves it
  (`evalscope/api/registry.py:89-127`).
- `eval/configs/qwen3_8_27b_nvfp4_gpqa_budgets.yaml` uses the same `dataset`,
  `judge_strategy`, `few_shot_num` and `dataset_hub` settings; nothing in the repo
  pins a dataset revision.

## 6. Open points

- EvalScope pins no dataset revision; the snapshot hash above is what `master` resolved
  to on 2026-10-02, not a reproducibility guarantee.
- The EvalScope converted-cache directory name is derived from the v1.10.0 source, not
  observed on disk here.
- Offline behavior of `MsDataset.load` with a warm ModelScope cache was not tested in
  this investigation.
