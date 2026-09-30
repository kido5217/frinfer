# Upstream sync

How this fork (`kido5217/ninfer-yarn`) stays current with upstream `Neroued/ninfer` while keeping
its own patches. Every fact below was verified during the first sync (2026-09-26, upstream tip
`e31bc99b`) and re-confirmed during the second sync (2026-09-30, upstream tip `d44ab584`).

## Ground rules

- **Upstream is read-only.** Never post issues, PRs, or comments to `Neroued/ninfer`, and never
  push to it; the `upstream` remote's push URL is set to `DISABLED` as a guard.
- **Sync source is `master` only.** Upstream's `dev` is the maintainer's rewritable WIP branch and
  is never a sync source. Upstream `master` is append-only (their ruleset enforces non-fast-forward
  plus required linear history and rebase-only PRs), so the merge base cannot be invalidated under
  us.
- **`gh` resolves to `upstream` once that remote exists.** Keep
  `gh repo set-default kido5217/ninfer-yarn` set and pass `--repo kido5217/ninfer-yarn` on every
  scripted `gh` call.
- **Never rebase or force-push `master`.** A sync is a merge; our patches stay ordinary commits on
  top.

## One-time setup (already applied)

- `git remote add upstream https://github.com/Neroued/ninfer`;
  `git remote set-url --push upstream DISABLED`.
- Repo settings: `allow_merge_commit=true`, `allow_squash_merge=true`, `allow_rebase_merge=false`.
- Ruleset `protect-master` (id 24025442): `pull_request.allowed_merge_methods = ["merge","squash"]`.
  The repo setting alone is not enough — a squash-only rule rejects merge commits with
  *"Merge commits are not allowed on this repository"*.
- `gh repo set-default kido5217/ninfer-yarn`.

## The sync

1. Refresh and record the new tip:
   ```bash
   git fetch upstream master
   git rev-parse --short upstream/master
   git rev-list --count HEAD..upstream/master   # 0 = already current
   ```
2. Branch off master: `git checkout -b sync/upstream-<date> master`.
3. Merge:
   ```bash
   git merge --no-edit \
     -m "Merge upstream Neroued/ninfer master (<tip>) into the fork" upstream/master
   ```
4. Resolve conflicts **per file**: adopt upstream's side, re-apply our additions on top; nothing of
   ours is dropped silently. Recompute the overlap set each sync:
   ```bash
   BASE=$(git merge-base master upstream/master)
   comm -12 <(git diff --name-only $BASE..master | sort) \
            <(git diff --name-only $BASE..upstream/master | sort)
   ```
   Historically: `AGENTS.md`, `bench/README.md`, `src/models/qwen3_5/execution/text.cpp`,
   `src/models/qwen3_5/program/planning/startup.cpp`.
5. Verify the merged tree:
   ```bash
   nix develop -c cmake --build build -j
   nix develop -c bash -c 'export LD_LIBRARY_PATH=$(dirname "$(readlink -f /run/opengl-driver/lib/libcuda.so.1)"):$LD_LIBRARY_PATH; ctest --test-dir build'
   nix develop -c python3.13 -m pytest tests/convert -q
   ```
   With a free GPU, also smoke the artifact (`qwen3_8_27b_nvfp4.ninfer`): a native run at
   `--max-context 262144` and a YaRN boundary run at `262145`. Upstream performs numerical work;
   its perf commits legitimately change kernel accumulation order, so native generations may
   differ across a sync while every oracle-based suite still passes. Report such differences
   instead of treating them as failures.
6. Land as a **true merge commit**: push the branch, open a PR
   (`gh pr create --repo kido5217/ninfer-yarn ...`), merge with `--merge` — **not squash** (a
   squash would drop the upstream ancestry and the next sync would re-apply the same commits) — and
   delete the branch.
7. Record: update the drift baseline below to the new upstream tip and note the sync, its
   verification, and any numerical deltas on the tracker.

## Drift alert

A session-side smart note (`Upstream drift`) surfaces when upstream `master` moves past the
recorded synced tip. When it fires, run the procedure above, then re-arm the note with the new
baseline tip.

Vendored third-party sources carry their own drift notes; for `third_party/llama-chat` see
[Vendored llama.cpp chat stack](llama-chat-vendor.md).

## Verification standard

The fork's own qualification applies after every sync: build, op suites, converter tests, and —
with a free GPU — the artifact smoke. End-to-end bit-identity across an upstream perf merge is not
expected; what must stay exact are the fork's patch areas, e.g. the YaRN `s <= 1` byte-identity
invariant, the renamed binaries, and the flake.
