# Upstream `Neroued/ninfer` — history and conventions

Research for ticket kido5217/ninfer-yarn#18. All findings were gathered read-only on 2026-09-26 from
a throwaway worktree (`research/upstream-conventions`) by URL fetch (no remote added) and the public
GitHub API. Snapshot of upstream at research time: `master` = `dev` =
`e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`.

Commands used (representative):

```sh
git fetch --no-tags https://github.com/Neroued/ninfer \
  'refs/heads/master:refs/remotes/upstream-fetch/master' \
  'refs/heads/dev:refs/remotes/upstream-fetch/dev'
gh api repos/Neroued/ninfer/rulesets/19769905
gh api repos/Neroued/ninfer/branches
gh api "repos/Neroued/ninfer/events?per_page=100"
git log --graph --format='%h | A:%ad | C:%cd | %s' --date=iso-strict bace20dc..e31bc99b
git rev-list --merges e31bc99b | wc -l
git cherry -v bace20dc e31bc99b
```

## 1. History stability — does upstream rewrite `master`?

**No. Since 2026-07-26, `master` is structurally append-only. What upstream rewrites is feature
branches: they are rebased before landing, which is why old author dates appear above newer
commits.**

Evidence:

- A repository ruleset named `master` (id `19769905`, `enforcement: active`, target `branch`,
  conditions `ref_name.include: ["~DEFAULT_BRANCH"]`, `bypass_actors: null`) applies since
  2026-07-26T19:05:09+03:00 (updated 2026-07-28) and contains exactly:
  `deletion`, `non_fast_forward`, `required_linear_history`, and `pull_request`
  (`required_approving_review_count: 1`, `require_code_owner_review: true`,
  `dismiss_stale_reviews_on_push: true`, `required_review_thread_resolution: true`,
  `allowed_merge_methods: ["rebase"]`). Force-pushes and merges are impossible on `master`;
  landing a PR must rebase its commits onto the current tip.
  (`gh api repos/Neroued/ninfer/rulesets/19769905`, `.../rules/branches/master`.)
- The reachable history contains **zero merge commits** (`git rev-list --merges e31bc99b | wc -l`
  = 0, of 1130 commits) and the 15 commits above the fork point form a single-parent chain.
- **The fork's ancestry is intact.** Fork point `bace20dc70249eed6402b66d4852c6c3f9612905`
  (2026-09-24) is still an ancestor of upstream `master`; commits of shared history missing from
  upstream: 0 (`git rev-list --count bace20dc ^e31bc99b`); root
  `410b9f6c2890d2d935c7abb503e13a6fd233fff3` is still an ancestor. The fork's clone reflog
  (`master@{6}: clone`) shows the fork copied `master` at `bace20dc`, and all 15 upstream commits
  since then sit on top of it — strictly append-only in the observed window.
- **The 2026-08-20 hint is a feature-branch rebase, not a master rewrite.** In the kda/gdn stack
  (`619e3f4c`, `6d333ce0`, `d4ea63ea`, `71a1cb0e`, `0784e76f`) author dates range 2026-08-20 to
  2026-09-26, but the committer date is identical for all five: `2026-09-26T21:05:32+08:00`.
  Public events show branch `feat/kda` created 2026-09-26T10:02:02Z and deleted
  2026-09-26T13:10:34Z, with the stack pushed to `dev` before reaching `master`.
- **PR landing is rebase-merge, verifiable by timestamps**: PR #302 merged 2026-09-23T08:51:39Z ↔
  commit `4c0fe48a` committer date 2026-09-23T16:51:39+08:00 (author date 2026-09-22T17:07:24-07:00);
  PR #281 merged 2026-09-23T13:56:55Z ↔ commit `594930e7` committer date 2026-09-23T21:56:54+08:00.
  Author dates survive; committer dates equal the merge time.
- **No duplicate cherry-picks**: `git cherry -v bace20dc e31bc99b` marks all 15 commits `+` (no
  equivalent patch below the fork point), and no commit subject repeats anywhere in the 1130-commit
  history. 205 of 1130 commits overall have committer date > author date — the deposit left by
  rebase landing.
- Caveat: the repo was created 2026-06-26, so a one-month window (2026-06-26 → 2026-07-26) predates
  the ruleset. It cannot affect this fork: the fork point and every shared ancestor are still
  reachable and unchanged.

## 2. `dev` vs `master`

**`dev` is the maintainer's unprotected work-in-progress staging branch; `master` is the shipped
state. At research time both tips were the same commit (`e31bc99b`) and therefore the same history.**

- `gh api repos/Neroued/ninfer/branches`: only two branches exist. `master`: `protected: true`.
  `dev`: `protected: false`. Both at `e31bc99b13f517c8aae70b997b7c4a49b4dcdc5d`.
- `CONTRIBUTING.md` (in the upstream tree) states it directly: "Target pull requests at the
  `master` branch. Before implementation, also review the current `dev` branch to avoid duplicating
  or conflicting with maintainer work that has not yet reached `master`."
- Public events (last 100) show all 9 PushEvents targeting `refs/heads/dev` (all 2026-09-26), plus
  creation/deletion of short-lived branch `feat/kda`; no push events target `master` in the window.
- Being unprotected, `dev` can be force-pushed/rewritten at will — it is a moving preview, not a
  stable sync source.

## 3. Contribution policy

**Upstream accepts outside patches in principle, but acceptance is discretionary and review is
explicitly not guaranteed.** `CONTRIBUTING.md` exists at the upstream root and defines a strict
issue-first workflow.

- Policy highlights (`git show e31bc99b:CONTRIBUTING.md`): every change starts with an Issue; wait
  for the maintainer to confirm scope and direction before implementing; one PR = one coherent
  decision with implementation, tests, and active documentation; PR template with linked Issue,
  design, and verification commands; review is "a limited maintainer resource and is not
  guaranteed"; a PR may be closed without line-by-line review (unconfirmed scope, missing evidence,
  bundled changes, unexplainable AI-generated code, etc.).
- Activity (GitHub search API, 2026-09-26): 154 PRs total — 29 merged, 125 unmerged; 36 PRs open.
  164 issues total, 63 open. External contributors have landed merged PRs (e.g. #302 by adubkov,
  #281, #271, #266 by MOVIBALE, #302 merged 2026-09-23).
- The fork owner's own PR #309 (`fix(frontend): keep quoted reasoning closes...`) was closed
  2026-09-25T23:29:14Z unmerged with **zero comments and zero reviews** — a concrete example of the
  "closed without detailed review" clause.
- Repo is public: Apache-2.0, created 2026-06-26, 2440 stars, 468 forks, issues and discussions
  enabled, description "High-performance single-GPU inference for selected model checkpoints and
  GPUs."

## 4. Releases and tags

**There is no release or tag flow at all.**

- `git ls-remote --tags https://github.com/Neroued/ninfer` → empty.
- `gh api repos/Neroued/ninfer/releases` → `[]` (length 0); `.../tags` → empty.
- `master` is the only versioned state; any sync must be pinned by commit SHA, not a version.

## 5. CI

**Upstream has no repository CI — no workflow runs on `master` pushes or PRs.**

- `.github/` contains only `FUNDING.yml` and `pull_request_template.md`; no `.github/workflows`
  directory exists in the tree, and no other CI config (GitLab, CircleCI, Jenkins, Buildkite,
  Azure Pipelines) is present (`git ls-tree -r --name-only e31bc99b`).
- `gh api repos/Neroued/ninfer/actions/workflows` lists only two GitHub-managed dynamic workflows:
  `dynamic/agents/copilot-pull-request-reviewer` (Copilot Code Review) and
  `dynamic/dependabot/update-graph`; recent runs are Copilot review runs (last 2026-08-18) and a
  Dependabot graph update (2026-06-26).
- Consequence: verification is manual, per the PR template and `CONTRIBUTING.md`; there is no
  upstream status check that a fork merge could break.

## Implications for merge-based sync

1. **Merging upstream `master` into fork `master` is stable.** Upstream `master` is append-only
   behind `non_fast_forward` + `required_linear_history` + rebase-only PR merges, so the fork's
   merge base cannot be invalidated or duplicated by an upstream rewrite. No need for
   `--rebase`/cherry-pick of upstream history.
2. **Now is a clean sync point.** At research time the fork is 6 commits ahead (`88381a8d`…`5df49b8a`)
   and upstream 15 commits ahead of the shared fork point `bace20dc` (2026-09-24). A trial merge of
   `upstream-fetch/master` into fork `master` in the throwaway worktree (`git merge --no-commit
   --no-ff`, then aborted) completed **without conflicts**: only `AGENTS.md`, `bench/README.md`,
   `src/models/qwen3_5/execution/text.cpp`, and `src/models/qwen3_5/program/planning/startup.cpp`
   auto-merged; 402 files changed, +19315/−16884.
3. **Never rebase or force-push the fork's published `master`, and never cherry-pick upstream
   commits in place of a merge.** Upstream *does* rewrite feature-branch commits on landing (PR
   commit SHAs are not stable until merged), so anything contributed upstream and also kept locally
   will exist under two SHAs — merges are the only safe integration path in both directions.
4. **Sync only from `upstream/master`; treat `dev` as a signal, not a source.** `dev` is unprotected
   and can be rewritten at any time. Optionally watch `dev` as an early warning of work heading to
   `master` (as upstream's own CONTRIBUTING advises contributors).
5. **Pin the sync target by SHA in the runbook** — upstream has no tags, no releases, and no
   version numbers to reference.
6. **The fork carries the whole verification burden.** With no upstream CI and no automated checks
   on `master` or PRs, a merged upstream change can only be validated by the fork's own build and
   tests. If contributing anything upstream, the bar is the one in the upstream `CONTRIBUTING.md`
   and PR template (linked Issue, scope confirmation, exact verification commands, stated
   limitations), and acceptance is at the maintainer's discretion — fork work should not depend on
   it.
