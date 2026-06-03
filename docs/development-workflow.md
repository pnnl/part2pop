# part2pop Development Workflow (Authoritative)

This document defines the required day-to-day development workflow for `part2pop`.
Follow it for all code, test, documentation, and release work unless a maintainer explicitly approves an exception.

## Core Rules (Non-Negotiable)

1. **One task = one branch.**
2. **One branch = one worktree.**
3. **No in-repo sandboxes.** Never create ad hoc scratch directories inside this repository.
4. **No copied repos inside the repo.** Never duplicate `part2pop` into subfolders.
5. **No giant patches.** Keep changes small, reviewable, and task-scoped.
6. **No blind staging.** Do not use broad staging patterns without review (for example `git add .`).
7. **No generated artifacts.** Do not commit generated files unless explicitly required by maintainers.
8. **AI/Cline command guardrail.** Cline/AI assistants **must not run commands unless explicitly authorized** by the user for that session/task.
9. **Test realism requirement.** Tests must exercise real `src/part2pop` code and realistic public API behavior.
10. **No fake population structures.** Do not add fake/dummy “population-like” test structures that bypass real modeling behavior.

---

## Standard Branch + Worktree Model

Default base for new work is:

- `origin/main`

Branch naming by work type:

- Release task: `release/<task>`
- Documentation task: `docs/<task>`

Required worktree path pattern:

- `../part2pop-<task>`

### Example: documentation task

- Branch: `docs/development-workflow`
- Worktree path: `../part2pop-development-workflow`
- Base: `origin/main`

### Example: release task

- Branch: `release/v0.6.0-rc1`
- Worktree path: `../part2pop-v0.6.0-rc1`
- Base: `origin/main`

---

## Git Safety Checks (Required)

Run these checks before editing, before staging, and before committing.

### 1) Before editing files

Confirm you are in the correct worktree and branch, and start clean.

```bash
git rev-parse --show-toplevel
git branch --show-current
git status --short
```

Expected:

- Repository root is the intended worktree path (`../part2pop-<task>`).
- Branch matches task branch (`docs/<task>` or `release/<task>`).
- No unexpected modified/untracked files.

### 2) Before staging

Inspect exactly what changed.

```bash
git status --short
git diff
```

Stage explicitly by file path only (no blind staging):

```bash
git add docs/development-workflow.md
```

### 3) Before committing

Verify staged content and commit scope.

```bash
git diff --staged
git status
```

Confirm:

- Staged diff matches task intent.
- No unrelated files are staged.
- Patch size is reviewable (not giant).

---

## Worktree Lifecycle

Use worktrees to isolate tasks cleanly.

### Create a task worktree

Documentation task example:

```bash
git fetch origin
git worktree add -b docs/<task> ../part2pop-<task> origin/main
```

Release task example:

```bash
git fetch origin
git worktree add -b release/<task> ../part2pop-<task> origin/main
```

### List worktrees

```bash
git worktree list
```

### Remove worktree after merge/close

From the primary repository checkout (not from inside the worktree being removed):

```bash
git worktree remove ../part2pop-<task>
git branch -d docs/<task>    # or release/<task>, after merge
```

If branch is not merged and must be discarded intentionally:

```bash
git branch -D docs/<task>
```

---

## Patch Size and Commit Hygiene

- Keep each branch focused on a single task outcome.
- Prefer multiple small commits over one large, mixed commit.
- Separate refactor from behavior change where possible.
- Do not mix release mechanics, docs edits, and functional source changes unless the task explicitly requires it.

---

## Test Policy for part2pop

When adding or modifying tests:

- Use real code paths in `src/part2pop`.
- Validate realistic behavior through the public API.
- Use realistic fixtures/data patterns aligned with scientific intent.
- Avoid fake/dummy population-like scaffolding that hides integration assumptions.

Do **not**:

- Mock away the core behavior under test to the point the test is meaningless.
- Introduce toy structures that resemble populations but are not valid for real usage patterns.

---

## Failure Protocol (Mandatory)

If you hit unexpected status, diffs, conflicts, or test failures, stop and triage before continuing.

### A) Unexpected `git status` output

1. Stop editing.
2. Inspect:
   - `git status --short`
   - `git diff`
3. If unrelated files are modified, isolate and revert only after confirmation.
4. Continue only when task scope is clean and understood.

### B) Unexpected diff content

1. Stop staging/committing.
2. Review file-by-file.
3. Split unrelated changes into separate branches/worktrees.
4. Re-check with `git diff` and `git diff --staged`.

### C) Merge/rebase conflicts

1. Pause feature work.
2. Resolve conflicts deliberately, preserving intended scientific/public API behavior.
3. Re-run status/diff checks.
4. Request maintainer review when conflict resolution is non-trivial.

### D) Test failures

1. Do not bypass or silence failures.
2. Determine whether failure is:
   - regression introduced by the branch,
   - stale assumptions in tests,
   - environment/data issue.
3. Fix root cause and keep tests realistic to real `part2pop` usage.
4. Re-verify clean status and scoped diffs before commit.

---

## AI Assistant Operating Constraint

For this repository workflow, AI assistants (including Cline) must follow a strict authorization model:

- No command execution without explicit user authorization.
- Documentation-only tasks must not modify source or tests.
- If authorization is unclear, ask first and wait.

This protects repository integrity and keeps human reviewers in control of side effects.
