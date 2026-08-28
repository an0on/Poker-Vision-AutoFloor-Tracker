# CLAUDE.md

## Project context
<!-- Fill in: what this project does, stack, architecture. -->

## Source of truth
Every implementation task must satisfy `/PRD.md` in this repo. Read it before
starting work. If a requirement is ambiguous, ask before implementing —
don't guess and don't expand scope beyond what's written there.

## Workflow
1. Create a feature branch from `main`: `feat/<short-description>` or `fix/<short-description>`.
2. Implement against the PRD.
3. Run tests locally before committing.
4. Commit using Conventional Commits (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`).
5. Push and open a PR. Never push directly to `main`.
6. Wait for the CI + Codex review gate (see `.github/workflows/ci-and-codex-review.yml`).
   If Codex posts a failing verdict, address only the listed findings and push
   again to the same branch — see "Automated fix loop" below.

## Conventions
- Code comments: English, regardless of the language used elsewhere in the project.
- Commit messages: Conventional Commits format.
- No force-push to shared branches.
- No direct commits to `main`.

## Automated fix loop
Codex reviews every PR against `PRD.md`. On failure it posts a PR comment
containing a `<!-- codex-verdict -->` marker followed by a JSON block:

```json
{
  "status": "fail",
  "findings": [
    { "file": "src/foo.ts", "line": 42, "issue": "...", "prd_ref": "REQ-3" }
  ]
}
```

When `.github/workflows/claude-autofix.yml` invokes you against such a comment:
- Parse the `findings` array only — don't touch anything not listed.
- Fix each finding against the referenced `prd_ref` requirement.
- Commit as `fix(review): address finding [iteration N/5]`, where N is one
  higher than the previous iteration commit on this branch.
- Never attempt more than 5 correction iterations on one PR. If the 5th
  attempt still fails, stop — a human takes over from there. Don't retry
  past the limit even if you believe you're close.

## Not allowed
- Direct pushes to `main`.
- Force-pushing shared branches.
- Merging PRs — the final merge is always done manually by the repo owner,
  never automated, even when all checks pass.
