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
5. Before opening a PR: run a local Codex review against the diff (see below).
6. Push and open a PR. Never push directly to `main`.
7. CI (`.github/workflows/ci.yml`) checks tests and build. That's the only
   automated gate — the content review is manual, done by the repo owner.

## Conventions
- Code comments: English, regardless of the language used elsewhere in the project.
- Commit messages: Conventional Commits format.
- No force-push to shared branches.
- No direct commits to `main`.

## Review before merge
Codex acts as a second, independent reviewer — run locally and manually,
not as an automated CI gate:

```bash
git diff main...HEAD > pr.diff
codex review --diff pr.diff --context PRD.md
```

This uses your ChatGPT subscription login (`codex login`), not an API key —
no per-token billing. Read the findings yourself; if something needs
fixing, describe it to Claude Code directly in your next message rather
than through any automated hand-off.

## Not allowed
- Direct pushes to `main`.
- Force-pushing shared branches.
- Merging PRs — the final merge is always done manually by the repo owner.
