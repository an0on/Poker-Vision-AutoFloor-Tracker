#!/usr/bin/env bash
# Configure branch protection for `main` via the GitHub CLI.
# Requires: `gh auth login` as a repo admin.
#
# Merge stays manual by design: auto-merge is an opt-in PR-level setting
# that's off by default, so simply never enabling it on a PR keeps the
# final merge in your hands even once every check is green.

set -euo pipefail

REPO="owner/repo"   # adjust
BRANCH="main"

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${REPO}/branches/${BRANCH}/protection" \
  -f required_status_checks.strict=true \
  -f required_status_checks.contexts[]="test" \
  -f required_status_checks.contexts[]="codex-review" \
  -f enforce_admins=true \
  -f required_pull_request_reviews.required_approving_review_count=1 \
  -f restrictions=null \
  -f allow_force_pushes=false \
  -f allow_deletions=false
