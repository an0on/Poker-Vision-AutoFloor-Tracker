# 0001 — Canonical table geometry comes from a fresh reference photo, not the archived v3 instance

## Status

Accepted (supersedes the original plan recorded in CLAUDE.md's decision
table).

## Context

CLAUDE.md's decision table, written before REQ-6/REQ-7 implementation
started, planned:

- `calibration/` v3 landscape → **Behalten**, migrate into the new
  Pydantic schema as the canonical geometry ("kanonische, bestätigte
  Tischgeometrie").
- `calibration/` v1/v2 → **Verwerfen**, moved to `docs/archive/`.

Once REQ-6/REQ-7 work started, `v3_landscape` turned out to not actually
match the physical table: its 10-seat layout differs from the real table's
seat division, and it carries no real-world units, camera model, or
homography — every field the `CalibrationAuthoring` schema needs. It was a
hand-typed/estimated instance, not a measurement of the actual table.

## Decision

Discard `v1`, `v2`, and `v3` alike as the geometry source. Author the
canonical geometry fresh from a reference photo of the physical table,
via feature-clicks (`calib mark-zones`, REQ-10a) rather than reusing or
correcting the old `v3` file. All three prior instances move to
`docs/archive/calibration/` for historical reference only; none of them
are loaded by `src/poker_vision/`.

## Consequences

- The "Behalten → migrate v3" row in CLAUDE.md's decision table is
  superseded by this ADR: v3 was not migrated, it was archived alongside
  v1/v2.
- Canonical geometry provenance is documented in
  [`docs/archive/README.md`](../archive/README.md), which explains why
  each archived file was superseded.
- `calib mark-zones` (REQ-10a) and `calib learn-table` (REQ-10b) are the
  only paths that produce or reproduce table geometry going forward; no
  code path re-derives geometry from the old pixel-space files.
