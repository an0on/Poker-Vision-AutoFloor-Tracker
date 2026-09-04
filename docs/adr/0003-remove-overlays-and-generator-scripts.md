# 0003 — Remove `overlays/` renderings and their generator scripts

## Status

Accepted. Implemented in [PR #47](https://github.com/an0on/Poker-Vision-AutoFloor-Tracker/pull/47).

## Context

CLAUDE.md's decision table called for five related removals, all tied to
the same superseded `v1`/`v2`/`v3` pixel-space calibration instances
(see [0001](0001-calibration-geometry-source-reference-photo.md)):

| Item | Table decision |
|---|---|
| `overlays/*.png` | Verwerfen (generated artifact; runtime overlay rendering exists since REQ-37) |
| `overlays/*.svg` | Behalten nur bei physischer Nutzung (→ `assets/`), sonst verwerfen |
| `build_rotated_raster*.py` | Verwerfen (rotation is a special case of the homography, REQ-8) |
| `build_landscape_calibration_instance*.py` | Ändern → one calibration CLI (REQ-9/REQ-10) |
| `build_runtime_json_v1.py` | Ändern → `calib compile` (REQ-9) |

None of the SVGs were ever used as a physical print template, and by the
time REQ-9/REQ-10/REQ-37 landed, every generator script's logic already
had a superseding home in the source tree.

## Decision

Delete `overlays/calibration_instances/`, `overlays/raster_drafts/`, and
`overlays/scripts/` outright — not archived, unlike the actual measured
calibration JSON in `docs/archive/` — since these were draft renderings
and one-off scripts with no reference value once their logic and
successor geometry already exist in the source tree. The removal and its
full reasoning per file are documented in the "`overlays/` (removed, not
archived here)" section of
[`docs/archive/README.md`](../archive/README.md).

## Consequences

- No `overlays/` directory exists in the source tree; runtime overlay
  rendering (REQ-37) and `calib compile`/`calib create`/`calib edit`
  (REQ-9/REQ-10) are the only paths that produce calibration renderings
  or runtime JSON.
- `pyproject.toml`'s ruff `extend-exclude` no longer references
  `overlays/scripts` (removed alongside the directory).
- This ADR is a pointer to `docs/archive/README.md` rather than a
  duplicate of its content — see that file for the per-item reasoning.
