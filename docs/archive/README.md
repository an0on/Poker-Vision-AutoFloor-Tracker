# Archive

Superseded calibration schema versions, kept for history (REQ-6, REQ-43).

- `calibration/poker_table_calibration_instance_current_table_v1_landscape.json`,
  `..._v2_landscape.json`, `..._v3_landscape.json` — the three prior pixel-space
  calibration instances. `_v3_landscape.json` in particular was originally meant
  to become the canonical geometry (see `CLAUDE.md`'s decision table), but its
  seat layout (10 seats) doesn't match the actual physical table's design
  (verified via feature-clicked measurement of `calibration/reference/`,
  see REQ-10a) and it carries no real-world units, camera model, or homography
  — every field the current `CalibrationAuthoring` schema (REQ-6/REQ-7) needs.
  Superseded by the reference-photo-derived geometry instead.
- `calibration/schemas/poker_table_calibration_schema_v1.json`,
  `calibration/runtime/poker_table_runtime_v1.json` — the v1 JSON Schema and a
  sample runtime instance for the old (pre-Pydantic) format.
- `poker_table_calibration_schema_v1.md`, `poker_table_runtime_v1.md` — docs
  for the same v1 format.

None of this is loaded by `src/poker_vision/` — the only calibration schema in
the source tree is `poker_vision.calibration.authoring.CalibrationAuthoring` /
`poker_vision.calibration.runtime.CalibrationRuntime`.

## `overlays/` (removed, not archived here)

Per CLAUDE.md's decision table (REQ-8, REQ-12): `overlays/calibration_instances/`
and `overlays/raster_drafts/` were rendered PNG/SVG overlays for the same
superseded `v1`/`v2`/`v3` pixel-space calibration instances archived above —
not source data in their own right, and never claimed anywhere as a physical
print template (the one case REQ-12 would have kept an SVG, moved to
`assets/`). `overlays/scripts/build_landscape_calibration_instance*.py`,
`build_rotated_raster_v2.py`, and `build_runtime_json_v1.py` are the
generator scripts behind them; their logic already lives in
`poker_vision.calibration.cli`/`.compile` (REQ-9/REQ-10's `calib compile`,
`calib create`/`edit`), and `build_rotated_raster*.py`'s rotation handling
is now just the homography's own rotation component (REQ-8) — both
decisions the table already called for ("Verwerfen" / "Ändern → ein
Kalibrierungs-CLI" / "Ändern → `calib compile`"), just not yet acted on.
Deleted outright rather than moved here: unlike the actual measured `v1`–
`v3` calibration JSON (kept above for its historical geometry), these were
draft renderings and one-off scripts with no reference value of their own
once their logic and successor geometry already exist in the source tree.
