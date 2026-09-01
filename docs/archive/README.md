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
