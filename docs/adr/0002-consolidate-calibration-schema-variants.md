# 0002 — One calibration schema with `schema_version`, not multiple variants

## Status

Accepted.

## Context

Before REQ-4/REQ-6, the project had accumulated several parallel
calibration schema/format variants: a pre-Pydantic JSON Schema
(`poker_table_calibration_schema_v1.json`) plus its Markdown
documentation, a corresponding runtime instance
(`poker_table_runtime_v1.json`) plus its Markdown documentation, and the
`v1`/`v2`/`v3` calibration instance files themselves, each with slightly
different structure as the format evolved ad hoc. CLAUDE.md's decision
table called this out as "Mehrere Schema-Varianten" and required
consolidation into one schema.

Multiple schema variants meant no single source of truth for what a valid
calibration file looks like, made validation inconsistent, and risked
silently loading stale or incompatible data.

## Decision

Replace all prior schema variants with exactly one pair of Pydantic v2
models: `poker_vision.calibration.authoring.CalibrationAuthoring` (the
editable, human-authored format) and
`poker_vision.calibration.runtime.CalibrationRuntime` (the precomputed,
loop-facing format produced by `calib compile`, REQ-9). Both carry a
`schema_version` field (REQ-4); unknown fields are rejected rather than
silently ignored.

The old JSON Schema, its runtime sample, and their Markdown docs are
archived under `docs/archive/` (`schemas/poker_table_calibration_schema_v1.json`,
`runtime/poker_table_runtime_v1.json`,
`poker_table_calibration_schema_v1.md`, `poker_table_runtime_v1.md`) for
historical reference only — see
[`docs/archive/README.md`](../archive/README.md).

## Consequences

- Every module that loads calibration data goes through the two Pydantic
  models above; there is no second code path that parses the old JSON
  Schema format.
- A calibration file with an unrecognized `schema_version` or an unknown
  field fails to load (REQ-4/REQ-11) instead of degrading silently.
- Future schema changes bump `schema_version` on the existing models
  rather than introducing a parallel variant.
