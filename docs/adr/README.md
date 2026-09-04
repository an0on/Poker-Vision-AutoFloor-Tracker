# Architecture Decision Records

Lightweight ADRs (Context / Decision / Consequences) for the architecture
decisions in CLAUDE.md's "Bestehender Stand – Entscheidung" table, per
REQ-43 (`/PRD.md`). Each row of that table is covered by either an ADR here
or an entry in [`docs/archive/README.md`](../archive/README.md) — AC-27
accepts either.

| ADR | Decision table row(s) covered |
|---|---|
| [0001](0001-calibration-geometry-source-reference-photo.md) | `calibration/` v3 landscape (Behalten → Pydantic), `calibration/` v1/v2 (Verwerfen) |
| [0002](0002-consolidate-calibration-schema-variants.md) | Mehrere Schema-Varianten (Ändern → ein Schema mit `schema_version`) |
| [0003](0003-remove-overlays-and-generator-scripts.md) | `overlays/` PNG, `overlays/` SVG, `build_rotated_raster*.py`, `build_landscape_calibration_instance*.py`, `build_runtime_json_v1.py` (all Verwerfen/Ändern) |
| [0004](0004-adopt-adrs-and-consolidate-notes-into-docs.md) | `docs/`, `notes/` (Behalten → `docs/` mit ADRs) |

New ADRs are numbered sequentially and are never edited to reverse a past
decision — a changed decision gets a new ADR that supersedes the old one and
says so explicitly.
