# 0004 — Adopt ADRs under `docs/`, fold `notes/` into `docs/`

## Status

Accepted.

## Context

CLAUDE.md's decision table planned to keep `docs/`/`notes/` and
"structure the decision history" under `docs/` with ADRs, but this was
never carried out (REQ-43): no `docs/adr/` existed, and `notes/` remained
a separate top-level directory with a single research file
(`tournament_director_integration_notes.md`) rather than living under
`docs/`. AC-27 requires an ADR or `docs/archive/README` entry for every
decision-table row, including this one, plus the Phase-0 result on
record.

## Decision

- Introduce `docs/adr/` for lightweight (Context/Decision/Consequences)
  ADRs, starting with this file and
  [0001](0001-calibration-geometry-source-reference-photo.md)–[0003](0003-remove-overlays-and-generator-scripts.md).
  New architecture decisions get a new numbered ADR here going forward,
  rather than being described only in commit messages or PR bodies.
- Move `notes/tournament_director_integration_notes.md` to
  `docs/notes/tournament_director_integration_notes.md`; remove the
  top-level `notes/` directory. Content is unchanged — it is external
  research (The Tournament Director integration), not project-decision
  history, so it lives under `docs/notes/` rather than `docs/adr/`.
- The Phase-0 result (image + Freigabevermerk) already lives under
  `docs/phase0/` (see `docs/phase0/README.md` and the
  "Phase-0-Freigabe" section of `/PRD.md`) — no change needed there.

## Consequences

- `docs/` is now the single top-level location for project documentation:
  `docs/adr/` (decisions), `docs/archive/` (superseded artifacts),
  `docs/notes/` (research notes), `docs/phase0/` (Phase-0 record), plus
  `docs/future-features.md` and `docs/abschlussprojekt_poker_vision_handover.md`.
- There is no more top-level `notes/` directory.
- `docs/abschlussprojekt_poker_vision_handover.md` is a dated snapshot of
  project state at handover time and is left untouched by this move —
  it is a historical record, not living documentation that needs to
  track the current directory layout.
