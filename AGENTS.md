# AGENTS.md — Abschlussprojekt Poker Vision

## Source of truth
Maßgeblich ist `/PRD.md`, ergänzt durch den Workflow in `/CLAUDE.md`. Bei
Widerspruch zwischen diesem Dokument und `PRD.md` gilt `PRD.md`. Geplante,
aber noch nicht implementierte Erweiterungen stehen in
`docs/future-features.md`, nicht hier.

## Ziel (v0.1 MVP)
Lokales CV-System (Top-Down-Kamera, 1 Tisch) mit Platzhalter-Detections
(Mock-Erkennung statt eigenem Modell), das end-to-end validiert:
- Seat Occupancy (Chip-Präsenz in `chip_zone`)
- Dealer-Button-Position → Seat
- Board State: Flop / Turn / River (Kartenanzahl in `board_zone`)
- Hand-Start/-Ende über Board leer ↔ nicht-leer

## Harte Annahmen
- Zielplattform zuerst: **MacBook Pro M4 Max, 36 GB RAM**; Inferenz auf
  `cpu`/`mps`, `cuda` in v0.1 ein reservierter, abgelehnter Wert (REQ-3)
- Primärformat: **Querformat**
- Kein End-to-End-Monolith; **modulare Pipeline** unter `src/poker_vision/`
  (REQ-1)

## Kalibrierung
Ein einziges Kalibrierungsschema (Pydantic v2, REQ-4/REQ-6/REQ-7):
`CalibrationAuthoring`/`CalibrationRuntime` in
`src/poker_vision/calibration/`. Enthält Kameraintrinsics + Distortion,
Homographie Pixel→Tischebene, Tischmaße/-einheit sowie Zonen
(`player_area`, `chip_zone` je Seat, global `board_zone` und
`dealer_area`) als Polygone in Tischkoordinaten.

Die älteren Kalibrierungs-JSONs (`calibration/*_v1_*.json`, `*_v2_*.json`,
`*_v3_*.json`, `calibration/runtime/poker_table_runtime_v1.json`) sind
Referenzgeometrie für die v3-landscape-Migration in ein neues Schema
(REQ-6, REQ-9) — sie sind kein Format, das die neuen
`CalibrationAuthoring`/`CalibrationRuntime`-Schemas direkt einlesen. Die
Migration selbst ist offene Arbeit (REQ-6/REQ-9/REQ-10), nicht Teil von
REQ-4.

## Erkennungslogik (v0.1)
### Seat Occupancy
`occupied`, wenn ≥ 1 stabiler `chip`-Track in der `chip_zone` liegt
(REQ-29). Kein zusätzliches Signal in v0.1 — siehe
`docs/future-features.md`.

### Dealer Button
- Detection-Klasse `dealer_button`, direkt erkannt
- Zuordnung per Point-in-Polygon (`dealer_area`) bzw. Nearest-Seat-Fallback
  unter Schwellwert (REQ-26, REQ-27)

### Board
- eine einzige `board_zone`
- 3 Karten => Flop, 4 Karten => Turn, 5 Karten => River (REQ-31)
- nur monoton steigende Übergänge innerhalb einer Hand

### Hand-Verlauf
`hand_started`/`hand_ended` ausschließlich über Board leer ↔ nicht-leer
(REQ-32, PRD.md Annahme A2). Keine granularere State Machine (kein
preflop/showdown/waiting/closed) in v0.1 — siehe
`docs/future-features.md`.

## Modellrichtung (v0.1)
Platzhalter-Erkennung ohne Training: `mock`-Detector aus Skriptdatei,
ArUco-Markern, oder einem vortrainierten COCO-Standardmodell mit
Klassen-Mapping (REQ-18–REQ-20). Ein eigenes YOLO-Modell (`yolo`-Detector,
CoreML-Export, MPS-Inferenz) existiert nur als Interface-Stub und ist
explizit out of scope für v0.1 (PRD.md).

## Wichtige Hinweise
- `chip_zone`/`player_area` in den alten Kalibrierungs-JSONs sind noch
  abgeleitet, nicht final manuell kalibriert; bei der Migration (REQ-6,
  REQ-9) gegen REQ-11 neu zu validieren.
- Geplante, aber noch nicht implementierte Funktionen stehen in
  `docs/future-features.md`, nicht in diesem Dokument.

## Wichtigste Dateien
- `/PRD.md` — Requirements und Acceptance Criteria (Source of truth)
- `/CLAUDE.md` — Workflow
- `docs/future-features.md` — Post-v0.1-Erweiterungen
- `src/poker_vision/calibration/` — Authoring-/Runtime-Schema (REQ-4)
- `calibration/poker_table_calibration_instance_current_table_v3_landscape.json`
  — Referenzgeometrie für die REQ-6-Migration
