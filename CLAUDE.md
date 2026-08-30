# CLAUDE.md

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

## Codex review loop
Codex acts as a second, independent reviewer — run locally and manually,
not as an automated CI gate:

```bash
codex review --base main
```

This uses your ChatGPT subscription login (`codex login`), not an API key —
no per-token billing. Read the findings yourself; if something needs
fixing, describe it to Claude Code directly in your next message rather
than through any automated hand-off.

Proceed through implementation → tests → commit → codex review loop
automatically, without pausing to ask permission at each step. Only stop
and ask when: (a) a scope decision is genuinely ambiguous, as described
above, or (b) you're ready to push and open the PR — at that point, ask
once, with the summary of what was found/fixed.

## Not allowed
- Direct pushes to `main`.
- Force-pushing shared branches.
- Merging PRs — the final merge is always done manually by the repo owner.

## Project context

- **Ziel:** Lokales CV-System (Top-Down-Kamera, 1 Tisch), das Chips, Karten und Dealer-Button erkennt, Sitzen zuordnet und Seat Occupancy + Handverlauf in Echtzeit als Events bereitstellt (späterer Konsument: "The Tournament Director").
- **Kernprinzip:** Starre Kalibrierung (JSON, v3 landscape als kanonische Geometrie) + dynamische Erkennung (YOLO). Alle Geometrie-Entscheidungen im **Tisch-Koordinatensystem** (nach Homographie), nie im rohen Pixelraum.
- **Plattform:** MacBook Pro M4 Max (36 GB), Apple Silicon. Inferenz auf **CPU/MPS**, **CUDA-Aufrufe sind verboten**. Windows + TD-Parallelbetrieb ist späteres Ausbauziel → Code bleibt geräte-agnostisch (Device-Auswahl zentral, keine Backend-spezifischen Pfade).
- **Kamera:** iPhone 15 Pro Max via macOS Continuity Camera, fixe Top-Down-Montage, Landscape.
- **Scope (MVP):**
  - Chips: nur **Präsenz** in `player_area`/`chip_zone` → Seat Occupancy. Keine Denomination, kein Stack-Counting.
  - Karten: nur **Anzahl** in der zentralen `board_zone` → Street (3 = Flop, 4 = Turn, 5 = River). Rang/Farbe ignoriert.
  - Dealer Button: Position → Seat.
- **Modell:** Eigener Datensatz (echte Chips, Karten, der konkrete Dealer Button), iteratives Training eines maßgeschneiderten YOLO. **v0.1** nutzt nur Platzhalter, um die Distanz-/Zuordnungsmathematik zu testen.

## Architecture

### Stack
- **Python 3.11+**, **uv** (Env/Pakete), **ruff**, **pytest**.
- **Ultralytics YOLO (v8/11)** für Training + Detection, integriertes **ByteTrack**. Training auf MPS; Runtime-Inferenz per **CoreML-Export** (Neural Engine) als Standardpfad, PyTorch-MPS als Fallback. **Kein** onnxruntime-gpu/TensorRT im Projekt; ONNX nur als portables Austauschformat für den späteren Windows-Pfad.
- **OpenCV** (AVFoundation-Backend für Continuity Camera, Undistortion, Homographie, Overlay), **NumPy**.
- **Pydantic v2** für alle Schemas (Kalibrierung, Config, Events), ein Schema mit `schema_version`.
- **FastAPI + uvicorn**: WebSocket-Event-Stream, REST-Status, MJPEG-Debug-Stream; statische HTML-Debug-Seite, kein Frontend-Framework.
- **Kein** Broker, keine DB, kein Docker (Kamera-Passthrough auf macOS unpraktikabel). Events als JSONL auf Platte (replaybar), State in-memory.

### Modulaufteilung (`src/poker_vision/`)
- `capture/` – Quellen: `continuity` (OpenCV/AVFoundation, Geräte-Index + Auflösungs-Cap, z. B. 1920×1080 für Inferenz), `video_file`, `image_dir`. Replay ist Pflicht für Tests.
- `calibration/` – Pydantic-Schema, Loader, Undistortion + Homographie, Zonen-Polygone (`player_area`/`chip_zone` je Seat, `board_zone`, Dealer-Bereich), CLI zum Erstellen/Editieren, `compile` (Authoring-JSON → Runtime-JSON).
- `detection/` – `Detector`-Interface mit zwei Implementierungen: `yolo` (CoreML/MPS) und **`mock`** (Platzhalter für v0.1: liefert Detections aus Datei/Skript oder aus ArUco-Markern statt echter Objekte). Klassenset MVP: `chip`, `card`, `dealer_button`. Output ausschließlich in Tischkoordinaten.
- `tracking/` – Track-IDs, Hysterese/Debounce (Objekt gilt erst nach N Frames als da/weg).
- `assignment/` – rein geometrisch: Point-in-Polygon je Zone, Nearest-Seat mit Distanzschwelle. Keine Spiellogik. **Das ist der Teil, den v0.1 mit `mock` durchtestet.**
- `state/` – State-Machine: Occupancy je Seat (Chips in Zone), Dealer-Seat, Street aus Kartenanzahl in `board_zone`, `hand_started/ended` (Board leer → nicht leer → leer). Emittiert typisierte Events.
- `export/` – Adapter: `websocket`, `jsonl`, `tournament_director` (Stub bis Windows-Phase).
- `debug/` – Overlay aus Kalibrierung + Live-State, MJPEG-Endpoint.
- `tools/` – Dataset-Pipeline: Frame-Sampling aus Continuity/Video, Export für Annotation (CVAT/Label Studio, extern), Trainings-Skript (MPS), CoreML-Export, Modell-Versionierung (`models/vX/` + Metriken-JSON).

### Datenfluss
1. `capture` → Frame (aufgelöst auf Inferenz-Größe)  
2. `calibration` → Undistort; Detection im Originalbild, danach Box-Mittelpunkte per Homographie in Tischebene (Warp-first nur nach A/B-Messung)  
3. `detection` (`mock` in v0.1, `yolo` ab v0.2) → Detections  
4. `tracking` → stabile Tracks mit Hysterese  
5. `assignment` → Track ↔ Seat/Zone  
6. `state` → Events (`seat_occupied/vacated`, `dealer_moved`, `street_changed`, `hand_started/ended`)  
7. `export` + `debug`  
- Stufen 3–6 laufen ohne Kamera per Replay/Mock in Tests.

### Bestehender Stand – Entscheidung
| Teil | Entscheidung | Begründung |
|---|---|---|
| `calibration/` v3 landscape | **Behalten** → ins Pydantic-Schema migrieren | Kanonische, bestätigte Tischgeometrie. |
| `calibration/` v1/v2 | **Verwerfen** (→ `docs/archive/`) | Überholt, drei Stände verwirren. |
| Mehrere Schema-Varianten | **Ändern** → ein Schema mit `schema_version` | Eine Quelle der Wahrheit. |
| `overlays/` PNG | **Verwerfen** (generiertes Artefakt) | Wird zur Laufzeit gerendert. |
| `overlays/` SVG | **Behalten nur bei physischer Nutzung** (→ `assets/`), sonst verwerfen | Nur als Druckvorlage Eigenwert. |
| `build_rotated_raster*.py` | **Verwerfen** | Rotation = Spezialfall der Homographie. |
| `build_landscape_calibration_instance*.py` | **Ändern** → ein Kalibrierungs-CLI | Logik wertvoll, Varianten-Wildwuchs nicht. |
| `build_runtime_json_v1.py` | **Ändern** → `calib compile` in `tools/` | Authoring/Runtime-Trennung ist richtig. |
| `docs/`, `notes/` | **Behalten** → `docs/` mit ADRs | Entscheidungshistorie strukturiert erhalten. |

### Externe Abhängigkeiten
- Runtime: `ultralytics`, `opencv-python`, `numpy`, `pydantic`, `fastapi`, `uvicorn`, `coremltools` (Export), `torch` (MPS-Build, nur Training/Fallback).
- Dev: `pytest`, `ruff`, `uv`; Annotation extern (CVAT oder Label Studio).
- Optional: `supervision` (Zonen-/Overlay-Utilities).

### Offene Risiken / Trade-offs
- **Continuity Camera:** Auto-Fokus/-Belichtung/Center-Stage des iPhones können Bild verschieben → Center Stage aus, Belichtung fixieren; Kalibrierung gegen Referenzmarker regelmäßig validieren. Für Windows später Kamerawechsel → `capture` muss austauschbar bleiben.
- **Datensatz:** Größtes Risiko bleibt Qualität/Menge (Ziel initial ~300–500 annotierte Frames unter realer Beleuchtung). v0.1-Mock verhindert, dass Geometrie-Bugs mit Modell-Bugs vermischt werden.
- **CoreML-Export:** Nicht jede Ultralytics-Version exportiert sauber (NMS-Handling). MPS-Fallback fest einplanen.
- **Okklusion durch Hände:** Hysterese Pflicht; optional Klasse `hand` → State einfrieren.
- **Street-Erkennung nur über Kartenanzahl:** Misdetections (z. B. 2 statt 3 Karten kurz sichtbar) → nur monoton steigende Übergänge innerhalb einer Hand akzeptieren, Rücksprung erst bei leerem Board.
- **Windows/TD-Phase:** Geräte-Auswahl (`cpu`/`mps`/später `cuda`) ausschließlich über Config; TD-Schnittstelle unbekannt → Adapter-Stub.
- **Datenschutz:** Nur Events persistieren, keine Frames; Hinweis am Tisch.
