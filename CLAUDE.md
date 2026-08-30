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
- `debug/` – Overlay aus Kalibrierung + Live-State, MJPEG-Endpoint; nimmt Live-Frames über `LatestFrameHub` vom Runner entgegen (siehe `### Pipeline-Runner`).
- `tools/` – Dataset-Pipeline: Frame-Sampling aus Continuity/Video, Export für Annotation (CVAT/Label Studio, extern), Trainings-Skript (MPS), CoreML-Export, Modell-Versionierung (`models/vX/` + Metriken-JSON).
- `runner/` – Pipeline-Orchestrierung: Frame-Loop, Lifecycle (Start/Stop/Signale), CLI-Einstiegspunkt, `FrameContext` (siehe `### Pipeline-Runner`). Einzige Abhängigkeitsrichtung: `runner` → alle Stufen; keine Stufe importiert `runner`.

### Datenfluss
1. `capture` → Frame (aufgelöst auf Inferenz-Größe)  
2. `calibration` → Undistort; Detection im Originalbild, danach Box-Mittelpunkte per Homographie in Tischebene (Warp-first nur nach A/B-Messung)  
3. `detection` (`mock` in v0.1, `yolo` ab v0.2) → Detections  
4. `tracking` → stabile Tracks mit Hysterese  
5. `assignment` → Track ↔ Seat/Zone  
6. `state` → Events (`seat_occupied/vacated`, `dealer_moved`, `street_changed`, `hand_started/ended`)  
7. `export` + `debug`  
- Stufen 3–6 laufen ohne Kamera per Replay/Mock in Tests.
- Die konkrete Laufzeit-Reihenfolge inkl. Fehlerpolitik orchestriert der Runner
  (siehe `### Pipeline-Runner`): Schritt 2 oben beschreibt die *Anwendung*
  vorberechneter Kalibrierungs-Matrizen innerhalb der `detection`-Stufe
  (REQ-17); das *Laden/Validieren* der Kalibrierung selbst ist dort explizit
  kein Per-Frame-Schritt, sondern läuft einmalig vor Loop-Start.

### Pipeline-Runner

#### Einordnung
- Neues eigenständiges Paket `src/poker_vision/runner/` — kein `pipeline.py` im Root.
  - Begründung: Runner umfasst mehrere Concerns (Loop, Lifecycle, CLI, Frame-Kontext);
    ein Paket hält das konsistent mit der bestehenden Stufen-Struktur.
- Dateien: `loop.py` (Orchestrierung), `lifecycle.py` (Start/Stop/Signale),
  `cli.py` (Einstiegspunkt), `context.py` (FrameContext).
- Abhängigkeitsrichtung strikt einseitig: runner → alle Stufen.
  Keine Stufe importiert runner; bestehende Module bleiben runner-agnostisch
  und unverändert (Ausnahme: debug bekommt eine Publish-Schnittstelle, s. u.).
- `__main__.py` / Console-Script `poker-vision` delegiert an `runner.cli`.

#### Threading-Modell (Annahme, bewusst konservativ)
- Pipeline-Loop läuft single-threaded und synchron: deterministisch, einfach
  testbar, keine Race-Conditions in der State Machine, ausreichend für MVP.
- Nebenläufig nur: MjpegDebugServer (eigener Thread) und ggf.
  WebSocket-Export (bereits im ExportManager gekapselt).

#### Frame-Iteration (Reihenfolge pro Frame)
1. capture: Frame holen
2. detection → tracking → assignment → state (Kernkette)
3. export: State-Events/Snapshot übergeben (fehlerisoliert, nie blockierend)
4. debug: Frame + Snapshot in FrameHub publizieren (latest-wins, nie blockierend)
- calibration ist KEIN Per-Frame-Schritt: einmalig beim Start laden + validieren,
  Fail-fast bei Fehler (Exit ≠ 0 vor Loop-Start).
- `FrameContext` (in `context.py`) wird vom Loop pro Frame intern erzeugt und
  nach jeder Stufe fortgeschrieben: frame_id, timestamp, Raw-Frame,
  Detections, Tracks, Zonen-Zuordnung, State-Snapshot, Stufen-Fehlerliste.
  Der Loop ruft jede Stufe weiterhin mit deren bestehender, typisierter
  Signatur auf und trägt das Ergebnis in den `FrameContext` ein — Stufen
  erhalten oder importieren `FrameContext` selbst nicht (sonst Verletzung
  der `runner → Stufen`-Abhängigkeitsrichtung aus `#### Einordnung`).
  Einheitliche interne Übergabe zwischen Loop-Schritten statt loser
  Parameter, keine Änderung an Stufen-APIs.

#### Fehlerpolitik je Stufe
- capture:
  - EOF bei video_file/image_dir → geordneter Shutdown, Exit-Code 0
    (normales Laufzeitende, kein Fehler).
  - Fehler bei Live-Quelle (continuity) → Retry mit Backoff; nach
    konfigurierbarem Timeout Abbruch mit Exit ≠ 0.
- detection/tracking/assignment/state (Kernkette):
  - Exception in einer dieser Stufen → gesamten Frame verwerfen, KEIN
    partielles Update. State Machine sieht nur vollständig verarbeitete
    Frames → Event-Sequence-Zähler und Hand-Lifecycle bleiben konsistent.
  - Voraussetzung dafür (Coding-Konvention, keine Copy-on-Write-/
    Transaktions-Maschinerie): jede Stufe berechnet ihr Update rein
    (Rückgabewert), ohne eigenen persistenten Zustand direkt zu mutieren
    (Tracking-Hysterese/Track-IDs, State-Machine-Zustand). Der Loop wendet
    die zurückgegebenen Updates erst an — committet sie in
    Pipeline-Reihenfolge —, nachdem die gesamte Kernkette für den Frame
    erfolgreich durchlaufen wurde, nicht bereits nach der einzelnen Stufe.
    Ein Fehler in einer späteren Stufe verhindert damit auch das Commit
    einer bereits erfolgreich berechneten früheren Stufe; nur so ist
    „kein partielles Update" oben tatsächlich über die gesamte Kernkette
    garantiert.
  - Fehler loggen + Fehlerzähler; nach N konsekutiven Fehlern (config,
    Default 30) Abbruch mit Exit ≠ 0. Einzelner erfolgreicher Frame
    resettet den Zähler.
- export: nutzt bestehende Fehlerisolation des ExportManager; Runner
  behandelt Export-Fehler nie als fatal.
- debug: best effort; Fehler beim Publizieren/Rendern beeinflussen den
  Loop nicht.

#### Backpressure / Pacing
- Live-Quelle: latest-frame-wins — ist die Verarbeitung langsamer als die
  Kamera, werden Frames gedroppt (immer aktuellster Frame).
- Umsetzung in `continuity`: interner Hintergrund-Thread liest kontinuierlich
  von der Kamera und schreibt in einen thread-sicheren Single-Slot-Puffer
  (dasselbe latest-wins-Pattern wie `LatestFrameHub` aus REQ-46, nicht
  `cv2.CAP_PROP_BUFFERSIZE` — dessen Verhalten ist backend-abhängig
  unzuverlässig). Der Puffer führt einen monoton steigenden Versions-/
  Sequenzzähler; `get_latest()` liefert jeden Frame genau einmal zurück
  (kein erneutes Ausliefern desselben Frames, falls der Loop schneller als
  die Kamera ist — verhindert doppelte `frame_id`s / doppelte
  Hysterese-Zählung). Der Loop selbst bleibt single-threaded und synchron:
  er ruft `get_latest()` auf diesem Puffer als kurzen blockierenden Wait
  mit Timeout ab (`threading.Condition`/`Event`, kein Busy-Spin, kein
  reines Sleep-Polling) statt `read()` direkt auf der Kamera; das Dropping
  passiert im Hintergrund-Thread der Capture-Implementierung, nicht im
  Loop. Gilt ausschließlich für `continuity`.
- Dateiquellen (`video_file`/`image_dir`): kein Drop, jeder Frame wird
  verarbeitet (deterministisch, wichtig für Regressionstests); kein
  Hintergrund-Thread, kein Versionszähler, unverändert synchroner
  Lesepfad. Kein Realtime-Pacing per Default; optionales `--realtime`-Flag
  als späteres Nice-to-have, nicht Teil der REQs.

#### capture ↔ debug: FrameHub
- Neues, thread-sicheres Single-Slot-Objekt `LatestFrameHub`
  (Ort: `runner/` oder `debug/`, Entscheidung: `debug/`, da es die
  Konsum-Schnittstelle des Debug-Servers ist):
  - `publish(frame, context_snapshot)` vom Loop (überschreibt, latest-wins).
  - `get_latest()` vom MjpegDebugServer.
  - Trägt denselben Versions-/Sequenzzähler wie der continuity-interne
    Puffer (gleiches Pattern), hier zur Vermeidung unnötigen erneuten
    Renderings: der per-Client-Streaming-Thread des MjpegDebugServers
    ruft `get_latest()` als kurzen blockierenden Wait mit Timeout auf eine
    neue Version auf, statt denselben Frame im Busy-Loop erneut zu
    rendern/encodieren. Blockiert wird dabei ausschließlich der jeweilige
    Debug-Client-Thread (läuft ohnehin eigenständig, siehe
    Threading-Modell) — nie der Loop oder `publish()` (siehe
    Fehlerpolitik: debug blockiert den Loop nie). Unabhängig davon: die
    blockierende Wait-mit-Timeout-Semantik aus Backpressure/Pacing für den
    continuity-Konsum durch den Loop betrifft einen separaten Puffer mit
    separatem Konsumenten.
- Overlay-Rendering bleibt im Debug-Server und passiert on-demand pro
  verbundenem Client: ohne Client keine Rendering-Kosten im System.
- Debug-Server wird vom Runner-Lifecycle gestartet/gestoppt (config-gesteuert
  aktivierbar wie die Export-Adapter), läuft nicht mehr standalone-only;
  der bestehende Standalone-Betrieb bleibt für isolierte Tests erhalten.

#### Lifecycle
- Start: Config laden (eine Datei, Pydantic-validiert) → Kalibrierung
  laden/validieren (fail-fast) → Stufen konstruieren → Debug-Server starten
  (falls aktiv) → Loop starten.
- Stop: SIGINT/SIGTERM setzen ein Shutdown-Flag; Loop beendet den aktuellen
  Frame vollständig, dann: capture schließen → export flush/close →
  debug stoppen → Exit. Zweites SIGINT = harter Abbruch.
- CLI: `poker-vision run --config <pfad>`; zusätzlich
  `poker-vision validate --config <pfad>` (Config + Kalibrierung prüfen,
  ohne Loop-Start).

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
