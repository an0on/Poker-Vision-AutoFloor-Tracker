# Product requirements document

## Feature: v0.1 MVP — Mock-Detection-Pipeline

### Goal
    Distanz-/Zuordnungsmathematik (capture → calibration → assignment → state) end-to-end mit Platzhalter-Detections validieren, bevor echtes YOLO-Training beginnt.

### Phase 0 – Sandbox Proof of Concept (Gate vor v0.1)

**Regel:** Kein Anlegen von Projektstruktur, Poker-Logik oder Kalibrierungs-Geometrie, bevor Phase 0 vom User verifiziert wurde. Ergebnis ist ein einzelnes Python-Skript, kein Teil des späteren Pakets.

#### Requirements Phase 0
- REQ-0.1: Genau eine alleinstehende Python-Datei; keine Ordnerstruktur, keine State Machine, kein JSON-Parser, keine Config-Datei, keine Kalibrierung, keine Homographie.
- REQ-0.2: Input ist ein statisches Bild mit beliebigem Hintergrund, auf dem zwei Platzhalter-Objekte liegen, die eine vortrainierte Standard-Klasse haben (z. B. Maus = „Dealer Button", Handy = „Chip-Haufen").
- REQ-0.3: Objekterkennung mit einem vortrainierten Standardmodell (z. B. YOLOv8n/COCO); kein Training, kein eigenes Modell, Device `cpu` oder `mps`, kein CUDA.
- REQ-0.4: Aus den Bounding Boxen beider Objekte wird jeweils der exakte Mittelpunkt (X/Y) im Pixelraum berechnet.
- REQ-0.5: Euklidische Distanz (Pythagoras) zwischen den beiden Mittelpunkten wird berechnet und ausgegeben.
- REQ-0.6: Nearest-Neighbor-Verknüpfung: Für das Objekt „Dealer Button" wird das nächstgelegene Objekt der anderen Klasse gewählt (bei nur zwei Objekten trivial, Logik muss aber als Nearest-Neighbor formuliert sein, nicht als feste Paarung).
- REQ-0.7: Output ist das Bild mit eingezeichneter Linie („Gummiband") zwischen den beiden Mittelpunkten, beiden Mittelpunkten als Marker und der Distanz als Text; wird gespeichert und/oder angezeigt.
- REQ-0.8: Werden nicht genau die zwei erwarteten Klassen gefunden, bricht das Skript mit klarer Meldung ab (kein stilles Weiterlaufen).
- REQ-0.9: Gate: Erst nach ausdrücklicher Freigabe des Ergebnisbilds durch den User beginnt die Arbeit an REQ-1 ff.

#### Acceptance criteria Phase 0
- AC-0.1 (REQ-0.1): Das Repository enthält für Phase 0 genau eine `.py`-Datei; kein `src/`, keine JSON/YAML-Dateien, keine weiteren Module.
- AC-0.2 (REQ-0.2, REQ-0.3): Auf einem vom User bereitgestellten Foto (Schreibtisch, Maus + Handy) werden beide Objekte mit ihrer COCO-Klasse erkannt; kein CUDA-Aufruf im Skript.
- AC-0.3 (REQ-0.4, REQ-0.5): Ausgegebene Mittelpunkte und Distanz stimmen mit einer manuellen Nachrechnung aus den Bounding-Box-Koordinaten überein (Toleranz 1 px).
- AC-0.4 (REQ-0.6): Bei einem Testbild mit drei Objekten (1 Maus, 2 Handys) verbindet das Gummiband die Maus mit dem näheren Handy.
- AC-0.5 (REQ-0.7): Ergebnisbild zeigt Linie, beide Mittelpunkte und Distanzwert; Datei wird geschrieben.
- AC-0.6 (REQ-0.8): Bild ohne eines der beiden Objekte → Abbruch mit Meldung, keine Ausgabedatei.
- AC-0.7 (REQ-0.9): User hat das Ergebnisbild gesichtet und die Phase explizit freigegeben (Vermerk in PRD.md mit Datum).

#### Phase-0-Freigabe

**Status: freigegeben am 2026-08-29 durch den Repo-Owner (an0on).**

Grundlage der Freigabe ist `phase0_poc.py` in der Fassung von Branch
`feat/phase-0-sandbox-poc`, ausgefuehrt auf `Test1.jpeg`. Referenzmaterial liegt
unter `docs/phase0/` (siehe `docs/phase0/README.md`).

Nachweis je Kriterium:

| AC | Nachweis |
|---|---|
| AC-0.1 | Genau eine neue `.py`-Datei (`phase0_poc.py`); kein `src/`, keine JSON/YAML-Dateien. |
| AC-0.2 | `Test1.jpeg`: `mouse` conf 0.744, `cell phone` conf 0.492; Device `mps`; `cuda` nur im Ablehnungspfad. |
| AC-0.3 | Mittelpunkte (2254.27, 2914.61) und (987.17, 3027.89); Distanz Skript 1272.16 px vs. manuell 1272.1559 px → Delta 0.004 px. |
| AC-0.4 | `Test1.jpeg --conf 0.10`: zwei `cell phone`-Detections; Gummiband zum naeheren (1272 px statt 2425 px), das fernere als "verworfen" markiert. |
| AC-0.5 | `docs/phase0/Test1_phase0.jpg` zeigt Linie, beide Mittelpunkte und Distanzwert. |
| AC-0.6 | `Test3.jpeg` (Maus als `sports ball` klassifiziert) und `Test4.jpeg` (keine Maus): Abbruch mit Meldung, Exit-Code 2, keine Ausgabedatei. |
| AC-0.7 | Dieser Vermerk. |

Damit ist das Gate aus REQ-0.9 passiert; die Arbeit an REQ-1 ff. ist ab hier zulaessig.

### Requirements (v0.1)

**Annahmen (bei Widerspruch bitte korrigieren):**
- A1: Sitzanzahl und Seat-IDs kommen aus der v3-landscape-Kalibrierung; keine Konfiguration außerhalb der Kalibrierung.
- A2: „Hand" beginnt für v0.1 mit dem ersten stabil erkannten Board (Preflop ist ohne Board-Karten nicht erkennbar) und endet bei stabil leerem Board.
- A3: Platzhalter-Erkennung in v0.1 nutzt ausschließlich Bordmittel ohne Training: Skriptdatei, OpenCV-ArUco oder ein vortrainiertes COCO-Standardmodell mit Klassen-Mapping (Fortführung des Phase-0-Ansatzes).

#### Grundlagen
- REQ-1: Neue Paketstruktur `src/poker_vision/` mit den Modulen `capture`, `calibration`, `detection`, `tracking`, `assignment`, `state`, `export`, `debug`, `tools`, `runner`; Toolchain Python ≥ 3.11, `uv`, `ruff`, `pytest`. Beginn erst nach Phase-0-Gate (REQ-0.9).
- REQ-2: Eine zentrale Config (Pydantic v2, `schema_version`) für Quelle, Device, Schwellwerte, Hysterese, Ports, Pfade; kein Modul liest Umgebungsvariablen oder Konstanten direkt.
- REQ-3: Device-Auswahl ausschließlich über Config mit Werten `cpu` | `mps` (`cuda` als reservierter, in v0.1 abgelehnter Wert); kein Modul enthält backend-spezifische Codepfade oder CUDA-Aufrufe.
- REQ-4: Alle Datenstrukturen (Kalibrierung Authoring + Runtime, Config, Detections, Events, State-Snapshot) sind Pydantic-v2-Modelle mit `schema_version`; unbekannte Felder werden abgelehnt.
- REQ-5: Sämtliche Geometrie-Entscheidungen (Tracking, Assignment, State) arbeiten ausschließlich in Tischkoordinaten; kein Modul hinter `detection` akzeptiert Pixelkoordinaten.

#### Calibration
- REQ-6: Ein einziges Kalibrierungsschema (Authoring) mit `schema_version`. Die kanonische Geometrie stammt aus dem Referenzfoto (`calibration/reference/`, siehe REQ-10a) — **nicht** aus den alten `v1`–`v3`-Pixel-Dateien, deren Sitzzahl (10 dort zufällig auch, aber andere Aufteilung/Zuordnung) durch das eigene, nachträglich per Feature-Klicks vermessene Referenzfoto ersetzt wird. Ersetzt die bisherigen Schema-Varianten in `calibration/`; v1/v2/v3 wandern nach `docs/archive/`.
- REQ-7: Schema enthält Kameraintrinsics + Distortion, Homographie Pixel→Tischebene, Tischmaße/-einheit, Zonen als Polygone in Tischkoordinaten: je Seat `player_area` und `chip_zone`, global `board_zone` und `dealer_area` (= die "Action Area", in der der physische Dealer-Button erkannt wird, siehe REQ-27/REQ-30 — kein kleiner fester Fleck, sondern die gesamte Fläche innerhalb des inneren Ovals); Seat-IDs stabil und eindeutig. Zusätzlich: genau ein Seat ist als `card_dealer_seat_id` markiert — die feste physische Kartengeber-Position (Standardfall: nicht durchgehend als nummerierter Spielersitz gezählt, siehe REQ-10a). Das ist eine reine Kalibrierungs-/Geometrie-Eigenschaft, unabhängig davon, ob an einem gegebenen Tag ein zehnter Spieler von dort aus mitspielt (Turnier-/Spielzustand, außerhalb des Scopes dieses Schemas).
- REQ-8: Undistortion + Homographie als eine Transformationsstufe; Rotation ist ausschließlich Bestandteil der Homographie. Ersetzt `build_rotated_raster*.py` (verworfen, keine Nachfolger).
- REQ-9: CLI `calib compile`: Authoring-JSON → Runtime-JSON (vorberechnete Matrizen, aufgelöste Polygone). Ersetzt `build_runtime_json_v1.py` (nach `tools/` migriert).
- REQ-10: Ein Kalibrierungs-CLI zum Erstellen/Editieren/Validieren der Authoring-JSON. Ersetzt alle `build_landscape_calibration_instance*.py`-Varianten durch genau ein Werkzeug.
- REQ-10a: Interaktives Klick-Tool (`calib mark-zones`) zum Authoring der Referenzfoto-Geometrie, statt Koordinaten von Hand zu tippen oder algorithmisch zu raten (automatische Farbsegmentierung ist bei diesem Tischdesign nachweislich unzuverlässig — benachbarte gleichfarbige Sitze sind nur durch eine schwache Naht statt einer kontrastreichen Linie getrennt). Nimmt ein beliebiges Bild entgegen (Referenzfoto oder frisch aufgenommener Live-Frame — dasselbe Werkzeug für beide Fälle):
  - Operator klickt die Eckpunkte aller 10 `player_area`-Polygone.
  - Operator klickt je 6 Punkte für das innere und das äußere Oval (Kapsel-/Stadion-Form: Bogenanfang, **Kreismittelpunkt** (nicht ein dritter Punkt auf der Kurve — der tatsächliche Mittelpunkt, von dem aus der Radius zu Anfang/Ende gemessen wird), Bogenende an jedem der zwei Enden), woraus Radius und Streckung exakt bestimmt und die Kurven als Polygon-Approximation (abgetastete Bogenpunkte) gerendert werden — nicht als grobe 4-Punkte-Rechteck-Näherung wie in den alten `v1`–`v3`-Dateien.
  - Operator klickt 4 Punkte für `board_zone`.
  - Operator markiert genau einen der 10 Sitze als `card_dealer_seat_id` (feste Kartengeber-Position, i. d. R. einer der beiden zentralen Sitze am Mittelstreifen).
  - Das Tool leitet daraus ab: `dealer_area` = Fläche innerhalb des inneren Ovals (schließt `chip_zone`/`player_area` per Konstruktion aus, da diese im Kranz außerhalb des inneren Ovals liegen); Seat-Nummerierung `seat_1..seat_10` im Uhrzeigersinn beginnend beim Sitz direkt nach dem als `card_dealer_seat_id` markierten Sitz, welcher damit selbst immer `seat_10` erhält (auch im Standardfall neun Spieler + Kartengeber — ob an `seat_10` tatsächlich ein zehnter Spieler sitzt, ist Turnier-/Spielzustand, nicht Teil der Kalibrierung); `chip_zone` je Sitz als konfigurierbarer Schrumpf-Faktor auf `player_area` (Default: 50 % Richtung Zentroid), vom Operator überschreibbar.
  - **Tischeinheiten:** `table.width/height` entsprechen exakt der Auflösung des Referenzfotos (1 Bildpixel = 1 nominale Tischeinheit), `homography` ist eine Identitätsabbildung (Bildpixel = Tischkoordinate) — keine echte Vermessung des physischen Tisches (bewusste Entscheidung, siehe Diskussion zu REQ-6/REQ-7 in diesem Repo; reale Maße wurden explizit als nicht nötig eingestuft). Konsequenz: jeder künftige Distanz-Schwellwert (z. B. REQ-27s `dealer_nearest_seat_max_distance`) ist in Referenzfoto-Pixel-Einheiten zu verstehen, nicht in echten Millimetern; `table.unit: "mm"` ist daher ein nomineller Platzhalter, kein gemessener Wert.
  - Ergebnis ist eine vollständige `CalibrationAuthoring`, die dieselbe REQ-11-Validierung wie `calib create`/`calib edit` durchläuft, bevor sie geschrieben wird.
- REQ-10b: `calib learn-table` — automatische Kalibrierung neuer physischer Tisch-Exemplare desselben Grunddesigns (Filzfarbe variiert, Geometrie ist bauartbedingt identisch). Nimmt ein neues Live-Foto entgegen und erzeugt daraus automatisch eine vollständige, REQ-11-valide `CalibrationRuntime`, ohne erneutes manuelles Marking:
  1. Graustufen-Feature-Matching (ORB oder AKAZE + BFMatcher/FLANN, RANSAC-Homographie) zwischen Referenzfoto und Live-Foto, Suchraum primär im Mittelstreifen (Kartenfeld-Umrisse + Kontur; Branding wie "DOPO POKER" darf einfließen, ist aber nicht Voraussetzung) — nicht farbbasiert, robust gegen Filzfarb-Variation und Tische ohne Branding.
  2. Die neue Bild→Tisch-Homographie ergibt sich als Verkettung: Live-Bild → (Feature-Match-Homographie) → Referenz-Bild → (bereits gelöste Referenz-Homographie) → Tischebene. Die Zonen selbst (bereits in Tischkoordinaten) werden dabei NICHT erneut transformiert — sie sind identisch zur Referenz; nur die Bild→Tisch-Homographie ändert sich pro Aufnahme. Kamera-Intrinsics, Distortion und Tischmaße werden unverändert von der Referenzkalibrierung übernommen (Annahme: gleiches Kameramodell/-aufbau für jede Aufnahme).
  3. Zu wenige/unzuverlässige Matches (RANSAC-Inlier-Anteil unter konfiguriertem Schwellwert) → Abbruch mit klarer Fehlermeldung statt einer unplausiblen Kalibrierung.
  Ergänzt REQ-9/REQ-10/REQ-10a, ersetzt sie nicht: `calib mark-zones` bleibt für das einmalige Referenzfoto-Authoring nötig.
- REQ-11: Validierung beim Laden (hart, kein Weiterlaufen): Polygone geschlossen und nicht degeneriert, `chip_zone` liegt in `player_area` desselben Seats, keine Überlappung zwischen `chip_zone`s verschiedener Seats, `board_zone` überlappt keine `chip_zone`, Homographie invertierbar, `card_dealer_seat_id` referenziert einen existierenden Seat.
- REQ-12: `overlays/*.png` werden nicht mehr versioniert (Laufzeit-Rendering, siehe REQ-37); `overlays/*.svg` nur dann nach `assets/`, wenn sie physisch als Druckvorlage genutzt werden, sonst verworfen.

#### Capture
- REQ-13: `Capture`-Interface mit Implementierungen `continuity`, `video_file`, `image_dir`; identische Frame-Ausgabe (Bild, Zeitstempel, laufender Frame-Index, Quellkennung).
- REQ-14: Konfigurierbarer Auflösungs-Cap (Default 1920×1080) mit erhaltenem Seitenverhältnis; Kalibrierung referenziert die Inferenzauflösung explizit.
- REQ-15: `video_file` und `image_dir` sind deterministisch (gleiche Eingabe → identische Frame-Sequenz und -Indizes) und laufen ohne Kamera.
- REQ-16: `continuity` wird über Geräteindex ausgewählt (AVFoundation); Fehlen der Kamera führt zu einem klaren Fehler, nicht zu einem Fallback auf eine andere Quelle. Keine Test-Abhängigkeit von Continuity-Hardware.

#### Detection
- REQ-17: `Detector`-Interface mit Output je Frame: Liste aus (Klasse ∈ {`chip`, `card`, `dealer_button`}, Konfidenz, Mittelpunkt in Tischkoordinaten, optional Box), plus Frame-Index; Pixel→Tisch-Transformation erfolgt in der Detection-Stufe, bevor Ergebnisse die Stufe verlassen. Mittelpunktberechnung übernimmt die in Phase 0 verifizierte Methode.
- REQ-18: `mock`-Detector, Modus A: Detections aus Skriptdatei (JSONL, Frame-Index → Detections, wahlweise in Pixel- oder Tischkoordinaten mit Kennzeichnung).
- REQ-19: `mock`-Detector, Modus B: ArUco-Marker im Bild, Mapping Marker-ID → Klasse aus Config; Marker-Zentrum = Objektmittelpunkt.
- REQ-20: `mock`-Detector, Modus C: vortrainiertes COCO-Standardmodell (z. B. YOLOv8n) mit Mapping COCO-Klasse → MVP-Klasse aus Config (z. B. `mouse` → `dealer_button`, `cell phone` → `chip`); kein Training, kein eigenes Modell.
- REQ-21: `mock`-Detector bietet konfigurierbare Störungen (Positions-Jitter, Frame-Dropout, Geister-Detections) mit festem Seed, um Hysterese/Tracking reproduzierbar zu testen.
- REQ-22: `yolo`-Detector (eigenes Modell) existiert nur als registrierte Interface-Implementierung ohne Modell; Auswahl in v0.1 liefert einen expliziten Fehler („nicht in v0.1 verfügbar").

#### Tracking
- REQ-23: Vergabe stabiler Track-IDs pro Klasse über Frames per Nearest-Matching in Tischkoordinaten mit Distanzschwelle (Config); ByteTrack wird erst mit dem eigenen `yolo`-Modell angebunden.
- REQ-24: Hysterese: Track gilt nach `n_on` aufeinanderfolgenden Frames als anwesend, nach `n_off` fehlenden Frames als abwesend; beide global konfigurierbar und je Klasse überschreibbar.
- REQ-25: Nur bestätigte (stabile) Tracks werden an `assignment` weitergegeben.

#### Assignment
- REQ-26: Point-in-Polygon je Zone für jeden stabilen Track: `chip` → `chip_zone`/`player_area` eines Seats, `card` → `board_zone`, `dealer_button` → `dealer_area` bzw. Seat.
- REQ-27: Nearest-Seat-Fallback nur für `dealer_button`: liegt der Button in keiner Zone, wird der nächstgelegene Seat (euklidische Distanz zum `player_area`-Zentroid, Tischeinheiten) gewählt, sofern unter Schwellwert; darüber → `unassigned`. Fortführung der Phase-0-Nearest-Neighbor-Logik in Tischkoordinaten.
- REQ-28: Ein Track wird höchstens einer Zone zugeordnet; bei Mehrfachtreffer deterministische Wahl (kleinste Zentroid-Distanz) und Warnlog. Keine Spiellogik im Modul.

#### State
- REQ-29: Seat-Occupancy: `occupied`, wenn ≥ 1 stabiler `chip`-Track in der `chip_zone` liegt; Events `seat_occupied` / `seat_vacated` nur bei Zustandswechsel.
- REQ-30: Dealer-Seat aus `dealer_button`-Zuordnung; Event `dealer_moved(from, to)` nur bei Seat-Wechsel; Verlust des Buttons ändert den Dealer-Seat nicht.
- REQ-31: Street aus Anzahl stabiler `card`-Tracks in `board_zone`: 3 → `flop`, 4 → `turn`, 5 → `river`; 1, 2 und > 5 erzeugen kein Event (Warnlog). Innerhalb einer Hand nur monoton steigende Übergänge; Rücksetzung erst bei stabil leerem Board.
- REQ-32: `hand_started` bei Übergang leeres Board → stabiles Board (≥ 1 Karte), `hand_ended` bei Übergang → stabil leeres Board; jede Hand erhält eine laufende `hand_id`.
- REQ-33: Jedes Event trägt `event_type`, `sequence` (monoton), `timestamp`, `frame_index`, `hand_id` (falls zutreffend), `seat` (falls zutreffend), typisierte Payload; State-Snapshot des Gesamtzustands ist jederzeit abfragbar.

#### Export
- REQ-34: `jsonl`-Adapter: append-only, ein Event pro Zeile, Datei pro Session; es werden ausschließlich Events persistiert, niemals Frames oder Bildausschnitte.
- REQ-35: `websocket`-Adapter (FastAPI/uvicorn): beim Verbinden vollständiger State-Snapshot, danach Events in Sequenzreihenfolge; REST `GET /status` (Snapshot) und `GET /health`.
- REQ-36: `tournament_director`-Adapter als Stub: Interface vorhanden, Aufrufe werden nur geloggt; keine Netzwerk- oder Windows-Abhängigkeit.
- REQ-37a: Adapter sind über Config einzeln aktivierbar; Ausfall eines Adapters stoppt die Pipeline nicht.

#### Debug
- REQ-37: MJPEG-Endpoint mit Overlay: Zonen aus Kalibrierung, stabile Tracks mit ID/Klasse, Zuordnung (Gummiband Track→Seat wie in Phase 0), Occupancy/Dealer/Street; über Config abschaltbar.
- REQ-38: Statische HTML-Debug-Seite (MJPEG + WebSocket-Eventliste), ohne Frontend-Framework und ohne Build-Schritt.

#### Replay, Tests, Qualität
- REQ-39: Pipeline-Stufen 3–6 (`detection`→`state`) laufen vollständig ohne Kamera über `video_file`/`image_dir` + `mock`; das Phase-0-Testbild ist als erstes Replay-Fixture enthalten.
- REQ-40: Testfixtures: mindestens ein Replay-Set (Frames + Mock-Skript) mit erwarteter Event-Sequenz für Occupancy, Dealer-Wechsel, Flop→Turn→River, Hand-Ende, sowie Störfälle (Dropout, Jitter, 2-Karten-Flackern).
- REQ-41: `ruff` ohne Befunde, `pytest` grün, keine `cuda`-Referenzen im Quellcode (automatisiert geprüft).
- REQ-42: Overhead der Stufen 3–6 pro Frame ≤ 10 ms auf dem Zielrechner bei ≤ 50 Detections/Frame (Messung im Replay).
- REQ-43: `docs/` enthält ADRs für die Architekturentscheidungen dieser Runde, das Phase-0-Ergebnis (Bild + Freigabevermerk) und `docs/archive/` für v1/v2-Kalibrierung und verworfene Skripte; `notes/` wird nach `docs/` überführt.

#### Runner

Entscheidung zur Aufteilung: DREI getrennte REQs statt einer Einheit.
Begründung: Frame-Loop ist headless mit Mocks unit-testbar, Lifecycle/CLI ist
ein Integrationstest-Thema, FrameHub berührt das bestehende debug-Modul —
getrennte REQs ergeben kleinere, unabhängig reviewbare PRs. Die Acceptance
Criteria stehen wegen ihres Umfangs direkt bei jedem REQ als Checkliste,
abweichend vom AC-N-Schema im nachfolgenden Abschnitt (siehe dortiger Hinweis).

##### REQ-44 — Frame-Loop-Orchestrierung
Der Runner orchestriert pro Frame die Kette
capture → detection → tracking → assignment → state → export → debug
gemäß der in CLAUDE.md definierten Fehlerpolitik.

Acceptance Criteria:
- [ ] `runner/loop.py` verarbeitet Frames strikt sequenziell in obiger
      Reihenfolge; Kalibrierung wird nur einmal beim Start geladen.
- [ ] Ein `FrameContext` wird vom Loop pro Frame intern erzeugt und nach
      jeder Stufe fortgeschrieben (frame_id, Detections, Tracks, Zonen-
      Zuordnung, State-Snapshot, Fehlerliste); Stufen behalten ihre
      bestehenden typisierten Signaturen und importieren `FrameContext`
      nicht (kein Bruch der `runner → Stufen`-Abhängigkeitsrichtung).
- [ ] Exception in detection/tracking/assignment/state verwirft den Frame
      ohne partielles State-Update (Test: State-Snapshot vor/nach
      Fehler-Frame identisch, Event-Sequence unverändert).
- [ ] Coding-Konvention (keine Copy-on-Write-/Transaktions-Maschinerie):
      jede Stufe der Kernkette berechnet ihr Update rein (Rückgabewert)
      statt eigenen persistenten Zustand direkt zu mutieren; der Loop
      committet die Updates (Tracking-Hysterese/Track-IDs, State-Machine-
      Zustand) in Pipeline-Reihenfolge erst, nachdem die gesamte
      Kernkette für den Frame erfolgreich war — ein Fehler in einer
      späteren Stufe verhindert damit auch das Commit einer bereits
      erfolgreich berechneten früheren Stufe (Test: Tracking-Update einer
      erfolgreichen `tracking`-Stufe wird NICHT übernommen, wenn die
      nachfolgende `assignment`- oder `state`-Stufe im selben Frame
      wirft). Betrifft insbesondere Tracking-Hysterese/Track-IDs
      (REQ-23, REQ-24) und State-Machine-Zustand (REQ-29–REQ-32);
      bestehende Implementierungen dieser REQs werden im Rahmen von
      REQ-44 daraufhin geprüft und bei Nichteinhaltung angepasst (kein
      eigenes REQ).
- [ ] N konsekutive Kernketten-Fehler (config, Default 30) beenden den Loop
      mit Fehlerstatus; ein Erfolgs-Frame resettet den Zähler (Test mit
      Mock-Detector, der gezielt Fehler wirft).
- [ ] Export-Fehler beenden den Loop nie (Test mit fehlerhaftem Adapter).
- [ ] EOF bei video_file/image_dir beendet den Loop regulär (Status "completed").
- [ ] Vollständiger Loop-Durchlauf ist headless testbar: mock-Detection,
      image_dir-Capture, jsonl-Export — ohne Kamera, GUI oder Netzwerk.
- [ ] Bei Live-Quelle (`continuity`) liest ein interner Hintergrund-Thread
      kontinuierlich und schreibt in einen thread-sicheren Single-Slot-Puffer
      mit monoton steigendem Versions-/Sequenzzähler (latest-wins, gleiches
      Pattern wie `LatestFrameHub` aus REQ-46, nicht
      `cv2.CAP_PROP_BUFFERSIZE`); `get_latest()` liefert jeden Frame genau
      einmal und blockiert kurz mit Timeout auf einen neuen Frame
      (`threading.Condition`/`Event`, kein Busy-Spin, kein reines
      Sleep-Polling) statt direkt `read()` auf der Kamera aufzurufen. Gilt
      nur für `continuity`; `video_file`/`image_dir` bleiben beim
      bestehenden synchronen, deterministischen Lesepfad ohne Dropping.
      Tests: (a) Verarbeitung langsamer als Frame-Rate der Quelle → Loop
      erhält stets den aktuellsten Frame, kein Backlog, kein Blockieren
      über den Timeout hinaus; (b) Verarbeitung schneller als Frame-Rate
      der Quelle → kein Frame wird zweimal verarbeitet (keine doppelten
      `frame_id`s, keine doppelte Hysterese-Zählung).

##### REQ-45 — CLI-Einstiegspunkt & Lifecycle
Der Runner ist über ein CLI startbar, config-gesteuert und fährt auf
SIGINT/SIGTERM sauber herunter.

Acceptance Criteria:
- [ ] `poker-vision run --config <pfad>` startet die Pipeline; alle
      Stufen-Konfigurationen kommen aus genau einer Pydantic-validierten Datei.
- [ ] `poker-vision validate --config <pfad>` prüft Config + Kalibrierung
      inkl. Zonen-Validierung und endet ohne Loop-Start (Exit 0/≠0).
- [ ] Ungültige Config oder Kalibrierung → Abbruch vor Loop-Start,
      Exit ≠ 0, verständliche Fehlermeldung.
- [ ] SIGINT/SIGTERM: aktueller Frame wird abgeschlossen, Exporte werden
      geflusht/geschlossen, Debug-Server gestoppt, Exit 0
      (Test: jsonl-Datei ist nach Signal vollständig und valide).
- [ ] Zweites SIGINT erzwingt sofortigen Abbruch.
- [ ] Exit-Codes: 0 = regulär/EOF/Signal, ≠ 0 = Config-/Kalibrierungs-/
      Fehlerschwellen-Abbruch (dokumentiert).

##### REQ-46 — Debug-Anbindung über LatestFrameHub
Der Debug-MJPEG-Stream zeigt Live-Frames der laufenden Pipeline über einen
thread-sicheren Single-Slot-Hub (latest-wins).

Acceptance Criteria:
- [ ] `LatestFrameHub` mit `publish(frame, snapshot)` / `get_latest()`;
      publish überschreibt immer (kein Queue-Backlog), thread-sicher
      (Test: konkurrierendes publish/get ohne Korruption/Deadlock).
- [ ] Runner publiziert nach jedem erfolgreich verarbeiteten Frame;
      Fehler-Frames werden nicht publiziert.
- [ ] Overlay-Rendering erfolgt on-demand im Debug-Server; ohne verbundenen
      Client findet kein Rendering statt (Test: Render-Funktion wird ohne
      Client nicht aufgerufen).
- [ ] Debug-Server wird per Config aktiviert/deaktiviert und vom
      Runner-Lifecycle gestartet/gestoppt; deaktiviert = keine Ports offen.
- [ ] Publizieren blockiert den Loop nicht messbar (latest-wins, kein Lock
      über Rendering-Dauer).
- [ ] Bestehender Standalone-Betrieb des MjpegDebugServers bleibt für
      isolierte Tests funktionsfähig.

### Acceptance criteria (v0.1)

- AC-1 (REQ-1, REQ-41): `uv sync && ruff check && pytest` läuft auf frischem Checkout fehlerfrei; Verzeichnisbaum entspricht der Modulaufteilung; Phase-0-Freigabe (AC-0.7) liegt vor dem ersten Commit unter `src/`.
- AC-2 (REQ-3, REQ-41): Config mit `device: cuda` wird mit klarer Fehlermeldung abgelehnt; grep nach `cuda` im `src/` liefert nur den Config-Validator.
- AC-3 (REQ-4): Laden einer Kalibrierung/Config mit falscher `schema_version` oder unbekanntem Feld schlägt fehl.
- AC-4 (REQ-6, REQ-43): Die aus dem Referenzfoto abgeleitete 10-Sitz-Geometrie ist im neuen Schema abgelegt; v1/v2/v3 und alle verworfenen Skripte liegen nur noch unter `docs/archive/`; kein alter Schema-Loader mehr im Quellcode.
- AC-5 (REQ-8): Ein um 180° gedrehter Referenzframe erzeugt nach Homographie dieselben Tischkoordinaten für bekannte Referenzpunkte (Toleranz konfiguriert, Default ≤ 1 % der Tischbreite).
- AC-6 (REQ-9, REQ-10): `calib compile` erzeugt aus der Authoring-JSON deterministisch dieselbe Runtime-JSON (Byte-gleich bei gleicher Eingabe); Kalibrierungs-CLI kann Zone anlegen, verschieben, validieren.
- AC-6a (REQ-10a): `calib mark-zones` erzeugt aus den geklickten Punkten (10 Sitz-Polygone, je 6 Bogenpunkte für inneres/äußeres Oval, 4 Board-Zone-Punkte, ein markierter `card_dealer_seat_id`) eine REQ-11-valide `CalibrationAuthoring`; Seat-Nummerierung ist im Uhrzeigersinn ab dem auf `card_dealer_seat_id` folgenden Sitz und deterministisch bei denselben Eingabepunkten.
- AC-6b (REQ-10b): Für zwei unterschiedliche Fotos desselben (realen oder simulierten) Tisches erzeugt `calib learn-table` Tischkoordinaten für bekannte Referenzpunkte (z. B. Karten-Umriss-Ecken), die innerhalb einer konfigurierten Toleranz (Default ≤ 1 % der Tischbreite) übereinstimmen; ein Foto mit zu wenigen/unzuverlässigen Matches bricht mit klarer Fehlermeldung ab statt eine Kalibrierung zu erzeugen; die resultierende `CalibrationRuntime` ist REQ-11-valide.
- AC-7 (REQ-11): Jede aufgeführte Validierungsverletzung ist durch einen Testfall abgedeckt und bricht mit benannter Regel ab.
- AC-8 (REQ-13, REQ-15): Zweimaliges Abspielen derselben `video_file`/`image_dir`-Quelle liefert identische Frame-Indizes und Zeitstempel-Reihenfolge.
- AC-9 (REQ-14): Bei Eingabe > Cap ist der ausgegebene Frame auf den Cap skaliert, Seitenverhältnis erhalten.
- AC-10 (REQ-17): Ein Detector-Output mit Pixelkoordinaten außerhalb der Tischebene wird von `tracking` abgelehnt (Typ-/Validierungsfehler), nicht stillschweigend verarbeitet.
- AC-11 (REQ-18, REQ-19, REQ-20): Mock-Skript, ArUco-Testbild und COCO-Modus liefern für denselben physischen Aufbau Detections mit ≤ 1 % Tischbreite Abweichung in Tischkoordinaten; Modus C reproduziert auf dem Phase-0-Bild dieselben Mittelpunkte wie das Phase-0-Skript (Toleranz 1 px im Pixelraum).
- AC-12 (REQ-21, REQ-24): Mit Dropout < `n_off` Frames entsteht kein `seat_vacated`; mit Dropout ≥ `n_off` genau eines; Geister-Detection < `n_on` Frames erzeugt kein `seat_occupied`.
- AC-13 (REQ-22): Auswahl `detector: yolo` bricht beim Start mit Hinweis auf v0.2 ab.
- AC-14 (REQ-23): Über eine Replay-Sequenz mit bewegtem Chip bleibt die Track-ID erhalten, solange die Bewegung pro Frame unter der Distanzschwelle liegt.
- AC-15 (REQ-26, REQ-28): Chip in `chip_zone` Seat 3 → Zuordnung Seat 3; Chip in `player_area` außerhalb `chip_zone` → keine Occupancy; Karte außerhalb `board_zone` → keine Street-Zählung.
- AC-16 (REQ-27): Dealer-Button außerhalb aller Zonen, aber unter Schwelle → nächster Seat; über Schwelle → `unassigned`, kein `dealer_moved`; Testbild mit zwei Kandidaten-Seats wählt den mit kleinerer euklidischer Distanz.
- AC-17 (REQ-29): Replay-Fixture „Chip rein/raus" erzeugt exakt die Sequenz `seat_occupied(S)`, `seat_vacated(S)` mit korrekten Frame-Indizes.
- AC-18 (REQ-30): Fixture „Button Seat 1 → Seat 2" erzeugt genau ein `dealer_moved(1, 2)`; Verschwinden des Buttons erzeugt kein Event, Snapshot zeigt weiterhin Seat 2.
- AC-19 (REQ-31): Fixture „3 → 4 → 5 Karten" erzeugt `street_changed` in der Reihenfolge flop, turn, river; Fixture „3 → 2 → 3" (Flackern) erzeugt genau ein `flop`-Event; Fixture „4 → 3" innerhalb einer Hand erzeugt kein Event.
- AC-20 (REQ-32): Fixture „leer → 3 → 5 → leer" erzeugt `hand_started`, dann Streets, dann `hand_ended`; zweite Hand erhält `hand_id + 1`.
- AC-21 (REQ-33, REQ-34): JSONL-Datei enthält lückenlos aufsteigende `sequence`-Werte; jede Zeile validiert gegen das Event-Schema; Datei enthält keine Bilddaten.
- AC-22 (REQ-35): WebSocket-Client erhält als erste Nachricht einen Snapshot, der gegen das Snapshot-Schema validiert; anschließende Events entsprechen der JSONL-Datei derselben Session.
- AC-23 (REQ-36, REQ-37a): Aktivierter TD-Stub loggt jedes Event; absichtlich fehlschlagender Adapter unterbricht weder JSONL noch WebSocket.
- AC-24 (REQ-37, REQ-38): MJPEG-Stream zeigt bei Replay Zonen, Track-IDs, Gummiband-Linien und aktuellen State; `debug.enabled: false` startet keinen MJPEG-Endpoint; HTML-Seite lädt ohne externe Abhängigkeiten.
- AC-25 (REQ-39, REQ-40): Alle Tests laufen in CI ohne Kamera; die genannten Fixtures inkl. Phase-0-Bild existieren mit hinterlegter Soll-Event-Sequenz.
- AC-26 (REQ-42): Benchmark-Test im Replay dokumentiert Median-Overhead der Stufen 3–6 ≤ 10 ms/Frame.
- AC-27 (REQ-43): Für jede Zeile der Entscheidungstabelle (Behalten/Ändern/Verwerfen) existiert ein ADR oder ein Eintrag in `docs/archive/README`; Phase-0-Ergebnis ist abgelegt.
- Hinweis: Für REQ-44–REQ-46 (`#### Runner`) sind die Acceptance Criteria wegen ihres Umfangs direkt beim jeweiligen REQ als Checkliste geführt statt in diesem AC-N-Schema.

### Out of scope (Phase 0 + v0.1)

- Eigenes YOLO-Modell (`yolo`-Detector-Logik, CoreML-Export, MPS-Inferenz eines trainierten Modells, ByteTrack-Anbindung).
- Datensatz-Pipeline: Frame-Sampling für Annotation, Annotation-Export (CVAT/Label Studio), Trainings-Skript, Modell-Versionierung `models/vX/`.
- Windows-Betrieb, `cuda`-Device, ONNX-Export, Kamerawechsel weg von Continuity.
- Funktionaler Tournament-Director-Adapter (nur Stub in v0.1).
- Chip-Denomination, Stack-Counting, Karten-Rang/-Farbe, Spieleraktionen (Bet/Fold), Pot-Erkennung.
- Klasse `hand` / Okklusionsbehandlung über Hysterese hinaus.
- Automatische Rekalibrierung oder Drift-Erkennung gegen Referenzmarker (nur manuelle Validierung per CLI).
- Mehrere Tische, mehrere Kameras.
- In Phase 0 zusätzlich: Homographie, Zonen, Tracking, Hysterese, Events, Export – alles erst ab v0.1.
- All-in-Button als zweites Seat-Occupancy-Signal und eine granularere Hand-State-Machine (preflop/showdown/waiting_for_new_hand/hand_closed) — ursprünglich in `AGENTS.md` skizziert, siehe `docs/future-features.md` für den Stand dieser Planung.
