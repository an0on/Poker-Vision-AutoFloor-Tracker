# PRD: Poker Vision AutoFloor Tracker

## 1. Systemarchitektur & Laufzeitumgebung

* **Hardware-Zielplattform:** MacBook Pro M4 Max, 36 GB RAM für lokale Verarbeitung (definiert in `abschlussprojekt_poker_vision_handover.md` und `AGENTS.md`). Berechnungen müssen zwingend auf Apple Silicon (CPU/MPS) laufen, ohne CUDA-Abhängigkeiten.


* **Architektur-Paradigma:** Modulare Pipeline, kein End-to-End-Monolith (Vorgabe aus `abschlussprojekt_poker_vision_handover.md` und `AGENTS.md`). Custom YOLO-Modell (Objekterkennung) kombiniert mit regelbasierter Geometrie- und Zustandslogik.


* **Kamera & Orientierung:** Videostream im Querformat (Landscape) (siehe `abschlussprojekt_poker_vision_handover.md` und `AGENTS.md`). Steht das Bild kopf, wird es um 180° gedreht, nicht gespiegelt.



## 2. Der Core-Merge: Kalibrierte Geometrie & Dynamisches Tracking

Das System liest fixe Zonen aus einer Kalibrierungsdatei (`poker_table_runtime_v1.json`) und kombiniert diese mit dynamischem YOLO-Tracking (`persist=True`) (siehe `poker_table_runtime_v1.json` und `AGENTS.md`).

**Eingebettete Geometrie-Regeln aus den Kalibrierungsdaten:**

* **Globale Zonen:** `outer_rail`, `inner_rail` und `action_area` werden als "Capsule" (Racetrack-Oval) aus exakt 4 Punkten auf den geraden Abschnitten berechnet: `top_left`, `top_right`, `bottom_right`, `bottom_left` (definiert in `poker_table_runtime_v1.md`, `poker_table_calibration_schema_v1.md` und `poker_table_calibration_instance_current_table_v1_landscape.json`).


* **Seat Wedges:** 10 Spielerplätze entstehen aus 10 Seat-Dividern (laut `poker_table_runtime_v1.md`, `poker_table_calibration_schema_v1.md` und `poker_table_calibration_instance_current_table_v2_landscape.json`). Jeder Divider besteht aus 2 Punkten: `outer_ring_point` und `inner_ring_point`.


* **Seat Nummerierung:** `seat_1` wird explizit definiert (z.B. oberer mittlerer Wedge zwischen Divider d2 und d3), die restlichen 9 Sitze werden im Uhrzeigersinn abgeleitet (siehe `poker_table_runtime_v1.md`, `abschlussprojekt_poker_vision_handover.md` und `poker_table_calibration_schema_v1.md`).


* **Board Zone:** Ein einziges 4-Punkt-Polygon in der Mitte: `top_left`, `top_right`, `bottom_right`, `bottom_left` (nachlesbar in `poker_table_runtime_v1.md`, `poker_table_calibration_schema_v1.md` und `poker_table_calibration_instance_current_table_v1_landscape.json`).


* **State Memory:** Um fehlende Fold-Erkennungen abzufangen, wird ein `player_memory` genutzt, um den Seat-Status bei Ghosting/Verdeckungen zu erhalten (Strategie aus `abschlussprojekt_poker_vision_handover.md`).



## 3. Domänen- & Geschäftslogik

* **Seat Occupancy Priorität:** 1. `chips` (primär), 2. `all_in_button` (starkes Zusatzsignal), 3. `face_down_cards_secondary` (sekundäre Verifikation) (festgelegt in `poker_table_runtime_v1.md`, `abschlussprojekt_poker_vision_handover.md` und `poker_table_calibration_schema_v1.md`).


* **Dealer Button Assignment:** Es existiert kein festes Boundary-Band (siehe `abschlussprojekt_poker_vision_handover.md` und `poker_table_calibration_schema_v1.md`). Der Button wird per Nearest-Neighbor der `nearest_player_area_or_nearest_seat_anchor` zugeordnet (Regel aus `poker_table_runtime_v1.md`, `abschlussprojekt_poker_vision_handover.md` und `poker_table_runtime_v1.json`).


* **Community Board State:** Zählung der Kartenkandidaten exakt in der `board_zone`: 3 Karten = `flop`, 4 = `turn`, 5 = `river` (definiert in `poker_table_runtime_v1.md`, `abschlussprojekt_poker_vision_handover.md` und `poker_table_runtime_v1.json`).


* **The Tournament Director (TD) Integration:** TD nutzt den Dealer Button als Referenz für Auto-Balancing (siehe `tournament_director_integration_notes.md`). Die Vision-Pipeline muss die Statusänderungen als strukturierten CSV- oder HTML-Export für Windows bereitstellen.



## 4. Iterative Entwicklungsmodule (Agent Instructions)

Die Entwicklung folgt einem strikten Micro-Iterations-Muster. Jede visuelle Logik wird erst auf statischen Bildern verifiziert (`vX.0`), bevor sie auf den Live-Feed übertragen wird (`vX.1`).

### v0.1: Nearest-Neighbor Prototyp (Baseline Math)

* **v0.1.0 (Static):** Einfache Testbilder mit Platzhaltern (z.B. Handy = Chips, Maus = Dealer Button). Erkennung der Objekte, Berechnung der Euklidischen Distanz, visuelles "Gummiband" zwischen Button und nächstem Chip-Haufen einzeichnen.
* **v0.1.1 (Live):** Übertragung dieser Logik auf den Live-Kamera-Feed (`cv2.VideoCapture`).

### v0.2: JSON Parser & Geometry Utilities

* **Logik:** Parser für die Datei `poker_table_runtime_v1.json` implementieren (gefordert in `AGENTS.md` und `poker_table_runtime_v1.json`). Geometrie-Helfer bauen: `point-in-polygon` (für Wedges und Board), `nearest-seat-anchor` und `nearest-seat-wedge` (siehe `abschlussprojekt_poker_vision_handover.md` und `AGENTS.md`).



### v0.3: Table Integration & Occupancy

* **v0.3.0 (Static):** Anwendung auf statische Overlays aus den Kalibrierungs-Instanzen (gefordert in `AGENTS.md`). Button per Nearest-Neighbor der Geometrie zuordnen (laut `poker_table_runtime_v1.md` und `abschlussprojekt_poker_vision_handover.md`). Prüfen, ob Chips innerhalb eines `seat_wedge_polygon` liegen.


* **v0.3.1 (Live):** Test der Geometrie-Zuordnung im Live-Video.

### v0.4: Board State Logic

* **v0.4.0 (Static):** Zählen der Kartenkandidaten exakt im Polygon der `board_zone` (Regel aus `poker_table_runtime_v1.md` und `AGENTS.md`).


* **v0.4.1 (Live):** Live-Test mit Glättung der Kartenanzahl über stabile Frames, um Statuswechsel ohne Flackern auszulösen (Vorgabe aus `poker_table_runtime_v1.md`).



### v0.5: Tracking Persistence & Ghosting Protection (Live Only)

* **Logik:** Einführung von YOLO Object Tracking mit festen IDs. Aufbau des `player_memory`, das bei Verdeckung eines Chip-Stacks dessen Status anhand der letzten X/Y-Koordinate aufrechterhält.

### v0.6: Poker State Machine & TD Export

* **Logik:** Zustandsmaschine implementieren: `waiting_for_new_hand` -> `preflop` -> `flop` -> `turn` -> `river` -> `showdown` -> `hand_closed` (definiert in `abschlussprojekt_poker_vision_handover.md` und `AGENTS.md`). Umwandlung des Endzustands in das CSV-Zielformat für die TD-Integration (siehe `tournament_director_integration_notes.md`).



## 5. Cross-Platform & Deployment Strategy (Addendum)

Das System wird initial für Apple Silicon entwickelt, muss aber architekturell auf einen nahtlosen Wechsel zu Windows (Laptops) und Edge-Geräten (Raspberry Pi) vorbereitet sein. Die strikte Trennung von Objekterkennung (YOLO) und regelbasierter Geometrie macht die Kernlogik OS-unabhängig.

### 5.1 Hardware- & Kamera-Abstraktion (Coding Rules)

* **Keine Hardcodierung:** Compute-Ziele (`device='mps'`, `cuda`, `cpu`), Kamera-Indizes (`cv2.VideoCapture()`) und Auflösungen dürfen nicht im Code hardcodiert werden. Sie müssen zwingend über eine zentrale `config.py` oder Umgebungsvariablen (`.env`) gesteuert werden.
* **Pfad-Handling:** Die Python-Bibliothek `pathlib` ist für alle Dateizugriffe zu nutzen, um plattformübergreifende Konflikte zwischen POSIX-Pfaden (Mac/Linux) und Windows-Pfaden auszuschließen.

### 5.2 Plattform-Spezifikationen für künftiges Deployment

* **Windows (Laptops/PCs):**
* **Inferenz:** Fallback auf `cpu` oder `cuda` (bei vorhandener NVIDIA-GPU).
* **Kamera:** OpenCV-Integration primär über DirectShow (`cv2.CAP_DSHOW`).
* **Strategischer Vorteil:** Da *The Tournament Director* exklusiv auf Windows läuft, ist dies die finale Zielplattform für einen reibungslosen Parallelbetrieb. Die von der Pipeline erstellten CSV/HTML-Exporte der Seat-Occupancy können direkt im lokalen Windows-Dateisystem abgelegt und vom Tournament Director für das Auto-Balancing verarbeitet werden.




* **Raspberry Pi / Edge Devices:**
* **Inferenz:** Da unoptimierte YOLOv8-Modelle für einen Live-Feed auf dem Pi zu ressourcenhungrig sind, muss die Pipeline architekturell den Import von leichtgewichtigen Export-Formaten (ONNX, TFLite, NCNN) unterstützen. Alternativ ist die Anbindung von Hardware-Beschleunigern (Hailo, Coral TPU) einzuplanen.
* **Kamera:** OpenCV-Integration über V4L2 (Video4Linux).


### 5.3 Edge-Computing Fallback: Low-Frequency Polling (Raspberry Pi)

Für leistungsschwache Edge-Geräte wie den Raspberry Pi (ohne Hardware-Beschleuniger) wird ein dedizierter "Polling-Modus" als Betriebsart definiert. Dieser Modus reduziert den Funktionsumfang und die Berechnungsfrequenz auf das absolute Minimum, das für die Tournament Director-Integration nötig ist.

* **Reduzierter Scope:** Fokus ausschließlich auf die Erkennung des **Dealer Buttons** und der **Seat Occupancy** (über Chips). Die dynamische Auswertung der Board-Karten (Flop, Turn, River) wird in diesem Modus deaktiviert, um Rechenzeit zu sparen.


* **Intervall-Erkennung (Polling):** Anstatt einen kontinuierlichen Live-Videostream (z.B. 30 FPS) auszuwerten, greift das System nur alle *X Sekunden* (z.B. alle 3 bis 5 Sekunden) ein einzelnes Standbild (Frame) ab.
* **Architektur-Auswirkung:**
* Das kontinuierliche YOLO-Tracking (`persist=True`) wird abgeschaltet.
* Das System nutzt stattdessen die rohe Inferenz (`model(frame)`) auf dem Einzelbild und wendet die statische Zonen-Mathematik (Punkt-in-Polygon für Chips, Nearest-Neighbor für den Button) aus Modul `v0.3` an.
* Das `player_memory` (Ghosting-Schutz) wird von einer framebasierten auf eine zeitbasierte Glättung umgestellt (z. B. "Wenn ein besetzter Platz für zwei aufeinanderfolgende Polling-Intervalle leer ist, melde Seat Open").
