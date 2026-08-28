# Abschlussprojekt Poker Vision — Übergabe für Coding-Agenten

## Projektziel
Ziel ist ein lokal laufendes Computer-Vision-System zur Erkennung von:
- Platzbelegung / Seat Occupancy
- Handablauf / State Progression
- Dealer Button Position
- Community Board State (Flop / Turn / River)

Zielplattform zuerst:
- **MacBook Pro M4 Max, 36 GB RAM**
- lokale Verarbeitung
- später evtl. weitere Deployments / andere Plattformen

## Betriebsannahmen
- Primärfall: **Videostream im Querformat**
- Tisch soll in einer **kanonischen Querformat-Orientierung** verarbeitet werden
- Falls Tisch im Videostream auf dem Kopf steht: bevorzugt **180° drehen**, nicht spiegeln, außer das Bild ist tatsächlich seitenverkehrt
- Tischlayout ist stark **zonen-/layoutbasiert**, nicht primär frei generisch

## Domänenlogik
### Seat Occupancy
Priorität der Besetzungserkennung:
1. `chips`
2. `all_in_button`
3. `face_down_cards_secondary`

Wichtig:
- Chips sind das primäre Besetzungssignal
- All-in-Button ist ein starkes Zusatzsignal
- verdeckte Karten sind nur sekundäre Verifikation
- fehlende Fold-Erkennung darf den Ablauf nicht destabilisieren

### Dealer Button
Aktuelle Entscheidung:
- **kein spezielles Dealer-Button-Boundary-Band nötig**
- Dealer Button wird direkt visuell erkannt
- danach Zuordnung zum **nächsten Spielerbereich / Seat-Wedge**
- Fallback: nächster `seat_anchor`

Aktuelle Runtime-Regel:
- `position_assignment = nearest_player_area_or_nearest_seat_anchor`

### Community Board
Es gibt **eine einzige Board-Zone**.

Zustandslogik über Kartenanzahl in dieser Zone:
- `3 cards => flop`
- `4 cards => turn`
- `5 cards => river`

Keine getrennten Flop-/Turn-/River-Zonen nötig.

## Kalibrierkonzept
### Globale Formen
Folgende Formen werden jeweils über **4 Punkte** definiert:
- `outer_rail`
- `inner_rail`
- `action_area`
- `board_zone`

### Bedeutung der 4-Punkt-Capsules
Für `outer_rail`, `inner_rail` und `action_area` werden je 4 Punkte gesetzt:
- `top_left`
- `top_right`
- `bottom_right`
- `bottom_left`

Ziel:
- lange obere Gerade
- lange untere Gerade
- linker Radius
- rechter Radius

Damit wird eine racetrack-/capsule-artige Tischgeometrie modelliert.

### Seat Divider
Jede Seat-Trennlinie wird über **2 Punkte** definiert:
- `outer_ring_point`
- `inner_ring_point`

Aus den Divider-Linien werden die Seat-Wedges automatisch abgeleitet.

### Seat 1
Am Ende der Kalibrierung muss **Seat 1 explizit festgelegt** werden.
Danach werden die restlichen Seats **im Uhrzeigersinn** nummeriert.

Aktuell gesetzter Default:
- `seat_1` = oberer mittlerer Wedge
- zwischen Divider `d2` und `d3`

## Aktueller Dateistand

### Projektordner lokal
`/opt/data/projects/abschlussprojekt_poker_vision`

### Google-Drive-Spiegelung
Projektordner:
https://drive.google.com/drive/folders/1bHE21KcjXXfDz_H4DR6KpRXi1wUGgKt9

### Wichtigste Dateien
#### Runtime
- `calibration/runtime/poker_table_runtime_v1.json`
- `docs/poker_table_runtime_v1.md`

#### Kalibrierung
- `calibration/poker_table_calibration_instance_current_table_v1_landscape.json`
- `calibration/poker_table_calibration_instance_current_table_v2_landscape.json`
- `calibration/poker_table_calibration_instance_current_table_v3_landscape.json`

#### Kalibrier-Schema
- `calibration/schemas/poker_table_calibration_schema_v1.json`
- `docs/poker_table_calibration_schema_v1.md`

#### Overlays
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v1_landscape.png`
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v1_landscape.svg`
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v2_landscape.png`
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v2_landscape.svg`
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v3_landscape.png`
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v3_landscape.svg`

#### Frühere Raster-Entwürfe
- `overlays/raster_drafts/poker_table_overlay_draft.png`
- `overlays/raster_drafts/poker_table_overlay_draft.svg`
- `overlays/raster_drafts/poker_table_raster_v2_rotated.json`
- `overlays/raster_drafts/poker_table_raster_v2_rotated.png`
- `overlays/raster_drafts/poker_table_raster_v2_rotated.svg`
- `overlays/raster_drafts/poker_table_raster_v3_rotated.json`
- `overlays/raster_drafts/poker_table_raster_v3_rotated.png`
- `overlays/raster_drafts/poker_table_raster_v3_rotated.svg`

#### Skripte
- `overlays/scripts/build_landscape_calibration_instance.py`
- `overlays/scripts/build_landscape_calibration_instance_v2.py`
- `overlays/scripts/build_landscape_calibration_instance_v3.py`
- `overlays/scripts/build_rotated_raster_v2.py`
- `overlays/scripts/build_runtime_json_v1.py`

#### Notizen
- `notes/tournament_director_integration_notes.md`

#### Backups
- `backups/abschlussprojekt_poker_vision_20260828_110531.tar.gz`
- `backups/abschlussprojekt_poker_vision_20260828_110550.tar.gz`

## Aktuelle Hauptdatei für Coding-Agenten
Die wichtigste operative Datei ist aktuell:

`/opt/data/projects/abschlussprojekt_poker_vision/calibration/runtime/poker_table_runtime_v1.json`

Sie basiert auf:

`/opt/data/projects/abschlussprojekt_poker_vision/calibration/poker_table_calibration_instance_current_table_v3_landscape.json`

## Struktur der Runtime-Datei
### Top-Level
- `schema_version`
- `runtime_model_id`
- `based_on_calibration`
- `image`
- `table`
- `seat_1_definition`
- `numbering_direction`
- `dealer_button_tracking`
- `board_state_logic`
- `seat_runtime`

### table
Enthält:
- `outer_rail`
- `inner_rail`
- `action_area`
- `board_zone`

### dealer_button_tracking
Aktuell:
- direkte Erkennung
- kein Boundary-Band
- Zuordnung per Nearest

### board_state_logic
- 3 Karten => Flop
- 4 Karten => Turn
- 5 Karten => River

### seat_runtime
Enthält `seat_1` bis `seat_10`.
Für jeden Seat:
- `seat_id`
- `divider_before`
- `divider_after`
- `seat_anchor`
- `seat_wedge_polygon`
- `player_area`
- `chip_zone`
- `card_presence_zone`
- `occupancy_priority`

## Wichtig zu den Zonen
### player_area
Grobe Spielerfläche je Sitz.

### chip_zone
Abgeleitete Unterzone innerhalb der `player_area`.
Aktuell noch aus Bounding-Boxen abgeleitet, nicht separat manuell kalibriert.

### card_presence_zone
Unterzone für Kartenpräsenz.
Aktuell ebenfalls abgeleitet, noch nicht separat manuell optimiert.

## Aktueller Reifegrad
### Schon entschieden / stabil
- Querformat als bevorzugte kanonische Orientierung
- 4-Punkt-Kalibrierung für Rail und Action Area
- 2-Punkt-Divider pro Sitz
- eine einzige Board-Zone
- Seat 1 explizit definierbar
- Dealer Button per direkter Erkennung + nearest assignment

### Noch prototypisch / später verfeinerbar
- `chip_zone`
- `card_presence_zone`
- spezielle `bet_spill_zone`
- spezielle `showdown_zone`
- evtl. `muck_zone`
- Homography-/Control-Point-basierte automatische Perspektivnormalisierung

## Empfehlungen für den nächsten Coding-Schritt
1. Parser für `poker_table_runtime_v1.json` bauen
2. Geometrie-Helfer implementieren:
   - point-in-polygon
   - nearest-seat-anchor
   - nearest-seat-wedge
   - board-zone card counting
3. erste Runtime-Pipeline aufsetzen:
   - Dealer Button Detection
   - Chip Detection
   - Card Detection im Board-Zone-Bereich
4. einfachen State-Machine-Layer ergänzen:
   - waiting_for_new_hand
   - preflop
   - flop
   - turn
   - river
   - showdown
   - hand_closed

## Technische Empfehlung für Modell-Setup
Für das Abschlussprojekt wurde als sinnvoll angesehen:
- **modulare Pipeline** statt End-to-End-Modell
- **YOLO-basiertes Custom Object Detection Modell** für:
  - Karten
  - Chips
  - Dealer Button
  - All-in Button
- regelbasierte Geometrie-/State-Logik darüber
- später ggf. Tracking-Komponente

## Wichtiger Hinweis zu Originalbildern
Die ursprünglichen Telegram-Cache-Bilder waren temporär und sind aktuell lokal nicht mehr vorhanden.
Die erzeugten Overlays, Kalibrierdateien und Runtime-Dateien sind aber vorhanden.

## Browser / Hermes Zugriff
Lokal geprüftes Dashboard:
`http://localhost:4860`

Extern geprüft:
`http://srv1780110.hstgr.cloud:4860`

Primärer Kontaktkanal des Nutzers ist aktuell aber Telegram.

## Kurzfassung für einen Coding-Agenten
Wenn du als Coding-Agent hier einsteigst, beginne mit:
1. `calibration/runtime/poker_table_runtime_v1.json`
2. `docs/poker_table_runtime_v1.md`
3. `calibration/poker_table_calibration_instance_current_table_v3_landscape.json`
4. `overlays/calibration_instances/poker_table_calibration_instance_current_table_v3_landscape.png`

Und behandle folgende Punkte als harte Projektannahmen:
- Querformat
- 10 Seats
- Seat Occupancy primär über Chips
- Dealer Button direkt erkennen, dann nearest player area zuordnen
- Board State über Kartenanzahl in einer einzigen Board-Zone
- Seat 1 explizit festlegen, danach clockwise nummerieren
