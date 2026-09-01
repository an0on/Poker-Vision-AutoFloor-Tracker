# Poker table runtime format v1

## Zweck
Diese Datei ist das Runtime-Format für die eigentliche Erkennungspipeline auf Basis der kalibrierten Tischgeometrie.

Referenzdatei:
`/opt/data/tmp/poker_table_runtime_v1.json`

Kalibrierbasis:
`/opt/data/tmp/poker_table_calibration_instance_current_table_v3_landscape.json`

## Gesamtidee
Die Kalibrierung beschreibt die Tischgeometrie.
Das Runtime-Format beschreibt die praktisch nutzbaren Zonen für die laufende Auswertung eines Videostreams.

## Top-Level-Felder
- `schema_version`: Versionsnummer des Runtime-Formats
- `runtime_model_id`: Name/ID dieses Runtime-Schemas
- `based_on_calibration`: Pfad zur verwendeten Kalibrierdatei
- `image`: Bildgröße und Orientierung
- `table`: globale Tischzonen
- `seat_1_definition`: Startpunkt der Sitznummerierung
- `numbering_direction`: Reihenfolge der Seats
- `dealer_button_tracking`: Zuordnungslogik für den Dealer Button
- `board_state_logic`: Logik für Flop / Turn / River
- `seat_runtime`: alle laufzeitrelevanten Seat-Zonen

## table
Enthält die globalen Tischzonen:
- `outer_rail`
- `inner_rail`
- `action_area`
- `board_zone`

### outer_rail
4-Punkt-Capsule für den äußeren Rail-Rand.

### inner_rail
4-Punkt-Capsule für den inneren Rail-Rand.

### action_area
4-Punkt-Capsule für die zentrale Action Area.

### board_zone
Ein einziges gleichschenkliges Viereck für alle Community Cards.

Triggerlogik:
- 3 erkannte Karten → `flop`
- 4 erkannte Karten → `turn`
- 5 erkannte Karten → `river`

## seat_1_definition
Legt fest, welcher Sitzbereich als `seat_1` gilt.
Alle weiteren Sitze werden im Uhrzeigersinn daraus abgeleitet.

## dealer_button_tracking
Aktuelle Logik:
- Dealer Button direkt visuell erkennen
- danach dem nächsten Spielerbereich zuordnen
- Fallback: nächster Seat-Anchor oder nächstes Seat-Wedge-Polygon

Wichtige Felder:
- `detector_required`
- `position_assignment`
- `fallback`

## seat_runtime
Enthält `seat_1` bis `seat_10`.
Jeder Seat hat seine eigenen Runtime-Zonen.

### Pro Seat enthalten
- `seat_id`
- `divider_before`
- `divider_after`
- `seat_anchor`
- `seat_wedge_polygon`
- `player_area`
- `chip_zone`
- `card_presence_zone`
- `occupancy_priority`

## Bedeutung der Seat-Felder
### seat_anchor
Ein repräsentativer Mittelpunkt des Seat-Wedges.
Nützlich für:
- Nearest-Zuordnungen
- Sitzlabeling
- Dealer-Button-Zuordnung

### seat_wedge_polygon
Das eigentliche Seat-Polygon zwischen zwei Divider-Linien.
Nützlich für:
- geometrische Zuordnung von Objekten zu Seats
- spätere Maskenbildung
- robustere Zuordnung als reine Mittelpunktlogik

### player_area
Grobe Spielerfläche pro Seat.
Gedacht für:
- Spielerpräsenz
- grobe Objektzuordnung
- Sitzbereichslogik

### chip_zone
Abgeleitete Unterzone innerhalb der Player-Area.
Gedacht für:
- Chip-Erkennung
- Occupancy-Signal
- Erkennung von Bet-/Call-Stacks im zugehörigen Bereich

### card_presence_zone
Abgeleitete Unterzone für Kartenpräsenz.
Gedacht für:
- offene / verdeckte Karten als sekundäres Signal
- Showdown-Erkennung als Zusatzsignal

## Occupancy-Logik
Priorität pro Seat:
1. `chips`
2. `all_in_button`
3. `face_down_cards_secondary`

Das bedeutet:
- Chips sind das primäre Besetzungssignal
- der All-in-Button ist ein starkes Zusatzsignal
- verdeckte Karten bestätigen nur sekundär

## Empfohlene Runtime-Nutzung
### Seat Occupancy
1. erkenne Chips / All-in-Button / Kartenobjekte
2. ordne sie dem nächsten `seat_wedge_polygon` oder `player_area` zu
3. bilde pro Seat einen Besetzungsstatus

### Dealer Button
1. Dealer Button direkt erkennen
2. Mittelpunkt des Buttons bestimmen
3. dem nächsten `player_area` oder `seat_anchor` zuordnen
4. daraus die aktuelle Button-Position ableiten

### Board State
1. alle Kartenkandidaten in `board_zone` zählen
2. Kartenanzahl in stabilen Frames glätten
3. Zustandswechsel auslösen:
   - 0→3 = Flop
   - 3→4 = Turn
   - 4→5 = River

## Grenzen der aktuellen v1
- `chip_zone` und `card_presence_zone` sind derzeit aus dem Seat-Bounding-Box-Modell abgeleitet
- noch keine separat manuell kalibrierten Spezialzonen für:
  - exakte Bet-Spill-Areas
  - Showdown-spezifische Kartenablage
  - Muck-Zone
- ideal für ersten Prototyp, später verfeinerbar

## Nächste sinnvolle Erweiterungen
- manuell gesetzte `chip_zone` pro Seat
- eigene `bet_spill_zone` pro Seat
- eigene `showdown_zone` pro Seat
- explizite `all_in_button_zone`
- Homography-/Control-Point-Normalisierung für neue Perspektiven
