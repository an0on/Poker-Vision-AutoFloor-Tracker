# AGENTS.md — Abschlussprojekt Poker Vision

## Ziel
Baue eine lokale Computer-Vision-Pipeline für Poker-Tischanalyse mit Fokus auf:
- Seat Occupancy
- Dealer Button Position
- Board State: Flop / Turn / River
- Handablauf über State Machine

## Harte Annahmen
- Zielplattform zuerst: **MacBook Pro M4 Max, 36 GB RAM**
- Primärformat: **Querformat**
- Tisch in kanonischer Querformat-Orientierung verarbeiten
- Falls Tisch auf dem Kopf steht: bevorzugt **180° drehen**
- Kein End-to-End-Monolith; **modulare Pipeline**

## Geometrie / Kalibrierung
Nutze diese Hauptdateien zuerst:
1. `calibration/runtime/poker_table_runtime_v1.json`
2. `docs/poker_table_runtime_v1.md`
3. `calibration/poker_table_calibration_instance_current_table_v3_landscape.json`
4. `docs/abschlussprojekt_poker_vision_handover.md`

Kalibrierlogik:
- `outer_rail`: 4 Punkte
- `inner_rail`: 4 Punkte
- `action_area`: 4 Punkte
- `board_zone`: 4 Punkte
- 10 Seat-Divider mit je 2 Punkten
- `seat_1` explizit definierbar, danach clockwise

## Erkennungslogik
### Seat Occupancy Priorität
1. `chips`
2. `all_in_button`
3. `face_down_cards_secondary`

### Dealer Button
- direkt visuell erkennen
- dann `nearest_player_area_or_nearest_seat_anchor`
- kein separates Boundary-Band nötig

### Board
- eine einzige `board_zone`
- 3 Karten => Flop
- 4 Karten => Turn
- 5 Karten => River

## Erwartete nächste Implementierungsschritte
1. Loader/Parser für `poker_table_runtime_v1.json`
2. Geometrie-Helfer:
   - point-in-polygon
   - nearest-seat-anchor
   - nearest-seat-wedge
3. Objekt-Erkennung für:
   - Dealer Button
   - Chips
   - Cards
   - All-in Button
4. State Machine:
   - waiting_for_new_hand
   - preflop
   - flop
   - turn
   - river
   - showdown
   - hand_closed

## Modellrichtung
Empfohlen:
- YOLO-basiertes Custom Detection Modell
- regelbasierte Geometrie- und Zustandslogik darüber
- optional später Tracking

## Wichtige Hinweise
- `chip_zone` und `card_presence_zone` sind aktuell noch abgeleitet, nicht final manuell kalibriert
- ursprüngliche Telegram-Cache-Bilder sind lokal nicht mehr vorhanden
- vorhandene Overlays/JSONs sind die maßgebliche Arbeitsgrundlage

## Wichtigste Dateien
- `calibration/runtime/poker_table_runtime_v1.json`
- `docs/poker_table_runtime_v1.md`
- `docs/abschlussprojekt_poker_vision_handover.md`
- `calibration/poker_table_calibration_instance_current_table_v3_landscape.json`
- `overlays/calibration_instances/poker_table_calibration_instance_current_table_v3_landscape.png`
