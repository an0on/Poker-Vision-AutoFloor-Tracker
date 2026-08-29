# Future features (post-v0.1)

Funktionen, die für das Projekt gewünscht sind, aber laut `/PRD.md`
ausdrücklich außerhalb des v0.1-MVP-Scopes liegen. Ursprünglich in
`AGENTS.md` (vor der PRD.md-getriebenen Planung) als Teil des aktuellen
Scopes beschrieben — hierher verschoben, damit `AGENTS.md` den
tatsächlichen v0.1-Stand widerspiegelt und Reviews nicht dagegen prüfen.
Nicht implementieren, bevor eine spätere Phase sie explizit in `PRD.md`
aufnimmt.

## All-in-Button als zweites Occupancy-Signal

Ursprünglich in `AGENTS.md` als Teil der Seat-Occupancy-Priorität
(`chips` → `all_in_button` → `face_down_cards_secondary`) und als eigene
Detection-Klasse vorgesehen. v0.1 erkennt Seat Occupancy ausschließlich
über Chip-Präsenz in der `chip_zone` (PRD.md REQ-29); `all_in_button` ist
keine MVP-Detection-Klasse (PRD.md REQ-17: nur `chip`, `card`,
`dealer_button`).

Bei Aufnahme in einen späteren Scope: neue `DetectionClass`-Variante,
zusätzliches Occupancy-Signal/-Priorität in der State-Machine,
entsprechende Config-Schwellwerte.

## Granulare Hand-State-Machine

Ursprünglich in `AGENTS.md` vorgesehen: `waiting_for_new_hand`, `preflop`,
`flop`, `turn`, `river`, `showdown`, `hand_closed`. v0.1 kennt nur
`hand_started`/`hand_ended` (Board leer ↔ nicht-leer, PRD.md REQ-32) plus
`street` aus der Kartenanzahl in der `board_zone` (`flop`/`turn`/`river`,
REQ-31). `preflop` ist ohne Board-Karten nicht erkennbar (PRD.md A2);
`showdown` sowie die Unterscheidung `waiting_for_new_hand` vs.
`hand_closed` sind ohne zusätzliche Signale (Kartenaufdecken,
Chip-Bewegung) mit der v0.1-Erkennung nicht möglich.

Bei Aufnahme in einen späteren Scope: zusätzliche `StateSnapshot`- und
Event-Felder, zusätzliche Detektionssignale zur Unterscheidung der
Zustände.
