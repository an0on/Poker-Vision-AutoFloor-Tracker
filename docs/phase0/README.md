# Phase 0 – Sandbox Proof of Concept (Ergebnis und Freigabe)

Ablage des Phase-0-Ergebnisses nach REQ-43. Der Freigabevermerk selbst steht in
`/PRD.md`, Abschnitt „Phase-0-Freigabe" (freigegeben am **2026-08-29**).

## Dateien

| Datei | Rolle |
|---|---|
| `Test1.jpeg` | Quellbild, unveraendert und in Originalaufloesung (3024×4032). Referenz fuer AC-0.2 bis AC-0.5. |
| `Test1_phase0.jpg` | Ergebnis bei Standardschwelle `--conf 0.25`: Gummiband, beide Mittelpunkte, Distanz. |
| `Test1_phase0_nearest_neighbour.jpg` | Ergebnis bei `--conf 0.10`: zwei `cell phone`-Kandidaten, Nearest-Neighbour-Wahl sichtbar (AC-0.4). |

`Test1.jpeg` bleibt bewusst unskaliert: AC-11 verlangt spaeter, dass der
`mock`-Detector im COCO-Modus auf diesem Bild dieselben Mittelpunkte
reproduziert wie das Phase-0-Skript (Toleranz 1 px im Pixelraum). Jede
Reskalierung wuerde diesen Vergleich unbrauchbar machen. Damit ist die Datei
zugleich das erste Replay-Fixture nach REQ-39.

Weitere Rohaufnahmen liegen unter `data/` und sind bewusst **nicht** versioniert
(`.gitignore`) – im Betrieb werden ausschliesslich Events persistiert, niemals
Frames (REQ-34).

## Reproduktion

Phase 0 hat per REQ-0.1 keine Projekt-Config, die Abhaengigkeiten kommen daher
aus einer Wegwerf-Umgebung:

```bash
uv run --no-project --with ultralytics --with opencv-python \
  python phase0_poc.py docs/phase0/Test1.jpeg -o /tmp/Test1_phase0.jpg
```

Gemessene Werte (Modell `yolov8n.pt`, COCO, Device `mps`):

```
mouse        conf=0.744  box=(1988.8566, 2716.4600, 2519.6831, 3112.7617)  centre=(2254.27, 2914.61)
cell phone   conf=0.492  box=( 624.1805, 2826.8689, 1350.1547, 3228.9136)  centre=( 987.17, 3027.89)
euklidische Distanz = 1272.16 px
```

## Uebertrag nach v0.1

- Die Mittelpunktberechnung (exakter Box-Mittelpunkt) wird von REQ-17 uebernommen.
- Die Nearest-Neighbour-Formulierung wird von REQ-27 uebernommen, dort aber in
  Tischkoordinaten statt im Pixelraum.
- Die Gummiband-Darstellung wird vom Debug-Overlay in REQ-37 uebernommen.

## Beobachtung fuer die Modellphase

Die COCO-Konfidenzen liegen ueber alle fuenf Testaufnahmen niedrig
(`cell phone` 0.28–0.49, `mouse` 0.34–0.74); in `Test3` wird die Maus als
`sports ball` klassifiziert, in `Test4` gar nicht gefunden. Das ist erwartbar –
Top-Down-Aufnahmen auf Teppich liegen ausserhalb der COCO-Trainingsverteilung –
und fuer Phase 0 ohne Belang, da nur die Distanz- und Zuordnungsmathematik
geprueft wurde. Es stuetzt aber die Annahme A3: der COCO-Modus taugt als
Platzhalter fuer v0.1 (REQ-20), das eigene Modell bleibt der Weg fuer v0.2.
