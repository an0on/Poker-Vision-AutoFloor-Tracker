# The Tournament Director – Integrationsnotizen

Erstellt aus offiziellen Quellen auf thetournamentdirector.net.

## Offizielle Quellen
- Hauptdokumentation: https://thetournamentdirector.net/assets/userguide/docs343.html
- FAQ: https://thetournamentdirector.net/faq.html
- Change Log: https://thetournamentdirector.net/changes.html

## Lokal gesicherte Extrakte / Caches
- Gesamtdoku (Cache): `/opt/data/cache/web/thetournamentdirector.net-dbe27a7824.md`
- FAQ (Cache): `/opt/data/cache/web/thetournamentdirector.net-b61f4eb204.md`
- Changelog (Cache): `/opt/data/cache/web/thetournamentdirector.net-34e547cbde.md`

## Relevante Erkenntnisse für spätere Integration

### 1) Plattform / Laufzeit
- Laut FAQ läuft The Tournament Director derzeit nur auf Windows.
- FAQ: "Is there a Mac version?" → "Not at this time. The Tournament Director currently runs only on Windows."

### 2) Automatisches Seating / Balancing
Aus Doku Abschnitt `12.5 Automatic Seating Management`:
- TD kann Seating/Balancing automatisch verwalten.
- Wichtige Einstellungen:
  - automatische Movement-Vorschläge
  - maximal erlaubte Player-Disparität vor Movement
  - automatisches Akzeptieren von Movement-Vorschlägen
  - zufällige Final-Table-Seating-Option
  - Option: Spieler möglichst nah an Seat 1 setzen
- TD triggert automatisches Rebalancing bei Turnierereignissen wie:
  - Buy-in
  - Bust-out
  - Rebuy
  - Undo solcher Aktionen
- Movement-Vorschläge können ganz oder teilweise akzeptiert werden.

### 3) Dealer Button relevant für Balancing
Aus Doku Abschnitt `12.8 Placing Dealer Buttons` und FAQ:
- Dealer Buttons beeinflussen, welche Spieler zum Balancing bewegt werden.
- Wenn Dealer Buttons gesetzt sind, versucht TD Spieler so zu bewegen, dass sie relativ zum Dealer Button denselben Abstand behalten.
- Praktischer Workflow laut Doku/FAQ:
  - erst wenn TD ein Balance-Suggestion-Dialog zeigt,
  - dann `Set Dealer Buttons` drücken,
  - Dealer Buttons setzen,
  - TD berechnet sofort neue Movement-Suggestion relativ zum Dealer Button neu.
- FAQ bestätigt denselben empfohlenen Workflow.

### 4) Locking Players / feste Dealer
Aus Doku Abschnitt `12.9 Locking Players` und FAQ:
- Spieler können auf Seats gelockt werden.
- Gelockte Spieler werden beim Rebalancing möglichst nicht bewegt.
- Nur wenn es keine andere Wahl gibt, werden sie trotzdem bewegt.
- Wenn ein gelockter Spieler bewegt werden muss, wandert der Lock mit.

### 5) Seats als unavailable markieren
Aus Doku Abschnitt `12.10 Making Seats Unavailable`:
- Seats können als unavailable markiert werden.
- TD behandelt unavailable Seats so, als existierten sie nicht.
- Praktisch wichtig, wenn Tische effektiv kleiner gemacht werden oder ein Seat für Nicht-Spieler/Dealer reserviert wird.

### 6) Kollapsreihenfolge von Tischen
Aus Doku Abschnitt `12.11 Controlling How Your Tables Collapse`:
- TD erlaubt Tabellen in Gruppen zu priorisieren:
  - zuerst kollabieren
  - zuletzt kollabieren
  - keine Präferenz
- Optional kann die Reihenfolge innerhalb der Listen fixiert werden.
- TD macht dabei nur "best effort"; nicht jede Situation kann exakt den Wunsch erfüllen.
- Frühere Final-Table-Designation wurde durch Collapse-Order ersetzt.

### 7) Seating komprimieren
Aus Doku Abschnitt `12.12 Compressing Seating`:
- Leere Sitze zwischen Spielern können eliminiert werden, ohne die Reihenfolge der Spieler zu verändern.
- Das ist wichtig, falls spätere Integration interne Sitzordnungen konsistent halten soll.

### 8) Letzten Balance-Vorgang ansehen
Aus Doku Abschnitt `12.13 Viewing the Last Balance`:
- TD kann den letzten Balance-/Movement-Vorgang erneut anzeigen.
- Das ist potenziell nützlich für Audit/Abgleich mit externer Bilderkennung.

### 9) Import mit Table Name + Seat Number
Aus Doku Abschnitt `Players Import`:
- Wenn beim CSV-Import die Spalten `Table Name` und `Seat Number` enthalten sind, können Spieler beim Import automatisch gesetzt werden.
- Voraussetzung: `Allow seating of players who have not bought-in` muss aktiviert sein.
- Das ist ein möglicher Integrationspfad, falls externe Logik Sitzpläne vorbereitet und per Import einspeist.

### 10) Exporting Tables / Seating Assignments
Aus Doku Abschnitt `12.20 Exporting Tables`:
- Tabellen und Seating Assignments können exportiert werden.
- Formate:
  - Diagram Format
  - Player List Format
- Generell unterstützt TD CSV/HTML-Exporte an vielen Stellen.
- Das spricht für eine Integrationsstrategie über Import/Export statt direkte In-Process-API.

### 11) Technologischer Unterbau
Aus Change Log 3.7:
- Die App wurde auf Electron umgestellt.
- Das kann später relevant sein für UI-Automation / Screen-Scraping / Dateibasiertes Bridging.

## Vorläufige Integrationshypothese
Aktuell gibt es in den gesicherten offiziellen Quellen keinen sofort sichtbaren öffentlichen API-Hinweis.
Daher sind die wahrscheinlichsten Integrationspfade:
1. **Screen-/Vision-basierter Parallelbetrieb** mit externer Logik
2. **CSV/HTML Import/Export** für Seating / Players / Reports
3. **UI-Automation auf Windows** falls direkte Datenschnittstelle fehlt
4. **Dokument-/Template-basierte Kopplung** über Exporte und Re-Importe

## Nächste sinnvolle Recherche
- prüfen, ob es offizielle Import-/Export-Formate für Seating Movements gibt
- prüfen, ob Forum/Support offizielle Aussagen zu externer Automatisierung/API enthält
- prüfen, welche Daten im Tournament-History-/Export enthalten sind
- prüfen, ob TDA/WSOP/EPT Seat-Balancing-Regeln als externe Policy-Engine modelliert werden sollten
