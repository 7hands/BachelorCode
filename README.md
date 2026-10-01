# Gestenerkennung von Taktfiguren mittels Multi-Camera-Arrays

Code zur Bachelorarbeit: Erkennung der Taktfigur eines Dirigenten (Ein-, Zwei-,
Drei-, Vier- und Sechs-Schlag-Takt) in Echtzeit aus Handlandmarks. Raspberry
Pis eines Multi-Camera-Arrays nehmen synchron Bilder auf, ein PC berechnet mit
MediaPipe die Landmarks und klassifiziert die Taktfigur mit TCN- bzw.
GCN-basierten Modellen.

## Dateien

| Datei | Aufgabe |
|---|---|
| `merkmale.py` | Aufbereitung der Landmark-Spur und Eingaben der Modelle (Training und live identisch) |
| `modelle.py` | alle Netze (TCN, ST-GCN, Stufe 1 und 2 der indirekten Modelle) und ihr Bauplan |
| `daten.py` | Datensatz laden, säubern, zwei Kameras fusionieren, gemittelte Bahnen |
| `training.py` | Splits, Training, Auswertung je Aufnahme (Vergleich der Fenstergrößen) |
| `takt_training.py` | Training und Auswertung je Takt, speichert die Modelle |
| `live_pc.py` | Echtzeitsystem, PC-Seite: Trigger, MediaPipe, Klassifikation |
| `live_pi_raw.py` | Echtzeitsystem, Pi-Seite: Bild auf Trigger aufnehmen und senden |
| `live_pi.py` | Variante mit MediaPipe auf dem Pi (Messung der Pi-Leistung) |
| `record_pc_raw.py`, `record_pi_raw.py` | Aufnahme des Datensatzes mit Metronom |
| `pruef_checkpoint.py` | prüft gespeicherte Modelle auf dem Live-Pfad, ohne Kamera |
| `landmarks_bild.py` | Live-Bild mit Landmarks, Bahn und vorhergesagter Taktart (Fotos) |
| `video_landmarks_rechts.py` | Landmarks der dirigierenden Hand aus den IDEA-Videos |
| `klick_zeiten.py`, `takt_klicks.py` | Klicks und Taktgrenzen aus der Tonspur der IDEA-Videos |
| `konfusion_sampo.py` | Konfusionsmatrizen je Takt auf den IDEA-Videos |
| `vergleich_paper.py` | Vergleich mit Chin-Shyurng et al. (2019): nur 2/4, 3/4, 4/4 |

## Einrichtung

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Erwartete Ordner im Arbeitsverzeichnis:

- `Datensatz/` – die Aufnahmen, `P<Person>_<Takt>-4_<bpm>bpm_<Zeitstempel>.npz`
  (Personen 1, 2, 3) mit `lm` (Kameras × Frames × 21 × 3), `t`, `clicks`, `cam_ids`,
  `bpm`, `meter`, `count_in_bars`. Ein Teil der Aufnahmen von Person 3 (Teil 3b,
  erkannt am Zeitstempel, siehe `daten.py`) stammt zum Teil von einer vierten Person
  und wird nur im Training und im Test auf ungesehene Personen verwendet.
- `models/` – gespeicherte Modelle aus `takt_training.py`
- für die IDEA-Auswertung: `video_npz/` und `takte_<Video>.csv` – nicht enthalten,
  weil die Videos nicht von mir stammen. Sie lassen sich aus den Videos des IDEA
  Open Movement Dataset (Lemouton, Bevilacqua et al. 2019) erzeugen:
  `video_landmarks_rechts.py --konf 0.02` (Landmarks), `klick_zeiten.py` und
  `takt_klicks.py` (Taktgrenzen aus den Metronomklicks der Tonspur).

Selbsttests ohne Daten: `python merkmale.py`, `python modelle.py`,
`python record_pc_raw.py --selftest`, `python live_pi.py --selfcheck`.

## Ergebnisse der Arbeit reproduzieren

Mit den gespeicherten Modellen in `models/` (ohne neues Training, Ergebnisse
identisch zu denen der Arbeit):

```bash
python takt_training.py --selfcheck       # Splits und Taktzahlen anzeigen
python takt_training.py --nur-auswerten   # Ablation je Takt -> takt_ergebnisse.csv
python konfusion_sampo.py                 # Konfusionsmatrizen und Tabelle je Video
python vergleich_paper.py                 # Vergleich mit Chin-Shyurng et al.
```

Neu trainieren (GPU empfohlen; die Zahlen können wegen nicht deterministischer
GPU-Rechnung leicht von denen der Arbeit abweichen):

```bash
python takt_training.py                   # alle Modelle, 4 Splits, 10 Seeds, je Takt
python training.py                        # Vergleich der Fenstergrößen, je Aufnahme
```

Achtung: `takt_training.py` ohne `--nur-auswerten` überschreibt die Modelle in `models/`.

Hinweis zu Tabelle 6.9 der Arbeit (Erkennungsrate je IDEA-Video): Sie stammt aus
`sampo_ergebnisse.csv`, einem früheren Lauf, der eigene Modelle trainiert und nicht
gespeichert hat, noch bevor die gemittelten Bahnen je Split nur aus den
Trainingsaufnahmen gebildet wurden. Die Tabelle je Video, die `konfusion_sampo.py`
aus den gespeicherten Modellen erzeugt (`ergebnis_je_video.csv`), weicht deshalb
in einzelnen Zellen um bis zu 6 Punkte ab, im Durchschnitt je Modell um höchstens
2 Punkte; die Rangfolge der Modelle ist dieselbe. Die Konfusionsmatrizen und alle
Zahlen im Text zu den IDEA-Videos entsprechen den gespeicherten Modellen.

Alle Einstellungen stehen oben in den Skripten und lassen sich als Flag
überschreiben, z. B. `--modelle pos4,TCN->TCN --seeds 3`.

## Ergebnisse der Arbeit (`ergebnisse/`)

| Datei | Inhalt | In der Arbeit |
|---|---|---|
| `fenster_vergleich.csv` | Training je Fenstergröße, Auswertung je Aufnahme | Tabellen 6.1–6.7 |
| `takt_ergebnisse.csv` | Ablation je Takt (aus `models/`, `--nur-auswerten`) | Ablation je Takt |
| `konfusion_sampo_takt.csv` | Konfusionsmatrizen auf den IDEA-Videos | Abbildungen 6.1–6.6 |
| `ergebnis_je_video.csv` | Erkennungsrate je IDEA-Video aus `models/` | vgl. Tabelle 6.9 |
| `sampo_ergebnisse.csv` | Erkennungsrate je IDEA-Video, früherer Lauf | Tabelle 6.9 |
| `vergleich_paper.csv` | nur 2/4, 3/4, 4/4, Vergleich mit Chin-Shyurng et al. | Vergleichstabelle |

Eine Zeile je Split, Modell, Fenster und Seed; Mittelwerte entstehen beim Lesen.
Splits: `ohne 1`, `ohne 2`, `ohne 3` (Person ungesehen) und `alle Personen (80/20)`.
Die Skripte schreiben neue CSVs ins Arbeitsverzeichnis, nicht nach `ergebnisse/`.

## Echtzeitsystem

```bash
# auf jedem Pi
python live_pi_raw.py --id 1 --ip <IP des PCs>
# auf dem PC
python live_pc.py --checkpoint models/TCN_TCN_alle_Personen_80_20_s0.pt --takt 4
```

Der PC sendet mit 25 Hz einen Aufnahmezeitpunkt, die Pis nehmen genau dann ein
Bild auf (820×616) und schicken es zurück. Ausgegeben wird die Mehrheit der
letzten neun Vorhersagen; mit `--takt` am Ende die Trefferquote.
