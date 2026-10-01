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

- `datensatz/` – die Aufnahmen, `<Person>_<Takt>-4_<bpm>bpm_<Zeitstempel>.npz`
  mit `lm` (Kameras × Frames × 21 × 3), `t`, `clicks`, `cam_ids`, `bpm`, `meter`, `count_in_bars`
- `models/` – gespeicherte Modelle aus `takt_training.py`
- für die IDEA-Auswertung: `video_npz/` und `takte_<Video>.csv`

Selbsttests ohne Daten: `python merkmale.py`, `python modelle.py`,
`python record_pc_raw.py --selftest`, `python live_pi.py --selfcheck`.

## Ergebnisse der Arbeit reproduzieren

```bash
python takt_training.py --selfcheck     # Splits und Taktzahlen anzeigen
python takt_training.py                 # alle Modelle, 4 Splits, 10 Seeds, je Takt
python training.py                      # Vergleich der Fenstergrößen, je Aufnahme
python konfusion_sampo.py               # Konfusionsmatrizen auf den IDEA-Videos
python vergleich_paper.py               # Tabelle zum Vergleich mit Chin-Shyurng et al.
```

Alle Einstellungen stehen oben in den Skripten und lassen sich als Flag
überschreiben, z. B. `--modelle pos4,TCN->TCN --seeds 3`.

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
