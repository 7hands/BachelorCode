"""Vergleich mit Chin-Shyurng et al. (2019): nur 2/4-, 3/4- und 4/4-Takte.

Wertet die gespeicherten Modelle aus models/ (takt_training.py) je Takt
aus, genau wie takt_training.py, aber nur auf Aufnahmen mit 2, 3 oder 4
Schlaegen. Zwei Varianten:
  alle5  - das Modell darf wie trainiert alle fuenf Klassen sagen
  nur234 - Argmax nur ueber 2, 3, 4 (wie das Paper-System, das nur diese kennt)

Testsets: "bekannt" = die 20 % aus "alle Personen (80/20)" (alle Personen im
Training), "ungesehen" = die gehaltene Person aus den Splits ohne <Person>.

  python vergleich_paper.py [--modelle pos4,TCN->TCN] [--seeds 10]
  python vergleich_paper.py --tabelle        # nur aus der CSV drucken
"""
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

import merkmale as MM
import takt_training as TT
import training as TR

MODELLE = TR._f("modelle", ["pos4", "tips6pos", "TCN->TCN"])
SEEDS = TR._f("seeds", 10)
CSV = TR._f("csv", "vergleich_paper.csv")
DREI = (2, 3, 4)
MODELLDIR = TR._f("modelldir", "models")
TESTS = [(TR.ALLE, "bekannt")] + [(f"ohne {p}", "ungesehen") for p in TR.LOPO]
NAMEN = {"pos4": "Ohne Ableitung, 2 Punkte", "tips6pos": "Ohne Ableitung, 6 Punkte",
         "TCN->TCN": "Indirekt mit TCN", "voll12": "Mit Ableitung"}
SPALTEN = ["modell", "gruppe", "split", "seed", "variante", "takt", "bpm", "n", "richtig"]


def logits(mn, mod, stufe1, rec, win, w1):
    """-> (Logits je Fenster, Fensterenden) wie takt_training.vorhersagen."""
    if stufe1:
        c = TR.kurve(*stufe1, rec, w1, mn == "GCN->TCN")
        ends = list(range(w1 + win, len(c) + 1, MM.STEP)) if c is not None else []
        xs = [c[e - win:e].T for e in ends]
    else:
        build = TR.bauer(mn)
        ends = list(range(win, len(rec["raw"]) + 1, MM.STEP))
        xs = [build(rec, e - win, e) for e in ends]
    if not ends:
        return None, []
    x = torch.tensor(np.stack(xs), dtype=torch.float32).to(TR.DEVICE)
    with torch.no_grad():
        return torch.cat([mod(x[i:i + 256]) for i in range(0, len(x), 256)]).cpu().numpy(), ends


def tabelle(zeilen):
    """Gepoolt ueber Seeds (jeder Takt eines Seeds zaehlt einmal)."""
    z = defaultdict(lambda: [0, 0])
    for r in zeilen:
        for k in ((r["modell"], r["gruppe"], r["variante"], int(r["takt"])),
                  (r["modell"], r["gruppe"], r["variante"], "ges"),
                  (r["modell"], r["gruppe"], r["variante"], "bpm", int(r["bpm"]))):
            z[k][0] += int(r["richtig"]); z[k][1] += int(r["n"])
    q = lambda k: f"{100 * z[k][0] / z[k][1]:.1f}" if z[k][1] else "–"
    modelle = [m for m in NAMEN if any(r["modell"] == m for r in zeilen)]
    for var, text in (("nur234", "Vorhersage nur 2/3/4 (wie das Paper)"),
                      ("alle5", "Vorhersage aus allen fuenf Klassen")):
        print(f"\n### Nur 2/4-, 3/4-, 4/4-Takte, je Takt [%] — {text}\n")
        print("| Modell | Personen | 2/4 | 3/4 | 4/4 | gesamt | Takte |")
        print("|---|---|---|---|---|---|---|")
        for mo in modelle:
            for gr in ("bekannt", "ungesehen"):
                k = (mo, gr, var)
                print(f"| {NAMEN[mo]} | {gr} | {q(k + (2,))} | {q(k + (3,))} | "
                      f"{q(k + (4,))} | **{q(k + ('ges',))}** | {z[k + ('ges',)][1]} |")
    bpms = sorted({int(r["bpm"]) for r in zeilen})
    print("\n### Nach Tempo, nur 2/3/4 vorhergesagt [%] (Takte in Klammern)\n")
    print("| Modell | Personen | " + " | ".join(f"{b} BPM" for b in bpms) + " |")
    print("|---|---|" + "---|" * len(bpms))
    for mo in modelle:
        for gr in ("bekannt", "ungesehen"):
            k = (mo, gr, "nur234", "bpm")
            print(f"| {NAMEN[mo]} | {gr} | " + " | ".join(
                f"{q(k + (b,))} ({z[k + (b,)][1]})" for b in bpms) + " |")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    if "--tabelle" in sys.argv:
        tabelle(list(csv.DictReader(open(CSV))))
        sys.exit()
    verteilung = TR.splits(TR.lade_daten())
    idx = [TR.KLASSEN.index(k) for k in DREI]
    zeilen = []
    for MODELL in MODELLE:
        for sp, gruppe in TESTS:
            _, tests = verteilung[sp]
            name = next(k for k in tests if k.startswith(gruppe[:4]))
            test = [r for r in tests[name] if int(r["meter"]) in DREI]
            for sd in range(SEEDS):
                mod, stufe1, win, w1 = TT.lade_checkpoint(os.path.join(MODELLDIR, TT.dateiname(MODELL, sp, sd)))
                z = defaultdict(lambda: [0, 0])
                for r in test:
                    lg, ends = logits(MODELL, mod, stufe1, r, win, w1)
                    if lg is None:
                        continue
                    bpm = int(re.search(r"(\d+)bpm", r["name"]).group(1))
                    alle5 = lg.argmax(1)
                    nur234 = np.asarray(idx)[lg[:, idx].argmax(1)]
                    for var, p in (("alle5", alle5), ("nur234", nur234)):
                        for takt, ok in TT.je_takt(p, ends, r, TT.RAND):
                            z[(var, takt, bpm)][0] += ok
                            z[(var, takt, bpm)][1] += 1
                for (var, takt, bpm), (ok, n) in z.items():
                    zeilen.append(dict(modell=MODELL, gruppe=gruppe, split=sp, seed=sd,
                                       variante=var, takt=takt, bpm=bpm, n=n, richtig=ok))
                del mod, stufe1
                print(f"  {MODELL:<9} {sp:<22} Seed {sd} fertig")
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SPALTEN); w.writeheader(); w.writerows(zeilen)
    print(f"-> {CSV}")
    tabelle(zeilen)
