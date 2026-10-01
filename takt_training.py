"""Training und Auswertung JE TAKT - die Ergebnisse der Arbeit.

Ein Takt = ein Testfall (Taktgrenzen aus den Metronom-Klicks), Genauigkeit =
richtige Takte / alle Takte. Je Takt entscheidet die Mehrheit der Fenster,
deren ENDE im Takt plus Rand liegt; ein Fenster darf nach hinten in die
vorigen Takte reichen, genau wie live.

Bewertet werden beide Testsets jedes Splits: die zurueckgehaltenen 20 % der
Trainingspersonen ("bekannt") und die gehaltene Person ("ungesehen"). Eine
Zeile je (Split, Modell, Fenster, Seed, Testset) mit Rohwerten. Die CSV wird
angehaengt, schon gerechnete Kombinationen werden uebersprungen. Die Modelle
werden als Checkpoints gespeichert; live_pc.py laedt sie.

  python takt_training.py --selfcheck
  python takt_training.py
  python takt_training.py --modelle pos4,TCN->TCN --seeds 3
  python takt_training.py --nur-auswerten     # gespeicherte Modelle aus models/
                                              # auswerten statt neu trainieren
"""

# ═══════════════════════════ EINSTELLUNGEN ═══════════════════════════════════

MODELLE = ["voll12", "pos4", "tips6pos", "GCN", "TCN->TCN", "GCN->TCN"]

FENSTER = [50]                  # direkte Modelle
KASKADE_W1 = [50]               # Stufe 1 der indirekten Modelle
KASKADE_W2 = [75]               # Stufe 2

SPLITS = None                   # None = alle: ohne <Person> je Person + 80/20

SEEDS = 10
RAND = 12                       # Frames Rand je Seite um den Takt (12 = 0,5 s)
AUCH_JE_AUFNAHME = True         # Mehrheit je Aufnahme zum Vergleich mitschreiben

CSV = "takt_ergebnisse.csv"
MODELLDIR = "models"            # leer = nichts speichern
MODELL_SPLIT = ""               # nur aus diesem Split speichern ("" = alle Splits)
MODELL_SEEDS = 0                # nur die ersten n Seeds speichern (0 = alle)

# ═════════════════════════════════════════════════════════════════════════════

import csv
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch

import daten as DS
import merkmale as MM
import modelle as MD
import training as TR

_f = TR._f
MODELLE = _f("modelle", MODELLE)
FENSTER = _f("fenster", FENSTER, int)
KASKADE_W1 = _f("kaskade_w1", KASKADE_W1, int)
KASKADE_W2 = _f("kaskade_w2", KASKADE_W2, int)
SPLITS = _f("splits", SPLITS) or TR.SPLITS
SEEDS = _f("seeds", SEEDS)
RAND = _f("rand", RAND)
AUCH_JE_AUFNAHME = _f("auch_je_aufnahme", AUCH_JE_AUFNAHME)
CSV = _f("csv", CSV)
MODELLDIR = _f("modelldir", MODELLDIR)
MODELL_SPLIT = _f("modell_split", MODELL_SPLIT)
MODELL_SEEDS = _f("modell_seeds", MODELL_SEEDS)

NUR_AUSWERTEN = "--nur-auswerten" in sys.argv
KASK = TR.KASK
SPALTEN = ["datensatz", "split", "testset", "gehalten", "modell", "fenster", "w1",
           "rand", "seed", "epochen", "block", "dilationen", "params", "takte_n",
           "takte_richtig", "takte_macro", "je_taktart", "aufnahmen_n",
           "aufnahmen_richtig", "sek"]


def plan():
    aus = []
    for sp in SPLITS:
        for mn in MODELLE:
            for w2 in (KASKADE_W2 if mn in KASK else FENSTER):
                for w1 in (KASKADE_W1 if mn in KASK else [0]):
                    aus.append((sp, mn, w2, w1))
    return aus


def je_takt(pred, ends, rec, rand):
    """-> Liste (Taktart, richtig) fuer jeden Takt der Aufnahme."""
    down = np.asarray(sorted(int(d) for d in rec["down"]))
    soll = TR.KLASSEN.index(int(rec["meter"]))
    e = np.asarray(ends)
    aus = []
    for a, b in zip(down[:-1], down[1:]):
        m = (e - 1 >= a - rand) & (e - 1 < b + rand)
        if m.any():
            aus.append((int(rec["meter"]),
                        Counter(pred[m].tolist()).most_common(1)[0][0] == soll))
    return aus


def checkpoint(mn, sp, sd, win, w1, mod, mi=None, md=None):
    """Alles, was live_pc.py zum Nachbauen braucht."""
    cpu = lambda m: {k: v.cpu() for k, v in m.state_dict().items()}
    ck = {"arch": mn, "klassen": TR.KLASSEN, "convs": TR.BLOCK_CONVS,
          "wn": TR.BLOCK_WEIGHTNORM, "fps": MM.FPS, "split": sp,
          "seed": sd, "quelle": "takt_training.py"}
    if mn in KASK:
        ck |= {"variante": "graph" if mn == "GCN->TCN" else "pos4",
               "w1": w1, "w2": win, "hop1": TR.HOP1,
               "dils1": list(TR.dils(w1)), "dils2": list(TR.dils(win)),
               "s1_ictus": cpu(mi), "s1_down": cpu(md), "s2": cpu(mod)}
    else:
        ck |= {"variante": "graph" if mn in MD.GRAPH else mn,
               "fenster": win, "dils": list(TR.dils(win)), "state": cpu(mod)}
    return ck


def lade_checkpoint(pfad):
    """Checkpoint -> (Modell, [Stufe-1-Ictus, Stufe-1-Downbeat] oder None, Fenster, W1),
    auf TR.DEVICE, fuer die Offline-Auswertung (konfusion_sampo, vergleich_paper)."""
    ck = torch.load(pfad, map_location="cpu")
    dev = TR.DEVICE
    if "s2" in ck:
        s1 = "beat_gcn" if ck["variante"] == "graph" else "beat"
        stufe1 = []
        for key in ("s1_ictus", "s1_down"):
            m = MD.baue(s1, tuple(ck["dils1"]), ck["convs"], ck["wn"], cin=4)
            m.load_state_dict(ck[key]); stufe1.append(m.eval().to(dev))
        m2 = MD.baue("clf", tuple(ck["dils2"]), ck["convs"], ck["wn"], cin=4)
        m2.load_state_dict(ck["s2"])
        return m2.eval().to(dev), stufe1, ck["w2"], ck["w1"]
    m = MD.baue(ck["arch"], tuple(ck["dils"]), ck["convs"], ck["wn"], len(ck["klassen"]))
    m.load_state_dict(ck["state"])
    return m.eval().to(dev), None, ck["fenster"], 0


def dateiname(mn, sp, sd):
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{mn}_{sp}_s{sd}").strip("_") + ".pt"


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    P = plan()
    print(f"Modelle {MODELLE}")
    print(f"Fenster {FENSTER} | indirekt W1 {KASKADE_W1} W2 {KASKADE_W2}")
    print(f"Splits {SPLITS} | {SEEDS} Seeds | {TR.EPOCHEN} Epochen | Rand {RAND} Frames")
    print(f"-> {len(P) * SEEDS} Durchgaenge, Ziel {CSV}")
    print(f"Modelle werden aus {MODELLDIR}/ geladen, nicht trainiert\n" if NUR_AUSWERTEN
          else f"Checkpoints: {MODELLDIR}/\n" if MODELLDIR else "Checkpoints: keine\n")

    verteilung = TR.splits(TR.lade_daten())
    if "--selfcheck" in sys.argv:
        for sp in SPLITS:
            tr, tests = verteilung[sp]
            print(f"  {sp:<24} Training {len(tr):>4} Spuren")
            for k, v in tests.items():
                n = sum(max(len(np.asarray(r["down"])) - 1, 0) for r in v)
                print(f"      {k:<26}{len(v):>4} Spuren, {n:>5} Takte")
        sys.exit()

    neu = not os.path.exists(CSV)
    fertig = set()
    if not neu:
        if next(csv.reader(open(CSV)), []) != SPALTEN:
            sys.exit(f"{CSV} hat einen anderen Spaltensatz - umbenennen oder --csv setzen.")
        for z in csv.DictReader(open(CSV)):
            fertig.add((z["split"], z["modell"], int(z["fenster"]),
                        int(z["w1"] or 0), int(z["seed"])))
        offen = sum(1 for sp, mn, win, w1 in P for sd in range(SEEDS)
                    if (sp, mn, win, w1, sd) not in fertig)
        print(f"{len(fertig)} Zeilen schon in {CSV}, {offen} Durchgaenge offen")
        if not offen:
            sys.exit(f"\nNichts zu tun - {CSV} enthaelt bereits alles. Fuer einen neuen "
                     f"Lauf die Datei umbenennen\noder --csv setzen (sonst werden auch "
                     f"keine Checkpoints geschrieben).")
    fh = open(CSV, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=SPALTEN)
    if neu:
        w.writeheader(); fh.flush()

    t_all = time.time()
    for sp, mn, win, w1 in P:
        if sp not in verteilung:
            print(f"  Split '{sp}' unbekannt - uebersprungen"); continue
        tr, tests = verteilung[sp]
        for sd in range(SEEDS):
            if (sp, mn, win, w1, sd) in fertig:
                continue
            t0 = time.time()
            torch.manual_seed(sd); np.random.seed(sd)
            kurven, npar, mi, md, X, Y = None, 0, None, None, None, None
            if NUR_AUSWERTEN:
                # gespeichertes Modell statt Training - liefert dieselben Zahlen
                mod, stufe1, _, _ = lade_checkpoint(os.path.join(MODELLDIR,
                                                                 dateiname(mn, sp, sd)))
                if stufe1:
                    mi, md = stufe1
                    npar = sum(p.numel() for m in stufe1 for p in m.parameters())
                    kurven = {r["name"]: TR.kurve(mi, md, r, w1, mn == "GCN->TCN")
                              for v in tests.values() for r in v}
            elif mn in KASK:
                graph = mn == "GCN->TCN"
                mi = TR.stufe1(tr, "ictus", sd, w1, graph)
                md = TR.stufe1(tr, "down", sd, w1, graph)
                npar = sum(p.numel() for p in mi.parameters()) \
                    + sum(p.numel() for p in md.parameters())
                kurven = {r["name"]: TR.kurve(mi, md, r, w1, graph)
                          for r in list(tr) + [x for v in tests.values() for x in v]}
                X, Y = TR.stufe2_daten(tr, kurven, w1, win)
                torch.manual_seed(sd)
                mod = TR.fit(MD.baue("clf", TR.dils(win), TR.BLOCK_CONVS,
                                     TR.BLOCK_WEIGHTNORM, cin=4).to(TR.DEVICE), X, Y)
            else:
                X, Y = TR.stapel(tr, mn, win)
                torch.manual_seed(sd)
                mod = TR.fit(TR.netz(mn, win).to(TR.DEVICE), X, Y)
            npar += sum(p.numel() for p in mod.parameters())
            if MODELLDIR and not NUR_AUSWERTEN and (not MODELL_SEEDS or sd < MODELL_SEEDS) \
                    and sp.startswith(MODELL_SPLIT or ""):
                os.makedirs(MODELLDIR, exist_ok=True)
                torch.save(checkpoint(mn, sp, sd, win, w1, mod, mi, md),
                           os.path.join(MODELLDIR, dateiname(mn, sp, sd)))

            for tname, trecs in tests.items():
                takte, auf = [], []
                for r in trecs:
                    p, ends = TR.vorhersagen(mod, mn, r, win, w1,
                                             kurven[r["name"]] if kurven else None)
                    if not len(p):
                        continue
                    takte += je_takt(p, ends, r, RAND)
                    auf.append(Counter(p.tolist()).most_common(1)[0][0]
                               == TR.KLASSEN.index(int(r["meter"])))
                if not takte:
                    continue
                richtig = np.mean([x for _, x in takte])
                je_art = {m: float(np.mean([x for mm, x in takte if mm == m]))
                          for m in sorted({m for m, _ in takte})}
                macro = float(np.mean(list(je_art.values())))
                w.writerow({
                    "datensatz": "/".join(DS.DIRS), "split": sp, "testset": tname,
                    "gehalten": tname.split(":")[1].strip() if ":" in tname else "",
                    "modell": mn, "fenster": win, "w1": w1 or "", "rand": RAND,
                    "seed": sd, "epochen": TR.EPOCHEN,
                    "block": f"{TR.BLOCK_CONVS}conv" + ("+wn" if TR.BLOCK_WEIGHTNORM else ""),
                    "dilationen": "-".join(map(str, TR.dils(win))), "params": npar,
                    "takte_n": len(takte), "takte_richtig": f"{richtig:.4f}",
                    "takte_macro": f"{macro:.4f}",
                    "je_taktart": {k: round(v, 3) for k, v in je_art.items()},
                    "aufnahmen_n": len(auf) if AUCH_JE_AUFNAHME else "",
                    "aufnahmen_richtig": f"{np.mean(auf):.4f}" if AUCH_JE_AUFNAHME else "",
                    "sek": f"{time.time() - t0:.0f}"})
                fh.flush()
                kurz = "bekannt" if tname.startswith("bekannt") else tname.split(":")[1].strip()
                print(f"  {sp:<22}{mn:<10}"
                      + (f"W1={w1:<3}W2={win:<4}" if w1 else f"WIN={win:<8}")
                      + f"s{sd} {kurz:<14} Takte {richtig:.3f} (macro {macro:.3f}, "
                        f"n={len(takte)})  Aufnahmen {np.mean(auf):.3f}")
            del mod, X, Y, kurven, mi, md
            if TR.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    fh.close()
    print(f"\nfertig in {(time.time() - t_all) / 60:.0f} min -> {CSV}")
