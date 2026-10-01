"""Konfusionsmatrix je Takt auf den sampo-Videos -> konfusion_sampo_takt.csv/.png

Zeilen: tatsaechliche Taktlaenge aus dem Klicktrack. Spalten: Vorhersage
(1, 2, 3, 4, 6). Je Takt entscheidet die Mehrheit der Fenster, die in ihm enden.
Die Zeilen sind auf 1 normiert, leere Felder stehen als 0.00 da.

Die Modelle kommen aus models/ (Checkpoints aus takt_training.py, 10 Seeds).
Die Landmarks der Videos erzeugt video_landmarks_rechts.py, die Taktgrenzen
(takte_<Video>.csv) takt_klicks.py aus den betonten Klicks der Tonspur.

  python konfusion_sampo.py [--split "alle Personen (80/20)"] [--seeds 10]
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import daten as DS
import merkmale as MM
import takt_training as TT
import training as TR

MODELLE = TR._f("modelle", ["voll12", "pos4", "tips6pos", "TCN->TCN", "GCN", "GCN->TCN"])
SPLITS = TR._f("splits", [TR.ALLE])
SEEDS = TR._f("seeds", 10)
MODELLDIR = TR._f("modelldir", "models")
CSV = TR._f("csv", "konfusion_sampo_takt.csv")
CSV_VIDEO = TR._f("csv_video", "ergebnis_je_video.csv")   # Tabelle je Video und Modell
VIDEOS = ["sampo_antoine_ircam_12", "sampo_antoine_ircam_13",
          "sampo_nikita_ircam_10", "sampo_nikita_ircam_11"]
SPUR = "_right_k002"
SOLL = [2, 3, 4, 6]        # 5er und 7er weggelassen: nicht im Klassenraum
NAMEN = {"voll12": "Mit Ableitung", "pos4": "Ohne Ableitung, 2 Punkte",
         "tips6pos": "Ohne Ableitung, 6 Punkte", "TCN->TCN": "Indirekt mit TCN",
         "GCN": "GCN", "GCN->TCN": "Indirekt mit GCN"}
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def takte(v):
    return [(float(r["start_s"]), float(r["ende_s"]), int(r["schlaege"]))
            for r in csv.DictReader(open(f"takte_{v}.csv"))]


def lade(mn, split, seed):
    return TT.lade_checkpoint(os.path.join(MODELLDIR, TT.dateiname(mn, split, seed)))


def bild(zaehl, mo, split):
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"wspace": 0.35})
    for j, ber in enumerate(("antoine", "nikita")):
        A = np.array([[zaehl[(mo, split, ber, s, p)] for p in TR.KLASSEN] for s in SOLL], float)
        Z = A / np.maximum(A.sum(1, keepdims=True), 1)
        a = ax[j]
        bi = a.imshow(Z, cmap="Greys", vmin=0, vmax=1)
        for y in range(len(SOLL)):
            for x in range(len(TR.KLASSEN)):
                a.text(x, y, f"{Z[y, x]:.2f}", ha="center", va="center",
                       fontsize=8, color="white" if Z[y, x] > 0.55 else "black")
        a.set_xticks(range(len(TR.KLASSEN)), [str(k) for k in TR.KLASSEN], fontsize=8)
        a.set_yticks(range(len(SOLL)), [str(s) for s in SOLL], fontsize=8)
        a.set_xlabel("Vorhersage", fontsize=8.5)
        if j == 0:
            a.set_ylabel("tatsächlich", fontsize=8.5)
        a.set_title(f"{ber.capitalize()} ({int(A.sum())} Entscheidungen)", fontsize=9, loc="left")
    cb = fig.colorbar(bi, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.tick_params(labelsize=7.5)
    # Bilder der Arbeit ohne Titel (Modellname steht in der Caption);
    # nur die Ausnahme-Splits bekommen einen, damit man sie auseinanderhaelt
    if split != TR.ALLE:
        fig.suptitle(f"{NAMEN.get(mo, mo)} — {split}, {SEEDS} Seeds", fontsize=9.5)
    # Standardsplit behaelt die alten Dateinamen, damit sie im LaTeX-Ordner
    # nur ersetzt werden muessen
    datei = ("konfusion_" + re.sub(r"[^A-Za-z0-9]+", "_", mo).strip("_")
             + ("" if split == TR.ALLE
                else "_" + re.sub(r"[^A-Za-z0-9]+", "_", split).strip("_").lower())
             + ".png")
    fig.savefig(datei, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"-> {datei}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    zaehl = defaultdict(int)                   # (modell, split, bereich, soll, pred) -> n
    if "--bilder" in sys.argv:                 # nur neu zeichnen, nichts rechnen
        for z in csv.DictReader(open(CSV)):
            zaehl[(z["modell"], z["split"], z["bereich"], int(z["soll"]),
                   int(z["vorhersage"]))] += int(z["anzahl"])
        for split in SPLITS:
            for mo in MODELLE:
                if any(k[0] == mo and k[1] == split for k in zaehl):
                    bild(zaehl, mo, split)
        raise SystemExit

    daten = {v: (DS.lade_video(f"video_npz/{v}{SPUR}.npz"), takte(v)) for v in VIDEOS}
    je_video = []                              # (modell, split, video, seed, takte, richtig)
    for split in SPLITS:
        for mo in MODELLE:
            for sd in range(SEEDS):
                mod, stufe1, win, w1 = lade(mo, split, sd)
                for v in VIDEOS:
                    rec, tk = daten[v]
                    if stufe1:
                        c = TR.kurve(*stufe1, rec, w1, mo == "GCN->TCN")
                        ends = list(range(w1 + win, len(c) + 1, MM.STEP))
                        xs = np.stack([c[e - win:e].T for e in ends])
                    else:
                        build = TR.bauer(mo)
                        ends = list(range(win, len(rec["raw"]) + 1, MM.STEP))
                        xs = np.stack([build(rec, e - win, e) for e in ends])
                    x = torch.tensor(xs, dtype=torch.float32).to(TR.DEVICE)
                    with torch.no_grad():
                        p = torch.cat([mod(x[i:i + 256]) for i in range(0, len(x), 256)]
                                      ).argmax(1).cpu().numpy()
                    zeit = (np.asarray(ends) - 1) / MM.FPS
                    n = ok = 0
                    for a, b, s in tk:
                        m = (zeit >= a) & (zeit < b)
                        if not m.any():
                            continue
                        pred = TR.KLASSEN[Counter(p[m].tolist()).most_common(1)[0][0]]
                        zaehl[(mo, split, "antoine" if "antoine" in v else "nikita", s, pred)] += 1
                        if s in TR.KLASSEN:            # 5er und 7er liegen ausserhalb
                            n += 1; ok += pred == s
                    je_video.append((mo, split, v, sd, n, ok))
                    del x
                del mod, stufe1
                if TR.DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  {split:<22} {mo:<11} Seed {sd} fertig")

    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["modell", "split", "bereich", "soll", "vorhersage", "anzahl"])
        for (mo, sp, ber, s, p), n in sorted(zaehl.items()):
            w.writerow([mo, sp, ber, s, p, n])
    print(f"-> {CSV}")
    with open(CSV_VIDEO, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["modell", "split", "video", "seed", "takte_n", "takte_richtig"])
        w.writerows(je_video)
    print(f"-> {CSV_VIDEO}")

    # Erkennungsrate je Takt und Video, Mittel ueber die Seeds (ohne 5er und 7er)
    print("\n| Modell | " + " | ".join(v.replace("sampo_", "").replace("_ircam", "")
                                       for v in VIDEOS) + " | Durchschnitt |")
    print("|---" * (len(VIDEOS) + 2) + "|")
    for split in SPLITS:
        for mo in MODELLE:
            q = [[100 * ok / n for m, sp, v, _, n, ok in je_video
                  if m == mo and sp == split and v == vid and n] for vid in VIDEOS]
            if all(q):
                werte = [np.mean(x) for x in q]
                print(f"| {NAMEN.get(mo, mo)} | " + " | ".join(f"{x:.1f}" for x in werte)
                      + f" | {np.mean(werte):.1f} |")

    for split in SPLITS:
        for mo in MODELLE:
            if any(k[0] == mo and k[1] == split for k in zaehl):
                bild(zaehl, mo, split)
