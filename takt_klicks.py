"""Takte aus dem Klicktrack: betonte Klicks = Taktanfang.

Die sampo-Tonspuren enthalten zwei Klicktypen, die sich an der Spitzenfrequenz
sauber trennen lassen (hier 1100 Hz normal, 1570 Hz betont; der betonte ist
zusätzlich lauter). Der Abstand zwischen zwei betonten Klicks ist die
Taktlänge in Schlägen -- damit gibt es für diese Videos eine echte
Taktannotation, die nicht aus der Bewegung stammt.

  python takt_klicks.py                 # sampo_antoine_ircam_12
  python takt_klicks.py --alle

Schreibt takte_<name>.csv (Takt, Start, Ende, Schläge, Tempo) und .png.
"""
import csv
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Frühere Handannotation von antoine 12 (Sekunden von, Sekunden bis, Takt) - nur zum Vergleich im
# Bild; sie ist teilweise falsch (die 6/4-Abschnitte sind 5-6-7).
ANTOINE_HAND = [(0, 13, 6), (13, 33, 4), (43, 56, 4), (60, 71, 6)]
SRC = "Data/conductor-follower-master/datasets/conductors_movies_with_sound"
SR, SCHWELLE, PAUSE = 48000, 1e-4, 48000 // 50
arg = lambda f, s: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else s
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def klicks(name):
    """-> (Zeiten, Spitzenfrequenzen, Pegel) aller Klicks."""
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", os.path.join(SRC, name + ".aif"),
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
    x = np.frombuffer(p.stdout, np.float32)
    idx = np.where(np.abs(x) > SCHWELLE)[0]
    schnitt = np.where(np.diff(idx) > PAUSE)[0]
    a = np.concatenate([[idx[0]], idx[schnitt + 1]])
    b = np.concatenate([idx[schnitt], [idx[-1]]])
    n = int(0.02 * SR); fen = np.hanning(n); f = np.fft.rfftfreq(4096, 1 / SR)
    frq = np.array([f[np.argmax(np.abs(np.fft.rfft(x[i:i + n] * fen[:len(x[i:i + n])], 4096)))]
                    for i in a])
    pegel = np.array([np.abs(x[i:j + 1]).max() for i, j in zip(a, b)])
    return a / SR, frq, pegel


def takte(t, betont):
    """-> Liste (Start, Ende, Schläge) zwischen aufeinanderfolgenden Akzenten."""
    k = np.where(betont)[0]
    return [(t[i], t[j], j - i) for i, j in zip(k[:-1], k[1:])]


def auswerten(name):
    t, frq, pegel = klicks(name)
    betont = frq > np.median(frq) * 1.2
    tk = takte(t, betont)
    with open(f"takte_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["takt", "start_s", "ende_s", "schlaege", "tempo_pro_min"])
        for i, (a, b, s) in enumerate(tk):
            w.writerow([i, f"{a:.3f}", f"{b:.3f}", s, f"{60 * s / (b - a):.1f}"])
    folge = [s for _, _, s in tk]
    print(f"{name}: {len(t)} Klicks, davon {betont.sum()} betont "
          f"({np.median(frq[betont]):.0f} Hz / {np.median(pegel[betont]):.3f} gegen "
          f"{np.median(frq[~betont]):.0f} Hz / {np.median(pegel[~betont]):.3f})")
    print(f"   {len(tk)} Takte, Längen: {folge}")
    zahl = {s: folge.count(s) for s in sorted(set(folge))}
    print(f"   Verteilung der Taktlängen: {zahl}   "
          f"(ausserhalb unserer Klassen 1,2,3,4,6: "
          f"{sum(v for k, v in zahl.items() if k not in (1,2,3,4,6))} Takte)")

    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    ax.vlines(t[~betont], 0, 0.6, color="0.55", lw=0.8)
    ax.vlines(t[betont], 0, 1.0, color="k", lw=1.4)
    for a, b, s in tk:
        ax.text((a + b) / 2, 1.08, str(s), ha="center", va="bottom", fontsize=6.5)
    for s0, s1, m in ANTOINE_HAND if name == "sampo_antoine_ircam_12" else []:
        ax.plot([s0, s1], [-0.28, -0.28], color="k", lw=2.5, solid_capstyle="butt")
        ax.text((s0 + s1) / 2, -0.52, f"{m}/4 (Hand)", ha="center", va="top", fontsize=6.5)
    ax.set_ylim(-0.8, 1.45); ax.set_yticks([]); ax.set_xlim(0, t[-1] + 2)
    ax.set_xlabel("Sekunden", fontsize=8); ax.tick_params(labelsize=7.5)
    ax.set_title(f"{name}: Schläge (grau), Taktanfänge (schwarz), Zahl = Schläge je Takt",
                 fontsize=8.5, loc="left")
    fig.savefig(f"takte_{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"   -> takte_{name}.csv / .png")
    return folge


if __name__ == "__main__":
    namen = (sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(SRC, "sampo_*.aif")))
             if "--alle" in sys.argv else [arg("--name", "sampo_antoine_ircam_12")])
    folgen = {n: auswerten(n) for n in namen}
    if len(folgen) > 1:
        erste = list(folgen.values())[0]
        gleich = all(v == erste for v in folgen.values())
        print(f"\nTaktfolge in allen {len(folgen)} Videos identisch: {gleich}")
