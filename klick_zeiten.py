"""Klickzeiten aus der Tonspur -- über die Lautstärke, ohne Tempoannahme.

Die sampo-Tonspuren sind digitale Stille mit kurzen Klicks darin (Median des
Pegels -120 dB, Klicks 20 ms lang bei etwa 0.12 Vollaussteuerung). Ein Klick
ist damit schlicht eine Stelle, an der überhaupt etwas zu hören ist.

Früher (klick_audio.py, §17) lief das über die Ableitung der RMS-Hüllkurve
und in §31 zusätzlich über ein angepasstes 500-ms-Raster. Beides war falsch:
die Ableitung erzeugt auf Stille Artefakte, und das Raster setzt ein
konstantes Tempo voraus -- das Tempo wechselt hier aber zwischen den Takten.

  python klick_zeiten.py                    # sampo_antoine_ircam_12
  python klick_zeiten.py --alle
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

SRC = "Data/conductor-follower-master/datasets/conductors_movies_with_sound"
SR = 48000
SCHWELLE = 1e-4          # alles darunter ist digitale Stille
PAUSE = SR // 50         # 20 ms Stille trennen zwei Klicks
arg = lambda f, s: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else s
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def lade(pfad):
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", pfad, "-f", "f32le",
                        "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.float32) if p.stdout else None


def impulse(x):
    """-> (Startzeiten, Dauern, Spitzenpegel) aller nicht stillen Abschnitte."""
    idx = np.where(np.abs(x) > SCHWELLE)[0]
    if len(idx) < 2:
        return np.array([]), np.array([]), np.array([])
    schnitt = np.where(np.diff(idx) > PAUSE)[0]
    a = np.concatenate([[idx[0]], idx[schnitt + 1]])
    b = np.concatenate([idx[schnitt], [idx[-1]]])
    pegel = np.array([np.abs(x[i:j + 1]).max() for i, j in zip(a, b)])
    return a / SR, (b - a) / SR, pegel


def auswerten(name):
    x = lade(os.path.join(SRC, name + ".aif"))
    if x is None:
        return print(f"{name}: Ton nicht lesbar")
    dauer_ges = len(x) / SR
    still = float(np.mean(np.abs(x) <= SCHWELLE))
    t, dauer, pegel = impulse(x)
    if len(t) < 4:
        return print(f"{name}: nur {len(t)} Impulse")
    if still < 0.5:
        print(f"{name}: nur {still:.0%} Stille -- das ist Musik, kein Klicktrack. "
              f"Übersprungen.")
        return
    d = np.diff(t)
    lokal = np.array([np.median(d[max(0, i - 3):i + 4]) for i in range(len(d))])
    verdacht = np.where(d > 1.6 * lokal)[0]
    with open(f"klicks_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["nr", "sekunden", "dauer_ms", "pegel", "abstand_ms"])
        for i, z in enumerate(t):
            w.writerow([i, f"{z:.4f}", f"{dauer[i]*1000:.0f}", f"{pegel[i]:.3f}",
                        f"{d[i-1]*1000:.1f}" if i else ""])
    print(f"{name}: {len(t)} Klicks, {still:.0%} der Tonspur ist Stille")
    print(f"   erster {t[0]:.2f}s, letzter {t[-1]:.2f}s (Video {dauer_ges:.1f}s) -- "
          f"davor/danach kein Klick")
    print(f"   Klickdauer {np.median(dauer)*1000:.0f} ms, Pegel {np.median(pegel):.3f}")
    print(f"   Abstand: Median {np.median(d)*1000:.0f} ms, Spanne "
          f"{d.min()*1000:.0f}-{d.max()*1000:.0f} ms "
          f"= {60/d.max():.0f}-{60/d.min():.0f}/min -- KEIN festes Tempo")
    if len(verdacht):
        print(f"   {len(verdacht)} auffällig lange Abstände (möglicherweise "
              f"ausgelassener Klick): " + ", ".join(f"{t[i]:.1f}s ({d[i]*1000:.0f}ms)"
                                                    for i in verdacht))

    fig, ax = plt.subplots(2, 1, figsize=(6.9, 3.8), gridspec_kw={"hspace": 0.6})
    ze = np.arange(len(x)) / SR
    ax[0].plot(ze[::10], np.abs(x[::10]), color="0.6", lw=0.4)
    ax[0].plot(t, pegel, "v", color="k", ms=3)
    ax[0].set_xlim(0, dauer_ges); ax[0].set_xlabel("Sekunden", fontsize=8)
    ax[0].set_ylabel("Amplitude", fontsize=8); ax[0].tick_params(labelsize=7.5)
    ax[0].set_title(f"{name}: {len(t)} Klicks, sonst Stille "
                    f"({still:.0%} der Tonspur)", fontsize=8.5, loc="left")
    ax[1].plot(t[1:], 60 / d, ".-", color="k", ms=3, lw=0.7)
    ax[1].set_xlim(0, dauer_ges); ax[1].set_xlabel("Sekunden", fontsize=8)
    ax[1].set_ylabel("Tempo [Klicks/min]", fontsize=8); ax[1].tick_params(labelsize=7.5)
    ax[1].set_title("Tempo aus dem Abstand zum vorigen Klick", fontsize=8.5, loc="left")
    ax[1].grid(alpha=0.3)
    fig.savefig(f"klicks_{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"   -> klicks_{name}.csv / .png")


if __name__ == "__main__":
    namen = ([os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(SRC, "*.aif")))]
             if "--alle" in sys.argv else [arg("--name", "sampo_antoine_ircam_12")])
    for n in namen:
        auswerten(n)
