"""Läuft ein Checkpoint aus takt_training.py auf dem Live-Pfad?

Schiebt Aufnahmen aus Datensatz/ Frame für Frame durch denselben Puffer,
Fensterbauer und dasselbe Modell wie live_pc.py und vergleicht mit der
bekannten Taktart. Damit ist ohne Kamera und Pi geprüft, dass die Datei
geladen, das Modell richtig gebaut und der Eingang richtig gefüllt wird.

  python pruef_checkpoint.py                          # alle in models/
  python pruef_checkpoint.py --datei models/pos4_....pt --n 6
"""
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch

import daten as DS
import live_pc as LP
import merkmale as MM

arg = lambda f, s: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else s
N = int(arg("--n", 8))


def pruefe(pfad, recs):
    model, win, build, ck = LP.lade(pfad)
    ok, mehrheit = [], []
    for r in recs:
        lm = r["r21"].reshape(len(r["r21"]), 21, 2)
        buf = [(i / MM.FPS, lm[i]) for i in range(len(lm))]
        p = []
        for ende in range(win + MM.MEDPAD, len(buf) + 1, 25):     # jede Sekunde
            f = LP.fenster(buf[:ende], win)
            if f is None:
                continue
            x = torch.tensor(np.asarray(build(f, 0, win))[None], dtype=torch.float32)
            with torch.no_grad():
                p.append(int(model(x).argmax(1)))
        if not p:
            continue
        soll = ck["klassen"].index(int(r["meter"]))
        ok.extend([q == soll for q in p])
        mehrheit.append(Counter(p).most_common(1)[0][0] == soll)
    return ck, win, np.mean(ok), np.mean(mehrheit), len(ok)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    proben = [r for r in DS.lade() if r["name"].endswith("#0")][:N]
    print(f"{len(proben)} Aufnahmen, Taktarten "
          f"{dict(sorted(Counter(int(r['meter']) for r in proben).items()))}\n")
    dateien = [arg("--datei", None)] if "--datei" in sys.argv \
        else sorted(glob.glob(os.path.join(arg("--dir", "models"), "*.pt")))
    for pf in dateien:
        try:
            ck, win, fen, mehr, n = pruefe(pf, proben)
        except Exception as e:                                   # noqa: BLE001
            print(f"{os.path.basename(pf):<44} FEHLER: {type(e).__name__}: {e}")
            continue
        art = "indirekt" if "s2" in ck else "direkt"
        print(f"{os.path.basename(pf):<44}{art:<9}Vorlauf {win:>4} Frames  "
              f"{ck['params_gesamt']:>6} Parameter | je Fenster {fen:.0%}, "
              f"je Aufnahme {mehr:.0%} ({n} Fenster)")
