"""Konduktor-Videos -> Landmarks NUR der rechten Hand (.npz).

Wie video_landmarks.py, aber mit num_hands=2 und Auswahl über die Händigkeit.
Bisher lief die Erkennung mit num_hands=1; MediaPipe gibt dann die sicherste
Hand zurück, und das war streckenweise die linke -- die Spur springt dann
zwischen den Händen hin und her.

MediaPipe beschriftet die Hände aus Sicht der gefilmten Person und setzt ein
NICHT gespiegeltes Bild voraus. Verlässlich ist das Etikett hier nicht. Gewählt wird deshalb die Hand, die
sich BEWEGT (90%-Quantil der Verschiebung je Frame) -- die ruhende Hand liegt
oft dauerhaft im Bild und wird sonst fälschlich verfolgt.

  python video_landmarks_rechts.py                     # nur sampo_antoine_ircam_12
  python video_landmarks_rechts.py --alle              # alle Videos in SRC
  python video_landmarks_rechts.py --alle
"""
import glob
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

SRC = "Data/conductor-follower-master/datasets/conductors_movies_with_sound"
OUT = "video_npz"
arg = lambda f, s: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else s
HAND = arg("--hand", "Right")
KONF = float(arg("--konf", 0.5))
TAG = arg("--tag", "")            # Anhang an den Dateinamen
MODUS = arg("--modus", "stetig")   # stetig | etikett


def landmarker():
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=KONF,
        min_hand_presence_confidence=KONF))   # sonst greift die Vorgabe 0.5


def leithand(frames):
    """Welches Etikett gehört zur dirigierenden Hand?

    Entschieden wird auf den Frames mit ZWEI erkannten Händen: dort ist die
    Zuordnung eindeutig, und die dirigierende Hand bewegt sich stärker als die
    ruhende (die z.B. nur die Seite umblättert). Median statt Quantil, weil bei
    niedriger Schwelle einzelne Fehltreffer jedes hohe Quantil kapern.
    """
    bahn = {}
    for i, kand in enumerate(frames):
        if len({k[0] for k in kand}) == 2:
            for nm, _, pts in kand:
                bahn.setdefault(nm, []).append((i, pts[0, 0], pts[0, 1]))
    beweg = {}
    for nm, v in bahn.items():
        a = np.array(v)
        d = np.linalg.norm(np.diff(a[:, 1:], axis=0), axis=1)[np.diff(a[:, 0]) == 1]
        beweg[nm] = float(np.median(d)) if len(d) > 20 else 0.0
    if not beweg or max(beweg.values()) == 0:
        return HAND, beweg
    return max(beweg, key=beweg.get), beweg


def spur(pfad, lmk):
    """Zwei Durchgänge: erst alle Treffer sammeln, dann die bewegte Hand wählen
    und ihr folgen. Treffer mit dem anderen Etikett werden übernommen, wenn sie
    auf der Bahn liegen -- MediaPipe vertauscht die Händigkeit hier oft."""
    cap = cv2.VideoCapture(pfad)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, spuren, zahl = [], {}, {"keine": 0, "nur Right": 0, "nur Left": 0, "beide": 0}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
        kand = [(h[0].category_name, h[0].score,
                 np.array([(p.x, p.y, p.z) for p in pts], float))
                for h, pts in zip(r.handedness, r.hand_landmarks)]
        for nm, _, pts in kand:
            spuren.setdefault(nm, []).append((i, pts[0, 0], pts[0, 1]))
        namen = [k[0] for k in kand]
        zahl["keine" if not namen else
             "beide" if len(set(namen)) == 2 else f"nur {namen[0]}"] += 1
        frames.append(kand); i += 1
    cap.release()

    leit, beweg = leithand(frames)
    lms, genommen, vor = [], {"Händigkeit": 0, "einzeln, Leithand": 0,
                              "einzeln, auf der Bahn": 0, "verworfen": 0}, None
    for kand in frames:
        pts = None
        if len({k[0] for k in kand}) == 2:                 # beide Hände sichtbar
            pts = max((k for k in kand if k[0] == leit), key=lambda k: k[1])[2]
            genommen["Händigkeit"] += 1
        elif kand:                                         # nur eine Hand sichtbar
            eigen = [k for k in kand if k[0] == leit]
            if eigen:
                pts = max(eigen, key=lambda k: k[1])[2]
                genommen["einzeln, Leithand"] += 1
            elif vor is not None:
                d, k = min(((np.linalg.norm(k[2][0, :2] - vor), k) for k in kand),
                           key=lambda x: x[0])
                if d < 0.09:                               # etwa 1.5 Handbreiten
                    pts = k[2]; genommen["einzeln, auf der Bahn"] += 1
                else:
                    genommen["verworfen"] += 1
            else:
                genommen["verworfen"] += 1
        if pts is None:
            lms.append([(np.nan,) * 3] * 21)
        else:
            lms.append(pts.tolist()); vor = pts[0, :2]
    rate = float(np.isfinite(np.array(lms, float)[:, 0, 0]).mean())
    return (np.array(lms, float), np.array([j / fps for j in range(len(lms))]),
            rate, fps, zahl, {k: [x[1] for x in v] for k, v in spuren.items()},
            {"leit": leit, "bewegung (2-Hand-Frames)": {k: round(v, 4) for k, v in beweg.items()},
             **genommen})


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    vids = (sorted(glob.glob(os.path.join(SRC, "*.mov"))) if "--alle" in sys.argv
            else [os.path.join(SRC, arg("--name", "sampo_antoine_ircam_12") + ".mov")])
    lmk = landmarker()
    for v in vids:
        name = os.path.basename(v)[:-4]
        t0 = time.time()
        lm, ts, rate, fps, zahl, xpos, genommen = spur(v, lmk)
        ziel = os.path.join(OUT, f"{name}_{HAND.lower()}{TAG}.npz")
        np.savez_compressed(ziel, lm=lm[None], t=ts, rate=rate, fps=fps, name=name)
        alt = os.path.join(OUT, name + ".npz")
        a = float(np.load(alt, allow_pickle=True)["rate"]) if os.path.exists(alt) else float("nan")
        print(f"{name}: {HAND} in {rate:.1%} der {len(ts)} Frames "
              f"(bisher mit einer Hand: {a:.1%})  {time.time()-t0:.0f}s")
        print(f"   Frames: {zahl}")
        print(f"   übernommen nach Etikett des Treffers: {genommen}")
        print("   mittlere x-Position je Etikett: "
              + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in sorted(xpos.items()))
              + "   (0 = links im Bild)")
        print(f"   -> {ziel}")
