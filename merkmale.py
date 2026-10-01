"""Merkmale aus den Handlandmarks - dieselbe Rechnung im Training und live.

Eine Landmark-Spur wird kausal aufbereitet (Luecken vorwaerts fuellen, Median
ueber die letzten 5 Frames) und daraus je Fenster die Eingabe eines Modells
gebaut. Jedes Fenster wird mit seiner EIGENEN Statistik normiert (Mittelwert
abziehen, isotrop skalieren) - es gibt keinen Zustand, der live erst
einschwingen muesste.

Eingaben der Modelle:
  voll12    Fingerspitzen-Median + Handgelenk, Position + 1. und 2. Ableitung  12 Kanaele
  pos4      dieselben zwei Punkte, nur Position                                 4 Kanaele
  tips6pos  5 Fingerspitzen einzeln + Handgelenk, nur Position                 12 Kanaele
  Graph     alle 21 Landmarks als (2, W, 21)

  python merkmale.py        # Selbsttest: offline == live
"""
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FPS = 25                # Trainingsraster
STEP = 5                # Auswerteraster in Frames
MEDPAD = 8              # Vorlauf im Live-Puffer, damit der Medianfilter warm ist
TIPS = [4, 8, 12, 16, 20]
WRIST = 0
TIPS6 = (4, 8, 12, 16, 20, 0)


def _ffill(x):
    """NaN vorwaerts fuellen (kausal); fuehrende NaN mit dem ersten gueltigen Wert."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return None
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    first = int(np.argmax(ok))
    y[:first] = x[first]
    return y


def _medfilt_causal(x, k=5):
    """Median der letzten k Werte - schaut nicht in die Zukunft."""
    xp = np.concatenate([np.full(k - 1, x[0]), x])
    return np.median(sliding_window_view(xp, k), axis=-1)


def points(lm_cam):
    """(F,21,>=2) -> (F,4) = [tips_h, tips_v, wrist_h, wrist_v], NaN gefuellt,
    noch nicht geglaettet. Live wird dazwischen auf 25 Hz resampled."""
    lm = np.asarray(lm_cam, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # Frames ohne Hand
        tips = np.nanmedian(lm[:, TIPS, :2], axis=1)
    wrist = lm[:, WRIST, :2]
    cols = []
    for arr in (tips, wrist):
        for k in range(2):
            c = _ffill(arr[:, k])
            if c is None:
                return None
            cols.append(c)
    return np.stack(cols, axis=1)


def smooth(pts):
    """(F,C) -> (F,C) kausal medianfiltert. Muss auf dem 25-Hz-Raster laufen."""
    return np.stack([_medfilt_causal(pts[:, k]) for k in range(pts.shape[1])], axis=1)


def resample(pts, ts, n, fps=FPS):
    """Unregelmaessig abgetastete Punkte -> die letzten n Frames auf 25-Hz-Raster.

    ts = echte Zeitstempel (s). Das Raster endet am neuesten Bild; fehlende
    Bilder werden linear interpoliert. None, wenn der Puffer das Fenster
    zeitlich nicht abdeckt."""
    ts = np.asarray(ts, float)
    grid = ts[-1] - np.arange(n - 1, -1, -1) / fps
    if grid[0] < ts[0]:
        return None
    return np.stack([np.interp(grid, ts, pts[:, k]) for k in range(pts.shape[1])], axis=1)


def raw_track(lm_cam):
    """(F,21,>=2) -> (F,4) zwei Punkte, NaN gefuellt + kausal geglaettet."""
    p = points(lm_cam)
    return None if p is None else smooth(p)


def track21(lm_cam):
    """(F,21,>=2) -> (F,42) alle Landmarks, gleiche Behandlung wie raw_track."""
    lm = np.asarray(lm_cam, float)[:, :, :2]
    cols = []
    for j in range(lm.shape[1]):
        for k in range(2):
            c = _ffill(lm[:, j, k])
            if c is None:
                return None
            cols.append(_medfilt_causal(c))
    return np.stack(cols, axis=1)


def _norm(p):
    """Mittelwert des Fensters abziehen, isotrop auf RMS 1 skalieren."""
    p = p - p.mean(axis=0)
    return p / (np.sqrt((p ** 2).mean()) + 1e-6)


def _deriv(p):
    """Position + kausale 1. und 2. Differenz: v[t]=p[t]-p[t-1], a[t]=v[t]-v[t-1]."""
    v = np.zeros_like(p); v[1:] = p[1:] - p[:-1]
    a = np.zeros_like(v); a[1:] = v[1:] - v[:-1]
    return np.concatenate([p, v, a], axis=1)


def feats(rec, s, e, variant):
    """Fenster [s:e) -> (W, C). Je Punkt eigene Normierung."""
    if variant in ("voll12", "pos4"):
        w = np.asarray(rec["raw"][s:e], float)
        out = [_norm(w[:, j:j + 2]) for j in range(0, w.shape[1], 2)]
        if variant == "voll12":
            out = [_deriv(p) for p in out]
        return np.concatenate(out, axis=1)
    if variant == "tips6pos":
        return np.concatenate([_norm(rec["r21"][s:e, 2 * j:2 * j + 2]) for j in TIPS6],
                              axis=1)
    raise ValueError(variant)


def graph_win(rec, s, e):
    """Fenster -> (2, W, 21). EIN Mittelwert und EINE Skala fuer die ganze Hand,
    damit die Handform erhalten bleibt."""
    w = rec["r21"][s:e].reshape(e - s, 21, 2)
    w = w - w.reshape(e - s, -1).mean()
    w = w / (np.sqrt((w ** 2).mean()) + 1e-6)
    return w.transpose(2, 0, 1)


def phase_target(ictus, T):
    """Soll-Phase 0..2pi zwischen aufeinanderfolgenden Schlaegen als (cos, sin).

    -> (target (T,2), valid (T,) bool). Am Schlag ist cos = 1, auf halbem Weg
    zum naechsten Schlag cos = -1; der Sinus unterscheidet die beiden Haelften.
    Gueltig nur zwischen zwei annotierten Schlaegen."""
    target = np.stack([np.ones(T), np.zeros(T)], axis=1)
    valid = np.zeros(T, dtype=bool)
    ic = np.array(sorted(int(f) for f in ictus if 0 <= f < T))
    for a, b in zip(ic[:-1], ic[1:]):
        if b <= a:
            continue
        ph = 2 * np.pi * (np.arange(a, b) - a) / (b - a)
        target[a:b, 0] = np.cos(ph)
        target[a:b, 1] = np.sin(ph)
        valid[a:b] = True
    if len(ic):
        valid[ic[-1]] = True
    return target, valid


if __name__ == "__main__":
    # offline (ganze Spur, dann Fenster) == live (nur der Puffer der letzten Frames)
    rng = np.random.default_rng(0)
    lm = np.full((400, 21, 3), np.nan)
    t = np.arange(400)
    for j in range(21):
        lm[:, j, 0] = 0.5 + 0.1 * np.sin(2 * np.pi * t / 30) + 0.002 * rng.standard_normal(400)
        lm[:, j, 1] = 0.5 + 0.2 * np.cos(2 * np.pi * t / 15) + 0.002 * rng.standard_normal(400)
    lm[50:55] = np.nan                                        # Hand fehlt kurz
    W = 50
    rec = {"raw": raw_track(lm)}
    off = feats(rec, 400 - W, 400, "voll12")
    live = feats({"raw": raw_track(lm[-(W + MEDPAD):])}, MEDPAD, W + MEDPAD, "voll12")
    assert np.allclose(off, live, atol=1e-9), np.abs(off - live).max()
    ts = np.arange(400) / FPS                                # gleichmaessig -> identisch
    rs = resample(points(lm), ts, W + MEDPAD)
    assert np.allclose(smooth(rs)[-W:], rec["raw"][-W:], atol=1e-9)
    print(f"merkmale OK: offline == live (diff {np.abs(off - live).max():.1e}), "
          f"Puffer braucht Fenster + {MEDPAD} Frames")
