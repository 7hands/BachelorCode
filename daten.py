"""Aufnahmen laden, säubern, fusionieren - plus gemittelte Bahnen.

SAEUBERN    Zwei Fehlerbilder der Landmark-Erkennung werden entfernt:
            Einzelspitzen (1-3 Frames weit weg, danach wieder zurück) und
            Ruhe-Klumpen (die zweite, ruhende Hand; erkannt an den Spruengen
            hinein und heraus). Luecken bis MAXLUECKE Frames werden linear
            interpoliert.
FUSION      Luecken einer Kamera werden aus der anderen ergänzt. Dazu wird auf
            den Frames, in denen beide sauber sind, eine affine Abbildung der
            zweiten Kamera in die Bildkoordinaten der ersten geschätzt.
GEMITTELT   Je (Person, Taktart) eine über alle Takte gemittelte Bahn, in
            mehreren Tempi neu abgetastet - künstliche, rauscharme Aufnahmen
            für das Training. Sie werden je Split nur aus den Trainings-
            aufnahmen gebildet (siehe training.splits).

Datensatz: Datensatz/<P1|P2|P3>_<Takt>-4_<bpm>bpm_<Zeitstempel>.npz, Felder
lm (Kameras, Frames, 21, 3), t, clicks, cam_ids, bpm, meter, count_in_bars.

  python daten.py        # Uebersicht ueber den geladenen Datensatz
"""
import glob
import os

import numpy as np

import merkmale as MM

DIRS = ("Datensatz",)
# Die drei Personen der Arbeit (1, 2, 3); im Dateinamen als Praefix P1_, P2_, P3_.
# Die Reihenfolge hier bestimmt, in welcher Folge der 80/20-Split seine
# Zufallsziehungen macht (training.splits) - nicht aendern.
PERSONEN = ("1", "3", "2")
PRAEFIX = "P"
# Teil 3b: Aufnahmen von Person 3, zu einem Teil von einer vierten Person, nicht
# trennbar. Sie tragen das Merkmal block2, weil sie nie im Testset "bekannt"
# landen duerfen (training.splits) und eigene gemittelte Bahnen bekommen.
# Erkannt werden sie am Zeitstempel am Ende des Dateinamens.
BLOCK2 = "3b"                # Bezeichnung in Ausgaben: Teil b von Person 3
BLOCK2_ZU = PERSONEN[1]
BLOCK2_ZEITEN = {1785330803, 1785331044, 1785331379, 1785331605, 1785331814,
                 1785331941, 1785332096, 1785332256, 1785332402, 1785332549,
                 1785332699, 1785332849, 1785332984, 1785333077, 1785333189,
                 1785333293, 1785333448}
TEMPI = (50, 60, 70, 85, 100, 120)
TAKTE_SYN = 16

# Saeubern
SPITZE = 0.05        # Abstand zur Nachbar-Geraden, ab dem es eine Spitze ist
LANGSAM = 0.006      # Schrittweite, unter der ein Frame als ruhend gilt
KLUMPEN = 0.05       # Radius der Ruhe-Haeufung
FERN = 0.12          # Mindestabstand des Klumpens zur bewegten Wolke
ANTEIL = 0.05        # Mindestanteil, ab dem ein Klumpen als solcher zaehlt
SPRUNG = 0.10        # Schrittweite, ab der ein Uebergang ein Sprung ist
QUOTE = 0.5          # Anteil der Uebergaenge, die Spruenge sein muessen
MAXLUECKE = 8
MAXREST = 0.05       # Restfehler der affinen Abbildung, ab dem nicht fusioniert wird


# --- Saeubern -------------------------------------------------------------------
def mitte(lm):
    return np.nanmean(np.asarray(lm, float)[:, :, :2], axis=1)


def finde(p):
    """-> (Maske Einzelspitzen, Maske Ruhe-Klumpen). Erst der Klumpen, dann die
    Spitzen auf den verbliebenen Frames."""
    n = len(p)
    da = ~np.isnan(p).any(axis=1)
    sp = np.zeros(n, bool); kl = np.zeros(n, bool)
    if da.sum() < 5:
        return sp, kl

    i = np.where(da)[0]
    d = np.full(n, np.inf)
    d[i[1:]] = np.linalg.norm(np.diff(p[i], axis=0), axis=1)
    ruhig = da & (d < LANGSAM)
    if ruhig.sum() >= ANTEIL * da.sum():
        q = p[ruhig]
        z = q[np.argmax([(np.linalg.norm(q - x, axis=1) < KLUMPEN).sum() for x in q])]
        nah = da & (np.linalg.norm(p - z, axis=1) < KLUMPEN)
        bewegt = da & (d > 5 * LANGSAM) & ~nah
        if bewegt.sum() > 20 and np.linalg.norm(z - np.median(p[bewegt], axis=0)) > FERN \
                and nah.sum() >= ANTEIL * da.sum():
            m = nah.astype(int); k = np.diff(m)
            rein = np.where(k == 1)[0]; raus_ = np.where(k == -1)[0] + 1
            ue = [np.linalg.norm(p[j + 1] - p[j]) for j in rein
                  if j + 1 < n and da[j] and da[j + 1]]
            ue += [np.linalg.norm(p[j] - p[j - 1]) for j in raus_
                   if j < n and da[j] and da[j - 1]]
            if ue and np.mean(np.array(ue) > SPRUNG) >= QUOTE:
                kl = nah

    rest = np.where(da & ~kl)[0]
    if len(rest) < 5:
        return sp, kl
    pr = p[rest]
    for lauf in (1, 2, 3):
        for k in range(len(rest) - lauf - 1):
            a, b = k, k + lauf + 1
            if rest[b] - rest[a] > lauf + 3:
                continue                       # zu grosse Luecke dazwischen
            if np.linalg.norm(pr[a] - pr[b]) > SPITZE:
                continue                       # Nachbarn selbst weit auseinander
            w = (np.arange(1, lauf + 1) / (lauf + 1))[:, None]
            soll = pr[a] * (1 - w) + pr[b] * w
            if np.linalg.norm(pr[a + 1:b] - soll, axis=1).min() > SPITZE:
                sp[rest[a + 1:b]] = True
    return sp, kl


def _interpoliere(lm, hab, maxluecke):
    """Luecken bis maxluecke Frames zwischen gueltigen Frames linear fuellen."""
    g = np.where(hab)[0]
    ip = np.zeros(len(lm), bool)
    for a, b in zip(g[:-1], g[1:]):
        if 1 < b - a <= maxluecke + 1:
            w = np.linspace(0, 1, b - a + 1)[1:-1, None, None]
            lm[a + 1:b] = lm[a] * (1 - w) + lm[b] * w
            ip[a + 1:b] = True
    return ip


def schneide(lm, maxluecke=MAXLUECKE):
    """-> (lm gesaeubert + interpoliert, Anteil Spitzen, Anteil Klumpen, Masken)."""
    lm = np.asarray(lm, float).copy()
    p = mitte(lm)
    sp, kl = finde(p)
    lm[sp | kl] = np.nan
    interp = _interpoliere(lm, ~np.isnan(lm).all(axis=(1, 2)), maxluecke)
    leer = np.isnan(lm).all(axis=(1, 2))
    n = max(len(p), 1)
    return lm, sp.sum() / n, kl.sum() / n, dict(sp=sp, kl=kl, interp=interp, leer=leer)


# --- Fusion zweier Kameras ------------------------------------------------------
def affin(a, b):
    """Schaetzt M mit b ~ M @ [a,1]. a,b: (N,2) -> (3,2), mittlerer Restfehler."""
    A = np.hstack([a, np.ones((len(a), 1))])
    M, *_ = np.linalg.lstsq(A, b, rcond=None)
    return M, float(np.linalg.norm(A @ M - b, axis=1).mean())


def fusioniere(lm0, lm1):
    """-> (lm in den Koordinaten von Kamera 0, Kennzahlen)."""
    n0, _, _, m0 = schneide(lm0)
    n1, _, _, m1 = schneide(lm1)
    gut0 = ~(m0["kl"] | m0["sp"] | m0["leer"])
    gut1 = ~(m1["kl"] | m1["sp"] | m1["leer"])
    beide = gut0 & gut1
    info = dict(vorher=float(gut0.mean()), beide=float(beide.mean()),
                rest=np.nan, nachher=float(gut0.mean()), genutzt=0.0)
    if beide.sum() < 30:
        return n0, info
    a = np.asarray(lm1, float)[beide][:, :, :2].reshape(-1, 2)
    b = np.asarray(lm0, float)[beide][:, :, :2].reshape(-1, 2)
    ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
    if ok.sum() < 60:
        return n0, info
    M, rest = affin(a[ok], b[ok])
    info["rest"] = rest
    if rest > MAXREST:
        return n0, info                       # Abbildung taugt nicht
    aus = np.asarray(n0, float).copy()
    fehlt = (~gut0) & gut1
    if fehlt.any():
        q = np.asarray(lm1, float)[fehlt][:, :, :2]
        f = np.concatenate([q, np.ones(q.shape[:2] + (1,))], axis=2)
        aus[fehlt, :, :2] = np.einsum("fkc,cd->fkd", f, M)
        aus[fehlt, :, 2:] = 0.0
    info["genutzt"] = float(fehlt.mean())
    info["nachher"] = float((gut0 | (fehlt & gut1)).mean())
    hab = gut0 | (fehlt & gut1)
    aus[~hab] = np.nan
    info["interp"] = float(_interpoliere(aus, hab, MAXLUECKE).mean())
    info["leer"] = float(np.isnan(aus).all(axis=(1, 2)).mean())
    return aus, info


# --- Laden ------------------------------------------------------------------------
def person_von(datei):
    """Person aus dem Dateinamen: 'P3_4-4_...' -> '3'."""
    p = datei.split("_")[0]
    return p[len(PRAEFIX):] if p.startswith(PRAEFIX) and p[len(PRAEFIX):] in PERSONEN \
        else "unbekannt"


def ist_block2(datei):
    """Gehoert die Aufnahme zu Teil 3b? Am Zeitstempel im Dateinamen erkannt."""
    try:
        return int(os.path.splitext(datei)[0].rsplit("_", 1)[-1]) in BLOCK2_ZEITEN
    except ValueError:
        return False


def _rang(quelle):
    """Feste Reihenfolge Teil 3b, 1, 3, 2 - so wurden die Modelle trainiert."""
    if quelle == BLOCK2:
        return 0
    return 1 + PERSONEN.index(quelle) if quelle in PERSONEN else len(PERSONEN) + 1


def _dateiordnung(pfad):
    """Ladereihenfolge: nach Teil 3b / Person, dann nach dem Rest des Namens."""
    b = os.path.basename(pfad)
    return _rang(BLOCK2 if ist_block2(b) else person_von(b)), b.split("_", 1)[-1]


def lade(sauber=True, dirs=DIRS):
    """-> Liste von Spuren (eine je Aufnahme und Kamera) mit raw (F,4),
    r21 (F,42), ictus/down (Frames), meter, name ('datei#kamera'), person und
    block2 (Aufnahme aus Teil 3b)."""
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz")), key=_dateiordnung):
            z = np.load(f, allow_pickle=True)
            if "cam_ids" not in z or z["lm"].shape[1] < 200 or "190bpm" in f:
                continue
            b = os.path.basename(f)
            m = int(z["meter"])
            cf = np.array([int(np.argmin(np.abs(z["t"] - c))) for c in z["clicks"]])
            for cam in range(z["lm"].shape[0]):
                if sauber:
                    lm = (fusioniere(z["lm"][cam], z["lm"][1 - cam])[0]
                          if z["lm"].shape[0] > 1 else schneide(z["lm"][cam])[0])
                else:
                    lm = np.asarray(z["lm"][cam], float)
                raw, r21 = MM.raw_track(lm), MM.track21(lm)
                if raw is None or r21 is None:
                    continue
                out.append(dict(raw=raw, r21=r21, ictus=cf, down=cf[::m], meter=m,
                                name=f"{b}#{cam}", person=person_von(b),
                                block2=ist_block2(b), syn=False))
    return out


def lade_video(pfad):
    """Landmarks eines Videos (video_landmarks_rechts.py, 30 fps) -> Spur mit
    raw/r21 auf dem 25-Hz-Raster, wie eine eigene Aufnahme. None ohne Hand."""
    z = np.load(pfad, allow_pickle=True)
    lm = z["lm"][0][:, :, :2]                       # (F,21,2)
    t = z["t"]
    grid = np.arange(t[0], t[-1], 1.0 / MM.FPS)
    out = np.empty((len(grid), 21, 2))
    for j in range(21):
        for k in range(2):
            c = lm[:, j, k]
            ok = np.isfinite(c)
            out[:, j, k] = np.interp(grid, t[ok], c[ok]) if ok.sum() > 8 else np.nan
    raw, r21 = MM.raw_track(out), MM.track21(out)
    if raw is None or r21 is None:
        return None
    return dict(raw=raw, r21=r21, name=str(z["name"]), rate=float(z["rate"]))


def mittelbahn(recs, n=90):
    """Eine ueber alle Takte gemittelte Bahn je Gruppe -> (n,4) und (n,42)."""
    r2, r21 = [], []
    for r in recs:
        d = r["down"]
        for a, b in zip(d[:-1], d[1:]):
            a, b = int(a), int(b)
            if b - a < 10 or b > len(r["raw"]):
                continue
            g = np.linspace(a, b - 1, n)
            r2.append(np.stack([np.interp(g, np.arange(a, b), r["raw"][a:b, k])
                                for k in range(r["raw"].shape[1])], axis=1))
            r21.append(np.stack([np.interp(g, np.arange(a, b), r["r21"][a:b, k])
                                 for k in range(r["r21"].shape[1])], axis=1))
    if not r2:
        return None, None
    return np.mean(r2, axis=0), np.mean(r21, axis=0)


def gemittelt(recs, tempi=TEMPI, takte=TAKTE_SYN):
    """Je (Person, Taktart) eine Mittelbahn, in mehreren Tempi als Aufnahme;
    Teil 3b bekommt eigene Mittelbahnen. 'quellen' haelt fest, aus welchen
    Spuren sie gemittelt wurde."""
    grp = {}
    for r in recs:
        quelle = BLOCK2 if r.get("block2") else r["person"]
        grp.setdefault((quelle, r["meter"]), []).append(r)
    out = []
    for (quelle, m), rs in sorted(grp.items(), key=lambda kv: (_rang(kv[0][0]), kv[0][1])):
        b2, b21 = mittelbahn(rs)
        if b2 is None:
            continue
        for bpm in tempi:
            f_takt = int(round(m * 60.0 / bpm * MM.FPS))
            if f_takt < 20:
                continue
            g = np.linspace(0, len(b2) - 1, f_takt)
            e2 = np.stack([np.interp(g, np.arange(len(b2)), b2[:, k])
                           for k in range(b2.shape[1])], axis=1)
            e21 = np.stack([np.interp(g, np.arange(len(b21)), b21[:, k])
                            for k in range(b21.shape[1])], axis=1)
            down = np.arange(takte) * f_takt
            ict = np.array([d + int(round(j * f_takt / m))
                            for d in down for j in range(m)])
            out.append(dict(raw=np.tile(e2, (takte, 1)), r21=np.tile(e21, (takte, 1)),
                            ictus=ict, down=down, meter=m,
                            name=f"mittel_{quelle}_{m}er_{bpm}bpm#0", person=rs[0]["person"],
                            block2=quelle == BLOCK2, syn=True,
                            quellen=[x["name"] for x in rs]))
    return out


if __name__ == "__main__":
    from collections import Counter
    recs = lade()
    print(f"{len(recs)} Spuren gesaeubert "
          f"({len({r['name'].split('#')[0] for r in recs})} Aufnahmen x Kameras)")
    print(f"Personen: {dict(Counter(r['person'] for r in recs))}")
    print(f"Taktarten: {dict(sorted(Counter(r['meter'] for r in recs).items()))}")
