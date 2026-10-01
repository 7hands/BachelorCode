"""Echtzeitsystem, PC-Seite: Trigger senden, Landmarks berechnen, Taktart ausgeben.

Ablauf je empfangenem Bild:
  1. Der PC sendet mit --trigger Hz einen Zielzeitpunkt (target_time) an die
     Pis; diese nehmen genau dann ein Bild auf (live_pi_raw.py).
  2. Das JPEG kommt zurück, MediaPipe berechnet hier die 21 Handlandmarks.
     Fehlt die Hand, wird die letzte gültige Hand wiederholt.
  3. Die Spur wird mit den ECHTEN Aufnahmezeiten auf das 25-Hz-Raster des
     Trainings interpoliert (Raster am neuesten Bild verankert), kausal
     medianfiltert und auf ein Fenster geschnitten - dieselbe Rechnung wie im
     Training (merkmale.py).
  4. Das Modell aus takt_training.py sagt die Taktart vorher; ausgegeben wird
     die Mehrheit der letzten neun Vorhersagen.

Die Pis können alternativ MediaPipe selbst rechnen und nur die Landmarks
schicken (live_pi.py) - dann entfällt Schritt 2 hier.

  python live_pc.py --checkpoint models/pos4_alle_Personen_80_20_s0.pt
  python live_pc.py --checkpoint models/TCN_TCN_alle_Personen_80_20_s0.pt --takt 3

Mit --takt wird am Ende die Trefferquote gegen die angesagte Taktart ausgegeben.
"""
import argparse
import time
from collections import Counter, deque

import numpy as np
import torch
import zmq

import merkmale as MM
import modelle as MD
from record_pc_raw import make_landmarker, detect

LEAD = 0.05               # Vorlauf des Zielzeitpunkts, damit die Pis ihn rechtzeitig sehen
MEHRHEIT = 9              # Ausgabe = Mehrheit der letzten neun Vorhersagen


def fenster(buf, win):
    """Puffer [(t, (21,2))] -> {'raw': (win,4), 'r21': (win,42)} oder None.

    Reihenfolge wie im Training: füllen -> auf das 25-Hz-Raster interpolieren
    -> glätten (braucht MEDPAD Vorlauf) -> auf genau win Zeilen schneiden."""
    if len(buf) < 8:
        return None
    n = win + MM.MEDPAD
    ts = np.array([b[0] for b in buf])
    lm = np.stack([b[1] for b in buf])                     # (T,21,2)
    p2 = MM.points(lm)
    if p2 is None:
        return None
    cols = []
    for j in range(lm.shape[1]):
        for k in range(2):
            c = MM._ffill(lm[:, j, k])
            if c is None:
                return None
            cols.append(c)
    p21 = np.stack(cols, axis=1)                           # (T,42)
    g2, g21 = MM.resample(p2, ts, n), MM.resample(p21, ts, n)
    if g2 is None or g21 is None:
        return None                                        # Puffer deckt das Fenster nicht
    return {"raw": MM.smooth(g2)[-win:], "r21": MM.smooth(g21)[-win:]}


def kaskade_build(mi, md, ck):
    """Fensterbauer der indirekten Modelle: fenster-dict mit W1+W2 Zeilen ->
    Phasenkurven (4, W2). Stufe 1 läuft auf Fenstern, deren Enden HOP1 Frames
    auseinanderliegen und am NEÜSTEN Frame verankert sind; von jedem Fenster
    werden die letzten HOP1 Werte übernommen - wie im Training (training.kurve)."""
    W1, W2, hop, var = ck["w1"], ck["w2"], ck["hop1"], ck["variante"]
    n = W1 + W2
    ein = MD.eingang("beat_gcn" if var == "graph" else "beat", var)

    def b(f, s, e):
        ends = list(range(n, W1 - 1, -hop))[::-1]
        x = torch.tensor(np.stack([ein(f, q - W1, q) for q in ends]), dtype=torch.float32)
        with torch.no_grad():
            oi, od = mi(x).numpy(), md(x).numpy()
        c = np.zeros((n, 4), np.float32)
        for k, q in enumerate(ends):
            lo = max(q - hop, W1 - hop); j = lo - (q - W1)
            c[lo:q, :2] = oi[k][:, j:].T
            c[lo:q, 2:] = od[k][:, j:].T
        return c[-W2:].T
    return b


def lade(pfad):
    """Checkpoint aus takt_training.py -> (Modell, Vorlauf in Frames, Fensterbauer, ck).
    Gebaut mit modelle.baue aus den Angaben im Checkpoint - wie im Training."""
    ck = torch.load(pfad, map_location="cpu")
    if "s2" in ck:                                         # indirektes Modell
        s1 = "beat_gcn" if ck["variante"] == "graph" else "beat"
        mi = MD.baue(s1, tuple(ck["dils1"]), ck["convs"], ck["wn"], cin=4)
        md = MD.baue(s1, tuple(ck["dils1"]), ck["convs"], ck["wn"], cin=4)
        m2 = MD.baue("clf", tuple(ck["dils2"]), ck["convs"], ck["wn"], cin=4)
        mi.load_state_dict(ck["s1_ictus"]); md.load_state_dict(ck["s1_down"])
        m2.load_state_dict(ck["s2"])
        for m in (mi, md, m2):
            m.eval()
        ck["params_gesamt"] = sum(sum(p.numel() for p in m.parameters()) for m in (mi, md, m2))
        return m2, ck["w1"] + ck["w2"], kaskade_build(mi, md, ck), ck
    m = MD.baue(ck["arch"], tuple(ck["dils"]), ck["convs"], ck["wn"], len(ck["klassen"]))
    m.load_state_dict(ck["state"]); m.eval()
    ck["params_gesamt"] = sum(p.numel() for p in m.parameters())
    return m, ck["fenster"], MD.eingang(ck["arch"], ck["variante"]), ck


def bericht(a, E, KL):
    print("=" * 60)
    if E["gefunden"]:
        sp = E["t_letzt"] - E["t_erst"]
        rate = f" = {len(E['gefunden']) / sp:.1f} fps" if sp > 1 else ""
        print(f"  Bilder empfangen   {len(E['gefunden'])} in {sp:.0f} s{rate}, "
              f"Hand gefunden in {np.mean(E['gefunden']):.0%}")
    if E["ms_bild"] and not E["pi_ms"]:
        b = np.array(E["ms_bild"])
        print(f"  Bild -> Landmarks  {b.mean():.1f} ms (p95 {np.percentile(b, 95):.1f}) "
              f"auf diesem Rechner")
    if E["pi_ms"]:
        print(f"  MediaPipe auf dem Pi  {np.mean(E['pi_ms']):.1f} ms je Bild")
    if E["ms_modell"]:
        print(f"  Fensterbau + Modell {np.mean(E['ms_modell']):.1f} ms je Vorhersage")
    for cid, d in E["pi"].items():
        print(f"  Pi {cid}: {d['fps_mittel']:.1f} fps, {d['ms_mittel']:.1f} ms MediaPipe, "
              f"Hand erkannt {d['erkannt']:.0%}")
    if not E["roh"]:
        return print("  Kein vollständiges Fenster - zu wenig Daten.")
    print(f"  Vorhersagen        {len(E['roh'])}, erste nach {E['t'][0] - E['t_erst']:.1f} s")
    if a.takt is None:
        return print("  (ohne --takt keine Trefferquote)")
    soll = KL.index(a.takt)
    roh, maj = np.array(E["roh"]), np.array(E["maj"])
    print(f"  Trefferquote       je Vorhersage {np.mean(roh == soll):.1%}, "
          f"Mehrheit({MEHRHEIT}) {np.mean(maj == soll):.1%}")
    for i, m in enumerate(KL):
        q = float(np.mean(maj == i))
        print(f"    {m}/4 {'<- Soll' if i == soll else '       '} {q:6.1%} |{'#' * int(q * 40):<40}|")


def main(a):
    model, win, build, ck = lade(a.checkpoint)
    KL = ck["klassen"]
    if a.takt is not None and a.takt not in KL:
        raise SystemExit(f"--takt {a.takt} nicht in {KL}")
    print(f"Modell {ck['arch']} ({ck['split']}, Seed {ck['seed']}) | Vorlauf {win} Frames "
          f"= {win / MM.FPS:.1f} s | {ck['params_gesamt']} Parameter")
    print(f"Raster {MM.FPS} Hz | Auslöserate {a.trigger} Hz | Kamera {a.cam}")
    landmarker = None                 # erst bauen, wenn wirklich Bilder kommen

    keep_s = 2.0 * (win + MM.MEDPAD) / MM.FPS
    buf, recent = deque(), deque(maxlen=MEHRHEIT)
    E = {"roh": [], "maj": [], "t": [], "gefunden": [], "ms_bild": [], "ms_modell": [],
         "pi_ms": [], "pi": {}, "t_erst": None, "t_letzt": None}

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB); pub.bind("tcp://*:5555")
    pull = ctx.socket(zmq.PULL); pull.bind("tcp://*:5556")
    poller = zmq.Poller(); poller.register(pull, zmq.POLLIN)
    print("\nWarte auf Pi(s) ...")
    time.sleep(2)
    print("Live. Dirigieren - Ctrl+C zum Stoppen.\n")

    next_trig = time.time()
    try:
        while True:
            now = time.time()
            if now >= next_trig:
                pub.send_json({"target_time": now + LEAD})
                next_trig = now + 1.0 / a.trigger
            if pull not in dict(poller.poll(5)):
                continue
            got = 0
            while True:
                try:
                    msg = pull.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if "stats" in msg:                 # Schlussmessung von live_pi.py
                    E["pi"][msg.get("camera_id")] = msg["stats"]
                    continue
                if msg.get("camera_id") != a.cam:
                    continue
                t = float(msg.get("capture_time") or msg.get("target_time") or time.time())
                tw = time.perf_counter()
                if "landmarks" in msg:             # Pi hat MediaPipe selbst gerechnet
                    lms = msg["landmarks"]
                    if msg.get("ms") is not None:
                        E["pi_ms"].append(float(msg["ms"]))
                else:
                    if landmarker is None:
                        landmarker = make_landmarker(a.delegate)
                    lms = detect(landmarker, msg["jpg"])
                E["ms_bild"].append((time.perf_counter() - tw) * 1000)
                E["gefunden"].append(lms is not None)
                if lms is not None:
                    buf.append((t, np.asarray(lms)[:, :2]))
                elif buf:
                    buf.append((t, buf[-1][1]))    # Hand fehlt: letzte wiederholen
                while buf and buf[0][0] < t - keep_s:
                    buf.popleft()
                E["t_erst"] = E["t_erst"] or t
                E["t_letzt"] = t
                got += 1
            if not got:
                continue

            _t = time.perf_counter()
            f = fenster(buf, win)
            if f is None:
                noch = max(0.0, win / MM.FPS - (E["t_letzt"] - E["t_erst"]))
                print(f"  sammle Daten ... noch {noch:4.1f} s      ", end="\r")
                continue
            x = torch.tensor(np.asarray(build(f, 0, win))[None], dtype=torch.float32)
            with torch.no_grad():
                p = torch.softmax(model(x), 1)[0]
            E["ms_modell"].append((time.perf_counter() - _t) * 1000)
            roh = int(p.argmax())
            recent.append(roh)
            cls = Counter(recent).most_common(1)[0][0]
            E["roh"].append(roh); E["maj"].append(cls); E["t"].append(E["t_letzt"])
            mark = "" if a.takt is None else (" +" if KL[cls] == a.takt else " -")
            print(f"  Takt: {KL[cls]}/4{mark}   (roh {KL[roh]}/4, {float(p.max()) * 100:3.0f} %)   ",
                  end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        pub.send_json({"target_time": time.time(), "stop": True})
        ende = time.time() + 2.0
        while time.time() < ende:          # live_pi.py schickt seine Messung erst nach dem Stop
            if pull in dict(poller.poll(200)):
                try:
                    m = pull.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                if "stats" in m:
                    E["pi"][m.get("camera_id")] = m["stats"]
        print("\nGestoppt.")
        bericht(a, E, KL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Modell aus takt_training.py")
    ap.add_argument("--cam", type=int, default=1, help="Kamera-ID des ausgewerteten Pis")
    ap.add_argument("--trigger", type=float, default=25.0,
                    help="Auslöserate in Hz, unabhängig vom 25-Hz-Trainingsraster")
    ap.add_argument("--delegate", default="auto", choices=("auto", "cpu", "gpu"))
    ap.add_argument("--takt", type=int, default=None,
                    help="angesagte Taktart -> am Ende Trefferquote")
    main(ap.parse_args())
