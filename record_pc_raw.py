"""Aufnahme-Hauptknoten (raw-Variante): der Pi schickt JPEGs, die
Landmark-Erkennung laeuft HIER auf dem PC. Entlastet den Pi -> hoehere FPS.

Der PC ist die Uhr: er sendet mit 25 Hz einen Zielzeitpunkt an die Pis und
spielt den Klicktrack; beide liegen auf DERSELBEN Zeitachse, die Klickzeiten
sind damit die Schlag-Labels. Nach 2 Takten Einzaehlen wird dirigiert. Die
Landmark-Erkennung laeuft nach der Aufnahme, bremst sie also nicht.

make_landmarker() und detect() benutzt auch live_pc.py.

Ergebnis: recordings/<Takt>-4_<bpm>bpm_<Zeitstempel>.npz mit
    lm      (Kameras, Frames, 21, 3)  normierte Bildkoordinaten, NaN = keine Hand
    t       (Frames,)                 Zeitstempel jedes Frames
    clicks  (Schlaege,)               Klickzeitpunkte = Schlag-Labels
    cam_ids, bpm, meter, count_in_bars

Benoetigt hand_landmarker.task im Arbeitsverzeichnis (siehe README).

  python record_pc_raw.py --selftest
  python record_pc_raw.py --pis 1 --bpm 100 --meter 4 --bars 16
"""
import argparse
import base64
import os
import subprocess
import time
import wave

import zmq
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Kalibrierkonstante fuer Audio-Ausgabelatenz + menschliche Vorwegnahme
# (typisch 20-80 ms); in den Aufnahmen der Arbeit 0.
CLICK_OFFSET = 0.0        # s
FPS = 25
LEAD = 0.05               # Vorlauf, damit die Pis den Trigger rechtzeitig sehen


def make_click(path, bpm, meter, beats, sr=44100):
    """Klicktrack als WAV. Downbeat hoeher -> die Eins ist hoerbar."""
    beat = 60.0 / bpm
    audio = np.zeros(int(sr * beat * (beats + 1)), dtype=np.float64)
    dur = int(0.012 * sr)
    t = np.arange(dur) / sr
    for k in range(beats):
        freq = 1800 if k % meter == 0 else 1100
        s = int(k * beat * sr)
        audio[s:s + dur] += np.sin(2 * np.pi * freq * t) * np.exp(-t * 250)
    pcm = (audio / np.abs(audio).max() * 0.8 * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return len(audio) / sr


def click_times(t0, bpm, meter, count_in, bars):
    """-> (alle Klicks, nur die Label-Klicks nach dem Count-in)."""
    beat = 60.0 / bpm
    n = (count_in + bars) * meter
    allc = t0 + np.arange(n) * beat + CLICK_OFFSET
    return allc, allc[count_in * meter:]


def pack(data):
    """data[target_time][cam] = 21 Punkte | None
    -> (lm (C,F,21,3), t (F,), cam_ids (C,)).

    Kamera-IDs beliebig: Zeilen folgen den sortierten IDs, cam_ids merkt sich
    die Zuordnung (fuer die Triangulation gegen die Kalibrierung noetig).
    """
    times = sorted(data.keys())
    cams = sorted({c for f in data.values() for c in f})
    idx = {c: i for i, c in enumerate(cams)}
    lm = np.full((len(cams), len(times), 21, 3), np.nan)
    for fi, t in enumerate(times):
        for cam, pts in data[t].items():
            if pts:                                # None = keine Hand erkannt
                lm[idx[cam], fi] = np.asarray(pts)
    return lm, np.array(times), np.array(cams)


def _build_landmarker(gpu):
    kw = {"delegate": python.BaseOptions.Delegate.GPU} if gpu else {}
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task", **kw),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    ))


def make_landmarker(delegate="auto", quiet=False):
    """delegate: "auto" (misst einmal, ~1 s) | "cpu" | "gpu".

    MediaPipes GPU-Pfad ist OpenGL-ES und haengt am Treiber - deshalb messen statt
    raten. Gemessen (bench_live.py, echtes Bild mit Hand):
      Ryzen 9 9900X + NVIDIA -> GL faellt auf llvmpipe (Software) zurueck:
          CPU  9.0 ms | GPU 66.8 ms  -> GPU 7.5x LANGSAMER
      Ryzen 5 5600H + Radeon (radeonsi, echter Treiber):
          CPU 42.4 ms | GPU 30.2 ms  -> GPU 1.4x schneller
    Bildgroesse ist egal (1640/1280/640 px gleich schnell), MediaPipe skaliert
    intern auf feste Modellgroesse.
    """
    if delegate in ("cpu", "gpu"):
        return _build_landmarker(delegate == "gpu")

    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    best, best_ms, best_name = None, float("inf"), "?"
    for gpu in (False, True):
        try:
            lmk = _build_landmarker(gpu)
            lmk.detect(img)                                   # warmup
            t = time.perf_counter()
            for _ in range(3):
                lmk.detect(img)
            ms = (time.perf_counter() - t) / 3 * 1000
        except Exception:                                     # noqa: BLE001 - GPU fehlt
            continue
        if ms < best_ms:
            best, best_ms, best_name = lmk, ms, "GPU" if gpu else "CPU"
    if best is None:
        return _build_landmarker(False)
    if not quiet:
        print(f"MediaPipe: {best_name}-Pfad gewaehlt ({best_ms:.1f} ms/Bild)")
    return best


def detect(landmarker, jpg_b64):
    """JPEG(base64, BGR) -> Liste von 21 (x,y,z) normiert, oder None."""
    buf = np.frombuffer(base64.b64decode(jpg_b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)          # zurueck zu BGR
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)         # picamera2 RGB888 war real BGR
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_image)
    if not res.hand_landmarks:
        return None
    return [(lm.x, lm.y, lm.z) for lm in res.hand_landmarks[0]]  # num_hands=1


def demo():
    """Self-Check ohne Hardware/Modell: JPEG-Roundtrip + BGR->RGB stimmen."""
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[:, :, 2] = 255                                 # rot in BGR
    ok, buf = cv2.imencode(".jpg", bgr)
    jpg = base64.b64encode(buf).decode("ascii")
    dec = cv2.imdecode(np.frombuffer(base64.b64decode(jpg), np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    assert rgb[..., 0].mean() > 200 and rgb[..., 2].mean() < 50, "BGR/RGB falsch herum"
    print("record_pc_raw OK: JPEG-Roundtrip + BGR->RGB korrekt")


def main(args):
    beat = 60.0 / args.bpm
    n_beats = (args.count_in + args.bars) * args.meter
    duration = n_beats * beat

    os.makedirs("recordings", exist_ok=True)
    name = args.name or f"{args.meter}-4_{args.bpm}bpm_{int(time.time())}"
    wav = os.path.join("recordings", f"{name}_click.wav")
    make_click(wav, args.bpm, args.meter, n_beats)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB); pub.bind("tcp://*:5555")
    pull = ctx.socket(zmq.PULL); pull.bind("tcp://*:5556")
    poller = zmq.Poller(); poller.register(pull, zmq.POLLIN)

    print(f"Warte auf {args.pis} Pi(s) ... (ZMQ braucht kurz zum Verbinden)")
    time.sleep(2)
    print(f"{args.meter}/4, {args.bpm} bpm, {args.count_in} Takte Count-in + "
          f"{args.bars} Takte dirigieren = {duration:.0f}s")
    input("ENTER zum Start (beim Count-in NICHT dirigieren) ...")

    t0 = time.time()
    subprocess.Popen(["aplay", "-q", wav],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, labels = click_times(t0, args.bpm, args.meter, args.count_in, args.bars)
    go = t0 + args.count_in * args.meter * beat

    raw, next_trig, said = {}, t0, False
    try:
        while time.time() < t0 + duration:
            now = time.time()
            if now >= next_trig:
                pub.send_json({"target_time": next_trig + LEAD})
                next_trig += 1.0 / FPS
            if not said and now >= go:
                print(">>> JETZT dirigieren <<<"); said = True
            if pull in dict(poller.poll(5)):
                m = pull.recv_json()
                raw.setdefault(m["target_time"], {})[m["camera_id"]] = m["jpg"]
    except KeyboardInterrupt:
        print("\nAbgebrochen.")

    pub.send_json({"target_time": time.time(), "stop": True})
    time.sleep(0.3)

    n_img = sum(len(v) for v in raw.values())
    print(f"\n{len(raw)} Frames empfangen ({n_img} Bilder). Erkenne Landmarks (offline) ...")
    landmarker = make_landmarker()
    data = {t: {cam: detect(landmarker, jpg) for cam, jpg in cams.items()}
            for t, cams in raw.items()}

    lm, t, cams = pack(data)
    if lm.size == 0:
        print("KEINE Daten empfangen -> Pi-IP/Verbindung pruefen (--ip, Ports 5555/5556).")
        return
    out = os.path.join("recordings", f"{name}.npz")
    np.savez(out, lm=lm, t=t, clicks=labels, cam_ids=cams,
             bpm=args.bpm, meter=args.meter, count_in_bars=args.count_in)

    det = np.isfinite(lm[..., 0]).mean() * 100
    print(f"gespeichert: {out}")
    print(f"  {lm.shape[1]} Frames ({lm.shape[1]/max(duration,1):.0f} fps), "
          f"{len(cams)} Kamera(s) {list(cams)}, {len(labels)} Ictus-Labels, "
          f"Hand erkannt in {det:.0f}% der Frames")
    if len(cams) < args.pis:
        print(f"  WARNUNG: nur {len(cams)} von {args.pis} erwarteten Kameras haben gesendet.")
    if det < 80:
        print("  WARNUNG: niedrige Erkennungsrate -> Licht/Abstand/Ausschnitt/JPEG-Qualitaet pruefen.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pis", type=int)
    ap.add_argument("--bpm", type=int, default=100)
    ap.add_argument("--meter", type=int, default=4)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--count-in", type=int, default=2)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.selftest:
        demo()
    elif a.pis:
        main(a)
    else:
        ap.error("--pis noetig (oder --selftest)")
