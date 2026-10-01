"""Pi-Worker für den Live-Betrieb MIT MediaPipe auf dem Pi.

Die Alternative zu live_pi_raw.py: dort schickt der Pi JPEGs und der PC rechnet
die Landmarks, hier rechnet der Pi selbst und schickt nur die 21 Punkte. Kein
Bildtransfer, keine Aufzeichnung, nichts wird zwischengespeichert - Bild
aufnehmen, HandLandmarker, sofort senden. Auf dem PC läuft dazu

    python live_pc.py --checkpoint models/<Modell>.pt --cam 0

und rechnet daraus live die Taktart. Das ist die echte Betriebssituation der
Variante "Landmarks auf dem Pi".

Nebenbei misst der Pi sich selbst, denn genau das ist die offene Frage: schafft
er die Rate überhaupt? Zwei Zahlen entscheiden das.

  Durchsatz       wie viele Frames pro Sekunde kommen wirklich durch.
  Bediente Trigger  der PC ruft mit --trigger Hz. Was der Pi nicht schafft,
                  fällt weg. 20% bediente Trigger heisst: vier von fünf
                  Aufnahmezeitpunkten existieren gar nicht.

Der Zeitstempel im Paket ist die ECHTE Aufnahmezeit, nicht der Sollzeitpunkt -
der PC interpoliert selbst auf sein 25-Hz-Raster. Ein langsamer und
unregelmässiger Pi liefert also brauchbare, nur dünnere Daten.

  python live_pi.py --id 1 --ip <IP des PCs>
  python live_pi.py --id 1 --ip <IP des PCs> --size 820x616   # halbe Auflösung
  python live_pi.py --id 1 --ip <IP des PCs>
  python live_pi.py --id 1 --ip <IP des PCs> --size 820x616   # halbe Auflösung --frei    # Trigger ignorieren, volle Last
  python live_pi.py --selfcheck                        # ohne Hardware
"""
import argparse
import sys
import time

import numpy as np


def zusammenfassung(ms, loop, dt, det, gesehen, bedient):
    """Rohmessungen -> Kennzahlen. ms/loop in ms, dt = Schleifenperioden in s."""
    ms, loop, dt = np.asarray(ms, float), np.asarray(loop, float), np.asarray(dt, float)
    return {"frames": len(ms), "erkannt": float(np.mean(det)) if len(det) else 0.0,
            "ms_mittel": float(ms.mean()), "ms_sd": float(ms.std()),
            "ms_p95": float(np.percentile(ms, 95)), "ms_max": float(ms.max()),
            "loop_mittel": float(loop.mean()), "loop_p95": float(np.percentile(loop, 95)),
            # fps aus der MITTLEREN Periode, nicht als Mittel der Kehrwerte:
            # sonst ziehen die schnellen Frames das Ergebnis nach oben.
            "fps_mittel": 1.0 / dt.mean(), "fps_min": 1.0 / dt.max(),
            "fps_sd": float(np.std(1.0 / dt)), "dt_sd_ms": float(dt.std() * 1000),
            "trigger_gesehen": gesehen, "trigger_bedient": bedient,
            "trigger_quote": bedient / gesehen if gesehen else float("nan")}


def drucke(d, ziel):
    print(f"  Frames                {d['frames']}")
    print(f"  Hand erkannt          {d['erkannt']:.0%}")
    print(f"  MediaPipe je Frame    {d['ms_mittel']:6.1f} ms  (sd {d['ms_sd']:.1f}, "
          f"p95 {d['ms_p95']:.1f}, max {d['ms_max']:.1f})")
    print(f"  Ganze Schleife        {d['loop_mittel']:6.1f} ms  (p95 {d['loop_p95']:.1f}) "
          f"- Aufnahme + MediaPipe + Senden")
    print(f"  Durchsatz             {d['fps_mittel']:6.1f} fps (sd {d['fps_sd']:.1f}, "
          f"schlechtester Frame {d['fps_min']:.1f})")
    print(f"  Periodenjitter        {d['dt_sd_ms']:6.1f} ms")
    if d["trigger_gesehen"]:
        print(f"  Trigger bedient       {d['trigger_bedient']} von {d['trigger_gesehen']} "
              f"= {d['trigger_quote']:.0%}")
    if d["fps_mittel"] < ziel:
        print(f"  -> reicht NICHT für {ziel:.0f} Hz: Faktor "
              f"{ziel / d['fps_mittel']:.1f} zu langsam.")
    else:
        print(f"  -> reicht für {ziel:.0f} Hz ({d['fps_mittel'] / ziel:.1f}x Reserve).")


def selfcheck():
    d = zusammenfassung([100.]*3, [120.]*3, [.1]*3, [True, False, True], 30, 3)
    assert d["frames"] == 3 and abs(d["fps_mittel"] - 10.) < 1e-9, d
    assert abs(d["erkannt"] - 2/3) < 1e-9 and abs(d["trigger_quote"] - .1) < 1e-9, d
    d = zusammenfassung([10, 30], [20, 40], [.05, .15], [1, 1], 0, 0)
    assert abs(d["fps_mittel"] - 10.) < 1e-9, "fps = 1/mean(dt), nicht mean(1/dt)"
    assert abs(d["fps_min"] - 1/.15) < 1e-9 and d["trigger_gesehen"] == 0, d
    print("selfcheck ok")


def sichtfeld(picam):
    """Wird wirklich der volle Sensor gelesen?

    Gleiches Seitenverhältnis garantiert das NICHT: picamera2 wählt zuerst
    einen Sensormodus, und manche Modi des IMX219 sind selbst schon Ausschnitte
    (1920x1080, 640x480). Erst danach wird skaliert. Deshalb hier nicht rechnen,
    sondern ScalerCrop gegen die volle Sensorfläche halten."""
    try:
        voll = tuple(picam.camera_properties.get("PixelArraySize", ()))
        cfg = picam.camera_configuration()
        roh = tuple(cfg.get("raw", {}).get("size", ()))
        crop = picam.capture_metadata().get("ScalerCrop")
    except Exception as e:
        return print(f"  Sichtfeld: nicht auslesbar ({type(e).__name__})")
    print(f"  Sensor {voll}, Modus {roh}, Ausgabe "
          f"{tuple(cfg['main']['size'])}")
    if not (crop and voll):
        return print("  Sichtfeld: ScalerCrop nicht gemeldet - ungeprüft")
    x, y, cw, chh = crop
    ax, ay = cw / voll[0], chh / voll[1]
    if ax > 0.97 and ay > 0.97:
        print(f"  Sichtfeld: VOLL ({ax:.0%} x {ay:.0%} des Sensors) - nur skaliert")
    else:
        print(f"  Sichtfeld: BESCHNITTEN auf {ax:.0%} x {ay:.0%} des Sensors "
              f"(Crop {crop})")
        print("             Die Landmarks sind damit anders normiert als im "
              "Training (1640x1232, voll).")


def main(a):
    import cv2
    import zmq
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from picamera2 import Picamera2

    w, h = (int(x) for x in a.size.split("x"))
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{a.ip}:5555")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://{a.ip}:5556")
    poller = zmq.Poller(); poller.register(sub, zmq.POLLIN)

    picam = Picamera2()
    picam.configure(picam.create_video_configuration({"format": "RGB888", "size": (w, h)}))
    # Seitenverhältnis gegen das Trainingsformat prüfen: Landmarks sind auf
    # [0,1] der Bildkanten normiert, ein anderes Verhältnis verzerrt die Figur
    # gegenüber allen Trainingsdaten - und fällt sonst nirgends auf.
    if abs(w / h - 4 / 3) > 0.02:
        print(f"WARNUNG: {w}x{h} ist {w/h:.2f}:1, Trainingsdaten sind 4:3 (1.33:1).")
        print("         Die Taktfigur ist damit gegenüber dem Training verzerrt.")
        print("         Empfohlen: 1640x1232 oder 820x616.")

    picam.start()
    sichtfeld(picam)
    picam.set_controls({"ExposureTime": a.exposure})     # wie record_pi.py
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.5))
    time.sleep(2)
    print(f"Pi {a.id} live {w}x{h} -> {a.ip}   "
          f"({'frei laufend' if a.frei else 'triggergesteuert'})")

    ms, loop, dt, det = [], [], [], []
    n = gesehen = bedient = 0
    letzte, t_start, t_druck, lauf = None, None, 0.0, True
    try:
        while lauf:
            target = None
            if a.frei:
                while sub in dict(poller.poll(0)):       # nur auf Stop hören
                    if sub.recv_json().get("stop"):
                        lauf = False
                if not lauf:
                    break
            else:
                # Bis zum NEÜSTEN Trigger durchspulen. Ist der Pi langsamer als die
                # Auslöserate, stauen sich die alten sonst auf und er fällt immer
                # weiter zurück. Was übersprungen wird, zählt als nicht bedient.
                if sub not in dict(poller.poll(20)):
                    continue
                while True:
                    try:
                        msg = sub.recv_json(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if msg.get("stop"):
                        lauf = False
                    elif msg.get("target_time") is not None:
                        target = float(msg["target_time"]); gesehen += 1
                if not lauf:
                    break
                if target is None:
                    continue
                while time.time() < target:              # gemeinsamer Zeitpunkt
                    time.sleep(0.001)

            t_loop = time.perf_counter()
            frame = np.ascontiguousarray(picam.capture_array("main"))
            t_mp = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # picamera2 RGB888 = real BGR
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            d_mp = (time.perf_counter() - t_mp) * 1000
            lms = [(l.x, l.y, l.z) for l in res.hand_landmarks[0]] \
                if res.hand_landmarks else None
            now = time.time()
            push.send_json({                              # SOFORT raus, vor der Buchhaltung
                "target_time": target if target is not None else now,
                "camera_id": a.id,
                "capture_time": now,                      # echte Zeit -> PC interpoliert
                "landmarks": lms,
                "ms": d_mp,
            })
            d_loop = (time.perf_counter() - t_loop) * 1000
            bedient += 1

            n += 1
            if n <= a.warm:                               # erste Frames sind langsamer
                continue
            if t_start is None:                           # Messbeginn, noch keine Periode
                t_start = letzte = now
                gesehen = bedient = 0
                continue
            dt.append(now - letzte); letzte = now
            ms.append(d_mp); loop.append(d_loop); det.append(lms is not None)

            if now - t_druck > 1.0 and len(dt) > 2:
                t_druck = now
                print(f"  {1/np.mean(dt[-30:]):5.1f} fps   {np.mean(ms[-30:]):5.1f} ms   "
                      f"erkannt {np.mean(det[-30:]):3.0%}   {now-t_start:5.0f}s ", end="\r")
    except KeyboardInterrupt:
        print("\nGestoppt.")
    finally:
        picam.stop()

    if len(dt) < 3:
        return print("\nZu wenige Frames für eine Aussage.")
    d = zusammenfassung(ms, loop, dt, det, gesehen, bedient)
    print("\n" + "=" * 62)
    print(f"PI {a.id}   {w}x{h}   Belichtung {a.exposure} us   "
          f"{'frei' if a.frei else 'Trigger'}")
    print("=" * 62)
    drucke(d, a.ziel)
    push.send_json({"camera_id": a.id, "stats": d})       # der PC druckt sie mit
    time.sleep(0.3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, default=1,
                   help="CAMERA_ID, pro Pi eindeutig. In allen Aufnahmen wurden "
                        "1 und 2 benutzt, nicht 0 und 1.")
    p.add_argument("--ip", default=None, help="IP des PCs (ausser bei --selfcheck nötig)")
    p.add_argument("--size", default="1640x1232",
                   help="BxH. Default = Sensormodus des IMX219 (4:3, volles "
                        "Sichtfeld) und damit identisch zu record_pi_raw.py, mit "
                        "dem 76 der 77 Trainingsaufnahmen entstanden sind. "
                        "820x616 ist exakt die Hälfte, also dasselbe Sichtfeld "
                        "bei halber Auflösung - schneller, ohne Geometriebruch. "
                        "NICHT 1280x720 nehmen: 16:9 beschneidet vertikal.")
    p.add_argument("--exposure", type=int, default=4000, help="us, wie record_pi.py")
    p.add_argument("--warm", type=int, default=10, help="Frames vor Messbeginn verwerfen")
    p.add_argument("--ziel", type=float, default=25.0, help="Zielrate für das Urteil")
    p.add_argument("--frei", action="store_true",
                   help="Trigger ignorieren und volle Last fahren. Misst die Obergrenze "
                        "der Hardware, ist aber nicht mehr kamerasynchron.")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()
    if not a.selfcheck and not a.ip:
        p.error("--ip nötig")
    sys.stdout.reconfigure(line_buffering=True)
    selfcheck() if a.selfcheck else main(a)
