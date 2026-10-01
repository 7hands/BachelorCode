"""Live-Bild mit den erkannten Handlandmarks - für Fotos und zur Kontrolle.

Zeigt das Kamerabild mit dem Handskelett, das MediaPipe erkennt (derselbe
HandLandmarker wie im Echtzeitsystem). Mit --zwei-punkte werden zusätzlich
die beiden Punkte markiert, die die TCN-Modelle bekommen (Median der
Fingerspitzen und Handgelenk), mit --spur die Bahn des Fingerspitzen-Medians
der letzten N Bilder - so wird die Taktfigur sichtbar.

  python landmarks_bild.py                         # Webcam 0
  python landmarks_bild.py --pi --cam 1            # Pi mit live_pi_raw.py
  python landmarks_bild.py --zwei-punkte --spur 50
  python landmarks_bild.py --checkpoint models/TCN_TCN_alle_Personen_80_20_s0.pt

Mit --checkpoint wird wie in live_pc.py die Taktart vorhergesagt (Mehrheit der
letzten neun Vorhersagen) und oben rechts ins Bild geschrieben.

Alle --alle Sekunden (Vorgabe 5) wird automatisch ein Bild gespeichert, sobald
eine Hand erkannt ist, in den Ordner bilder/ - zum Aussuchen. --alle 0 schaltet
das ab. Tasten: Leertaste oder s = sofort speichern, q oder Esc = beenden. Für ein Foto mit dem Pi dort in voller Auflösung
aufnehmen: python live_pi_raw.py --id 1 --ip <PC> --size 1640x1232
(dann reicht eine niedrige Rate, z. B. --trigger 10 hier).
"""
import argparse
import base64
import os
import sys
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

from modelle import BONES
from merkmale import TIPS, WRIST
import torch

import live_pc as LP
from merkmale import FPS, MEDPAD
from record_pc_raw import make_landmarker

GRUEN, ROT, GELB = (80, 200, 60), (40, 40, 230), (0, 215, 255)


def erkenne(landmarker, bgr):
    """BGR-Bild -> (21,2) normierte Bildkoordinaten wie im Training, oder None."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(rgb)))
    if not res.hand_landmarks:
        return None
    return np.array([(lm.x, lm.y) for lm in res.hand_landmarks[0]])


class Takt:
    """Vorhersage der Taktart wie live_pc.py: Puffer -> Fenster -> Modell -> Mehrheit."""

    def __init__(self, pfad):
        self.model, self.win, self.build, ck = LP.lade(pfad)
        self.klassen = ck["klassen"]
        self.buf, self.letzte = deque(), deque(maxlen=LP.MEHRHEIT)
        self.halten = 2.0 * (self.win + MEDPAD) / FPS
        print(f"Modell {ck['arch']}: erste Vorhersage nach {self.win / FPS:.1f} s")

    def neu(self, t, pts):
        """Ein Bild dazu -> Text für die Anzeige."""
        if pts is not None:
            self.buf.append((t, pts))
        elif self.buf:
            self.buf.append((t, self.buf[-1][1]))      # Hand fehlt: letzte wiederholen
        while self.buf and self.buf[0][0] < t - self.halten:
            self.buf.popleft()
        f = LP.fenster(self.buf, self.win)
        if f is None:
            return "Takt: ..."
        x = torch.tensor(np.asarray(self.build(f, 0, self.win))[None], dtype=torch.float32)
        with torch.no_grad():
            self.letzte.append(int(self.model(x).argmax(1)))
        return f"Takt: {self.klassen[Counter(self.letzte).most_common(1)[0][0]]}/4"


def schreibe(bild, text):
    """Text oben rechts auf dunklem Kasten."""
    h, w = bild.shape[:2]
    gr = w / 900
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, gr, max(1, round(2 * gr)))
    rand = round(12 * gr)
    x0, y0 = w - tw - 2 * rand, rand
    cv2.rectangle(bild, (x0, y0), (w - rand, y0 + th + bl + rand), (0, 0, 0), -1)
    cv2.putText(bild, text, (x0 + rand // 2, y0 + th + rand // 2), cv2.FONT_HERSHEY_SIMPLEX,
                gr, (255, 255, 255), max(1, round(2 * gr)), cv2.LINE_AA)
    return bild


def zeichne(bild, pts, zwei_punkte, spur):
    """Skelett, optional die zwei Modellpunkte und die Bahn, direkt ins Bild."""
    s = max(1, round(bild.shape[1] / 1000))              # Strichstärke mit der Auflösung
    if len(spur) > 1:
        cv2.polylines(bild, [np.int32(spur)], False, GELB, 2 * s, cv2.LINE_AA)
    if pts is None:
        return bild
    pts = np.asarray(pts) * [bild.shape[1], bild.shape[0]]
    p = np.int32(pts)
    for a, b in BONES:
        cv2.line(bild, tuple(p[a]), tuple(p[b]), GRUEN, s, cv2.LINE_AA)
    for q in p:
        cv2.circle(bild, tuple(q), 2 * s, GRUEN, -1, cv2.LINE_AA)
    if zwei_punkte:
        for q in (np.median(np.asarray(pts)[TIPS], axis=0), pts[WRIST]):
            cv2.circle(bild, tuple(np.int32(q)), 4 * s, ROT, -1, cv2.LINE_AA)
    return bild


def bilder_webcam(nr):
    print(f"Öffne Webcam {nr} ...", flush=True)
    # V4L2 direkt: der Standardweg über GStreamer bleibt unter Linux teils hängen
    cap = cv2.VideoCapture(nr, cv2.CAP_V4L2) if sys.platform.startswith("linux") \
        else cv2.VideoCapture(nr)
    if not cap.isOpened():
        raise SystemExit(f"Webcam {nr} nicht gefunden - andere Nummer mit --kamera "
                         f"probieren (ls /dev/video*) oder --pi nehmen")
    ok, bgr = cap.read()
    if not ok:
        raise SystemExit(f"Webcam {nr} liefert keine Bilder - andere Nummer mit --kamera "
                         f"probieren (Laptops haben oft video0 und video1)")
    print(f"Webcam {nr} offen: {bgr.shape[1]}x{bgr.shape[0]}", flush=True)
    yield time.time(), bgr
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield time.time(), bgr


def bilder_pi(cam, rate):
    """Wie live_pc.py: Trigger senden, JPEG von live_pi_raw.py empfangen."""
    import zmq
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB); pub.bind("tcp://*:5555")
    pull = ctx.socket(zmq.PULL); pull.bind("tcp://*:5556")
    poller = zmq.Poller(); poller.register(pull, zmq.POLLIN)
    print("Warte auf den Pi ...")
    naechster = time.time()
    try:
        while True:
            if time.time() >= naechster:
                pub.send_json({"target_time": time.time() + 0.05})
                naechster = time.time() + 1.0 / rate
            if pull not in dict(poller.poll(5)):
                continue
            msg = pull.recv_json()
            if msg.get("camera_id") == cam and "jpg" in msg:
                buf = np.frombuffer(base64.b64decode(msg["jpg"]), np.uint8)
                yield (float(msg.get("capture_time") or time.time()),
                       cv2.imdecode(buf, cv2.IMREAD_COLOR))   # der Pi schickt BGR
    finally:
        pub.send_json({"target_time": time.time(), "stop": True})


def main(a):
    print(f"Lade MediaPipe ({a.delegate}) ...", flush=True)
    landmarker = make_landmarker(a.delegate)
    takt = Takt(a.checkpoint) if a.checkpoint else None
    quelle = bilder_pi(a.cam, a.trigger) if a.pi else bilder_webcam(a.kamera)
    spur = deque(maxlen=a.spur or 1)
    os.makedirs("bilder", exist_ok=True)
    naechstes = time.time() + a.alle
    print("Leertaste/s = speichern, q/Esc = beenden"
          + (f" | automatisch alle {a.alle:g} s nach bilder/" if a.alle else ""))
    for t, bgr in quelle:
        pts = erkenne(landmarker, bgr)
        if a.spur and pts is not None:
            spur.append(np.median(pts[TIPS], axis=0) * [bgr.shape[1], bgr.shape[0]])
        bild = zeichne(bgr.copy(), pts, a.zwei_punkte, list(spur) if a.spur else [])
        if takt:
            schreibe(bild, takt.neu(t, pts))
        cv2.imshow("Landmarks", bild)
        taste = cv2.waitKey(1) & 0xFF
        automatisch = a.alle and pts is not None and time.time() >= naechstes
        if taste in (ord(" "), ord("s")) or automatisch:
            name = os.path.join("bilder", f"landmarks_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(name, bild)
            naechstes = time.time() + a.alle
            print(f"-> {name} ({bild.shape[1]}x{bild.shape[0]}, Hand "
                  f"{'erkannt' if pts is not None else 'NICHT erkannt'})")
        if taste in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", action="store_true", help="Bilder vom Pi statt von der Webcam")
    ap.add_argument("--cam", type=int, default=1, help="Kamera-ID des Pis (mit --pi)")
    ap.add_argument("--trigger", type=float, default=25.0, help="Bildrate vom Pi in Hz")
    ap.add_argument("--kamera", type=int, default=0, help="Webcam-Nummer (ohne --pi)")
    ap.add_argument("--zwei-punkte", action="store_true", dest="zwei_punkte",
                    help="Fingerspitzen-Median und Handgelenk rot markieren")
    ap.add_argument("--spur", type=int, default=0,
                    help="Bahn des Fingerspitzen-Medians der letzten N Bilder zeigen")
    ap.add_argument("--checkpoint", default=None,
                    help="Modell aus takt_training.py - dann steht die Taktart im Bild")
    ap.add_argument("--alle", type=float, default=5.0,
                    help="automatisch alle N Sekunden speichern (0 = aus)")
    ap.add_argument("--delegate", default="cpu", choices=("auto", "cpu", "gpu"),
                    help="cpu reicht hier; auto probiert auch die GPU, deren Start "
                         "mit manchen Treibern hängen bleibt")
    main(ap.parse_args())
