"""Pi-Worker (raw): schickt nur JPEG-Bilder, die Landmark-Erkennung läuft auf
dem PC. Der Pi macht KEINE Inferenz mehr -> deutlich höhere FPS.

Sonst identisch zu record_pi.py: PC ist die Uhr, target_time kommt zurück,
capture_time misst den Jitter. Das Bild wird ROH (BGR, wie picamera2 RGB888 es
liefert) als JPEG verschickt -> der PC dreht es vor MediaPipe nach RGB.

  python record_pi_raw.py --id 0 --ip <IP des PCs>
"""
import argparse
import base64
import time

import cv2
import numpy as np
import zmq
from picamera2 import Picamera2

p = argparse.ArgumentParser()
p.add_argument("--id", type=int, required=True, help="CAMERA_ID, pro Pi eindeutig")
p.add_argument("--ip", required=True, help="IP des PCs")
p.add_argument("--quality", type=int, default=80, help="JPEG-Qualität 1-100")
p.add_argument("--size", default="1640x1232",
               help="BxH. Default = volle-FOV 4:3 (2x2-binned IMX219) -> passt zur "
                    "3280x2464-Kalibrierung. Kleiner = weniger Bandbreite, aber 4:3 halten!")
p.add_argument("--exposure", type=int, default=1500,
               help="Belichtung in us. 4000=4ms gab ~44px Bewegungsschmier an schnellen "
                    "Ictus-Frames (Haupt-Ausreisserquelle). 1000-2000 halbiert/drittelt das. "
                    "ACHTUNG: kürzer = dunkler -> mit --gain/Licht gegensteuern.")
p.add_argument("--gain", type=float, default=25.0,
               help="AnalogueGain, kompensiert die kürzere Belichtung. Höher = heller "
                    "aber rauschiger. An der Erkennungsrate justieren.")
args = p.parse_args()
w, h = (int(x) for x in args.size.split("x"))

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect(f"tcp://{args.ip}:5555")
sub.setsockopt_string(zmq.SUBSCRIBE, "")
push = ctx.socket(zmq.PUSH)
push.connect(f"tcp://{args.ip}:5556")
poller = zmq.Poller()
poller.register(sub, zmq.POLLIN)

picam = Picamera2()
picam.configure(picam.create_video_configuration({"format": "RGB888", "size": (w, h)}))
picam.start()
picam.set_controls({"AeEnable": False, "ExposureTime": args.exposure,
                    "AnalogueGain": args.gain})
time.sleep(2)
print(f"Pi {args.id} (raw) bereit -> {args.ip}")

enc = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

try:
    while True:
        if sub not in dict(poller.poll(10)):
            continue
        msg = sub.recv_json()
        target = msg.get("target_time")
        if msg.get("stop"):
            print("Stop empfangen.")
            break

        while time.time() < target:             # auf den gemeinsamen Zeitpunkt warten
            time.sleep(0.001)

        frame = np.ascontiguousarray(picam.capture_array("main"))  # BGR (picamera2 RGB888)
        ok, buf = cv2.imencode(".jpg", frame, enc)                  # BGR-JPEG, PC dreht -> RGB
        push.send_json({
            "target_time": target,              # die gemeinsame Uhr, NICHT ein Zähler
            "camera_id": args.id,
            "capture_time": time.time(),        # echte Auslösezeit -> Jitter messbar
            "jpg": base64.b64encode(buf).decode("ascii"),
        })
except KeyboardInterrupt:
    print("\nPi gestoppt.")
finally:
    picam.stop()
