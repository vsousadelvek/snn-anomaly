"""Demo: anomalia sintetica + score de reconstrucao da SNN em tempo real.

Uso: python -m src.demo ../snn-tracker/raw/c02.mp4 -o demo/anomaly.mp4
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from .events import DVSSimulator
from .model import SpikingAE
from .make_data import WINDOW

EV_W, EV_H = 64, 64
CALIB_FRAMES = 100     # janelas p/ calibrar threshold


def inject_anomaly(frame, t, total):
    """Objeto piscando (strobe) cruzando a cena: eventos ON/OFF densos
    num padrao impossivel pro modelo 'normal'."""
    frac = t / total
    if not (0.40 <= frac <= 0.72):
        return frame, False
    p = (frac - 0.40) / 0.32
    h, w = frame.shape[:2]
    cx = int(w * 0.1 + p * w * 0.8)
    cy = int(h * (0.30 + 0.4 * p))
    f = frame.copy()
    on = (t // 2) % 2 == 0
    r = 55
    cv2.rectangle(f, (cx - r, cy - r), (cx + r, cy + r),
                  (255, 255, 255) if on else (0, 0, 0), -1)
    return f, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="demo/anomaly.mp4")
    args = ap.parse_args()

    ckpt = torch.load("checkpoints/spiking_ae.pt", map_location="cpu",
                      weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpikingAE().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (1200, 400))

    dvs = DVSSimulator(size=(EV_W, EV_H), threshold=0.15)
    buf = deque(maxlen=WINDOW)
    scores, baseline = [], []
    threshold = None

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, anomaly = inject_anomaly(frame, t, total)
        t += 1
        buf.append(dvs.process(frame))

        score = 0.0
        if len(buf) == WINDOW:
            acc = np.stack([np.clip(sum(buf), 0, 1),
                            np.clip(-sum(buf), 0, 1)], dtype=np.float32)
            x = torch.from_numpy(acc).unsqueeze(0).to(device)
            with torch.no_grad():
                score = float(((model(x) - x) ** 2).mean())
            scores.append(score)
            if threshold is None:
                baseline.append(score)
                if len(baseline) >= CALIB_FRAMES:
                    threshold = np.mean(baseline) + 4 * np.std(baseline)

        # painel esquerdo: camera + alerta
        view = cv2.resize(frame, (400, 400))
        # centro: eventos
        ev = np.zeros((EV_H, EV_W, 3), np.uint8)
        if buf:
            e = buf[-1]
            ev[e > 0] = (255, 255, 0); ev[e < 0] = (255, 0, 255)
        ev_img = cv2.resize(ev, (400, 400), interpolation=cv2.INTER_NEAREST)

        # direito: grafico de score
        panel = np.zeros((400, 400, 3), np.uint8)
        cv2.putText(panel, "ANOMALY SCORE", (100, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (120, 120, 120), 2)
        if scores:
            mx = max(max(scores), threshold or 0) * 1.15
            pts = scores[-200:]
            for i, s in enumerate(pts):
                y = int(370 - 300 * (s / mx))
                cv2.circle(panel, (20 + i * 2, y), 1, (0, 255, 255), -1)
            if threshold:
                ty = int(370 - 300 * (threshold / mx))
                cv2.line(panel, (0, ty), (400, ty), (0, 0, 255), 1)
                cv2.putText(panel, "threshold", (300, ty - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cur = scores[-1]
            alarming = threshold and cur > threshold
            cv2.putText(panel, f"{cur:.4f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if alarming else (0, 255, 0), 2)
            if alarming:
                cv2.rectangle(view, (0, 0), (400, 400), (0, 0, 255), 8)
                cv2.putText(view, "ANOMALIA", (100, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        cv2.putText(view, "camera", (10, 390), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(ev_img, "eventos (DVS)", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(np.hstack([view, ev_img, panel]))

    cap.release()
    out.release()
    n_anom = sum(1 for s in scores[CALIB_FRAMES:] if threshold and s > threshold)
    print(f"salvo: {args.out} | threshold={threshold:.5f} | "
          f"{n_anom} janelas acima do threshold")


if __name__ == "__main__":
    main()
