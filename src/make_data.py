"""Extrai event frames de videos 'normais' -> dataset .npy

Uso: python -m src.make_data ../snn-tracker/raw/c02.mp4 ../snn-tracker/raw/c03.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from .events import DVSSimulator

WINDOW = 10          # frames acumulados por event frame
SIZE = (64, 64)


def video_to_frames(path, max_windows=400):
    cap = cv2.VideoCapture(str(path))
    dvs = DVSSimulator(size=SIZE, threshold=0.15)
    buf, out = [], []
    while len(out) < max_windows:
        ok, f = cap.read()
        if not ok:
            break
        buf.append(dvs.process(f))
        if len(buf) == WINDOW:
            acc = np.stack([np.clip(sum(buf), 0, 1),
                            np.clip(-sum(buf), 0, 1)]).astype(np.float32)
            out.append(acc)
            buf = buf[WINDOW // 2:]   # overlap 50%
    cap.release()
    return out


def main():
    out_dir = Path("data/normal")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for vid in sys.argv[1:]:
        frames = video_to_frames(vid)
        for i, x in enumerate(frames):
            np.save(out_dir / f"{Path(vid).stem}_{i:04d}.npy", x)
        total += len(frames)
        print(f"{vid}: {len(frames)} event frames")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
