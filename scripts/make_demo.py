"""Render a synthetic scene, track the red objects and save docs/demo.gif.

The scene has two red balls on crossing paths, a blue square, a green ball
and a white box. Only the two red balls should be tracked, and each should
keep its own ID.

    python scripts/make_demo.py [--colour red] [--video docs/demo.mp4]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

from colourtrack import CentroidTracker, detect
from colourtrack.__main__ import annotate

ROOT = Path(__file__).resolve().parent.parent
W, H, N = 480, 320, 90


def background(rng) -> np.ndarray:
    x = np.linspace(60, 140, W, dtype=np.float32)
    base = np.repeat(np.repeat(x[None, :, None], H, 0), 3, 2)
    base += rng.normal(0, 2, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def render(i: int, bg: np.ndarray) -> np.ndarray:
    f = bg.copy()
    t = i / N
    cv.rectangle(f, (200, 20), (280, 90), (235, 235, 235), -1)                # white box
    cv.circle(f, (int(40 + 400 * t), int(160 + 70 * math.sin(6.3 * t))), 20, (30, 30, 220), -1)
    cv.circle(f, (int(440 - 400 * t), int(110 + 50 * math.cos(6.3 * t))), 16, (20, 20, 200), -1)
    cv.rectangle(f, (int(380 - 60 * t), 220), (int(420 - 60 * t), 260), (200, 60, 20), -1)
    cv.circle(f, (int(120 + 30 * math.cos(9 * t)), 70), 18, (40, 190, 40), -1)
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--colour", default="red")
    p.add_argument("--video", help="also write an mp4")
    args = p.parse_args()

    bg = background(np.random.default_rng(0))
    tracker, frames = CentroidTracker(), []
    for i in range(N):
        frame = render(i, bg)
        frames.append(annotate(frame, tracker.update(detect(frame, args.colour))))
    print(f"{tracker._next_id - 1} {args.colour} objects tracked")

    gif = [Image.fromarray(cv.cvtColor(f, cv.COLOR_BGR2RGB))
           .convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames[::2]]
    out = ROOT / "docs" / "demo.gif"
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=66, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    if args.video:
        writer = cv.VideoWriter(args.video, cv.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for f in frames:
            writer.write(f)
        writer.release()


if __name__ == "__main__":
    main()
