"""Detect red, green or blue objects in a frame.

Same method as matlab/detect_colour_blobs.m:

    difference = channel - grayscale    (bright only where that colour dominates)
    -> 3x3 median filter -> threshold -> drop blobs < min_area -> connected components
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

CHANNEL = {"red": 2, "green": 1, "blue": 0}          # OpenCV frames are BGR
THRESHOLD = {"red": 0.24, "green": 0.05, "blue": 0.15}  # same as the MATLAB code


@dataclass(frozen=True)
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float

    @property
    def centroid(self) -> tuple[float, float]:
        return self.cx, self.cy


def colour_mask(frame: np.ndarray, colour: str, threshold: float | None = None,
                min_area: int = 300) -> np.ndarray:
    """Binary mask (uint8 0/255) of pixels where ``colour`` dominates."""
    if colour not in CHANNEL:
        raise ValueError("colour must be red, green or blue")
    t = THRESHOLD[colour] if threshold is None else threshold
    # MATLAB rgb2gray weights; computed in float to avoid uint8 wrap-around
    gray = frame.astype(np.float32) @ np.array([0.1140, 0.5870, 0.2989], np.float32)
    diff = np.clip(frame[..., CHANNEL[colour]].astype(np.float32) - gray, 0, 255).astype(np.uint8)
    diff = cv.medianBlur(diff, 3)
    mask = (diff > t * 255).astype(np.uint8) * 255
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    return mask


def detect(frame: np.ndarray, colour: str, threshold: float | None = None,
           min_area: int = 300) -> list[Blob]:
    """Return the blobs of ``colour`` in a BGR frame, largest first."""
    mask = colour_mask(frame, colour, threshold, min_area)
    n, _, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [
        Blob(int(x), int(y), int(w), int(h), int(a), float(cx), float(cy))
        for (x, y, w, h, a), (cx, cy) in zip(stats[1:], centroids[1:])
    ]
    return sorted(blobs, key=lambda b: -b.area)
