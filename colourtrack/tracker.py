"""Keep the same ID for an object from frame to frame.

The MATLAB version only detects blobs in each frame independently. This
tracker links them over time with greedy nearest-centroid matching: each
existing track takes the closest new detection within ``max_distance``
pixels, unmatched detections start new tracks, and tracks that go missing
for more than ``max_missing`` frames are dropped. Each track keeps a trail
of recent positions so the motion path can be drawn.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detect import Blob


@dataclass
class Track:
    id: int
    blob: Blob
    trail: deque = field(default_factory=lambda: deque(maxlen=40))
    missing: int = 0

    def speed(self) -> float:
        """Average movement in pixels per frame over the trail."""
        if len(self.trail) < 2:
            return 0.0
        pts = np.array(self.trail)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())


class CentroidTracker:
    def __init__(self, max_distance: float = 80, max_missing: int = 10):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, blobs: list[Blob]) -> list[Track]:
        """Match this frame's blobs to existing tracks; returns visible tracks."""
        unmatched = list(range(len(blobs)))
        pairs = []
        for tid, track in self.tracks.items():
            for j in unmatched:
                d = np.hypot(track.blob.cx - blobs[j].cx, track.blob.cy - blobs[j].cy)
                if d <= self.max_distance:
                    pairs.append((d, tid, j))
        pairs.sort()

        used_tracks, used_blobs = set(), set()
        for _, tid, j in pairs:
            if tid in used_tracks or j in used_blobs:
                continue
            track = self.tracks[tid]
            track.blob, track.missing = blobs[j], 0
            track.trail.append(blobs[j].centroid)
            used_tracks.add(tid)
            used_blobs.add(j)

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    del self.tracks[tid]

        for j in range(len(blobs)):
            if j not in used_blobs:
                track = Track(self._next_id, blobs[j])
                track.trail.append(blobs[j].centroid)
                self.tracks[self._next_id] = track
                self._next_id += 1

        return [t for t in self.tracks.values() if t.missing == 0]
