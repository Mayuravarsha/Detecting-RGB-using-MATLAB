"""Track coloured objects from a webcam or a video file.

    python -m colourtrack --colour red                # webcam 0
    python -m colourtrack --colour blue --source clip.mp4 --output out.mp4
"""

from __future__ import annotations

import argparse
import sys

import cv2 as cv

from .detect import CHANNEL, detect
from .tracker import CentroidTracker

BOX = (0, 255, 255)       # yellow, as in the MATLAB version
MARK = (255, 0, 255)      # magenta


def annotate(frame, tracks):
    out = frame.copy()
    for t in tracks:
        b = t.blob
        cv.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), BOX, 2)
        pts = [tuple(map(int, p)) for p in t.trail]
        for p, q in zip(pts, pts[1:]):
            cv.line(out, p, q, MARK, 2)
        cv.drawMarker(out, (int(b.cx), int(b.cy)), MARK, cv.MARKER_CROSS, 14, 2)
        cv.putText(out, f"#{t.id} X:{int(b.cx)} Y:{int(b.cy)}", (b.x, max(12, b.y - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(out, f"#{t.id} X:{int(b.cx)} Y:{int(b.cy)}", (b.x, max(12, b.y - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="colourtrack", description=__doc__.splitlines()[0])
    p.add_argument("--colour", choices=sorted(CHANNEL), default="red")
    p.add_argument("--source", default="0", help="webcam index or video file")
    p.add_argument("--output", help="save the annotated video here")
    p.add_argument("--threshold", type=float, help="override the colour's default threshold (0-1)")
    p.add_argument("--min-area", type=int, default=300)
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args(argv)

    cap = cv.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print(f"cannot open {args.source}", file=sys.stderr)
        return 1
    tracker, writer, frames = CentroidTracker(), None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        tracks = tracker.update(detect(frame, args.colour, args.threshold, args.min_area))
        out = annotate(frame, tracks)
        if args.output:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*"mp4v"),
                                        cap.get(cv.CAP_PROP_FPS) or 25, (w, h))
            writer.write(out)
        if not args.no_display:
            cv.imshow("colourtrack (esc to quit)", out)
            if cv.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv.destroyAllWindows()
    print(f"processed {frames} frames, {tracker._next_id - 1} objects seen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
