import cv2 as cv
import numpy as np
import pytest

from colourtrack import CentroidTracker, detect
from colourtrack.detect import colour_mask

BGR = {"red": (30, 30, 220), "green": (40, 200, 40), "blue": (220, 60, 30)}


def canvas(value=110):
    return np.full((240, 320, 3), value, np.uint8)


@pytest.mark.parametrize("colour", ["red", "green", "blue"])
def test_detects_each_colour_and_only_that_colour(colour):
    f = canvas()
    for i, (name, bgr) in enumerate(BGR.items()):
        cv.circle(f, (60 + 100 * i, 120), 25, bgr, -1)
    blobs = detect(f, colour)
    assert len(blobs) == 1
    expected_x = 60 + 100 * list(BGR).index(colour)
    assert abs(blobs[0].cx - expected_x) < 2 and abs(blobs[0].cy - 120) < 2


def test_white_gray_and_black_are_ignored():
    f = canvas()
    cv.rectangle(f, (10, 10), (100, 100), (255, 255, 255), -1)
    cv.rectangle(f, (150, 10), (250, 100), (0, 0, 0), -1)
    for colour in BGR:
        assert detect(f, colour) == []


def test_small_blobs_removed_by_min_area():
    f = canvas()
    cv.circle(f, (100, 100), 5, BGR["red"], -1)     # ~80 px
    cv.circle(f, (200, 100), 20, BGR["red"], -1)    # ~1250 px
    assert len(detect(f, "red", min_area=300)) == 1
    assert len(detect(f, "red", min_area=10)) == 2


def test_salt_noise_is_filtered():
    f = canvas()
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, 240, 400), rng.integers(0, 320, 400)
    f[ys, xs] = BGR["red"]
    assert detect(f, "red", min_area=1) == []


def test_mask_has_no_uint8_wraparound():
    f = canvas(0)
    f[..., 2] = 10          # dark red: channel < gray is impossible here, but
    f[..., 0] = 250         # a blue pixel must not wrap to a large red value
    assert colour_mask(f, "red", min_area=1).max() == 0


def test_invalid_colour():
    with pytest.raises(ValueError):
        detect(canvas(), "purple")


def test_tracker_keeps_ids_through_motion_and_crossing():
    tracker = CentroidTracker(max_distance=60)
    ids = []
    for step in range(20):
        f = canvas()
        cv.circle(f, (40 + 12 * step, 80), 18, BGR["red"], -1)     # moving right
        cv.circle(f, (280 - 12 * step, 170), 18, BGR["red"], -1)   # moving left
        tracks = tracker.update(detect(f, "red"))
        ids.append({t.id: round(t.blob.cy) for t in tracks})
    assert all(len(frame) == 2 for frame in ids)
    assert ids[0] == ids[-1] == {1: 80, 2: 170}


def test_tracker_survives_short_occlusion_and_forgets_long_ones():
    tracker = CentroidTracker(max_distance=50, max_missing=3)
    blob = detect(cv.circle(canvas(), (100, 100), 20, BGR["red"], -1), "red")
    tracker.update(blob)
    for _ in range(3):
        assert tracker.update([]) == []
    assert [t.id for t in tracker.update(blob)] == [1]
    for _ in range(4):
        tracker.update([])
    assert [t.id for t in tracker.update(blob)] == [2]


def test_speed_estimate():
    tracker = CentroidTracker()
    for step in range(5):
        f = cv.circle(canvas(), (50 + 10 * step, 100), 20, BGR["blue"], -1)
        tracks = tracker.update(detect(f, "blue"))
    assert tracks[0].speed() == pytest.approx(10, abs=0.5)
