#!/usr/bin/env python3
"""Shot boundaries and per-shot motion.

A cut is a frame whose colour histogram breaks from the previous frame's
(Bhattacharyya distance above CUT). A shot is STATIC when the picture barely
changes inside it (median frame-to-frame difference below STATIC_DIFF on a
0-255 scale, after downscaling): end cards and packshots. Static shots are
where pseudonym mode may lay an invented text logo over a blurred one; on a
moving shot an overlay would swim, so those are blurred only.
"""
import numpy as np

CUT = 0.45
STATIC_DIFF = 1.5
MIN_SHOT_FRAMES = 4


def analyze(frame_iter, fps):
    import cv2
    prev_hist, prev_small = None, None
    cuts, diffs = [0], []
    n = 0
    for k, fr in frame_iter:
        small = cv2.resize(fr, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_hist is not None:
            d = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > CUT and k - cuts[-1] >= MIN_SHOT_FRAMES:
                cuts.append(k)
            diffs.append(float(np.mean(np.abs(gray - prev_small))))
        else:
            diffs.append(0.0)
        prev_hist, prev_small = hist, gray
        n = k + 1
    shots = []
    for i, a in enumerate(cuts):
        b = (cuts[i + 1] - 1) if i + 1 < len(cuts) else n - 1
        inner = diffs[a + 1: b + 1] or [0.0]
        med = float(np.median(inner))
        shots.append({"shot": i + 1, "f0": a, "f1": b,
                      "start_ms": int(a * 1000 / fps),
                      "end_ms": int((b + 1) * 1000 / fps),
                      "motion": round(med, 2),
                      "static": med < STATIC_DIFF})
    return shots


def shot_of(shots, k):
    for s in shots:
        if s["f0"] <= k <= s["f1"]:
            return s
    return shots[-1]
