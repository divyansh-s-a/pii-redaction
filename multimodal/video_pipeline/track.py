#!/usr/bin/env python3
"""Carry sampled-frame regions to every frame of their shot.

Each region found on a sampled frame is tracked FORWARD to the next sample
and BACKWARD to the previous one by template matching (normalised
cross-correlation on a downscaled grey frame, searched around the last
position). Every frame gets the union of what reaches it from both sides,
so a region seen on either neighbouring sample covers the frames between.
When the match is weak (zoom, occlusion, motion blur) the box HOLDS its last
position and grows, a little more each frame — over-blurring is the safe
direction. Tracking never crosses a shot cut.
"""
import numpy as np

WORK_W = 640            # tracking resolution (width)
MATCH_MIN = 0.45
GROW_PER_FRAME = 0.03   # margin added per frame while holding
GROW_MAX = 0.5
PAD = 0.10              # final padding, each side, relative to box size
PAD_PX = 6


def _to_px(box, w, h):
    x0, y0, x1, y1 = box
    return [x0 * w, y0 * h, x1 * w, y1 * h]


def _track(frames, start, stop, step, box):
    """box (px, tracking scale) from frames[start] to frames[stop) by step.
    Returns {index: (box, held)}."""
    import cv2
    h, w = frames[0].shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x1, x0 + 8)), min(h, max(y1, y0 + 8))
    tmpl = frames[start][y0:y1, x0:x1]
    out = {}
    cur = [x0, y0, x1, y1]
    grow = 0.0
    if tmpl.size == 0 or tmpl.std() < 2:          # flat patch: just hold
        for t in range(start + step, stop, step):
            grow = min(GROW_MAX, grow + GROW_PER_FRAME)
            out[t] = (_grown(cur, grow, w, h), True)
        return out
    tw, th = x1 - x0, y1 - y0
    for t in range(start + step, stop, step):
        m = int(max(tw, th) * 0.75) + 16
        sx0, sy0 = max(0, cur[0] - m), max(0, cur[1] - m)
        sx1, sy1 = min(w, cur[2] + m), min(h, cur[3] + m)
        win = frames[t][sy0:sy1, sx0:sx1]
        held = True
        if win.shape[0] >= th and win.shape[1] >= tw:
            res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
            _mn, mx, _ml, loc = cv2.minMaxLoc(res)
            if mx >= MATCH_MIN:
                cur = [sx0 + loc[0], sy0 + loc[1],
                       sx0 + loc[0] + tw, sy0 + loc[1] + th]
                held, grow = False, 0.0
        if held:
            grow = min(GROW_MAX, grow + GROW_PER_FRAME)
        out[t] = (_grown(cur, grow, w, h), held)
    return out


def _grown(b, g, w, h):
    dx, dy = (b[2] - b[0]) * g, (b[3] - b[1]) * g
    return [max(0, b[0] - dx), max(0, b[1] - dy),
            min(w, b[2] + dx), min(h, b[3] + dy)]


def track_shot(small_frames, samples, full_w, full_h):
    """small_frames: grey frames of ONE shot at tracking scale.
    samples: {local_index: [region, ...]} with normalised boxes.
    Returns {local_index: [region-with-'px' (full resolution, padded)]}."""
    n = len(small_frames)
    if not n:
        return {}
    sh, sw = small_frames[0].shape[:2]
    keys = sorted(samples)
    per = {k: [] for k in range(n)}
    for si, s in enumerate(keys):
        prev_s = keys[si - 1] if si > 0 else -1
        next_s = keys[si + 1] if si + 1 < len(keys) else n
        for rg in samples[s]:
            if not rg.get("box"):
                continue
            b = _to_px(rg["box"], sw, sh)
            per[s].append((rg, b, False))
            for t, (bb, held) in _track(small_frames, s, next_s, 1, b).items():
                per[t].append((rg, bb, held))
            for t, (bb, held) in _track(small_frames, s, prev_s, -1,
                                        b).items():
                per[t].append((rg, bb, held))
    sx, sy = full_w / sw, full_h / sh
    out = {}
    for t, items in per.items():
        regs = []
        for rg, b, held in items:
            x0, y0, x1, y1 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            px = (x1 - x0) * PAD + PAD_PX
            py = (y1 - y0) * PAD + PAD_PX
            regs.append(dict(rg, held=held, px=[
                int(max(0, x0 - px)), int(max(0, y0 - py)),
                int(min(full_w, x1 + px)), int(min(full_h, y1 + py))]))
        out[t] = regs
    return out


def small_gray(frame):
    import cv2
    h, w = frame.shape[:2]
    s = WORK_W / w if w > WORK_W else 1.0
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (int(w * s), int(h * s))) if s < 1 else g


def _self_test():
    import cv2
    rng = np.random.default_rng(1)
    bg = (rng.random((180, 320)) * 40).astype(np.uint8)
    frames = []
    for t in range(20):
        f = bg.copy()
        x = 20 + t * 8
        cv2.putText(f, "ZORVEX", (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    255, 2)
        frames.append(f)
    # sample at 0 and 19 only; the box must follow the text in between
    b0 = [16 / 320, 70 / 180, 140 / 320, 110 / 180]
    res = track_shot(frames, {0: [{"box": b0, "kind": "brand_text"}]},
                     320, 180)
    mid = res[10][0]["px"]
    assert mid[0] <= 20 + 10 * 8 <= mid[2], mid
    assert not res[10][0]["held"]
    print("track self-test: OK")


if __name__ == "__main__":
    _self_test()
