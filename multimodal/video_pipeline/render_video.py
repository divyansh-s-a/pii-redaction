#!/usr/bin/env python3
"""Pixel changes for the video arm.

blur(): pixelate then Gaussian-blur a region — text and faces unreadable at
any size (a light blur alone leaves large logos legible).
plate(): pseudonym mode, STATIC shots only — an opaque plate in the region's
own average colour with the invented name in plain type, laid over the
blurred original. Never used on moving shots (it would swim) and never for a
region without a mapping entry (there is no invented name to show).
"""
from pathlib import Path

import numpy as np

FONT_CANDIDATES = ["/System/Library/Fonts/Helvetica.ttc",
                   "/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/Library/Fonts/Arial.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
PLATE_KINDS = {"logo", "brand_text", "signage", "url_handle"}


def blur(frame, px):
    import cv2
    x0, y0, x1, y1 = px
    roi = frame[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return
    # block size from the SHORT side: a thin line of text (a phone number)
    # needs blocks a third of its height, or the digits stay guessable
    k = max(6, min(w, h) // 3)
    small = cv2.resize(roi, (max(1, w // k), max(1, h // k)),
                       interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y0:y1, x0:x1] = cv2.GaussianBlur(pix, (0, 0),
                                           sigmaX=max(3, k * 0.6))


def _font(size):
    from PIL import ImageFont
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                continue
    return ImageFont.load_default()


def plate_image(w, h, text, bg_bgr):
    """An RGB plate w x h with `text` fitted inside, contrasting colour."""
    from PIL import Image, ImageDraw
    bg = tuple(int(c) for c in bg_bgr[::-1])
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg = (20, 20, 20) if lum > 140 else (245, 245, 245)
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    size = max(8, int(h * 0.62))
    while size > 8:
        f = _font(size)
        bb = d.textbbox((0, 0), text, font=f)
        if bb[2] - bb[0] <= w * 0.9 and bb[3] - bb[1] <= h * 0.8:
            break
        size -= 2
    f = _font(size)
    bb = d.textbbox((0, 0), text, font=f)
    d.text(((w - (bb[2] - bb[0])) / 2 - bb[0],
            (h - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=fg)
    return np.asarray(im)[:, :, ::-1].copy()


def plate(frame, px, text, cache=None):
    x0, y0, x1, y1 = px
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return
    key = (w, h, text)
    if cache is not None and key in cache:
        img = cache[key]
    else:
        bg = frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        img = plate_image(w, h, text, bg)
        if cache is not None:
            cache[key] = img
    frame[y0:y1, x0:x1] = img


STILL_GROWTH = 1.3     # union of a region's boxes / its median box area


def regions_still(regions_by_frame, shot):
    """True when every region of the shot stays put: the union of a region's
    tracked boxes over the shot is barely larger than one box. A small
    title sliding over a flat background leaves the FRAME almost unchanged,
    so frame difference alone cannot call a shot static."""
    clusters = _clusters(regions_by_frame, shot, lambda rg: True)
    return all(c["growth"] <= STILL_GROWTH for c in clusters)


def _clusters(regions_by_frame, shot, keep):
    clusters = []
    for t in range(shot["f0"], shot["f1"] + 1):
        for rg in regions_by_frame.get(t, []):
            if not keep(rg):
                continue
            a = (rg["px"][2] - rg["px"][0]) * (rg["px"][3] - rg["px"][1])
            for c in clusters:
                if (c["entity"] == rg.get("entity")
                        and c["kind"] == rg.get("kind")
                        and _iou(c["last"], rg["px"]) > 0.2):
                    c["px"] = [min(c["px"][0], rg["px"][0]),
                               min(c["px"][1], rg["px"][1]),
                               max(c["px"][2], rg["px"][2]),
                               max(c["px"][3], rg["px"][3])]
                    c["areas"].append(a)
                    c["last"] = rg["px"]
                    break
            else:
                clusters.append({"entity": rg.get("entity"),
                                 "kind": rg.get("kind"), "px": list(rg["px"]),
                                 "last": rg["px"], "areas": [a]})
    for c in clusters:
        med = sorted(c["areas"])[len(c["areas"]) // 2] or 1
        ua = (c["px"][2] - c["px"][0]) * (c["px"][3] - c["px"][1])
        c["growth"] = ua / med
    return clusters


def static_plates(regions_by_frame, shot, em):
    """For a static shot: one plate per mapped, voiced, text-like region
    that stays put, at the union of its tracked boxes."""
    if em is None:
        return []

    def voiced_text(rg):
        e = em.by_id.get(rg.get("entity") or "")
        return (e is not None and e["render"] == "voice" and e["replace"]
                and rg.get("kind") in PLATE_KINDS)
    out = []
    for c in _clusters(regions_by_frame, shot, voiced_text):
        if c["growth"] > STILL_GROWTH:
            continue                       # it moves: blur only
        out.append({"entity": c["entity"],
                    "text": em.by_id[c["entity"]]["replace"],
                    "px": c["px"]})
    # one plate per entity per place: boxes from different readers (an OCR
    # word, a model's logo box) overlap and would stack plates on each other
    merged = []
    for p in sorted(out, key=lambda p: -(p["px"][2] - p["px"][0])
                    * (p["px"][3] - p["px"][1])):
        for m in merged:
            if m["entity"] == p["entity"] and _overlap(m["px"], p["px"]):
                m["px"] = [min(m["px"][0], p["px"][0]),
                           min(m["px"][1], p["px"][1]),
                           max(m["px"][2], p["px"][2]),
                           max(m["px"][3], p["px"][3])]
                break
        else:
            merged.append(dict(p, px=list(p["px"])))
    return merged


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
