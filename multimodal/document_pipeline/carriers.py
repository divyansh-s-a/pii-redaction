#!/usr/bin/env python3
"""Composite carriers: rubber stamps and ink signatures.

A stamp carries provider identity AND the sign-off date a verifier needs, so
the original plan called for EXTRACT -> DESTROY -> RESTAMP. Nothing is written
back any more, so the restamp third is gone and the date must be SPARED
instead: the carrier is destroyed as up to four rectangles AROUND the date.
Owner's rule, verbatim: "What I asked you not to redact there is only date.
Everything else needs to be removed."

Sparing one small VERIFIED region is the opposite of the failure this
replaces. Before this pass, a box landed on a stamp's centre and left the
ring readable -- "DR AGARWALS HEALTHCARE LTD" was legible on five pages --
and on one page the box covered the printed word "Signature" while every
stroke of the signature beside it stood untouched.

WHY COLOUR, AND WHY NAIVE COLOUR FAILED BEFORE
A pigment that has been through a scanner is BOTH chromatic and attenuated,
so it takes two clauses, not one:
    spread = max(R,G,B) - min(R,G,B) >= SAT   pigment, not neutral grey
    max(R,G,B) <= VMAX                        ink, not a bright print tint
VMAX is what defeats born-digital pages, whose red emphasis text is pure
(255,0,0) and therefore excluded outright: measured on the reference bundle,
the four control pages drop from ~3-4k chromatic pixels to ZERO. SAT is what
defeats scanner chroma noise on black print, which sits at spread 0-10 with
the page's 99th percentile at 44, against stamp pigment at 60-124.

This is why an earlier attempt to seed a stamp detector from
handwriting.ink_boxes() failed and was abandoned: on page 17 it returned
466x20pt PRINTED TEXT LINES. It was not finding coloured print -- it was
finding grey print through a threshold loose enough to admit scanner noise.

MEASURED on the 29-page reference bundle, by an independent auditor who
re-ran the detector and rendered every box:
  * 12 of 29 pages fire, ONE region each, and all 12 are genuine stamps.
    Zero false positives across the document.
  * The four clinical/bill control pages yield 0 pixels, 0 regions.
  * Impression coverage >= 94% on every page after grow_box below.
  * Dates recovered and verified spared end to end.

KNOWN LIMITS, stated rather than buried:
  * A loose pen signature with no chromatic pigment is invisible here
    (page 3 of the reference bundle). Ink detection still covers that path.
  * A whole cancelled cheque can form one chromatic component and is then
    dropped by the size gates.
  * The spared date slit re-exposes 8-14% of the stamp's ink as ring-arc
    fragments. No letters are legible, but it is not zero.
  * Where a stamp is struck ON a chart, destroying it costs the chart: on
    the reference bundle's OCT page that is ~1.6% of the page, including
    some axis tick labels. Unavoidable -- the ink is physically on top.
"""
import re
import sys
from pathlib import Path

import numpy as np
import fitz
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
# LAST insert wins position 0. This directory must beat the audio arm's
# pipeline/, which has its own redact.py: in the wrong order a later plain
# `import redact` resolved to the AUDIO redactor and crashed the suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))

DETECT_DPI, OCR_DPI = 150, 600
SAT, VMAX      = 40, 210     # pigment spread / dark enough to be ink
OCR_SUPPRESS   = 35          # whiten pigment before OCR, to un-occlude print
CLOSE_PT       = 6.0
MIN_SIDE       = 40.0
MIN_LONG       = 45.0
MAX_SIDE       = 260.0
MAX_AREA_FRAC  = 0.06
ASPECT         = (0.35, 3.0)
FILL           = (0.03, 0.55)
KEEP_MAX_FRAC  = 0.45        # a spared date may not be most of the carrier

# CANDIDATES, NOT VERDICTS. On a page written in blue pen, colour says
# nothing about what is a stamp: the first version of this pass cut 37 blobs
# of a doctor's handwritten clinical notes (15% of the page) as "stamps" under
# the insurance pack, while the real hospital seal -- pale, spread ~26, max
# ~220 -- was too faint for its gates and stayed readable. So colour now only
# PROPOSES regions, with looser thresholds that reach a faded seal, and the
# model says what each region is before anything is cut.
CAND_SAT, CAND_VMAX = 20, 235
CAND_MIN_SIDE = 30.0     # stamp-sized and up; smaller marks (a signature, a
CAND_MIN_LONG = 36.0     # tick) are inkmarks.py's job, which keeps a merged
CAND_MAX_SIDE = 300.0    # blob of handwriting from being cut as one "stamp"
CAND_MAX_AREA_FRAC = 0.08
CAND_ASPECT = (0.25, 4.0)
CAND_FILL = (0.02, 0.60)
A4_WIDTH_PT = 595.0      # every size above is for an A4 page and SCALES with
                         # the real one: a 1700pt-wide scan rejected its 345pt
                         # seal as "too big" against a fixed 260pt limit

# the region kinds this pass cuts, and the pack semantic each one carries
CARRIER_CUT = {"logo_or_seal": "sem:visual_mark",
               "signature": "sem:signature",
               "handwritten_personal_data": "sem:ink_identity"}

# grow_box: a SECOND, looser pass used only to extend an ALREADY ACCEPTED
# box. Detection stays strict, so no new region can be created and the
# negative controls cannot change. Without it the palest edge of an
# impression falls outside the box: on the reference bundle page 21 that
# left "Dr. Aga" of "Dr. Agarwal's Eye Hospital" plainly legible at 78.4%
# coverage. With it, 99.3%.
GROW_SAT, GROW_VMAX = 25, 235
GROW_HALO_PT        = 70.0
GROW_MAX_SIDE       = 320.0
GROW_MAX_FRAC       = 0.06

MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
DATE_RE = [
    re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}"),
    re.compile(r"\d{1,2}[\s.\-]*" + MON + r"[\s\'.\-]*\d{2,4}", re.I),
    re.compile(MON + r"[\s.\-]*\d{1,2}[\s,\'.\-]*\d{2,4}", re.I),
]


def _rgb(page, dpi, clip=None):
    z = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip, colorspace=fitz.csRGB)
    a = np.frombuffer(pix.samples, dtype=np.uint8)
    return a.reshape(pix.height, pix.width, 3).astype(np.int16), pix


def _shape(m):
    """Diagnostics only: ring-ness and row/col coverage of the blob."""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return 0.0, 0.0, 0.0
    cy, cx = ys.mean(), xs.mean()
    r = np.hypot(ys - cy, xs - cx)
    R = r.max() or 1.0
    return (float((r > 0.6 * R).mean()),
            float((m.sum(1) > 0.05 * m.shape[1]).mean()),
            float((m.sum(0) > 0.05 * m.shape[0]).mean()))


def carriers(page, dpi=DETECT_DPI):
    """-> [{bbox: fitz.Rect in PDF pt, fill, ring, rowcov, colcov, ...}]"""
    a, _ = _rgb(page, dpi)
    mx, mn = a.max(2), a.min(2)
    M = ((mx - mn) >= SAT) & (mx <= VMAX)
    sc = 72.0 / dpi
    k = max(3, int(round(CLOSE_PT / sc)) | 1)
    C = ndi.binary_dilation(ndi.binary_closing(M, np.ones((k, k))), np.ones((5, 5)))
    lab, _n = ndi.label(C)
    parea = (page.rect.width * page.rect.height) or 1.0
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        h = (sl[0].stop - sl[0].start) * sc
        w = (sl[1].stop - sl[1].start) * sc
        if min(w, h) < MIN_SIDE or max(w, h) < MIN_LONG or max(w, h) > MAX_SIDE:
            continue
        if not (ASPECT[0] <= w / h <= ASPECT[1]):
            continue
        sub = M[sl] & (lab[sl] == i)
        fill = float(sub.sum()) / max(1.0, (w / sc) * (h / sc))
        if not (FILL[0] <= fill <= FILL[1]):
            continue
        r = fitz.Rect(sl[1].start * sc, sl[0].start * sc,
                      sl[1].stop * sc, sl[0].stop * sc)
        frac = (r.width * r.height) / parea
        if frac > MAX_AREA_FRAC:
            continue
        ring, rowc, colc = _shape(sub)
        out.append({"bbox": r, "fill": round(fill, 3), "ring": round(ring, 3),
                    "rowcov": round(rowc, 3), "colcov": round(colc, 3),
                    "ink_px": int(sub.sum()), "page_area_pct": round(100 * frac, 3)})
    out.sort(key=lambda d: -d["bbox"].width * d["bbox"].height)
    keep = []                                   # drop nested duplicates
    for d in out:
        if any(fitz.Rect(k["bbox"]).contains(d["bbox"]) for k in keep):
            continue
        keep.append(d)
    return keep


def date_in(page, rect, tag="x"):
    """OCR the carrier with pigment whitened; -> (date_text, keep_rect, ocr_lines)."""
    import objective
    r = fitz.Rect(rect) + (-6, -6, 6, 6)
    r.intersect(page.rect)
    a, pix = _rgb(page, OCR_DPI, clip=r)
    mx, mn = a.max(2), a.min(2)
    a[(mx - mn) >= OCR_SUPPRESS] = 255          # strip the stamp, keep black print
    path = f"/tmp/_carrier_{tag}.png"
    fitz.Pixmap(fitz.csRGB, pix.width, pix.height,
                a.astype(np.uint8).tobytes(), False).save(path)
    got = objective.vision(path, "text") or {}
    items = got.get("text") or []
    seen = [it.get("text", "") for it in items]
    for it in items:
        s = it.get("text") or ""
        for rx in DATE_RE:
            m = rx.search(s)
            if not m:
                continue
            lo, hi = m.span()
            pos, boxes = 0, []
            for t in (it.get("tokens") or []):
                tt = t.get("text") or ""
                j = s.find(tt, pos)
                if j < 0:
                    continue
                pos = j + len(tt)
                if j < hi and j + len(tt) > lo and t.get("box"):
                    boxes.append(t["box"])
            b = boxes or ([it["box"]] if it.get("box") else [])
            if not b:
                continue
            x0 = min(q["x0"] for q in b); x1 = max(q["x1"] for q in b)
            y0 = min(q["y0"] for q in b); y1 = max(q["y1"] for q in b)
            kr = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                           r.x0 + x1 * r.width, r.y0 + y1 * r.height)
            kr.intersect(fitz.Rect(rect))
            if kr.is_empty:
                continue
            if (kr.width * kr.height) > KEEP_MAX_FRAC * rect.width * rect.height:
                continue                        # fail closed: keep is too big
            return m.group(0), kr, seen
    return None, None, seen


def minus(carrier, keep):
    """carrier minus keep -> <=4 rects. keep=None -> the whole carrier."""
    c = fitz.Rect(carrier)
    if keep is None:
        return [c]
    k = fitz.Rect(keep); k.intersect(c)
    if k.is_empty:
        return [c]
    out = []
    if k.y0 > c.y0: out.append(fitz.Rect(c.x0, c.y0, c.x1, k.y0))
    if k.y1 < c.y1: out.append(fitz.Rect(c.x0, k.y1, c.x1, c.y1))
    if k.x0 > c.x0: out.append(fitz.Rect(c.x0, k.y0, k.x0, k.y1))
    if k.x1 < c.x1: out.append(fitz.Rect(k.x1, k.y0, c.x1, k.y1))
    return [q for q in out if q.width > 0.5 and q.height > 0.5]




def grow_box(page, rect, dpi=DETECT_DPI):
    """Extend an ACCEPTED carrier to the full extent of its impression.

    Detection above is deliberately strict, which clips the palest edge of an
    impression. This re-thresholds loosely but ONLY inside a halo around a box
    that already passed, and keeps only components touching it -- so it can
    never invent a region, and the negative controls are untouched by
    construction.
    """
    c = fitz.Rect(rect)
    k = (page.rect.width / A4_WIDTH_PT) or 1.0
    h = GROW_HALO_PT * k
    halo = fitz.Rect(c.x0 - h, c.y0 - h, c.x1 + h, c.y1 + h)
    halo.intersect(page.rect)
    if halo.is_empty:
        return c
    try:
        a, _ = _rgb(page, dpi, clip=halo)
    except Exception:
        return c
    mx, mn = a.max(2), a.min(2)
    M = ((mx - mn) >= GROW_SAT) & (mx <= GROW_VMAX)
    M = ndi.binary_opening(M, np.ones((2, 2)))
    lab, n = ndi.label(ndi.binary_dilation(M, np.ones((17, 17))))
    if not n:
        return c
    sc = 72.0 / dpi
    # which labels does the accepted box already touch?
    i0 = max(0, int((c.y0 - halo.y0) / sc)); i1 = int((c.y1 - halo.y0) / sc)
    j0 = max(0, int((c.x0 - halo.x0) / sc)); j1 = int((c.x1 - halo.x0) / sc)
    inside = set(np.unique(lab[i0:i1, j0:j1])) - {0}
    if not inside:
        return c
    grown = fitz.Rect(c)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if i not in inside or sl is None:
            continue
        grown |= fitz.Rect(halo.x0 + sl[1].start * sc,
                           halo.y0 + sl[0].start * sc,
                           halo.x0 + sl[1].stop * sc,
                           halo.y0 + sl[0].stop * sc)
    grown.intersect(page.rect)
    parea = (page.rect.width * page.rect.height) or 1.0
    if (max(grown.width, grown.height) > GROW_MAX_SIDE * k
            or (grown.width * grown.height) / parea > GROW_MAX_FRAC):
        return c                        # refuse a growth that ran away
    return grown


def candidates(page, dpi=DETECT_DPI):
    """Coloured, stamp-sized regions -> [fitz.Rect]. PROPOSALS only."""
    k = (page.rect.width / A4_WIDTH_PT) or 1.0
    a, _ = _rgb(page, dpi)
    mx, mn = a.max(2), a.min(2)
    M = ((mx - mn) >= CAND_SAT) & (mx <= CAND_VMAX)
    sc = 72.0 / dpi
    kk = max(3, int(round(CLOSE_PT * k / sc)) | 1)
    C = ndi.binary_dilation(ndi.binary_closing(M, np.ones((kk, kk))), np.ones((5, 5)))
    lab, _n = ndi.label(C)
    parea = (page.rect.width * page.rect.height) or 1.0
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        h = (sl[0].stop - sl[0].start) * sc
        w = (sl[1].stop - sl[1].start) * sc
        if (min(w, h) < CAND_MIN_SIDE * k or max(w, h) < CAND_MIN_LONG * k
                or max(w, h) > CAND_MAX_SIDE * k):
            continue
        if not (CAND_ASPECT[0] <= w / h <= CAND_ASPECT[1]):
            continue
        sub = M[sl] & (lab[sl] == i)
        fill = float(sub.sum()) / max(1.0, (w / sc) * (h / sc))
        if not (CAND_FILL[0] <= fill <= CAND_FILL[1]):
            continue
        r = fitz.Rect(sl[1].start * sc, sl[0].start * sc,
                      sl[1].stop * sc, sl[0].stop * sc)
        if (r.width * r.height) / parea > CAND_MAX_AREA_FRAC:
            continue
        out.append(r)
    out.sort(key=lambda r: -r.width * r.height)
    keep = []
    for r in out:                                   # drop nested duplicates
        if not any(fitz.Rect(q).contains(r) for q in keep):
            keep.append(r)
    return keep


def findings(page, pack=None, dpi=DETECT_DPI, workdir=None):
    """Propose by colour, classify, then cut: carrier-minus-date, fail closed.

    Only what the model calls a stamp/seal, a signature or handwritten
    personal data is cut; a handwritten clinical note in blue pen is left
    exactly as it is. A cut carrier keeps the owner's rule -- everything goes
    except a date found inside it -- and if no date is found the whole
    carrier goes.
    """
    out = []
    try:
        cands = candidates(page, dpi=dpi)
    except Exception:
        return out
    if not cands:
        return out
    import inkmarks
    verdicts, unclassified, dropped = inkmarks.classify_regions(page, cands, workdir)
    if unclassified or dropped:
        # Loud, never silent. Open, not closed: on a blue-pen page these
        # candidates are mostly clinical handwriting, and cutting them all on
        # one failed request is the exact failure this rewrite removes.
        print(f"    CARRIER CLASSIFICATION DEGRADED: {len(unclassified)} "
              f"unclassified, {len(dropped)} over the cap -- left UNCUT", flush=True)
    for idx, (rect, kind, _why) in enumerate(verdicts):
        if kind not in CARRIER_CUT:
            continue
        rect = grow_box(page, rect, dpi=dpi)
        try:
            date_text, keep, _seen = date_in(page, rect, tag=f"p{page.number}_{idx}")
        except Exception:
            date_text, keep = None, None
        boxes = minus(rect, keep)
        if not boxes:
            continue
        cat = CARRIER_CUT[kind]
        out.append({
            "categories": [cat], "text": f"[{kind.replace('_', ' ')}]",
            "bbox": [round(v, 2) for v in (boxes[0].x0, boxes[0].y0,
                                           boxes[0].x1, boxes[0].y1)],
            "all_boxes": [[round(v, 2) for v in (b.x0, b.y0, b.x1, b.y1)]
                          for b in boxes],
            "n_boxes": len(boxes),
            "source": "carrier_impression", "locator": "chromatic_ink",
            "action": (pack.action_for([cat]) if pack is not None else "REDACT"),
            "reason": (f"coloured region; model says {kind}; "
                       + (f"date {date_text!r} spared" if date_text
                          else "no date found - whole carrier destroyed")),
        })
    return out
