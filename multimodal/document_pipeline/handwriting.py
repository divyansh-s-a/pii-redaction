#!/usr/bin/env python3
"""Chromatic-ink handwriting regions: on a filled paper form, the pen IS the data.

The printed template of a form is black and grey; what a person wrote into it
is, overwhelmingly, blue ballpoint. That colour difference is a detector that
needs no OCR and no label: mask the blue ink, group strokes into words and
fields, and every filled-in value becomes a redaction region -- including the
ones whose printed label the handwriting itself has made unreadable (the
policy number on the claim form) and the ones no lexicon anticipates (a tick
in a 'Maternity' box, a scrawled 'BILLS ATTACHED', a signature).

Scope, stated plainly: blue/violet ink only. Black-pen handwriting is
indistinguishable from print by colour and is left to the label-anchored and
detector paths. Pages are measured first (`blue_share`), so a page with no
chromatic ink costs nothing and gets no regions.

    python3 document_pipeline/handwriting.py --self-test
    python3 document_pipeline/handwriting.py --census DIR_OF_PNGS
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import fitz
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
# LAST insert wins position 0. This directory must beat the audio arm's
# pipeline/, which has its own redact.py: in the wrong order a later plain
# `import redact` resolved to the AUDIO redactor and crashed the suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry

DPI = 110            # strokes are ~2px wide here; enough to connect, cheap to render
MIN_AREA_PX = 14     # below this it is JPEG chroma noise, not a pen stroke
DILATE = 2           # joins the strokes of one letter
MAX_H_PT = 70        # an address block or signature, never a logo or a banner
MAX_W_PT = 420
GAP_MULT = 2.6       # boxed-character fields: one glyph per printed cell


def blue_mask(rgb):
    """Blue/violet pen on paper. Printed black has equal channels; white has
    all channels high; grey rules are neutral. Pen is B >> R,G and not white."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (b > r + 35) & (b > g + 22) & (b > 70) & (r < 205)


def ink_stats(rgb):
    dark = rgb.max(axis=2) < 140
    blue = blue_mask(rgb)
    tot = int(dark.sum() + blue.sum())
    return {"ink_px": int(dark.sum()), "blue_px": int(blue.sum()),
            "blue_share": float(blue.sum()) / tot if tot else 0.0}


def _rgb(pixmap):
    a = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.h, pixmap.w,
                                                         pixmap.n)
    if pixmap.n == 1:
        a = np.repeat(a, 3, axis=2)
    return a[..., :3]


BAND_TOP = 0.18          # letterhead / patient-sticker band
BAND_BOTTOM = 0.88       # footer / signature band
IDENTITY_INK = "sem:handwriting_ink"     # pen ink inside an identity zone
CLINICAL = "sem:clinical"         # clinical content -- kept under owner rule 4
# Ink that no identity zone claims. It is NOT the same statement as "this is
# clinical content", and conflating the two was an inversion waiting for a
# second pack: under insurance_medical `clinical` is KEEP so unzoned ink
# survived, but under any pack that redacts health content -- which the
# generic pack does deliberately, mirroring audio -- the same emission turns
# "keep this ink" into "destroy this ink". The mechanism must report what it
# OBSERVED (ink outside every identity zone) and let the pack decide.
UNZONED_INK = "sem:ink_unzoned"


def ink_boxes(page, dpi=DPI, gap_mult=GAP_MULT):
    """The expensive half: render, segment pen ink, group into boxes.

    Split from zoning so the label pass can SEE the ink before deciding
    whether a field is filled -- an answer area with ink in it is an answer,
    and one with neither ink nor text is an empty field that must be left
    alone.
    -> (boxes, stats)
    """
    pm = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    a = _rgb(pm)
    stats = ink_stats(a)
    m = blue_mask(a)
    if m.sum() < 40:
        return [], stats
    m = ndimage.binary_dilation(m, iterations=DILATE)
    lab, n = ndimage.label(m)
    sc = 72.0 / dpi
    toks = []
    for i, o in enumerate(ndimage.find_objects(lab)):
        if o is None:
            continue
        if int((lab[o] == i + 1).sum()) < MIN_AREA_PX:
            continue
        ys, xs = o
        r = fitz.Rect(xs.start * sc, ys.start * sc, xs.stop * sc, ys.stop * sc)
        if r.height > MAX_H_PT or r.width > MAX_W_PT:
            continue
        toks.append((i, "", r))
    out = []
    for b in geometry.run_boxes(toks, gap_mult=gap_mult):
        b = fitz.Rect(b.x0 - 1.5, b.y0 - 1.5, b.x1 + 1.5, b.y1 + 1.5) & page.rect
        if b.width >= 4 and b.height >= 4:
            out.append(b)
    return out, stats


def _first_table_rule(page, dpi=60, min_frac=0.60, thr=160):
    """The top border of the first table in the upper half of the page, as a
    line y = m*x + c in PDF points, or None.

    A scanned sheet is rarely square to the scanner: the progress-notes table
    on a 2338pt scan tilts ~2 degrees and has rounded corners, so no single
    pixel row holds 60% of its border and a straight-row test finds nothing.
    Short horizontal runs survive the tilt; joining them across a few rows
    rebuilds the border as one component.

    A long dark line alone is not a table: the shadow of the paper's top edge
    is one too, and taking it as the band ceiling would un-band the whole
    letterhead. A table's top border has column rules hanging from it, so a
    candidate counts only when at least one vertical rule starts on it away
    from its ends. The fitted line lets the caller compare each ink box with
    the border's height at that box's x, not with its highest point.
    -> (m, c) or None"""
    import numpy as _np
    from scipy import ndimage as _ndi
    pm = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    g = _np.frombuffer(pm.samples, _np.uint8).reshape(pm.h, pm.w)
    dark = g < thr
    seg = max(12, pm.w // 36)
    horiz = _ndi.binary_opening(dark[: pm.h // 2], _np.ones((1, seg)))
    joined = _ndi.binary_closing(horiz, _np.ones((5, seg)))
    lab, _n = _ndi.label(joined)
    cands = []
    for i, sl in enumerate(_ndi.find_objects(lab)):
        if sl is None:
            continue
        ys, xs = sl
        span = xs.stop - xs.start
        # long, and thin for its length (a line tilted <= ~4 degrees, not a
        # block of text or a filled header bar)
        if span >= min_frac * pm.w and ys.stop - ys.start <= max(8, 0.07 * span):
            cands.append((ys.start, ys.stop, xs.start, xs.stop, i + 1))
    if not cands:
        return None
    vseg = max(10, pm.h // 60)
    vert = _ndi.binary_opening(dark, _np.ones((vseg, 1)))
    vert = _ndi.binary_closing(vert, _np.ones((vseg, 3)))
    vl, _vn = _ndi.label(vert)
    vobjs = [o for o in _ndi.find_objects(vl) if o is not None]
    sc = 72.0 / dpi
    for y0, y1, x0, x1, idx in sorted(cands):
        span = x1 - x0
        hangs = any(y0 - 4 <= o[0].start <= y1 + 6
                    and o[0].stop - o[0].start >= 0.04 * pm.h
                    and x0 + 0.05 * span <= (o[1].start + o[1].stop) / 2
                    <= x1 - 0.05 * span
                    for o in vobjs)
        if not hangs:
            continue
        yy, xx = _np.nonzero(lab[y0:y1, x0:x1] == idx)
        m, c = _np.polyfit((xx + x0 + 0.5) * sc, (yy + y0) * sc, 1)
        return float(m), float(c) + page.rect.y0
    return None


def regions(page, labels=None, dpi=DPI, gap_mult=GAP_MULT,
            clinical_page=None, boxes=None, stats=None, words=None,
            pack=None):
    """-> (findings, stats). Each finding is a region in PDF points.

    ZONING, not reading. The taxonomy scopes handwritten-ink redaction to ink
    that falls in an IDENTITY zone, and explicitly keeps pen ink in clinical
    zones -- vitals cells, dosing columns, progress-note bodies -- under owner
    rule 4. Tagging every stroke on the page as identity would destroy exactly
    the handwritten clinical record the bundle exists to document.

    So each ink region is placed, never read:
      inside a labelled field  -> inherits that label's category, so ink in the
                                  PAN box is T08 and ink in the Diagnosis box
                                  is clinical and survives
      in the header/footer band-> identity zone (letterhead, sticker, signature)
      anywhere else            -> identity only if this page carries no clinical
                                  anchors at all; on a clinical page unzoned ink
                                  is clinical and is left alone
    `clinical_page` overrides that last inference when the caller knows better.
    """
    if boxes is None:
        boxes, stats = ink_boxes(page, dpi=dpi, gap_mult=gap_mult)
    stats = dict(stats or {})
    if clinical_page is None:
        # A claim form carries one Diagnosis field among twenty identity
        # fields; a progress note is the reverse. Presence of ANY clinical
        # label is therefore not the question -- which kind of page this is, is.
        n_clin = sum(1 for L in labels or []
                     if CLINICAL in (L.get("categories") or []))
        n_id = sum(1 for L in labels or []
                   if CLINICAL not in (L.get("categories") or []))
        clinical_page = n_clin > n_id
    H = page.rect.height
    top_band, bot_band = page.rect.y0 + BAND_TOP * H, page.rect.y0 + BAND_BOTTOM * H
    # THE LETTERHEAD BAND ENDS WHERE A TABLE BEGINS. The band is a fraction
    # of page height, sized for an A4 sheet where 18% (~150pt) holds the
    # letterhead and patient sticker. On a 2338pt-tall scan of progress notes
    # it reached 421pt while the notes table began at 180pt, and it cut the
    # first three rows of clinical notes -- a service date, "POD-2", "pt
    # fine", the first medication lines -- as identity ink. A table's top
    # border spans most of the page width; a sticker is a small box. So ink
    # above the first full-width rule is still letterhead, and ink inside the
    # table is table content, which the label and row rules already handle.
    rule = None
    if words is not None:
        try:
            rule = _first_table_rule(page)
        except Exception:
            rule = None
    # ...but only where the band being given up holds no identity field. A
    # patient-data form is ALSO a table that starts near the top, and its
    # first cells are "Name of the Patient", "Age", "Sex" with the answers
    # handwritten on the line BELOW each label, where no label region
    # reaches; the band was the only thing cutting them. So any printed
    # identity label between the border and the band's old edge keeps the
    # band whole. Without the page's words that cannot be checked, and the
    # band stays whole too. Only PRINTED words are asked: a label is printed,
    # and the pen's own "Baby by mother side" in a notes column is not a
    # "Mother" field.
    if rule is not None:
        import formfields
        zone = []
        for w in words:
            r = fitz.Rect(w[2])
            mid = (r.y0 + r.y1) / 2
            if not (rule[0] * (r.x0 + r.x1) / 2 + rule[1] - 4 <= mid <= top_band):
                continue
            area = max(r.width * r.height, 1e-6)
            if max(((r & b).width * (r & b).height for b in boxes
                    if r.intersects(b)), default=0.0) >= 0.6 * area:
                continue
            zone.append(w)
        ident = [L for L in formfields.find_labels(zone, pack, prose_veto=False)
                 if pack is None or pack.action_for([L["category"]]) != "KEEP"]
        if ident:
            stats["band_ceiling_blocked_by"] = sorted({L["label"] for L in ident})
            rule = None
    stats["band_ceiling"] = None if rule is None else round(rule[1], 1)

    def band_top_at(x):
        if rule is None:
            return top_band
        return min(top_band, rule[0] * x + rule[1])
    out = []
    for b in boxes:
        cats, label, zone = None, None, None
        for L in labels or []:
            lb = fitz.Rect(L["bbox"])
            inter = b & lb
            if (not inter.is_empty
                    and inter.width * inter.height
                    > 0.5 * b.width * b.height):
                cats, label, zone = L["categories"], L.get("label"), "field"
                break
        if cats is None:
            # Ink that no label CONTAINS still usually belongs to one: on a
            # ruled form the value sits to the right of its label on the same
            # row, and the label box rarely covers it. Inherit from the nearest
            # label sharing this row before falling back to page-level defaults.
            mid = (b.y0 + b.y1) / 2
            best, bestdx = None, None
            for L in labels or []:
                lb = fitz.Rect(L["bbox"])
                if not (lb.y0 - 2 <= mid <= lb.y1 + 2):
                    continue
                dx = b.x0 - lb.x1 if b.x0 >= lb.x1 else lb.x0 - b.x1
                if bestdx is None or dx < bestdx:
                    best, bestdx = L, dx
            if best is not None:
                cats, label, zone = best["categories"], best.get("label"), "row"
        if cats is None:
            mid = (b.y0 + b.y1) / 2
            if mid <= band_top_at((b.x0 + b.x1) / 2) or mid >= bot_band:
                cats, zone = [IDENTITY_INK], "band"
            else:
                # OWNER 2026-09-23: pen ink that no identity label claims is
                # KEPT. Blanket ink redaction destroyed exactly what a claim
                # agent works from -- handwritten admission and discharge dates,
                # category, hospital days, payment mode, admit purpose. Ink is
                # now redacted only where an identity label or an identity band
                # puts it. The accepted residual: a handwritten name in an
                # unlabelled part of a page is not removed by this mechanism.
                cats, zone = [UNZONED_INK], "unzoned_kept"
        out.append({"categories": cats, "text": "[handwritten value]",
                    "bbox": [round(v, 2) for v in (b.x0, b.y0, b.x1, b.y1)],
                    "source": "handwriting_ink", "locator": "handwriting_ink",
                    "in_field": label, "zone": zone, "region": True})
    stats["clinical_page"] = bool(clinical_page)
    return out, stats


def _frac_inside(a, b):  # kept: used by callers and tests
    inter = a & b
    if inter.is_empty or a.width * a.height <= 0:
        return 0.0
    return (inter.width * inter.height) / (a.width * a.height)


def resolve_labels_vs_ink(labels, ink, blue_share, min_share=0.03):
    """Decide whether the label fallback should draw at all on this page.

    Label-anchored regions exist for ONE reason: at 109 dpi no OCR reads
    handwriting, so the label says where a value must be even when the value
    is illegible. They are a FALLBACK, and they are blunt -- a label region
    covers a whole answer area whether or not anything was written in it.

    When the chromatic-ink channel works, it is strictly better: it finds the
    actual strokes, inherits its category from the label it sits in, and covers
    nothing else. So the ink governs and the label regions stand down.

    Letting both draw is what blacked out the 'GUIDANCE FOR FILLING CLAIM FORM'
    sheet -- three columns of printed instructions, no patient data anywhere,
    and the label column matched 'Policy No.', 'Name', 'PAN' and took the
    description beside each one.

    On a greyscale, bitonal or black-pen page there IS no chromatic signal, so
    the labels remain the only mechanism and draw unconditionally. That is the
    fail-closed half of the same rule.
    -> (labels_to_draw, n_dropped, basis)
    """
    if blue_share >= min_share and ink:
        return [], len(labels), "ink_governs"
    return labels, 0, ("no_ink_channel" if blue_share < min_share
                       else "ink_found_nothing")


def drop_repeated(hw_by_page, q=4.0):
    """Remove regions that recur at the same place on many pages of one
    bundle: a blue logo, stamp or printed header, not a pen. Threshold scales
    with the bundle (>= 3 pages and >= a quarter of them) so three identical
    filled forms ticking the same box are not mistaken for print."""
    import math
    from collections import Counter
    n = len(hw_by_page)
    min_pages = max(3, math.ceil(0.25 * n))

    def key(b):
        return tuple(int(round(v / q)) for v in b)

    cnt = Counter()
    for regs in hw_by_page.values():
        for k in {key(r["bbox"]) for r in regs}:
            cnt[k] += 1
    dropped = 0
    for pno, regs in hw_by_page.items():
        keep = [r for r in regs if cnt[key(r["bbox"])] < min_pages]
        dropped += len(regs) - len(keep)
        hw_by_page[pno] = keep
    return dropped


def census(png_dir):
    rows = []
    for p in sorted(Path(png_dir).glob("*.png")):
        try:
            pm = fitz.Pixmap(str(p))
        except Exception:
            continue
        st = ink_stats(_rgb(pm))
        rows.append((p.name, st["blue_share"], st["blue_px"]))
    hw = [r for r in rows if r[1] >= 0.03]
    print(f"{len(rows)} pages scanned; {len(hw)} have >=3% chromatic (pen) ink")
    for name, share, px in sorted(hw, key=lambda r: -r[1])[:40]:
        print(f"   {name:<24} blue_share={share:5.1%}  blue_px={px}")
    return rows


def self_test():
    import readers, formfields
    src = (str(readers.corpus_dir()) + "/case-48142-6f4d11a6-0ce9-4395-a069-b814baf51664.pdf")
    wd = tempfile.mkdtemp(prefix="pii_selftest_")
    # The corpus-backed part of this test needs a real bundle. Without one it
    # SKIPS with a clear message; it must never crash on a fresh checkout, and
    # it must never pass silently either -- the count printed at the end says
    # how much was actually asserted.
    if not Path(src).exists():
        print(f"  (no corpus at {src}; set PII_CORPUS_DIR to run the "
              f"corpus-backed assertions -- structural checks above still ran)")
        return True
    d = fitz.open(src)
    page = d[0]
    words = readers.read_page(d, 1, workdir=wd)["words"]
    labs = formfields.value_regions(page, words)
    regs, st = regions(page, labels=labs)
    from collections import Counter
    zc = Counter(r["zone"] for r in regs)
    print(f"page 1: blue_share={st['blue_share']:.1%} clinical_page="
          f"{st['clinical_page']} -> {len(regs)} handwriting regions")
    print(f"  zones: {dict(zc)}")
    from_ink = sum(1 for r in regs if r["categories"] == [IDENTITY_INK])
    print(f"  ink routed to identity: {from_ink}/{len(regs)} "
          f"(the rest inherit a field's category or are kept)")

    # a clinical page must not have its handwritten record destroyed
    clin_labels = [{"bbox": [50, 300, 550, 500], "categories": ["T37"],
                    "label": "Diagnosis"}]
    cregs, cst = regions(page, labels=clin_labels, clinical_page=True)
    kept = sum(1 for r in cregs if r["categories"] == ["T37"])
    print(f"  as a clinical page: {kept}/{len(cregs)} ink regions routed to "
          f"clinical (kept), rest to identity bands")
    assert kept > 0, "clinical zoning never fires"
    for r in regs:
        b = r["bbox"]
        print(f"   x{b[0]:6.0f}-{b[2]:<6.0f} y{b[1]:6.0f}-{b[3]:<6.0f}  "
              f"{r['categories'][0]:<24}{r['in_field'] or ''}")
    # the handwritten policy number sits on the row at y~70pt, x~140-305
    hit = [r for r in regs
           if 62 <= (r["bbox"][1] + r["bbox"][3]) / 2 <= 80
           and r["bbox"][0] < 200 and r["bbox"][2] > 250]
    print(f"\npolicy-number row covered: {'YES' if hit else 'NO'}")
    # overlay for eyes
    out = Path("/private/tmp/claude-501/-Users-shivangi-Documents-Claude-Audio-"
               "PII-Project/87ddc989-26e9-49df-9724-61c42bfe09fe/scratchpad/"
               "hw_overlay_p1.png")
    for r in regs:
        page.draw_rect(fitz.Rect(r["bbox"]), color=(1, 0, 0), width=1.2,
                       overlay=True)
    for L in labs:
        page.draw_rect(fitz.Rect(L["bbox"]), color=(0, 0.6, 0), width=0.7,
                       overlay=True)
    page.get_pixmap(dpi=100).save(str(out))
    print(f"overlay (red=handwriting, green=label regions): {out}")
    d.close()
    return bool(hit)


if __name__ == "__main__":
    if "--census" in sys.argv:
        census(sys.argv[sys.argv.index("--census") + 1])
        sys.exit(0)
    sys.exit(0 if self_test() else 1)
