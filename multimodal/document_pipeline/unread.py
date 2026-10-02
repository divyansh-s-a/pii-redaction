#!/usr/bin/env python3
"""Text that NO reader in this pipeline has seen, classified before it is cut.

Apple Vision supports no Indic script at all -- not Kannada, Hindi, Tamil,
Telugu, Marathi or Bengali. Indian identity documents are bilingual by law, so
on a SCANNED Aadhaar page the English name and address are redacted correctly
and the identical text in the state language beside them is never even
tokenised. It cannot be reached by a prompt, a category or a model, because
those glyphs never enter the word stream.

(The BORN-DIGITAL half of the same problem is handled deterministically and
for free by objective.scan_script: there the text IS in the PDF's own layer
with exact boxes.  This module is only for pages where that layer is absent.)

The shape of the fix keeps the architectural rule intact:

  GEOMETRY comes from Vision's VNDetectTextRectangles, which reports WHERE
  text is without reading it, so it sees these scripts perfectly. Subtracting
  the regions the recogniser DID account for leaves exactly the unread text.

  CLASSIFICATION comes from the model, which chooses among regions that
  already have boxes. It never emits a coordinate.

Blanking every unread region instead was measured and rejected: on this
corpus it blackens a mean 3.1% of page area and up to 16.3% of a single page,
and it destroys handwritten clinical values on pages with no Indic at all.
Reading a region before cutting it is what makes the difference between a
redaction and a hole.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
# LAST insert wins position 0. This directory must beat the audio arm's
# pipeline/, which has its own redact.py: in the wrong order a later plain
# `import redact` resolved to the AUDIO redactor and crashed the suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitz
import common
import objective

REGION_DPI = 300        # the crops are small; read them on their own terms
MIN_W_PT = 12.0         # narrower than this is a tick or speckle
MIN_H_PT = 3.0          # a card's small print is 4-6pt tall. At 6.0 this
                        # gate threw away a TPA's company name on
                        # an insurance card: Vision DETECTED the 101x4pt line,
                        # the recogniser produced nothing for it, and the one
                        # pass built for exactly that case discarded it as
                        # speckle. A printed rule is under ~1.5pt; the region
                        # detector is text-specific, so rules rarely reach
                        # here at all.
COVER_FRAC = 0.30       # a region this covered by known words is not unread
BATCH = 7               # crops per request: small enough to be counted right
MAX_REGIONS = 84        # per page, i.e. at most 12 requests. It was 14 in ONE
                        # request, and a dense scanned claim form carries 60:
                        # the other 46 were reported but never classified and
                        # never cut. Beyond 84 the page is noise, and the
                        # remainder is still reported rather than hidden.

# Second geometry source: text lines found in the PIXELS. Vision's region
# detector does not see every script -- on a scanned Aadhaar card it returned
# 11 regions and none on the Telugu name line, whose ink is as dark as the
# English name beside it -- so a line nothing detected was never classified
# and never cut.
INK_DPI = 150
INK_CONTRAST = 60       # darker than the local paper by this much is ink
INK_BG_PT = 24.0        # local paper estimated over this window
INK_SLACK = 25.0        # a line's darkest stroke vs the page's recognised type
INK_MAX_SAT = 40.0      # neutral ink only; coloured pen is handwriting's
INK_RULE_PT = 40.0      # printed rules longer than this are removed first
INK_LINE_H = (6.0, 40.0)
INK_GAP_PT = 14.0       # glyphs and words closer than this join one line

_SCHEMA = {
    "type": "object",
    "properties": {"regions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "text": {"type": "string"},
            "is_personal": {"type": "boolean"},
            "kind": {"type": "string"},
        },
        "required": ["index", "text", "is_personal", "kind"]}}},
    "required": ["regions"]}

_PROMPT = """Each image is one line cropped from a scanned Indian insurance or
identity document. They are in reading order and numbered from 0.

For EACH image, transcribe the text and decide whether it is PERSONAL DATA
about an individual.

is_personal = true for: a person's name, a parent's or spouse's name, a
postal address, a phone number, an identity number (Aadhaar, VID, enrolment,
policy, member), a date of birth, an AGE, a SEX or GENDER value (in any
script, e.g. MALE / FEMALE), a signature, AND the name of the person's
employer, insurer, third-party administrator, hospital or laboratory --
including its WEBSITE or email domain and a line listing its services or
branches under its name. Those identify whose record this is, and they go.

is_personal = false for: printed FIELD LABELS such as "Address:", "Date of
Birth", "Name" or "Enrolment No." in any script; government or issuer
boilerplate such as "Government of India" or "Unique Identification
Authority of India"; mottos; column headings; clinical test names, values,
units and reference ranges; money amounts; and anything unreadable.

A label is not personal data even when it sits directly beside the value it
names. Judge every image ONLY by the text visible inside it: never carry a
word over from another image, and a fragment you cannot read is not
personal. Transcribe the script as printed; give the kind in English."""


def unread_regions(page, words, dpi=REGION_DPI, workdir=None, ocr=True):
    """-> [fitz.Rect] of detected text that no reader accounted for.

    `ocr` says the words came from OCR of a scan. Only then are the pixel
    lines added: on a born-digital page every glyph is in the text layer and
    objective.scan_script already finds the ones in other scripts."""
    wd = Path(workdir or "/tmp")
    wd.mkdir(parents=True, exist_ok=True)
    img = wd / f"_regions_p{page.number}.png"
    try:
        page.get_pixmap(dpi=dpi).save(str(img))
        got = objective.vision(img, "textregions")
    except Exception:
        return []
    finally:
        img.unlink(missing_ok=True)
    if got.get("error"):
        return []
    W, H = page.rect.width, page.rect.height
    known = [r for _, _, r in words]
    out = []
    for reg in got.get("textregions", []):
        r = fitz.Rect(reg["x0"] * W, reg["y0"] * H,
                      reg["x1"] * W, reg["y1"] * H)
        if r.width < MIN_W_PT or r.height < MIN_H_PT:
            continue
        # How much of this region did the recogniser already explain? A
        # region the Latin reader covered is text we have already handled.
        area = max(r.width * r.height, 1.0)
        seen = 0.0
        for k in known:
            q = r & k
            if not q.is_empty:
                seen += max(0.0, q.x1 - q.x0) * max(0.0, q.y1 - q.y0)
        if seen / area > COVER_FRAC:
            continue
        out.append(r)
    if ocr:
        try:
            lines = ink_lines(page, known)
        except Exception:
            lines = []
        for r in lines:
            # already a Vision region that survived: nothing new
            if any(_area(r & b) > 0.5 * _area(r) for b in out):
                continue
            out.append(r)
    return out


def _area(r):
    return 0.0 if r.is_empty else r.width * r.height


def _same_line_cover(r, known):
    """Share of r's width explained by words whose vertical CENTRE is inside
    r. Vision's token boxes run ~1.8x the glyph height, so by area the DOB
    line's tokens "explained" the sex line below it and the only box that
    reached it was dropped; a word only explains its own line."""
    xs = []
    for k in known:
        cy = (k.y0 + k.y1) / 2
        if r.y0 <= cy <= r.y1 and k.x1 > r.x0 and k.x0 < r.x1:
            xs.append((max(k.x0, r.x0), min(k.x1, r.x1)))
    xs.sort()
    tot, cur = 0.0, None
    for a, b in xs:
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        tot += cur[1] - cur[0]
    return tot / max(r.width, 1.0)


def ink_lines(page, known):
    """Printed text lines found from the pixels that no word explains
    -> [fitz.Rect]. Geometry only; the model still decides what they are."""
    import numpy as np
    from scipy import ndimage as ndi
    sc = 72.0 / INK_DPI
    pm = page.get_pixmap(dpi=INK_DPI, colorspace=fitz.csGRAY)
    g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
    pc = page.get_pixmap(dpi=INK_DPI, colorspace=fitz.csRGB, alpha=False)
    rgb = np.frombuffer(pc.samples, np.uint8).reshape(pc.h, pc.w, 3).astype(int)
    sat = rgb.max(2) - rgb.min(2)
    # ink relative to its own local paper, not a fixed level: a phone photo
    # of a card is lit unevenly and its type is grey, not black
    k = max(3, int(INK_BG_PT / sc))
    bg = ndi.grey_closing(g, size=(k, k))
    dark = (bg.astype(int) - g.astype(int)) > INK_CONTRAST
    L = max(3, int(INK_RULE_PT / sc))
    rules = (ndi.binary_opening(dark, np.ones((1, L)))
             | ndi.binary_opening(dark, np.ones((L, 1))))
    dark &= ~ndi.binary_dilation(rules, np.ones((3, 3)))
    # A candidate must be as dark as this page's RECOGNISED type: paint
    # streaks, logo edges and blur contrast with white paper too, but never
    # with a stroke as dark as real print.
    lums = []
    for r in known:
        sub = g[max(0, int(r.y0 / sc)):int(r.y1 / sc) + 1,
                max(0, int(r.x0 / sc)):int(r.x1 / sc) + 1]
        if sub.size:
            lums.append(float(sub.min()))
    ref = float(np.median(lums)) + INK_SLACK if lums else 255.0
    # ...and sized like it. Fragments under half the read type's height are
    # what the classifier hallucinated names on (one 'T' of a nurse's hand).
    if known:
        mh = float(np.median([r.height for r in known]))
        lo, hi = max(INK_LINE_H[0], 0.5 * mh), 1.5 * mh
    else:
        lo, hi = INK_LINE_H
    gx = max(1, int(INK_GAP_PT / sc))
    lab, _n = ndi.label(ndi.binary_dilation(dark, np.ones((3, gx))))
    out = []
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        sub = dark[sl]
        ys, xs = np.nonzero(sub)
        if not len(ys):
            continue
        r = fitz.Rect((sl[1].start + xs.min()) * sc, (sl[0].start + ys.min()) * sc,
                      (sl[1].start + xs.max() + 1) * sc,
                      (sl[0].start + ys.max() + 1) * sc)
        if not (lo <= r.height <= hi):
            continue
        if r.width < MIN_W_PT or r.width < 2 * r.height:
            continue
        if float(g[sl][sub].min()) > ref:
            continue
        # Neutral ink only: coloured pen is handwriting, which handwriting.py
        # and inkmarks.py own, and on a clinical page it is the doctor's
        # notes the insurance pack must keep.
        if float(sat[sl][sub].mean()) > INK_MAX_SAT:
            continue
        if _same_line_cover(r, known) > COVER_FRAC:
            continue
        # A dashed or dotted cut-line survives the rule opening. Its ink sits
        # in one thin band of rows; type spreads over x-height and ascenders.
        rows = sub.sum(1)
        kb = max(3, int(0.2 * len(rows)))
        band = max(rows[i:i + kb].sum()
                   for i in range(max(1, len(rows) - kb + 1)))
        if band > 0.6 * max(rows.sum(), 1):
            continue
        out.append(r)
    return out


def _ask_batch(page, batch, dpi, wd, engine, depth=0):
    """batch = [(region_index, rect)] -> verdicts keyed by region_index.

    A reply that does not account for every image in its batch is not trusted
    to be aligned with them. Rather than cut the whole batch, SPLIT IT AND ASK
    AGAIN: the model counts small sets reliably, and the first version of this
    -- one request for up to 14 crops, failing closed on any miscount -- cut
    every region on the densest pages, labels included, on 3 of 4 runs.
    Only a region that still cannot be classified on its own fails closed.
    """
    paths = []
    for n, (_i, r) in enumerate(batch):
        pad = fitz.Rect(r.x0 - 2, r.y0 - 2, r.x1 + 2, r.y1 + 2) & page.rect
        f = wd / f"_reg_p{page.number}_{depth}_{n}_{int(r.x0)}_{int(r.y0)}.png"
        try:
            page.get_pixmap(dpi=dpi, clip=pad).save(str(f))
            paths.append(f)
        except Exception:
            paths.append(None)
    live = [(batch[n][0], f) for n, f in enumerate(paths) if f is not None]
    try:
        if not live:
            return []
        # Label every image: sent as a bare list the model loses count and
        # shifts every verdict after the first one it skips.
        parts = [{"text": _PROMPT}]
        for n, (_i, f) in enumerate(live):
            parts.append({"text": f"Image {n}:"})
            parts.append({"image_path": str(f), "mime": "image/png"})
        raw = common.generate(engine or common.resolve_model(), parts,
                              _SCHEMA, thinking_budget=1024)
        got = json.loads(raw).get("regions", [])
        if sorted({int(g.get("index", -1)) for g in got}) != list(range(len(live))):
            raise ValueError(f"answered for {len({g.get('index') for g in got})}"
                             f" of {len(live)}")
        for g in got:
            g["index"] = live[int(g["index"])][0]
        return got
    except Exception as exc:
        if len(live) > 1 and depth < 4:
            mid = len(batch) // 2
            return (_ask_batch(page, batch[:mid], dpi, wd, engine, depth + 1)
                    + _ask_batch(page, batch[mid:], dpi, wd, engine, depth + 1))
        # FAIL CLOSED, and say so in capitals so a run log cannot hide it:
        # unread text on an identity page is not something to wave through.
        print(f"    UNREAD CLASSIFIER FAILED on {len(live)} region(s) "
              f"({type(exc).__name__}: {str(exc)[:50]}); treating as personal",
              flush=True)
        return [{"index": i, "text": "", "is_personal": True,
                 "kind": "unclassified"} for i, _f in live]
    finally:
        for f in paths:
            if f is not None:
                f.unlink(missing_ok=True)


def classify(page, rects, dpi=REGION_DPI, workdir=None, engine=None):
    """Crop each region, ask in small batches, -> [(rect, text, kind)] personal.

    Regions the model calls a label, boilerplate or clinical content are
    returned as NOT personal and left alone, which is the entire reason this
    is not a blanking pass.
    """
    if not rects:
        return []
    wd = Path(workdir or "/tmp")
    wd.mkdir(parents=True, exist_ok=True)
    todo = list(enumerate(rects[:MAX_REGIONS]))
    got = []
    for s0 in range(0, len(todo), BATCH):
        got += _ask_batch(page, todo[s0:s0 + BATCH], dpi, wd, engine)
    out = []
    for g in got:
        try:
            i = int(g["index"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < len(rects) and g.get("is_personal"):
            out.append((rects[i], str(g.get("text", ""))[:80],
                        str(g.get("kind", ""))[:40]))
    return out


def _field_decision(r, zones, pack):
    """'keep' if r sits inside a KEEP field and no cut field, else None.

    A labelled field already carries the pack's decision for what is in it.
    The classifier sees one crop and no label: on a claim form it called the
    handwritten bank name and branch personal data and one IFSC cell an
    account-number digit, and both were cut although the label pass had
    zoned them as bank fields to KEEP. The cut side wins any overlap, so this
    can only stop a cut inside a field the pack keeps, never cause a leak.
    """
    if not zones or pack is None:
        return None                   # no zones -> no deferral: fail closed
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    inside_keep = inside_cut = False
    for z in zones:
        b = z.get("bbox")
        if not b:
            continue
        if not (b[0] <= cx <= b[2] and b[1] - 2 <= cy <= b[3] + 2):
            continue
        act = pack.action_for(z.get("categories") or [])
        if act == "KEEP":
            if not z.get("multiline"):
                inside_keep = True
        else:
            inside_cut = True
    return "keep" if inside_keep and not inside_cut else None


def findings(page, words, pack=None, dpi=REGION_DPI, workdir=None, zones=None,
             ocr=True):
    """The full pass -> findings ready for the redactor."""
    rects = unread_regions(page, words, dpi=dpi, workdir=workdir, ocr=ocr)
    if not rects:
        return []
    if len(rects) > MAX_REGIONS:
        print(f"    {len(rects)} unread regions on this page; classifying the "
              f"first {MAX_REGIONS}. The rest are reported, not redacted.",
              flush=True)
    out = []
    for r, text, kind in classify(page, rects, dpi=dpi, workdir=workdir):
        if _field_decision(r, zones, pack) == "keep":
            # Deferred to the field, and RECORDED, so the choice is visible.
            out.append({
                "categories": ["sem:non_latin_script"],
                "text": text or "[unread text]",
                "bbox": [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)],
                "n_boxes": 0, "source": "unread_region",
                "locator": "vision_region", "action": "KEEP",
                "reason": f"model says {kind}, but it sits in a labelled field "
                          f"the pack keeps; the field's decision stands",
            })
            continue
        out.append({
            "categories": ["sem:non_latin_script"],
            "text": text or "[unread text]",
            "bbox": [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)],
            "all_boxes": [[round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]],
            "n_boxes": 1,
            "source": "unread_region", "locator": "vision_region",
            "action": (pack.action_for(["sem:non_latin_script"])
                       if pack is not None else "REDACT"),
            "reason": f"text no reader transcribed; model says {kind}",
        })
    return out
