#!/usr/bin/env python3
"""Ink on a SCANNED page that no reader explained, classified before it is cut.

A black-ink signature on a scanned document is invisible to every other pass:
it is not text, so no detector sees it; it has no pigment, so the stamp pass
cannot see it; it is not its own image, so the embedded-mark pass cannot see
it; and Vision's text-region detector does not report it. On the review
bundle case-47999 that left the card holder's signature readable on all three
insurance cards of a scanned page, in both packs.

The geometry signal that remains is RESIDUAL INK: dark pixels left after
removing everything a reader accounted for -- OCR words, detected text
regions and long printed rules. It finds those signatures (3 of 3). It also
fires on 20 of 30 pages of that bundle, including clinical pages, so it can
NEVER redact on geometry alone: that would destroy a doctor's handwritten
readings, which the insurance pack is required to keep.

So the same shape as unread.py: pixels give the box, the model gives the
verdict, and it never emits a coordinate. It is asked what a mark IS, not
whether it is sensitive. Measured on four pages before this was written:

  ID-card page     11 marks -> 3 signature, 3 logo, 5 printed fragment
  A-scan (clinical) 8 marks -> 8 printed fragment              -> 0 cut
  scanned page     14 marks -> 12 printed fragment, 2 stray    -> 0 cut
  claim form       14 marks -> 3 handwritten personal data, 10 printed, 1 stray

Only signatures, logos/seals and handwritten personal data are cut.

FAILURE IS OPEN, AND LOUD. unread.py fails closed because unread text is
rarely clinical. Residual ink routinely IS clinical handwriting, so treating
every mark as personal when the classifier is down would black out a page of
doctor's notes on one timed-out request. The page is marked instead, so the
gap is visible in the run summary rather than silent.
"""
import json
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

import common
import objective

DPI = 150
DARK = 130
RULE_PT = 40.0          # a stroke this long and straight is a printed rule
BLOB_W = (12.0, 90.0)   # a signature, a seal, a handwritten entry
BLOB_H = (8.0, 50.0)
BLOB_DENSITY = (0.03, 0.40)
SIG_MIN_W, SIG_MIN_H = 18.0, 10.0   # a signature, not one boxed character
SIG_MAX_PER_STRIP = 3               # more in one unread strip: handwriting
CROP_DPI = 300
BATCH = 7               # marks per request: small enough to be counted right
MAX_BATCHES = 14        # per page (98 marks). It was 6 (42), and a nursing
                        # chart or handwritten claim form carries 50-95 marks:
                        # staff signatures past mark 42 were never classified
                        # and never cut. Beyond 98 the rest are still REPORTED
                        # in a page marker, never silently dropped.

# Charting grids: a medication-administration record or nursing chart is a
# lattice of small ruled cells, and the pen entries in those cells are either
# clinical (a time, a dose, a tick) or the sign-off of whoever gave the dose --
# a name, initials, a staff number, sometimes the patient's own signature.
GRID_MIN_V = 5          # vertical rules crossing the lattice
GRID_MIN_H = 6          # horizontal rules crossing it
GRID_MAX_PITCH_W = 0.12 # median column pitch, as a share of page width
GRID_MAX_PITCH_H = 0.06 # median row pitch, as a share of page height
GRID_MIN_PITCH_A4 = 16.0  # median column pitch floor, in A4-width points
ENTRY_W = (6.0, 90.0)   # one cell's entry; cells are small, so smaller
ENTRY_H = (5.0, 50.0)   # than a free-standing residual mark
GRID_MAX_BATCHES = 20   # per page, on top of the residual budget
GROUP_MAX_DENSITY = 0.60  # a dense signature scrawl runs to ~45% ink
EXTRA_MAX_BATCHES = 2   # unread-strip marks: their own budget (14)
RULE_SEG_PT = 12.0      # grid rules: shortest straight run of a leaning rule
GRID_CTX_PT = 40.0      # grid entries: context margin cap, about one cell
GRID_TALL_ROWS = 1.6    # a date or time is ~1 row; name-over-number is 2

CUT = {"signature": "sem:signature",
       "logo_or_seal": "sem:visual_mark",
       "handwritten_personal_data": "sem:ink_identity"}

_SCHEMA = {"type": "object", "properties": {"marks": {"type": "array", "items": {
    "type": "object", "properties": {
        "index": {"type": "integer"},
        "kind": {"type": "string", "enum": [
            "signature", "logo_or_seal", "handwritten_personal_data",
            "handwritten_clinical_value", "printed_text_fragment",
            "stray_mark"]},
        "why": {"type": "string"}},
    "required": ["index", "kind", "why"]}}},
    "required": ["marks"]}

_PROMPT = """Each image is a piece of a scanned Indian insurance claim document,
numbered from 0. In each, ONE mark is outlined in RED. Classify that outlined
mark by what it IS, using the surrounding page for context.

signature: a person's handwritten sign-off scrawl. In a medication,
  nursing or observation chart this includes whatever a nurse or doctor
  writes in a cell to record WHO gave or checked a dose -- a name, initials
  or a staff/employee number, alone or together -- and a patient's or
  relative's signature written in such a cell.
logo_or_seal: a printed emblem, crest or logo, OR a rubber-stamp impression of
  any shape -- a round hospital seal, a rectangular doctor's stamp with a name
  or registration number -- including a faint, smudged or MIRROR-IMAGE
  (reversed) impression transferred from a facing page.
handwritten_personal_data: a handwritten name, phone number, email, address
  or ID number written in a FIELD for one -- a form box, a "Name" or "ID No."
  column, beside a "Phone" or "Address" label. A DATE counts here only when
  it is clearly labelled or placed as a DATE OF BIRTH. Handwriting that is
  part of running sentences of a clinical or nursing note is
  handwritten_clinical_value, even if a single word looks like a name.
handwritten_clinical_value: a doctor's or technician's handwritten medical
  value, reading, dose, tick or note (6/6, +2.50, 120/80, 1 tab BD), AND the
  date or time of a visit, note entry, admission, procedure or discharge --
  a date in the date column of a progress note is a service date, not
  personal data. Also: marks PLOTTED on a chart's graph grid (dots, crosses,
  the joined line of a temperature or pulse curve); circled or ticked
  letters and numbers recording intake, output, urine, stools or a yes/no;
  and any handwritten money amount, bill number, quantity or count.
printed_text_fragment: part of printed text, a printed character in a box,
  or an empty printed box.
stray_mark: a smudge, speck, fold line or accidental stroke.

Each label gives the region's size, where it sits and the page's normal text
height. A single letter or symbol far LARGER than the body text -- especially
near a corner or header -- is a logo_or_seal (an emblem), not a text fragment.
If a region holds a stamp or signature together with other writing, classify
it by the stamp or signature. Classify what the region IS, not whether it is
sensitive."""


def residual_marks(page, words, covered=(), workdir=None, split=False):
    """-> [fitz.Rect] of dark marks no reader and no other pass accounted for.

    First every mark outside the words and outside every region Vision's
    detector called text. Then, APPENDED so the page's classification budget
    reaches them only after those, signature-sized marks inside a text region
    the reader could NOT read: that region belongs to unread.py, which asks
    whether its TEXT is personal, and a doctor's signature beside a dose
    frequency ("1-1-1  <sig>") fell between the two -- detected as text here,
    read as a harmless frequency there, never judged as a mark. Only
    signature-sized marks: a boxed claim form's unread character cells would
    otherwise add some 500 candidates and crowd the cap.

    split=True returns (primary, extra) so the caller can give the extra
    marks their own small budget."""
    sc = 72.0 / DPI
    pm = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
    g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w).astype(int)
    dark = g < DARK
    expl = np.zeros_like(dark)          # words, read regions, other passes
    unread_zone = np.zeros_like(dark)   # regions detected but not read

    def blank(r, pad=2.0, into=None):
        m = expl if into is None else into
        x0 = max(0, int((r.x0 - pad) / sc)); x1 = min(pm.w, int((r.x1 + pad) / sc) + 1)
        y0 = max(0, int((r.y0 - pad) / sc)); y1 = min(pm.h, int((r.y1 + pad) / sc) + 1)
        m[y0:y1, x0:x1] = True

    for _i, _t, r in words:
        blank(r)
    wd = Path(workdir or "/tmp")
    wd.mkdir(parents=True, exist_ok=True)
    img = wd / f"_inkregions_p{page.number}.png"
    try:
        page.get_pixmap(dpi=200).save(str(img))
        got = objective.vision(img, "textregions")
    finally:
        img.unlink(missing_ok=True)
    W, H = page.rect.width, page.rect.height
    known = [r for _i, _t, r in words]
    unread_regs = []
    for reg in got.get("textregions", []):
        tr = fitz.Rect(reg["x0"] * W, reg["y0"] * H, reg["x1"] * W, reg["y1"] * H)
        area = max(tr.width * tr.height, 1.0)
        seen = sum(max(0.0, (tr & k).width) * max(0.0, (tr & k).height)
                   for k in known if tr.intersects(k))
        if seen / area > 0.30:
            blank(tr)
        else:
            blank(tr, into=unread_zone)
            unread_regs.append(tr)
    for r in covered:                         # already cut by another pass
        blank(fitz.Rect(r), pad=0.0)
    L = max(3, int(RULE_PT / sc))

    def blobs(left, min_w, min_h):
        rules = (ndi.binary_opening(left, np.ones((1, L)))
                 | ndi.binary_opening(left, np.ones((L, 1))))
        left = left & ~ndi.binary_dilation(rules, np.ones((3, 3)))
        lab, _n = ndi.label(ndi.binary_dilation(left, np.ones((5, 5))))
        found = []
        for sl in ndi.find_objects(lab):
            if sl is None:
                continue
            w = (sl[1].stop - sl[1].start) * sc
            h = (sl[0].stop - sl[0].start) * sc
            if not (min_w <= w <= BLOB_W[1] and min_h <= h <= BLOB_H[1]):
                continue
            if not (BLOB_DENSITY[0] <= float(left[sl].mean()) <= BLOB_DENSITY[1]):
                continue
            found.append(fitz.Rect(sl[1].start * sc, sl[0].start * sc,
                                   sl[1].stop * sc, sl[0].stop * sc))
        return found

    out = blobs(dark & ~expl & ~unread_zone, BLOB_W[0], BLOB_H[0])
    more = []
    if unread_zone.any():
        extra = [r for r in blobs(dark & ~expl, SIG_MIN_W, SIG_MIN_H)
                 if not any((r & o).width * (r & o).height
                            > 0.5 * r.width * r.height for o in out)]
        # a signature stands alone in its strip; a strip of several such
        # marks is a line of handwriting, which the text and label passes own
        def strip(r):
            c = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            return next((i for i, u in enumerate(unread_regs) if u.contains(c)), None)
        per = {}
        for r in extra:
            per.setdefault(strip(r), []).append(r)
        for k, rs in per.items():
            if k is None or len(rs) <= SIG_MAX_PER_STRIP:
                more.extend(rs)
    more.sort(key=lambda r: -r.width * r.height)     # likeliest signatures first
    return (out, more) if split else out + more


def charting_grids(page):
    """Regions of the page ruled as a lattice of small cells -> [fitz.Rect]."""
    import formfields
    V = formfields._vertical_rules(page)
    Hs = formfields._horizontal_rules(page)
    if len(V) < GRID_MIN_V or len(Hs) < GRID_MIN_H:
        return []
    def crosses(v, h):
        return h[1] - 3 <= v[0] <= h[2] + 3 and v[1] - 3 <= h[0] <= v[2] + 3
    gv = [v for v in V if sum(crosses(v, h) for h in Hs) >= GRID_MIN_H]
    gh = [h for h in Hs if sum(crosses(v, h) for v in gv) >= GRID_MIN_V]
    if len(gv) < GRID_MIN_V or len(gh) < GRID_MIN_H:
        return []
    # The lattice is the longest run of EVENLY spaced columns. A chart's
    # printed side panel (drug, dose, route, the prescriber's box) is ruled
    # too, but in a few wide columns; its entries belong to the label and
    # residual passes, not to this one.
    xs = sorted({round(v[0]) for v in gv})
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 2]
    if not gaps:
        return []
    med = sorted(gaps)[len(gaps) // 2]
    if med > GRID_MAX_PITCH_W * page.rect.width:
        return []                   # a form's few big boxes, not a chart
    if med < GRID_MIN_PITCH_A4 * page.rect.width / 595.0:
        # graph paper and character boxes: a temperature plot's hour columns
        # (~10pt), a claim form's digit boxes (13-15pt). No one signs in a
        # cell that narrow, and read there the classifier cut plotted pulse
        # dots, circled intake letters and a claimed amount.
        return []
    best, run = (0, 0), [0, 0]
    for i in range(1, len(xs)):
        # 2x: a chart's first data column is often wider than the rest
        if xs[i] - xs[i - 1] <= max(2.0 * med, 8.0):
            run[1] = i
            if run[1] - run[0] > best[1] - best[0]:
                best = (run[0], run[1])
        else:
            run = [i, i]
    if best[1] - best[0] < GRID_MIN_V - 1:
        return []
    x0, x1 = xs[best[0]], xs[best[1]]
    rows = [h for h in gh
            if min(h[2], x1) - max(h[1], x0) >= 0.5 * (x1 - x0)]
    ys = sorted(h[0] for h in rows)
    py = sorted(b - a for a, b in zip(ys, ys[1:]) if b - a > 2)
    if len(rows) < GRID_MIN_H or not py or \
            py[len(py) // 2] > GRID_MAX_PITCH_H * page.rect.height:
        return []
    grid = fitz.Rect(x0, ys[0], x1, ys[-1]) & page.rect
    grid.row_pitch = py[len(py) // 2]           # read by grid_entries
    grid.col_rules = xs[best[0]:best[1] + 1]    # the lattice's column rules
    return [grid]


_GATE_SCHEMA = {"type": "object", "properties": {
    "has_signoffs": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["has_signoffs", "why"]}

_GATE_PROMPT = """This image is a ruled chart or table from a scanned Indian
hospital record. Answer has_signoffs = true if ANY of its cells hold a
HANDWRITTEN sign-off: a person's name, initials, signature or staff/employee
number written to record who gave, checked or received something (a nurse
signing for a dose, a patient or relative signing a cell). Answer false if
the cells hold only printed text, plotted curves or dots, or handwritten
clinical values -- times, doses, readings, ticks, circled letters."""


# Chart cells get their own question. The general mark prompt offers
# "handwritten clinical value" as the home for any number, and a nurse's
# staff number judged alone went there every time -- a 5-digit number, six times out of
# six -- while the name above it, the one clue, sat on a ruled line.
GRID_CUT = {"signature": "sem:signature", "name_or_initials": "sem:signature",
            "staff_number": "sem:signature"}

_GRID_SCHEMA = {"type": "object", "properties": {"marks": {"type": "array", "items": {
    "type": "object", "properties": {
        "index": {"type": "integer"},
        "kind": {"type": "string", "enum": [
            "signature", "name_or_initials", "staff_number", "date", "time",
            "dose_or_reading", "printed_text_fragment", "stray_mark"]},
        "why": {"type": "string"}},
    "required": ["index", "kind", "why"]}}},
    "required": ["marks"]}

_GRID_PROMPT = """Each image is a piece of a medication, nursing or observation
chart from a scanned Indian hospital record, numbered from 0. In each, ONE
entry in the chart's grid is outlined in RED. Say what that entry IS.

signature: a handwritten sign-off scrawl -- a nurse's, doctor's, patient's or
  relative's.
name_or_initials: a person's handwritten name or initials.
staff_number: a staff or employee number -- typically 4 to 6 digits, written
  under, beside or instead of a name to record WHO gave or checked a dose.
  In the day columns of a medication chart a bare number like this is a staff
  number: doses are written in the drug panel, not in the day cells.
date: a date (9/8/26, 10/8).
time: a time of day (8AM, @9am, 2pm, 22:00).
dose_or_reading: a dose, quantity or clinical reading (1 tab, 5ml, 120/80,
  98.6), a tick or a cross, or a code from the chart's own key or legend
  (P, OD, V, RD, W) -- a code the form defines is not anyone's initials.
printed_text_fragment: printed text or an empty printed box.
stray_mark: a smudge, speck or accidental stroke.

If the outline holds a name or number together with a signature, answer by
the name, number or signature. Say what the entry IS, not whether it is
sensitive."""


def grid_has_signoffs(page, grids, workdir=None):
    """One look at the whole lattice before judging its entries one by one.

    A printed results table or a CTG trace is ruled like a chart too, and
    judging its hundred-odd cells costs some 17 requests to learn nothing.
    FAILS OPEN: if the question cannot be answered, the entries are judged
    anyway -- a failed request may cost time, never a sign-off."""
    wd = Path(workdir or "/tmp")
    wd.mkdir(parents=True, exist_ok=True)
    G = grids[0]
    for g in grids[1:]:
        G = G | g
    clip = fitz.Rect(G.x0 - 20, G.y0 - 20, G.x1 + 20, G.y1 + 20) & page.rect
    dpi = max(72, min(200, int(1600 * 72 / max(clip.width, clip.height, 1))))
    f = wd / f"_gridgate_p{page.number}.png"
    try:
        page.get_pixmap(dpi=dpi, clip=clip).save(str(f))
        raw = common.generate(common.resolve_model(),
                              [{"text": _GATE_PROMPT},
                               {"image_path": str(f), "mime": "image/png"}],
                              _GATE_SCHEMA, thinking_budget=1024)
        return bool(json.loads(raw).get("has_signoffs", True))
    except Exception:
        return True
    finally:
        f.unlink(missing_ok=True)


def grid_entries(page, grids, covered=()):
    """Pen entries inside charting-grid cells -> [fitz.Rect], INCLUDING the
    ones a reader transcribed.

    residual_marks() skips ink that OCR read, leaving it to the text
    detector. In a chart that is where sign-offs are lost: OCR reads
    a name, a staff number, "name/number" as loose words, and a word list stripped of
    its grid does not say that each is the nurse who gave the 3pm dose. Here
    every entry is judged on its IMAGE, in context, by the same classifier.
    Rules are cut out first so an entry never joins its neighbour's cell."""
    sc = 72.0 / DPI
    pm = page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
    g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w).astype(int)
    bg = ndi.median_filter(g, size=(1, 15))
    ink = (g < DARK) | ((bg - g) > 60)
    # lengths scale with the page: a 1181pt-wide scan's handwriting is twice
    # an A4 page's, and a fixed 12pt run took the downstrokes of "D", "H" and
    # "1" for rules, eating a nurse's name written across a ruled line
    zoom = max(1.0, page.rect.width / 595.0)
    L = max(3, int(RULE_PT * zoom / sc))
    rules = (ndi.binary_opening(ink, np.ones((1, L)))
             | ndi.binary_opening(ink, np.ones((L, 1))))
    # A skewed photo's rules lean, and a straight 40pt opening leaves the
    # stair-steps behind: on one medication chart they joined every entry of
    # the lower grid into a single 1117x527pt blob, discarded as too big,
    # sign-offs and all. Short runs bridged sideways follow the lean.
    k = max(12, int(RULE_SEG_PT * zoom / sc))
    rules |= ndi.binary_closing(ndi.binary_opening(ink, np.ones((k, 1))),
                                np.ones((k, 3)))
    rules |= ndi.binary_closing(ndi.binary_opening(ink, np.ones((1, k))),
                                np.ones((3, k)))
    rules = ndi.binary_dilation(rules, np.ones((3, 3)))
    ink &= ~rules
    inside = np.zeros_like(ink)
    for G in grids:
        inside[max(0, int(G.y0 / sc)):int(G.y1 / sc) + 1,
               max(0, int(G.x0 / sc)):int(G.x1 / sc) + 1] = True
    ink &= inside
    joined = ndi.binary_dilation(ink, np.ones((5, 5))) & ~rules
    lab, _n = ndi.label(joined)
    frags = []
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        w = (sl[1].stop - sl[1].start) * sc
        h = (sl[0].stop - sl[0].start) * sc
        # Pieces are filtered only for being specks or rule; size and density
        # are judged on the GROUP. A dense scrawl (43% ink) and the 4pt tail
        # of a signature were dropped piece by piece and left readable.
        if w * h < 6.0 * zoom * zoom or w > ENTRY_W[1] * zoom or h > ENTRY_H[1] * zoom:
            continue
        # A stretch of a CURVED rule -- the page bends away from the camera
        # and its lines stop being straight enough to be removed -- keeps
        # its ink in one thin band of rows; writing spreads over its height.
        rows_ink = ink[sl].sum(1)
        kb = max(2, int(0.3 * len(rows_ink)))
        band = max(rows_ink[i:i + kb].sum()
                   for i in range(max(1, len(rows_ink) - kb + 1)))
        if band > 0.75 * max(rows_ink.sum(), 1):
            continue
        if _line_like(ink[sl]) or _line_like(ink[sl].T):
            continue
        frags.append(fitz.Rect(sl[1].start * sc, sl[0].start * sc,
                               sl[1].stop * sc, sl[0].stop * sc))
    pitch = min((getattr(G, "row_pitch", 12.0) for G in grids), default=12.0)
    cols = sorted(x for G in grids for x in getattr(G, "col_rules", []))
    # Group on ALL the ink, then drop groups another pass already cut. Carving
    # the other passes' boxes out first split sign-off blocks, and the piece
    # left over -- a name over a staff number -- no longer read as a sign-off.
    cov = [fitz.Rect(r) for r in covered]
    out = []
    for g in _group_entries(frags, pitch, cols):
        if (g.width < ENTRY_W[0] * zoom or g.height < ENTRY_H[0] * zoom
                or g.width > ENTRY_W[1] * zoom):
            continue
        if g.width < 8.0 * zoom and g.height > max(20.0 * zoom, 4 * g.width):
            continue            # pieces of a leaning rule, stacked
        x0, y0 = max(0, int(g.x0 / sc)), max(0, int(g.y0 / sc))
        dens = float(ink[y0:int(g.y1 / sc) + 1, x0:int(g.x1 / sc) + 1].mean())
        if not (BLOB_DENSITY[0] <= dens <= GROUP_MAX_DENSITY):
            continue
        a = max(g.width * g.height, 1e-6)
        done = sum((g & c).width * (g & c).height for c in cov if g.intersects(c))
        if done < 0.8 * a:
            out.append(g)
    return out


def _group_entries(frags, pitch, cols=()):
    """Pieces of one sign-off -> one candidate.

    Judged one piece at a time, a staff number split by the pen into single
    digits was read as a date, a 5-digit staff number alone as a clinical value and a
    nurse's name alone as a drug. A sign-off is a name and a number stacked
    in adjacent rows of one column, or side by side on one line; judged as
    that whole, it reads as what it is. Pieces join when they sit side by
    side on a line with a small gap, or stack in the same column with at
    most one row between them -- and never across a column rule: the next
    column is another day's dose, the time column a clinical value."""
    import bisect
    n = len(frags)
    col = [bisect.bisect(list(cols), (f.x0 + f.x1) / 2) for f in frags]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = frags[i]
        for j in range(i + 1, n):
            b = frags[j]
            if col[i] != col[j]:
                continue
            xo = min(a.x1, b.x1) - max(a.x0, b.x0)
            yo = min(a.y1, b.y1) - max(a.y0, b.y0)
            side = yo > 0.4 * min(a.height, b.height) and -xo <= 0.5 * pitch
            stack = (xo > 0.3 * min(a.width, b.width)
                     and -yo <= 0.6 * pitch)
            if side or stack:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        r = groups.get(find(i))
        groups[find(i)] = fitz.Rect(frags[i]) if r is None else r | frags[i]
    # a block taller than a dose-time slot is split back at row bounds by
    # the caller's cap on size; here only a hard ceiling keeps the crop sane
    out = []
    for r in groups.values():
        if r.height > 12 * pitch:
            out.extend(f for f in frags if r.contains(f))
        else:
            out.append(r)
    return out


def _line_like(sub, max_run_px=5):
    """True for ink that is only ruled line: in nearly every column it is one
    or two thin horizontal strands, and it runs the box's full width. Every
    handwritten entry has some taller stroke -- an 'l', a '1', a loop."""
    cols = sub.any(0)
    if cols.mean() < 0.85:
        return False
    thin = 0
    for c in np.nonzero(cols)[0]:
        col = sub[:, c].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], col, [0]))))
        runs = edges[1::2] - edges[::2]
        if len(runs) <= 2 and runs.max() <= max_run_px:
            thin += 1
    return thin >= 0.9 * cols.sum()


def _where(page, r):
    """A coarse, human position: 'top-left corner', 'middle', ..."""
    cx = (r.x0 + r.x1) / 2 / (page.rect.width or 1)
    cy = (r.y0 + r.y1) / 2 / (page.rect.height or 1)
    v = "top" if cy < 0.2 else ("bottom" if cy > 0.8 else "middle")
    h = "left" if cx < 0.33 else ("right" if cx > 0.67 else "centre")
    return f"{v}-{h}"


def _classify(page, rects, workdir, text_pt=None, margin_cap=None,
              prompt=None, schema=None):
    wd = Path(workdir or "/tmp")
    parts, paths = [{"text": prompt or _PROMPT}], []
    for i, r in enumerate(rects):
        # CONTEXT, WITH THE TARGET OUTLINED. A tight crop of one handwritten
        # word is ambiguous: on a nursing-notes page, words in the middle of
        # the clinical narrative were cut as "handwritten personal data"
        # because, alone, a stray word can look like a name. Showing the
        # surrounding lines makes a sentence look like a sentence and a name
        # box look like a name box. The red outline marks which mark is being
        # asked about; it is drawn on the classifier's copy only, never on
        # the document.
        m = max(20.0, 1.5 * r.height)
        if margin_cap is not None:
            m = min(m, margin_cap)
        ctx = fitz.Rect(r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m) & page.rect
        f = wd / f"_ink_p{page.number}_{i}.png"
        pm = page.get_pixmap(dpi=CROP_DPI, clip=ctx)
        from PIL import Image, ImageDraw
        im = Image.frombytes("RGB" if pm.n < 4 else "RGBA", (pm.w, pm.h), pm.samples).convert("RGB")
        z = CROP_DPI / 72.0
        box = [(r.x0 - ctx.x0) * z - 3, (r.y0 - ctx.y0) * z - 3,
               (r.x1 - ctx.x0) * z + 3, (r.y1 - ctx.y0) * z + 3]
        ImageDraw.Draw(im).rectangle(box, outline=(255, 0, 0), width=max(2, int(z)))
        im.save(str(f))
        paths.append(f)
        # LABEL EVERY IMAGE. Sent as a bare list, the model lost count: on
        # the review bundle it answered for 10 of 11 images, every index after
        # the one it skipped slid down by one, a printed fragment inherited
        # the "signature" verdict and the real signature got none. The run
        # was deterministic, so it would have failed the same way every time.
        # ...AND GIVE IT SCALE. A tight crop hides how big a mark is: a
        # hospital's stylised "C" emblem, 49pt tall on a page of 9pt text and
        # sitting in the top-left corner where logos go, was classified as
        # "a printed letter C" and kept. Size and position are what make it a
        # logo, so the label carries both.
        tctx = (f", about {text_pt:.0f}pt body text on this page" if text_pt else "")
        parts.append({"text": f"Image {i}: the mark INSIDE THE RED OUTLINE is "
                              f"{r.width:.0f}x{r.height:.0f}pt, {_where(page, r)} "
                              f"of the page{tctx}:"})
        parts.append({"image_path": str(f), "mime": "image/png"})
    try:
        raw = common.generate(common.resolve_model(), parts, schema or _SCHEMA,
                              thinking_budget=1024)
        got = json.loads(raw).get("marks", [])
        # A reply that does not account for every image cannot be trusted to
        # be aligned with them; raising here takes the loud failure path.
        if sorted({int(g.get("index", -1)) for g in got}) != list(range(len(rects))):
            raise ValueError(f"classifier answered for "
                             f"{len({g.get('index') for g in got})} of "
                             f"{len(rects)} marks; indices not trusted")
        return got
    finally:
        for f in paths:
            f.unlink(missing_ok=True)


def classify_regions(page, rects, workdir=None, cap=None, text_pt=None,
                     margin_cap=None, prompt=None, schema=None):
    """-> (verdicts, unclassified, dropped). verdicts = [(rect, kind, why)].

    Shared by this module and carriers.py: both find candidate marks from
    pixels, and neither may cut on geometry or colour alone. Small labelled
    batches; a miscount splits the batch and asks again rather than
    abandoning it; a region that still cannot be classified on its own is
    returned as UNCLASSIFIED so the caller can say so -- never guessed at.
    """
    cap = BATCH * MAX_BATCHES if cap is None else cap
    todo, dropped = list(rects[:cap]), list(rects[cap:])
    verdicts, unclassified = [], []

    def ask(chunk, depth=0):
        try:
            return [(chunk, g) for g in _classify(page, chunk, workdir, text_pt,
                                                  margin_cap, prompt, schema)]
        except Exception:
            if len(chunk) > 1 and depth < 4:
                mid = len(chunk) // 2
                return ask(chunk[:mid], depth + 1) + ask(chunk[mid:], depth + 1)
            unclassified.extend(chunk)
            return []

    for start in range(0, len(todo), BATCH):
        for chunk, g in ask(todo[start:start + BATCH]):
            try:
                i = int(g["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(chunk):
                verdicts.append((chunk[i], g.get("kind"), g.get("why", "")))
    return verdicts, unclassified, dropped


def findings(page, words, reader, pack=None, covered=(), workdir=None):
    """-> (findings, marker_or_None). Scanned pages only."""
    if reader not in ("vision_ocr", "ocr_empty"):
        return [], None
    try:
        rects, extra = residual_marks(page, words, covered=covered,
                                      workdir=workdir, split=True)
    except Exception:
        rects, extra = [], []
    # The unread-strip marks are a second net over ink no pass used to look
    # at; they get their own small budget, largest first, and never take a
    # slot from the primary candidates. Those beyond it were never examined
    # before this net existed and are not reported as a lost cap.
    extra = extra[:BATCH * EXTRA_MAX_BATCHES]
    # Inside a charting grid the grid pass owns the ink. A residual candidate
    # there must not also be carved out of the grid pass's input: when the
    # classifier then KEPT it, the carve-out had split a sign-off block and
    # the part it removed was never cut.
    try:
        grids = charting_grids(page)
        cells = (grid_entries(page, grids, covered=covered) if grids else [])
        if cells and not grid_has_signoffs(page, grids, workdir):
            cells = []
    except Exception:
        grids, cells = [], []
    # the grid pass takes over the lattice only when it actually runs there;
    # otherwise the residual candidates inside it stay with the residual pass
    if cells:
        def in_grid(r):
            a = max(r.width * r.height, 1e-6)
            return any((r & G).width * (r & G).height >= 0.5 * a
                       for G in grids if r.intersects(G))
        rects = [r for r in rects if not in_grid(r)]
        extra = [r for r in extra if not in_grid(r)]
    if not rects and not cells and not extra:
        return [], None
    cap = BATCH * MAX_BATCHES
    hs = sorted(r.y1 - r.y0 for _i, _t, r in words if r.y1 > r.y0)
    text_pt = hs[len(hs) // 2] if hs else None
    verdicts, unclassified, dropped = ([], [], [])
    if rects:
        verdicts, unclassified, dropped = classify_regions(
            page, rects, workdir, cap, text_pt=text_pt)
    gv, gu, gd = ([], [], [])
    if cells:
        # TWO OPINIONS, IN DIFFERENT COMPANY. A verdict depends on the batch
        # an entry is judged in: with one entry fewer on the page, the same
        # name / number / name / number block, stable as a signature when
        # judged three times, was read as the date "21/2/18" and kept. Each
        # entry is judged twice, the second time in reversed order so its
        # batch neighbours differ, and it is cut if either judgement says so.
        # Context is capped at about a cell either way; a tall block's usual
        # 1.5x-height margin took in half the page.
        cap_g = BATCH * GRID_MAX_BATCHES
        gkw = dict(text_pt=text_pt, margin_cap=GRID_CTX_PT,
                   prompt=_GRID_PROMPT, schema=_GRID_SCHEMA)
        first, gu, gd = classify_regions(page, cells, workdir, cap_g, **gkw)
        second, gu2, _ = classify_regions(page, list(reversed(cells[:cap_g])),
                                          workdir, cap_g, **gkw)
        best = {}
        for r, kind, why in first + second:
            key = tuple(round(v, 1) for v in r)
            if key not in best or (kind in GRID_CUT
                                   and best[key][1] not in GRID_CUT):
                best[key] = (r, kind, why)
        # A date or a time is one line. A pen block spanning several rows of
        # a sign-off grid is a name over a staff number, perhaps with a
        # signature under it; when the model calls such a block a "date" it
        # has latched onto the date its outline also takes in (the header
        # "20/4" just above a name over a staff number), and the block is cut anyway.
        pitch = min((getattr(G, "row_pitch", 12.0) for G in grids), default=12.0)
        gv = []
        for r, kind, why in best.values():
            # not "time": a chart's time column holds big, freely written
            # "6pm"s that span two rows and are what they say they are
            if (kind in ("date", "dose_or_reading")
                    and r.height > GRID_TALL_ROWS * pitch):
                why = (f"a {r.height / pitch:.1f}-row pen block judged {kind!r}; "
                       f"a {kind} is one line")
                kind = "signature"
            gv.append((r, kind, why))
        judged = set(best)
        gu = [r for r in gu + gu2 if tuple(round(v, 1) for v in r) not in judged]
    ev, eu = [], []
    if extra:
        ev, eu, _ed = classify_regions(page, extra, workdir, len(extra),
                                       text_pt=text_pt)
    unclassified, dropped = unclassified + gu + eu, dropped + gd
    out = []
    for src, found, cut in (("residual_ink", verdicts + ev, CUT),
                            ("grid_ink", gv, GRID_CUT)):
        for r, kind, _why in found:
            if kind not in cut:
                continue
            cat = cut[kind]
            out.append({
                "categories": [cat], "text": f"[{kind.replace('_', ' ')}]",
                "bbox": [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)],
                "all_boxes": [[round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]],
                "n_boxes": 1, "source": src, "locator": src,
                "action": (pack.action_for([cat]) if pack is not None
                           else "REDACT"),
                "reason": (f"ink no reader explained; model says {kind}"
                           if src == "residual_ink" else
                           f"pen entry in a charting-grid cell; model says {kind}"),
            })
    marker = None
    if unclassified:
        marker = {"categories": [], "text": "", "bbox": None, "action": "KEEP",
                  "source": "ink_classifier_unavailable",
                  "locator": "page_marker",
                  "reason": f"{len(unclassified)} residual mark(s) could not be "
                            f"classified even alone and were left UNCUT"}
    elif dropped:
        marker = {"categories": [], "text": "", "bbox": None, "action": "KEEP",
                  "source": "ink_marks_capped", "locator": "page_marker",
                  "reason": f"{len(dropped)} mark(s) beyond the per-page "
                            f"classification cap were not classified"}
    return out, marker
