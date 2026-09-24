#!/usr/bin/env python3
"""P5 — objective signals: PII evidence that needs no model and no human.

These form evidence tier T0 of the benchmark reference. A checksum that passes,
a QR that decodes, a face the OS detects — none of these are opinions, so
recall measured against them is not self-graded. Judge-only findings stay in a
separate tier and are never mixed into these numbers.

Measured on this corpus (2026-09-17):
  * the Aadhaar Secure QR decodes to the FULL identity - name, guardian,
    district, state, PIN, DOB - so redacting the printed number while leaving
    the QR intact leaks everything. Objective, reproducible.
  * a dense Aadhaar QR needs >= 300 dpi to decode from a page render; at
    120 dpi it is invisible to the decoder. Extracting the native embedded
    image (1000x1000) decodes at confidence 1.00 with no rendering at all.
"""
import gzip
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

HELPER = Path(__file__).resolve().parent / "vision_helper.swift"
QR_MIN_DPI = 300        # measured floor for dense Aadhaar QR from a render

# ---------------------------------------------------------------- checksums

_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]]
_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(num):
    """Aadhaar's check digit. A pass is strong evidence the 12 digits were read
    correctly AND are a real Aadhaar - which is why the per-reader pass rate is
    itself a headline result: it separates 'can localise' from 'can transcribe'.
    """
    digits = [int(c) for c in re.sub(r"\D", "", str(num))]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


def luhn_ok(num):
    ds = [int(c) for c in re.sub(r"\D", "", str(num))]
    if not 12 <= len(ds) <= 19:
        return False
    tot, alt = 0, False
    for d in reversed(ds):
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        tot += d
        alt = not alt
    return tot % 10 == 0


CARD_MIN_GROUP = 3


def card_shape_ok(s):
    """Is this laid out like a card number, or merely long enough to be one?

    Luhn alone is a weak filter: roughly one in ten digit runs passes it, so
    the CHECK DIGIT decides almost nothing on its own and the SHAPE has to do
    the rest. Measured on a real claim bundle: a growth-chart row reading
    "10 13 16 19 22 25 28 31 34" is eighteen digits, passes Luhn, and was
    redacted as a payment card -- destroying a row of a clinical chart.

    Real cards are written as one run of digits or in groups of four (Amex
    4-6-5). None is written in nine two-digit groups, so a run whose separated
    groups are shorter than three digits is a table, not a card.
    """
    groups = [g for g in re.split(r"[ -]+", str(s).strip()) if g]
    if len(groups) == 1:
        return True
    return all(len(g) >= CARD_MIN_GROUP for g in groups)


def card_ok(s):
    return card_shape_ok(s) and luhn_ok(s)


_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_ok(s):
    s = str(s).strip().upper()
    if not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]", s):
        return False
    tot = 0
    for i, ch in enumerate(s[:14]):
        v = _GST_CHARS.index(ch) * (2 if i % 2 else 1)
        tot += v // 36 + v % 36
    return _GST_CHARS[(36 - tot % 36) % 36] == s[14]


# Role accounts: a mailbox indexed per ORGANISATION, not per person. The
# discriminator the owner's bank rule uses, applied to mailboxes.
ROLE_ACCOUNT = re.compile(
    r"^(info|support|contact|admin|sales|enquir(y|ies)|care|customercare|"
    r"helpdesk|help|billing|accounts|office|mail|reception|feedback|"
    r"grievance|noreply|no-reply|donotreply|hr|careers|service|services|"
    r"claims|marketing|webmaster|postmaster)[._-]?\d*$", re.I)


def email_semantic(value):
    """A person's mailbox, or an organisation's front desk?

    `sunita.rao@gmail.com` identifies a person; `info@apollohospitals.com` on a
    letterhead identifies the issuer and, under a generic pack, is the thing a
    reader needs to know where the document came from. Emitting one key for
    both meant either personal mail survived or every letterhead was destroyed.
    """
    local = str(value).split("@", 1)[0]
    return "sem:org_contact" if ROLE_ACCOUNT.match(local) else "sem:contact"


def pan_semantic(value):
    """PAN's 4th character IS the holder type -- deterministic, not a guess.

    P=individual, C=company, H=HUF, F=firm, A=AOP, T=trust, B/G/J/L. A
    hospital's own PAN on a bill footer is an institutional registration code;
    the claimant's PAN is a government ID. Same ten characters, opposite rules,
    and the distinction is printed in the value itself.
    """
    v = str(value).strip().upper()
    return "sem:gov_id" if len(v) > 3 and v[3] == "P" else "sem:org_reg_code"


# pattern -> (SEMANTIC key, validator).
#
# These are semantic keys, never a pack's own ids and never the audio
# taxonomy's names. They used to be v0.2 names (EMAIL, TAX_ID,
# FINANCIAL_ACCOUNT_INFO), which no pack's `semantics` block maps -- so
# action_for failed closed and EVERY objective hit became REDACT. That
# silently destroyed IFSC codes under insurance_medical, whose T33 exists
# precisely to keep them, and it would have destroyed every letterhead under
# the generic pack. A mechanism that names categories the packs cannot read
# does not merely mis-file a hit; it inverts the pack's KEEP rules.
PATTERNS = {
    "AADHAAR": (r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)", "sem:gov_id",
                verhoeff_ok),
    "PAN":     (r"\b[A-Z]{5}\d{4}[A-Z]\b", pan_semantic, lambda s: True),
    "IFSC":    (r"\b[A-Z]{4}0[A-Z0-9]{6}\b", "sem:bank_common",
                lambda s: True),
    "GSTIN":   (r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b",
                "sem:org_reg_code", gstin_ok),
    "CARD":    (r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "sem:card", luhn_ok),
    "PHONE_IN": (r"(?<!\d)[6-9]\d{9}(?!\d)", "sem:contact", lambda s: True),
    "EMAIL":   (r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b", email_semantic,
                lambda s: True),
    "UPI":     (r"\b[\w.-]{3,}@(?:ybl|oksbi|okhdfcbank|okicici|paytm|ibl|axl|"
                r"upi|apl)\b", "sem:bank_personal", lambda s: True),
}


# SHAPE gates EMISSION; the checksum only grades EVIDENCE. They are different
# questions and were conflated. A card that fails Luhn may still be a card --
# one mis-OCR'd digit on a 72 dpi scan breaks the check digit, and dropping it
# would leak. A run of nine two-digit groups is not a damaged card; it is a
# table, and emitting it destroyed a row of a clinical chart. So a checksum
# failure lowers the evidence tier, while a shape failure means the value was
# never of this kind at all.
SHAPE = {"CARD": card_shape_ok}


def scan_text(text):
    """-> list of {kind, value, category, checksum_ok, span}. `checksum_ok`
    distinguishes proof from mere shape: a format match whose checksum fails is
    NOT objective evidence and must fall to a lower tier, because one mis-OCR'd
    digit on a 72 dpi scan breaks it."""
    hits = []
    for kind, (pat, cat, check) in PATTERNS.items():
        shape = SHAPE.get(kind)
        for m in re.finditer(pat, text):
            val = m.group(0)
            if shape and not shape(val):
                continue                 # not this kind of value at all
            hits.append({"kind": kind, "value": val,
                         "category": cat(val) if callable(cat) else cat,
                         "checksum_ok": bool(check(val)),
                         "span": [m.start(), m.end()]})
    return hits


# ------------------------------------------------------------ apple vision

def vision(image_path, mode="all"):
    """Barcodes / faces / Latin text with normalised top-left boxes."""
    try:
        r = subprocess.run(["swift", str(HELPER), str(image_path), mode],
                           capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout) if r.stdout.strip() else {
            "error": r.stderr[-200:]}
    except Exception as exc:
        return {"error": str(exc)[:200]}


def decode_aadhaar_qr(payload):
    """Aadhaar Secure QR: big integer -> gzip -> 0xff-delimited fields.

    Returns the FIELD COUNT and which identity slots are populated, never the
    values - this runs inside a benchmark, and the whole point is that these
    values should not be spreading further.
    """
    s = str(payload)
    if not s.isdigit():
        return None
    n = int(s)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    for fn in (gzip.decompress, zlib.decompress,
               lambda b: zlib.decompress(b, -15)):
        try:
            raw = fn(raw)
            break
        except Exception:
            continue
    else:
        return None
    fields = raw.split(b"\xff")
    text_fields = [f.decode("utf-8", "replace") for f in fields[:20]]
    return {
        "field_count": len(fields),
        "decoded_bytes": len(raw),
        "has_embedded_face": raw.find(b"\xff\xd8\xff") > 0
                             or raw.find(b"\x00\x00\x00\x0cjP") > 0,
        "nonempty_text_fields": sum(1 for f in text_fields if f.strip()),
    }


def page_qr_and_faces(doc, page_no, dpi=QR_MIN_DPI, workdir=None):
    """Objective visual signals for one page.

    Tries the NATIVE embedded images first (an e-Aadhaar QR is its own 1000x1000
    XObject and decodes at confidence 1.00 with no rasterisation), then falls
    back to a >=300 dpi page render for scans where the QR is baked into the
    page bitmap.
    """
    import fitz
    workdir = Path(workdir or ".")
    workdir.mkdir(parents=True, exist_ok=True)
    page = doc[page_no]
    found = {"qr": [], "faces": [], "source": None}

    for im in page.get_images(full=True):
        xref, w, h = im[0], im[2], im[3]
        if w != h or w < 300:
            continue
        p = workdir / f"_qr_{xref}.png"
        try:
            fitz.Pixmap(doc, xref).save(str(p))
        except Exception:
            continue
        res = vision(p, "barcodes")
        for b in res.get("barcodes", []):
            b["native_xref"] = xref
            b["aadhaar"] = decode_aadhaar_qr(b.get("payload", ""))
            found["qr"].append(b)
        p.unlink(missing_ok=True)
    if found["qr"]:
        found["source"] = "native_xobject"

    render = workdir / f"_pg_{page_no}.png"
    page.get_pixmap(dpi=dpi).save(str(render))
    res = vision(render, "all")
    if not found["qr"]:
        for b in res.get("barcodes", []):
            b["aadhaar"] = decode_aadhaar_qr(b.get("payload", ""))
            found["qr"].append(b)
        if found["qr"]:
            found["source"] = f"page_render_{dpi}dpi"
    found["faces"] = res.get("faces", [])
    found["latin_text_regions"] = len(res.get("text", []))
    render.unlink(missing_ok=True)
    return found


if __name__ == "__main__":
    # self-test: checksums on published specimen values, no corpus needed
    assert verhoeff_ok("234123412346"), "known-good Aadhaar specimen failed"
    assert not verhoeff_ok("234123412347"), "bad check digit accepted"
    assert not verhoeff_ok("12345"), "wrong length accepted"
    assert luhn_ok("4539578763621486") and not luhn_ok("4539578763621487")
    assert gstin_ok("27AAPFU0939F1ZV"), "known-good GSTIN failed"
    assert not gstin_ok("27AAPFU0939F1ZX"), "bad GSTIN check digit accepted"
    hits = scan_text("PAN ABCDE1234F IFSC HDFC0000523 call 9876543210")
    kinds = {h["kind"] for h in hits}
    assert {"PAN", "IFSC", "PHONE_IN"} <= kinds, kinds

    # THE CONTRACT: every emitted category is a `sem:` key. A bare name is
    # unmappable by any pack, fails closed to REDACT, and inverts KEEP rules.
    by = {h["kind"]: h["category"] for h in scan_text(
        "PAN AAAPL1234C PAN AABCT3518Q IFSC HDFC0000523 9876543210 "
        "info@x.com GSTIN 29AABCT3518Q1ZV 4111111111111111")}
    assert all(v.startswith("sem:") for v in by.values()), by
    assert by["IFSC"] == "sem:bank_common", "IFSC must be protectable"
    assert by["GSTIN"] == "sem:org_reg_code"
    assert by["EMAIL"] == "sem:org_contact", "role account read as personal"
    # a numeric table row is not a payment card, whatever Luhn says
    assert card_shape_ok("4111 1111 1111 1111") and card_shape_ok("4111111111111111")
    assert card_shape_ok("3782 822463 10005"), "Amex grouping rejected"
    assert not card_shape_ok("10 13 16 19 22 25 28 31 34"), "chart row read as card"
    assert not any(h["kind"] == "CARD" for h in
                   scan_text("10 13 16 19 22 25 28 31 34")), "chart row emitted"
    assert any(h["kind"] == "CARD" for h in
               scan_text("card 4111 1111 1111 1111")), "real card dropped"
    assert email_semantic("sunita.rao@gmail.com") == "sem:contact"
    # PAN holder type: 4th char P = individual, anything else = institution
    assert pan_semantic("AAAPL1234C") == "sem:gov_id"
    assert pan_semantic("AABCT3518Q") == "sem:org_reg_code"
    print("objective self-test OK (verhoeff, luhn, gstin, scan, sem: keys)")


# ---------------------------------------------------------------- QR by shape
# Apple Vision DECODES a QR, and on a photocopied Aadhaar card it fails: the
# card on case-48142 p24 yields 0 barcodes at 300 dpi while the symbol is
# plainly there. But redaction never needed the payload -- it needed the
# RECTANGLE. A QR carries three 7x7 concentric "finder patterns" at its
# corners, and a scanline crossing one always reads dark:light:dark:light:dark
# in a 1:1:3:1:1 ratio. That is a shape, not a decode, so it survives the blur
# and skew that defeat the decoder.
QR_RATIO_TOL = 0.55
QR_MIN_UNIT = 1.4          # px per module; below this the scan is hopeless


def _runs(row):
    """(value, start, length) for each run of equal values in a bool row."""
    out, n = [], len(row)
    i = 0
    while i < n:
        j = i
        while j < n and row[j] == row[i]:
            j += 1
        out.append((bool(row[i]), i, j - i))
        i = j
    return out


def _finder_hits(dark):
    """Centres of scanlines matching the 1:1:3:1:1 finder ratio, per axis.
    -> [(x, y, axis)]"""
    hits = []
    for axis in (0, 1):
        arr = dark if axis == 0 else dark.T
        step = max(1, arr.shape[0] // 400)
        for y in range(0, arr.shape[0], step):
            rs = _runs(arr[y])
            for k in range(len(rs) - 4):
                w = rs[k:k + 5]
                if not (w[0][0] and not w[1][0] and w[2][0] and not w[3][0]
                        and w[4][0]):
                    continue
                L = [r[2] for r in w]
                unit = (L[0] + L[1] + L[3] + L[4]) / 4.0
                if unit < QR_MIN_UNIT:
                    continue
                exp = [unit, unit, 3 * unit, unit, unit]
                if all(abs(a - b) <= QR_RATIO_TOL * b for a, b in zip(L, exp)):
                    cx = w[0][1] + sum(L) / 2.0
                    hits.append((cx, y, axis) if axis == 0
                                else (y, cx, axis))
    return hits


def locate_qr_by_pattern(page, dpi=200, min_hits=4, min_pt=28.0):
    """-> [fitz.Rect] of probable QR symbols, located WITHOUT decoding.

    Two filters carry the whole thing. A genuine finder pattern is a SQUARE
    concentric target, so a scanline finds it running horizontally AND
    vertically through the same spot -- ordinary dense text throws the
    1:1:3:1:1 ratio on one axis only, and demanding both removes essentially
    all of it. And a QR that encodes anything is physically large, so a cluster
    smaller than a couple of centimetres is not a symbol.
    """
    import numpy as np
    import fitz
    from scipy import ndimage
    pm = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
    dark = a < 128
    hits = _finder_hits(dark)
    if len(hits) < min_hits:
        return []
    H, W = dark.shape
    pts = [np.zeros(dark.shape, bool), np.zeros(dark.shape, bool)]
    for x, y, ax in hits:
        xi, yi = int(x), int(y)
        if 0 <= yi < H and 0 <= xi < W:
            pts[ax][yi, xi] = True
    both = pts[0] | pts[1]
    grp = ndimage.binary_dilation(both, iterations=max(4, int(dpi / 25)))
    lab, n = ndimage.label(grp)
    out, s = [], 72.0 / dpi
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        h_hits = int(pts[0][sl].sum())
        v_hits = int(pts[1][sl].sum())
        if h_hits < 2 or v_hits < 2 or h_hits + v_hits < min_hits:
            continue                       # must be seen on BOTH axes
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (0.5 <= w / max(h, 1) <= 2.0):
            continue                       # a QR is square-ish
        if w * s < min_pt or h * s < min_pt:
            continue                       # too small to be a real symbol
        # A QR is near-BITONAL: every module is paper-white or ink-black. An
        # ultrasound is mid-grey speckle, and its noise throws the 1:1:3:1:1
        # ratio often enough to cluster. Without this test the detector blacked
        # out a whole scan panel on a foetal ultrasound -- destroying exactly
        # the clinical imagery owner rule 4 protects.
        # Measured on this corpus: real QR symbols score extreme 0.58-0.63
        # with std 88-89, foetal ultrasound speckle 0.00-0.12 with std 22-40.
        # The gap is wide, so the thresholds sit in the middle of it rather
        # than snug against the QR side.
        patch = a[ys, xs]
        extreme = float(((patch < 60) | (patch > 200)).mean())
        if extreme < 0.35 or float(patch.std()) < 60:
            continue
        pad = 0.12
        out.append(fitz.Rect((xs.start - w * pad) * s, (ys.start - h * pad) * s,
                             (xs.stop + w * pad) * s, (ys.stop + h * pad) * s)
                   & page.rect)
    return out


# ------------------------------------------------------------ cheque signature
# A cancelled cheque earns its place in a claim bundle by evidencing the payout
# account: bank, branch and IFSC, all printed top-left. The drawer's signature
# is bottom-right, is a biometric identifier, and carries nothing an adjudicator
# needs.
#
# It cannot be found the usual ways. It has no OCR tokens to anchor to, and the
# ink channel is useless here because a cheque's cyan security guilloche pushes
# the chromatic mask to 38% of the page, drowning the signature among background
# artefacts. So this is deliberately LAYOUT-based: cheques are a standardised
# instrument, and on the measured example the signature sits at x 81-89%,
# y 58-70% of the cheque's own rect.
# An Indian cheque is ~8.0 x 3.67 in, so ~2.2:1. Measured on this corpus the
# real cheque is 2.24 while the wide Aadhaar-card strips that also matched a
# loose filter are 2.9-3.7, so the band is deliberately tight.
CHEQUE_ASPECT = (1.95, 2.65)
CHEQUE_MIN_PX = 600
SIG_ZONE = (0.62, 0.45, 1.0, 0.92)      # x0, y0, x1, y1 as fractions


def cheque_rects(page):
    """Placement rects of images shaped like a cheque."""
    import fitz
    out = []
    for im in page.get_images(full=True):
        w, h = im[2], im[3]
        if w < CHEQUE_MIN_PX or not (CHEQUE_ASPECT[0] <= w / max(h, 1)
                                     <= CHEQUE_ASPECT[1]):
            continue
        try:
            out.extend(page.get_image_rects(im[0]))
        except Exception:
            pass
    return out


def cheque_signature_zones(page):
    """-> [fitz.Rect] covering the signature area of each cheque on the page.

    Generous by design: the zone is wider than any one signature, because the
    alternative is leaving a biometric identifier in a document that exists to
    prove a bank account.
    """
    import fitz
    out = []
    for r in cheque_rects(page):
        x0 = r.x0 + SIG_ZONE[0] * r.width
        y0 = r.y0 + SIG_ZONE[1] * r.height
        x1 = r.x0 + SIG_ZONE[2] * r.width
        y1 = r.y0 + SIG_ZONE[3] * r.height
        z = fitz.Rect(x0, y0, x1, y1) & page.rect
        if z.width > 8 and z.height > 8:
            out.append(z)
    return out
