#!/usr/bin/env python3
"""Turn a taxonomy action into pixels.

The first build had exactly one path -- fill the box black -- so every
PSEUDONYMIZE, GENERALIZE and MASK_PARTIAL row in a pack would have silently
degraded to REDACT and the taxonomy's whole utility argument would have been
false while appearing to work. This module is that missing vocabulary.

OWNER DECISION 2026-09-23: NOTHING IS EVER WRITTEN INTO THE DOCUMENT.

An earlier build wrote pseudonyms and coarsened values back into the vacated
boxes. Measured on one 16-page run, that produced three distinct failures:
  * a GENERALIZE row whose coarsener could not handle the value returned the
    value unchanged, so '28 years 8 mons 27 days' was destroyed and then
    RE-WRITTEN into the page -- the pipeline removing PII and restoring it;
  * entity resolution split one clinician into [DR_1] and [DR_2], so the
    redacted file ASSERTED two doctors where there was one, with nothing in it
    to reveal the error;
  * internal placeholder strings ('[Town field]', '[handwritten value]')
    escaped into the deliverable as though they were document content.
A redacted document that contains text the original never had is making claims
of its own, and every such claim is a new way to be wrong.

So every destructive action is now the same pixel operation -- opaque black --
and the actions differ only in the audit record and the sidecar. The taxonomy
still distinguishes them, because what a value IS still governs reporting, the
token map and any future policy; it simply no longer changes what the reader
sees. Nothing is added to the page, ever.

    REDACT / MASK_PARTIAL / PSEUDONYMIZE / GENERALIZE   opaque black box
    KEEP                                                no annotation at all

    python3 document_pipeline/actions.py --self-test
"""
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))

BLACK = (0, 0, 0)
PAD = 1.0

DESTRUCTIVE = {"REDACT", "MASK_PARTIAL", "PSEUDONYMIZE", "GENERALIZE"}


# ---------- generalizers: value -> coarsened value ----------
_PIN = re.compile(r"\b(\d{3})\d{3}\b")
_AGE = re.compile(r"\b(\d{1,3})\s*(?:y|yr|yrs|year|years)?\b", re.I)
OCCUPATION_CLASS = [
    (r"doctor|physician|surgeon|nurse|medical|dentist|pharmac", "health professional"),
    (r"engineer|developer|software|technician|architect", "technical professional"),
    (r"teacher|professor|lecturer|tutor", "education professional"),
    (r"business|self\s*employ|propriet|merchant|trader|shop", "self-employed"),
    (r"service|employee|staff|clerk|officer|manager|executive", "salaried employee"),
    (r"house\s*(wife|maker)|homemaker", "homemaker"),
    (r"student", "student"),
    (r"retire|pension", "retired"),
    (r"farmer|agricult|cultivat", "agriculture"),
]


_SUB_YEAR = re.compile(r"\b(month|mon|mth|week|wk|day)s?\b", re.I)


def age_band(value):
    v = str(value)
    # An age recorded in months, weeks or days is an infant's, and the exact
    # figure is clinically load-bearing (dosing, growth percentile, neonatal
    # windows). Banding it to '15-19' because the number reads 18 would be
    # both wrong and destructive.
    if _SUB_YEAR.search(v):
        return v.strip()
    m = _AGE.search(v)
    if not m:
        return None
    a = int(m.group(1))
    if a >= 90:
        return "90+"
    if a < 2:
        return v.strip()
    lo = (a // 5) * 5
    return f"{lo}-{lo + 4}"


def generalize(value, categories=()):
    """Coarsen a value for a GENERALIZE row. None means 'cannot coarsen'."""
    v = str(value or "").strip()
    if not v:
        return None
    cats = set(categories or ())
    if "T16" in cats or "PATIENT_GEO_COARSE" in cats:
        if _PIN.search(v):
            return _PIN.sub(lambda m: m.group(1) + "xxx", v)
        return v                        # city/state survive verbatim per the pack
    if "T18" in cats or "AGE_SEX_RELATIONSHIP" in cats:
        return age_band(v) or v
    if "T19" in cats or "OCCUPATION_AND_EMPLOYER" in cats:
        low = v.lower()
        for pat, label in OCCUPATION_CLASS:
            if re.search(pat, low):
                return label
        return "occupation withheld"
    # unknown GENERALIZE row: coarsening is undefined, so fail to the safe side
    return None


_ALNUM = re.compile(r"[A-Za-z0-9]+")


def added_text(before, after):
    """Alphanumeric runs present in `after` that are not in `before`.

    The invariant for a redaction-only pipeline: a redacted page can LOSE
    content and can lose punctuation around what it lost, but it can never gain
    a character sequence. Compared on alphanumeric runs so that 'Balance:'
    becoming 'Balance' when the figure beside it dies is correctly seen as loss,
    not as an addition.
    """
    src = " ".join(_ALNUM.findall(str(before or "")))
    return [t for t in _ALNUM.findall(str(after or "")) if t not in src]


def _page_words(page):
    """Word rects for this page, cached on the page object.

    apply_action runs once per box and a 45-page bundle carries ~1000 of them,
    so re-extracting the word list each time would dominate the run.
    """
    try:
        cached = getattr(page, "_pii_word_rects", None)
        if cached is not None:
            return cached
    except Exception:
        pass
    rects = [fitz.Rect(w[:4]) for w in page.get_text("words")]
    try:
        page._pii_word_rects = rects
    except Exception:
        pass
    return rects


def _clip_to_neighbours(page, r, target):
    """Never let padding reach a line the box did not already cover.

    apply_redactions deletes every glyph whose box INTERSECTS the rect, with no
    minimum overlap -- so a rectangle reaching 0.5pt into the line above
    deletes that whole line. Measured: a phantom `City` field on a court filing
    sat at y 284.3-299.3 while the line above ended at y 284.8, and the 0.5pt
    touch destroyed the hearing date. Padding is worth keeping (it covers
    antialiasing at the box edge) but it must stop at the line boundary.

    `target` is the UNPADDED rect: any glyph it already overlaps is what we
    meant to redact, and is never used to clip.
    """
    top, bot = r.y0, r.y1
    for wr in _page_words(page):
        if wr.x1 <= r.x0 or wr.x0 >= r.x1:
            continue                                  # not beneath the box
        if wr.y1 > target.y0 and wr.y0 < target.y1:
            continue                                  # a glyph we are aiming at
        if wr.y1 <= target.y0:
            top = max(top, min(wr.y1 + 0.05, target.y0))      # line above
        elif wr.y0 >= target.y1:
            bot = min(bot, max(wr.y0 - 0.05, target.y1))      # line below
    if bot - top < 0.5:
        return fitz.Rect(r.x0, target.y0, r.x1, target.y1)
    return fitz.Rect(r.x0, top, r.x1, bot)


def apply_action(page, rect, action, *, text=None, pad=PAD, cosmetic=False):
    """Annotate one box. -> dict describing what was done (for the audit log).

    `text` is accepted and IGNORED. It stays in the signature so callers that
    still compute a token keep working, and so the audit can record what WOULD
    have been written -- but nothing reaches the page. See the module docstring.

    Callers apply_redactions() once per page after annotating; that is what
    actually destroys content.
    """
    r = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    r = _clip_to_neighbours(page, r, fitz.Rect(rect))
    r = r & page.rect
    info = {"action": action, "wrote": None, "degraded": None,
            "suppressed_text": text or None}
    if action == "KEEP" or r.is_empty:
        info["action"] = "KEEP"
        return info

    if cosmetic:                        # negative control: cover, destroy nothing
        page.draw_rect(r, color=BLACK, fill=BLACK, overlay=True)
        return info

    if action not in DESTRUCTIVE:
        # an action the code does not implement must not become a silent no-op
        info["degraded"] = f"unknown_action:{action}"
        info["action"] = "REDACT"
    page.add_redact_annot(r, fill=BLACK)
    return info


def self_test():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Patient: MARTIN GRIEVES", fontsize=11)
    page.insert_text((72, 130), "PIN: 600049   Age: 28 years", fontsize=11)
    page.insert_text((72, 160), "Balance: 12345.67", fontsize=11)

    assert generalize("600049", ["T16"]) == "600xxx"
    assert generalize("28 years", ["T18"]) == "25-29"
    assert generalize("18 months", ["T18"]) == "18 months", "infant age coarsened"
    assert generalize("95", ["T18"]) == "90+"
    assert generalize("Software Engineer", ["T19"]) == "technical professional"
    assert generalize("value", ["T99_UNKNOWN"]) is None, "unknown row must fail safe"
    print("  generalizers OK")

    before_text = page.get_text()
    i1 = apply_action(page, fitz.Rect(115, 90, 215, 104), "PSEUDONYMIZE",
                      text="[PATIENT_1]")
    i2 = apply_action(page, fitz.Rect(96, 120, 137, 134), "GENERALIZE",
                      text=generalize("600049", ["T16"]))
    i3 = apply_action(page, fitz.Rect(115, 150, 190, 164), "MASK_PARTIAL")
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS,
                          text=fitz.PDF_REDACT_TEXT_REMOVE)
    got = page.get_text()
    print("  page text after:", repr(" ".join(got.split())))
    assert "MARTIN" not in got, "pseudonymised name survived"
    assert "600049" not in got, "generalised value survived"
    assert "12345.67" not in got, "masked value survived"
    # THE INVARIANT: the page gained nothing. Every token in the output was
    # already in the input.
    added = added_text(before_text, got)
    assert not added, f"text was added to the document: {added}"
    assert not any(i["wrote"] for i in (i1, i2, i3)), "something was written"
    assert i1["suppressed_text"] == "[PATIENT_1]", "audit lost what was suppressed"
    print("  nothing added to the page (checked token-by-token)")
    doc.close()
    # padding must not reach the neighbouring line
    d2 = fitz.open(); pg2 = d2.new_page()
    pg2.insert_text((72, 100), "Hearing listed for 04 August 2026", fontsize=10)
    pg2.insert_text((72, 113), "Before the City Civil Judge", fontsize=10)
    tgt = None
    for w in pg2.get_text("words"):
        if w[4] == "City":
            tgt = fitz.Rect(w[:4])
    apply_action(pg2, tgt, "REDACT")
    pg2.apply_redactions()
    after2 = pg2.get_text()
    assert "04 August 2026" in after2, \
        f"padding destroyed the line above: {after2!r}"
    assert "City" not in after2, "the target survived"
    print("  padding clipped at the line boundary "
          "(line above intact, target gone)")
    d2.close()
    print("actions self-test OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
