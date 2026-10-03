#!/usr/bin/env python3
"""Bundle-wide value propagation: the same value gets the same treatment on
every page it appears on.

A claim bundle repeats one patient's name, policy number and UHID across
dozens of pages in different renderings -- crisp on a born-digital invoice,
smeared on a fax, handwritten on the form, stamped on a discharge slip. A
per-page detector redacts it where it happens to read it and misses it
elsewhere, which is both a leak (one surviving occurrence undoes every other
redaction of the same value) and the inconsistency that makes a delivered
dataset untrustworthy to annotate.

So: learn the value set from the pages where detection was CONFIDENT, then
hunt those strings across every page, including pages whose own detection pass
found nothing.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
# LAST insert wins position 0. This directory must beat the audio arm's
# pipeline/, which has its own redact.py: in the wrong order a later plain
# `import redact` resolved to the AUDIO redactor and crashed the suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect
import geometry
import matching

# Categories whose values are stable identifiers worth hunting bundle-wide.
# Excluded on purpose: AGE, DATE_OF_BIRTH, FINANCIAL_INFO and anything whose
# surface form is a bare number -- propagating "28" or "1997" across 200 pages
# blacks out page numbers, quantities and dosages.
PROPAGATE = {
    "FULL_NAME", "ADDRESS", "PHONE", "EMAIL", "INSURANCE_ID", "ASSIGNED_ID",
    "GOVERNMENT_ID", "TAX_ID", "BANK_ACCOUNT_NUMBER", "PASSPORT_NUMBER",
    "DRIVERS_LICENSE", "VEHICLE_REGISTRATION", "PAYMENT_CARD_NUMBER",
    "STUDENT_ID", "USERNAME_HANDLE",
}
MIN_LEN = 5
# a value must carry some signal beyond being a word: a digit, or two capitals,
# or a separator. "Female" and "Regular" never become bundle-wide redactions.
INFORMATIVE = re.compile(r"\d|[A-Z].*[A-Z]|[@/#-]")
COMMON = {"male", "female", "regular", "normal", "patient", "hospital",
          "india", "private", "limited", "general", "insurance", "claim",
          "total", "amount", "date", "name", "address", "report", "test"}


def name_categories(pack):
    """Categories a pack marks propagatable AND whose row is about a NAME."""
    if pack is None:
        return {"FULL_NAME"}
    out = {"FULL_NAME"}
    for c in pack.rows():
        if c.get("propagate") and "NAME" in str(c.get("name", "")).upper():
            out.add(c["id"])
            out.add(c["name"])
    return out


def _name_like(text):
    """A capitalised run of tokens, none of which is a dictionary word.

    This is the test that lets a single-word personal name past the
    INFORMATIVE gate without letting a common word through with it. The
    system dictionary is the discriminator formfields already uses for OCR
    repair: 'Zarinabanu', 'Gulnaaz', 'Meharunisa' and 'Khajapeer' are absent from
    it, while 'Female', 'Regular', 'Hospital' and 'Surgeon' -- precisely the
    words the gate exists to block -- are all present.
    """
    import formfields
    toks = [t for t in re.split(r"[\s.,]+", str(text).strip()) if t]
    if not toks:
        return False
    for t in toks:
        if not t[:1].isupper():
            return False
        if t.lower() in formfields.SYSTEM_WORDS:
            return False
    return True


def propagatable(pack):
    """Category ids a pack says are worth hunting bundle-wide.

    Must come from the PACK. The original hard-coded set held AUDIO taxonomy
    names (FULL_NAME, ADDRESS, ...) while packs emit T01..T40, so the
    intersection was empty and this entire mechanism was INERT for every pack
    run -- the bundle-wide consistency guarantee silently did nothing, and the
    only values ever propagated came from the regex pass, which still uses the
    old names.
    """
    if pack is None:
        return set(PROPAGATE)
    ids = {c["id"] for c in pack.rows() if c.get("propagate")}
    names = {c["name"] for c in pack.rows() if c.get("propagate")}
    return ids | names | set(PROPAGATE)


def harvest(page_findings, min_len=MIN_LEN, readers_by_page=None, pack=None):
    """-> {normalised_value: {'text','categories','pages'}} worth propagating.

    Provenance gate: Vision OCR over handwriting yields fragments like
    'S OP N A I' or '86/0898 3 122', and hunting those across a bundle is
    noise at best and a false match at worst. A value is TRUSTED when it was
    read from a real text layer or passed a checksum; an untrusted reading is
    kept only if a second page independently produced the same string, which
    is evidence in itself that the reading is right.
    """
    readers_by_page = readers_by_page or {}
    allowed = propagatable(pack)
    name_cats = name_categories(pack)
    vals = {}
    for pno, finds in page_findings.items():
        reader = readers_by_page.get(pno, "text_layer")
        for f in finds:
            cats = [c for c in (f.get("categories") or []) if c in allowed]
            if not cats:
                continue
            if f.get("source") == "label_anchored" or f.get("region"):
                continue                       # a region has no value string
            txt = (f.get("text") or "").strip()
            if len(txt) < min_len or txt.startswith("["):
                continue
            key = matching.normalize(
                txt, "id" if matching.is_identifier(txt) else "name")
            if key in COMMON:
                continue
            if not INFORMATIVE.search(txt):
                # INFORMATIVE demands a digit, two capitals or a separator,
                # which no single-word personal name has: 'Zarinabanu' carries
                # one capital and nothing else, so it was never learned, while
                # 'K R Hari Prasad,' cleared the gate on its capitals alone.
                # The bundle-wide guarantee therefore did not cover the very
                # values it exists for, and that name kept reappearing on any
                # page the detector happened to miss. A name-shaped token in a
                # NAME category is evidence enough on its own.
                if not (set(cats) & name_cats and _name_like(txt)):
                    continue
            # a checksum-verified id is worth propagating on its own; anything
            # else needs the detector not to have hedged
            if f.get("checksum_ok") is False and f.get("confidence") == "low":
                continue
            trusted = reader == "text_layer" or f.get("checksum_ok") is True
            e = vals.setdefault(key, {"text": txt, "categories": set(),
                                      "pages": set(), "trusted": False})
            e["categories"].update(cats)
            e["pages"].add(pno)
            e["trusted"] = e["trusted"] or trusted
            if len(txt) > len(e["text"]):
                e["text"] = txt                # keep the fullest rendering
    return {k: {"text": v["text"], "categories": sorted(v["categories"]),
                "pages": sorted(v["pages"]), "trusted": v["trusted"]}
            for k, v in vals.items()
            if v["trusted"] or len(v["pages"]) >= 2}


def apply_to_page(page, words, values, existing, action_for):
    """Boxes for every bundle value found on this page that isn't covered."""
    have = []
    for f in existing:
        for b in (f.get("all_boxes") or ([f["bbox"]] if f.get("bbox") else [])):
            have.append(b)
    out = []
    for key, v in values.items():
        try:
            rects = matching.find_runs(words, v["text"])
        except Exception:
            continue
        for r in rects:
            box = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
            if geometry.too_large(box, page.rect):
                continue
            if geometry.covered_by_any(box, have, threshold=0.90):
                continue
            have.append(box)
            out.append({
                "categories": v["categories"], "text": v["text"],
                "bbox": box, "source": "propagated",
                "locator": "bundle_value",
                "action": action_for(v["categories"]),
                "seen_on_pages": v["pages"][:6],
            })
    return out


def self_test():
    pf = {
        1: [{"categories": ["FULL_NAME"], "text": "MARTIN GRIEVES",
             "confidence": "high", "bbox": [0, 0, 1, 1]},
            {"categories": ["AGE"], "text": "28", "bbox": [0, 0, 1, 1]},
            {"categories": ["FULL_NAME"], "text": "Female", "bbox": [0, 0, 1, 1]}],
        7: [{"categories": ["INSURANCE_ID"], "text": "BLR-VI-52045-0011-001",
             "bbox": [0, 0, 1, 1]},
            {"categories": ["FULL_NAME"], "text": "Martin Grieves",
             "bbox": [0, 0, 1, 1]}],
        9: [{"categories": ["ADDRESS"], "text": "[Address field]",
             "source": "label_anchored", "bbox": [0, 0, 1, 1]}],
    }
    pf[1].append({"categories": ["FULL_NAME"], "text": "S OP N A I",
                  "confidence": "medium", "bbox": [0, 0, 1, 1]})
    pf[1].append({"categories": ["PHONE"], "text": "86/0898 3 122",
                  "bbox": [0, 0, 1, 1]})
    readers = {1: "vision_ocr", 7: "text_layer", 9: "vision_ocr"}
    got = harvest(pf, readers_by_page=readers)
    keys = sorted(got)
    print("harvested:", keys)
    K = lambda v: matching.normalize(
        v, "id" if matching.is_identifier(v) else "name")
    name_k, pol_k = K("MARTIN GRIEVES"), K("BLR-VI-52045-0011-001")
    assert name_k in got, "name not harvested"
    assert got[name_k]["pages"] == [1, 7], "pages not merged"
    assert pol_k in got, "policy id not harvested"
    assert not any(k == K("28") for k in got), "bare number propagated"
    assert K("Female") not in got, "common word propagated"
    assert not any(g["text"].startswith("[") for g in got.values()), \
        "label region propagated as a value"
    assert K("S OP N A I") not in got, "handwriting OCR fragment harvested"
    assert K("86/0898 3 122") not in got, "single-page untrusted phone harvested"
    assert got[name_k]["trusted"], "text-layer sighting not trusted"
    # the OCR-tolerant matcher must merge a degraded reading, not split it
    pf2 = {1: [{"categories": ["FULL_NAME"], "text": "MARTIN GRIEVES",
                "confidence": "high", "bbox": [0, 0, 1, 1]}],
           5: [{"categories": ["FULL_NAME"], "text": "MARTlN GRlEVES",
                "bbox": [0, 0, 1, 1]}]}
    merged = harvest(pf2, readers_by_page={1: "text_layer", 5: "vision_ocr"})
    assert len(merged) == 1, "a 109 dpi misread split into a second value"

    # a pack's own ids must be harvestable, or the mechanism is inert
    class _P:
        def rows(self):
            return [{"id": "T01", "name": "PATIENT_NAME", "propagate": True}]
    packed = harvest({1: [{"categories": ["T01"], "text": "Harriet Vance",
                           "confidence": "high", "bbox": [0, 0, 1, 1]},
                          {"categories": ["T01"], "text": "Harriet Vance",
                           "bbox": [0, 0, 1, 1]}]},
                     readers_by_page={1: "text_layer"}, pack=_P())
    assert packed, "pack-declared categories are not harvested"
    print(f"  pack categories harvestable: {sorted(packed)}")
    print("propagate self-test OK "
          f"({len(got)} values, name seen on {got[name_k]['pages']}; "
          "OCR variants merge)")
    return True


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
