#!/usr/bin/env python3
"""Synthesize non-insurance documents WITH ground truth.

The pipeline is validated on one domain. `generic_document` -- the pack that
catches everything else -- has never run on a document it was written for, so
nothing is known about it beyond "it redacts more than insurance_medical".

Synthetic data is the only way to fix that honestly, because we place every
value and therefore know its category and its exact box. That turns recall and
precision from judge-estimated into TRUE, and lets us manufacture the cases
the insurance corpus simply does not contain.

Two things keep it from being a toy:

* identifiers PASS their real checksums, generated through the SAME validators
  the pipeline uses, so the checksum enforcement path genuinely fires. In the
  real corpus only 10% of Aadhaar-shaped and 11% of card-shaped strings
  validate -- synthetic IDs that fail would make that path's recall fiction.

* pages are DEGRADED to match the real corpus, which is 79% image-only scans
  at a median 109 dpi with 42% below 100. A crisp born-digital PDF passes every
  test and proves nothing. Each document therefore ships in both renditions:
  born-digital, and the same page rasterised, downsampled, JPEG-crushed and
  noised.

Real QR and portrait rasters are lifted from the insurance corpus (owner's
suggestion) so the visual detectors meet genuine artifacts.

    python3 synth/make_corpus.py --out out/synth
"""
import argparse
import json
import random
import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "document_pipeline"))
import objective

FONT = "helv"
BOLD = "hebo"


# ---------------------------------------------------------------- identifiers
def _with_checksum(prefix_digits, validator, length):
    """Brute-force the final character until the pipeline's OWN validator
    accepts it -- generator and checker can then never drift apart."""
    for tail in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        cand = prefix_digits + tail
        if len(cand) == length and validator(cand):
            return cand
    return None


def make_aadhaar(rng):
    for _ in range(200):
        base = "".join(str(rng.randint(0, 9)) for _ in range(11))
        v = _with_checksum(base, objective.verhoeff_ok, 12)
        if v:
            return v
    return None


def make_card(rng):
    for _ in range(200):
        base = "4" + "".join(str(rng.randint(0, 9)) for _ in range(14))
        v = _with_checksum(base, objective.luhn_ok, 16)
        if v:
            return v
    return None


def make_pan(rng):
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ("".join(rng.choice(L) for _ in range(5))
            + "".join(str(rng.randint(0, 9)) for _ in range(4))
            + rng.choice(L))


FIRST = ["Arjun", "Meera", "Daniel", "Priya", "Thomas", "Anita", "Rohan",
         "Elena", "Marcus", "Sofia", "Kiran", "Laura"]
LAST = ["Whitfield", "Okonkwo", "Berenson", "Castellanos", "Nair", "Halvorsen",
         "Duarte", "Oyelaran", "Marchetti", "Sandoval"]
STREET = ["14 Copperfield Lane", "88 Harbour Road", "3 Wren Close",
          "220 Mill Street", "7 Alderney Court"]
CITY = [("Bristol", "BS1 4TR"), ("Leeds", "LS2 8QJ"), ("Dundee", "DD1 4HN"),
        ("Galway", "H91 XK54"), ("Cork", "T12 YN60")]
COMPANY = ["Nordvale Logistics Ltd", "Brightmoor Analytics", "Kestrel Foods plc",
           "Ashgrove Engineering", "Ravenhill Media Group"]
ROLE = ["Operations Analyst", "Site Supervisor", "Account Manager",
        "Quality Engineer", "Logistics Coordinator"]


def person(rng):
    city, post = rng.choice(CITY)
    return {
        "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "street": rng.choice(STREET), "city": city, "postcode": post,
        "email": None, "phone": f"07{rng.randint(100,999)} {rng.randint(100000,999999)}",
        "dob": f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1960,2000)}",
        "aadhaar": make_aadhaar(rng), "card": make_card(rng), "pan": make_pan(rng),
        "employee_id": f"EMP{rng.randint(10000,99999)}",
        "account": str(rng.randint(10**11, 10**12 - 1)),
        "salary": f"{rng.randint(28,85)},{rng.randint(0,999):03d}",
        "company": rng.choice(COMPANY), "role": rng.choice(ROLE),
    }


# ------------------------------------------------------------------ templates
def _put(page, gt, x, y, text, *, size=10, bold=False, cat=None, label=None):
    """Draw text; if it is PII, record its TRUE box by searching for it back."""
    page.insert_text((x, y), text, fontsize=size,
                     fontname=BOLD if bold else FONT)
    if cat:
        hits = page.search_for(text)
        box = None
        for h in hits:
            if abs(h.y1 - y) < size * 1.2 and abs(h.x0 - x) < 4:
                box = h
                break
        gt.append({"value": text, "category": cat, "label": label,
                   "page": page.number + 1,
                   "bbox": [round(v, 2) for v in (box or fitz.Rect(x, y - size, x + 80, y + 2))]})


def offer_letter(doc, p, gt):
    pg = doc.new_page(width=595, height=842)
    _put(pg, gt, 60, 70, p["company"], size=15, bold=True)
    _put(pg, gt, 60, 88, "Human Resources  |  Offer of Employment", size=9)
    pg.draw_line(fitz.Point(60, 98), fitz.Point(535, 98))
    y = 130
    for lab, val, cat in [
            ("Candidate name:", p["name"], "PERSON_NAME"),
            ("Address:", p["street"], "CONTACT_CHANNEL"),
            ("City / Postcode:", f"{p['city']} {p['postcode']}", "CONTACT_CHANNEL"),
            ("Phone no:", p["phone"], "CONTACT_CHANNEL"),
            ("Date of birth:", p["dob"], "DATE_OF_BIRTH"),
            ("Employee id:", p["employee_id"], "PERSONAL_IDENTIFIER"),
            ("PAN:", p["pan"], "PERSONAL_IDENTIFIER"),
            ("Position:", p["role"], None),
            ("Annual salary:", f"GBP {p['salary']}", None),
            ("Start date:", "02/11/2026", None)]:
        _put(pg, gt, 60, y, lab, size=10, bold=True)
        _put(pg, gt, 190, y, val, size=10, cat=cat, label=lab.rstrip(":"))
        y += 22
    _put(pg, gt, 60, y + 20, "This offer is subject to satisfactory references and", size=9)
    _put(pg, gt, 60, y + 34, "proof of right to work. Please sign and return.", size=9)
    return pg


def bank_statement(doc, p, gt):
    pg = doc.new_page(width=595, height=842)
    _put(pg, gt, 60, 70, "MERIDIAN BUILDING SOCIETY", size=14, bold=True)
    _put(pg, gt, 60, 86, "Branch: Fenchurch   Sort code: 40-22-18   BIC: MERIGB2L", size=9)
    _put(pg, gt, 60, 120, "Account holder:", size=10, bold=True)
    _put(pg, gt, 170, 120, p["name"], size=10, cat="PERSON_NAME",
         label="Account holder")
    _put(pg, gt, 60, 138, "Account number:", size=10, bold=True)
    _put(pg, gt, 170, 138, p["account"], size=10, cat="PERSONAL_IDENTIFIER",
         label="Account number")
    _put(pg, gt, 60, 156, "Card number:", size=10, bold=True)
    _put(pg, gt, 170, 156, p["card"], size=10, cat="PERSONAL_IDENTIFIER",
         label="Card number")
    y = 200
    for h, x in (("Date", 60), ("Description", 140), ("Out", 360), ("In", 430),
                 ("Balance", 500)):
        _put(pg, gt, x, y, h, size=10, bold=True)
    pg.draw_line(fitz.Point(60, y + 4), fitz.Point(545, y + 4))
    y += 20
    bal = 4210.55
    for dsc, out_, in_ in [("Card payment - Tesco", "42.10", ""),
                           ("Salary - " + p["company"], "", "2,480.00"),
                           ("Direct debit - Council tax", "168.00", ""),
                           ("Transfer to savings", "500.00", ""),
                           ("Refund - Rail ticket", "", "31.40")]:
        bal += (float(in_.replace(",", "")) if in_ else 0) - (float(out_) if out_ else 0)
        _put(pg, gt, 60, y, "0%d/10/2026" % (y % 9 + 1), size=9)
        _put(pg, gt, 140, y, dsc, size=9)
        _put(pg, gt, 360, y, out_, size=9)
        _put(pg, gt, 430, y, in_, size=9)
        _put(pg, gt, 500, y, f"{bal:,.2f}", size=9)
        y += 18
    return pg


def id_card_page(doc, p, gt, assets):
    """An employee ID card: portrait + QR, both REAL rasters from the corpus."""
    pg = doc.new_page(width=595, height=842)
    card = fitz.Rect(90, 120, 505, 400)
    pg.draw_rect(card, color=(0.2, 0.2, 0.2), width=1.2)
    _put(pg, gt, 110, 155, p["company"], size=13, bold=True)
    _put(pg, gt, 110, 172, "EMPLOYEE IDENTITY CARD", size=9)
    photo = fitz.Rect(110, 190, 190, 290)
    if assets.get("photo"):
        pg.insert_image(photo, filename=str(assets["photo"]))
        gt.append({"value": "[portrait]", "category": "BIOMETRIC_OR_SIGNATURE",
                   "label": "photo", "page": pg.number + 1,
                   "bbox": [round(v, 2) for v in photo]})
    _put(pg, gt, 210, 205, "Name:", size=9, bold=True)
    _put(pg, gt, 265, 205, p["name"], size=10, cat="PERSON_NAME", label="Name")
    _put(pg, gt, 210, 225, "Emp ID:", size=9, bold=True)
    _put(pg, gt, 265, 225, p["employee_id"], size=10, cat="PERSONAL_IDENTIFIER",
         label="Emp ID")
    _put(pg, gt, 210, 245, "Role:", size=9, bold=True)
    _put(pg, gt, 265, 245, p["role"], size=10)
    _put(pg, gt, 210, 265, "Nat. ID:", size=9, bold=True)
    _put(pg, gt, 265, 265, p["aadhaar"], size=10, cat="PERSONAL_IDENTIFIER",
         label="Nat. ID")
    qr = fitz.Rect(400, 280, 480, 360)
    if assets.get("qr"):
        pg.insert_image(qr, filename=str(assets["qr"]))
        gt.append({"value": "[QR]", "category": "MACHINE_READABLE_CODE",
                   "label": "qr", "page": pg.number + 1,
                   "bbox": [round(v, 2) for v in qr]})
    _put(pg, gt, 110, 330, "Valid to 31/12/2027. Property of the issuer.", size=8)
    return pg


# ----------------------------------------------------------------- degradation
def scanned_copy(doc, rng):
    """Build a NEW document that is `doc` rasterised, downsampled and noised.

    A fresh document rather than page-by-page mutation of the original: the
    delete-and-reinsert loop churned the shared object graph and left one
    three-page file whose every page failed to render ("invalid ICC
    colorspace"), so OCR read ZERO tokens and it silently scored 0/15. That
    would have looked like a pipeline failure instead of a generator bug.

    Targets the real corpus rather than a guess: median 109 dpi, p25 72, 42%
    below 100. A crisp synthetic page exercises none of the machinery that
    matters.
    -> (new_doc, [dpi per page])
    """
    out = fitz.open()
    dpis = []
    for pno in range(doc.page_count):
        dpi = rng.choice([72, 85, 100, 109, 120, 150])
        dpis.append(dpi)
        src = doc[pno]
        rect = fitz.Rect(src.rect)
        pm = src.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)
        gen = np.random.default_rng(rng.randint(0, 10 ** 6))
        noisy = np.clip(a.astype(np.int16)
                        + rng.randint(4, 12) * gen.standard_normal(a.shape),
                        0, 255).astype(np.uint8)
        pix = fitz.Pixmap(fitz.csRGB, pm.w, pm.h, bytearray(noisy.tobytes()),
                          False)
        page = out.new_page(width=rect.width, height=rect.height)
        page.insert_image(rect, pixmap=pix)
    return out, dpis


# ------------------------------------------------------------------------ main
def build(out_dir, seed=7):
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets = {}
    for k, f in (("qr", "qr.png"), ("photo", "portrait.png")):
        p = HERE / "assets" / f
        if p.exists():
            assets[k] = p

    manifest = []
    for i, (name, builders) in enumerate([
            ("hr_offer", [offer_letter, id_card_page]),
            ("bank_file", [bank_statement, offer_letter]),
            ("mixed_employment", [offer_letter, bank_statement, id_card_page])]):
        doc = fitz.open()
        gt = []
        p = person(rng)
        for b in builders:
            if b is id_card_page:
                b(doc, p, gt, assets)
            else:
                b(doc, p, gt)
        # one rendition born-digital, one degraded to a scan
        crisp = out / f"{name}_digital.pdf"
        doc.save(crisp, garbage=4, deflate=True)
        sdoc, dpis = scanned_copy(doc, rng)
        scanned = out / f"{name}_scanned.pdf"
        sdoc.save(scanned, garbage=4, deflate=True)
        sdoc.close()
        doc.close()
        for variant, path, d in (("digital", crisp, None), ("scanned", scanned, dpis)):
            manifest.append({"file": path.name, "variant": variant,
                             "dpi": d, "person": p["name"],
                             "items": gt})
        print(f"  {name:<20}{crisp.name:<28}{scanned.name:<28}"
              f"{len(gt):>3} ground-truth items  dpi={dpis}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n = sum(len(m["items"]) for m in manifest)
    print(f"\n{len(manifest)} PDFs, {n} ground-truth PII items -> {out}/manifest.json")
    return manifest


def self_test():
    rng = random.Random(1)
    a = make_aadhaar(rng)
    c = make_card(rng)
    assert a and objective.verhoeff_ok(a), f"Aadhaar checksum invalid: {a}"
    assert c and objective.luhn_ok(c), f"card checksum invalid: {c}"
    print(f"  generated Aadhaar {a} (Verhoeff OK), card {c} (Luhn OK)")
    print("synth self-test OK")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/synth")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    build(a.out, a.seed)
