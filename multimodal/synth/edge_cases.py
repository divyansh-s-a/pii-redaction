#!/usr/bin/env python3
"""Documents that exercise the paths real data does not reach.

The insurance corpus is 24 bundles of one genre from one country. Whole
mechanisms in the pipeline have therefore never met the input they were written
for, and several of them fail SILENTLY rather than loudly -- a fallback that
returns nothing reads exactly like a clean page.

Each case here is a document plus the assertion it exists to check. They are
deliberately small and ugly; realism is `make_corpus.py`'s job, coverage is this
one's.

    python3 synth/edge_cases.py --out out/edge
"""
import argparse
import json
import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "document_pipeline"))

NAME = "Harriet Vance"
MRN = "MRN-88213-04"
BLUE = (0.12, 0.18, 0.62)
BLACK = (0.05, 0.05, 0.05)


def _pen(page, x, y, w=90, colour=BLUE, seed=3):
    """A pen-like squiggle: sparse sweeping strokes, not glyphs."""
    rng = np.random.default_rng(seed)
    pts = [(x + i * w / 12.0, y + float(rng.normal(0, 3.2))) for i in range(13)]
    for a, b in zip(pts, pts[1:]):
        page.draw_line(fitz.Point(*a), fitz.Point(*b), color=colour, width=1.3)


def _hdr(page, title):
    page.insert_text((60, 60), title, fontsize=13, fontname="hebo")
    page.draw_line(fitz.Point(60, 68), fitz.Point(535, 68))


# --------------------------------------------------------------------- cases
def rotated(doc, rot):
    pg = doc.new_page(width=595, height=842)
    _hdr(pg, f"ROTATED {rot} DEGREES")
    pg.insert_text((60, 120), "Patient name:", fontsize=11, fontname="hebo")
    pg.insert_text((170, 120), NAME, fontsize=11)
    pg.insert_text((60, 145), "MRN:", fontsize=11, fontname="hebo")
    pg.insert_text((170, 145), MRN, fontsize=11)
    pg.set_rotation(rot)
    return {"expect": "name and MRN still located and removed despite rotation",
            "values": [NAME, MRN]}


def invisible_ocr(doc):
    """A scan with a hidden OCR text layer under it -- BOTH must be destroyed.

    The corpus has 26 such pages. Redacting only the pixels leaves the value
    fully extractable; redacting only the text leaves it fully readable.
    """
    tmp = fitz.open()
    t = tmp.new_page(width=595, height=842)
    _hdr(t, "SCANNED WITH HIDDEN TEXT LAYER")
    t.insert_text((60, 120), f"Patient name: {NAME}", fontsize=11)
    t.insert_text((60, 145), f"MRN: {MRN}", fontsize=11)
    pm = t.get_pixmap(dpi=110)
    tmp.close()
    pg = doc.new_page(width=595, height=842)
    pg.insert_image(pg.rect, pixmap=pm)
    # render_mode 3 = invisible: what every OCR-over-scan PDF does
    pg.insert_text((60, 120), f"Patient name: {NAME}", fontsize=11,
                   render_mode=3)
    pg.insert_text((60, 145), f"MRN: {MRN}", fontsize=11, render_mode=3)
    return {"expect": "hidden text layer AND pixels both destroyed",
            "values": [NAME, MRN]}


def bitonal_scan(doc):
    """Greyscale/bitonal: the chromatic ink channel is NOT APPLICABLE here, so
    the label fallback must take over rather than reporting 'no ink'."""
    tmp = fitz.open()
    t = tmp.new_page(width=595, height=842)
    _hdr(t, "BITONAL FAX SCAN")
    t.insert_text((60, 120), "Patient name:", fontsize=11, fontname="hebo")
    t.insert_text((60, 145), "MRN:", fontsize=11, fontname="hebo")
    _pen(t, 170, 118, colour=BLACK, seed=5)
    _pen(t, 170, 143, colour=BLACK, seed=9)
    pm = t.get_pixmap(dpi=100, colorspace=fitz.csGRAY)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
    bw = np.where(a < 160, 0, 255).astype(np.uint8)
    rgb = np.repeat(bw[:, :, None], 3, axis=2)
    pix = fitz.Pixmap(fitz.csRGB, pm.w, pm.h, bytearray(rgb.tobytes()), False)
    tmp.close()
    pg = doc.new_page(width=595, height=842)
    pg.insert_image(pg.rect, pixmap=pix)
    return {"expect": "black-pen values on a bitonal page fall back to "
                      "label-anchored regions, not silently to nothing",
            "values": []}


def black_pen(doc):
    pg = doc.new_page(width=595, height=842)
    _hdr(pg, "BLACK BALLPOINT ON A FORM")
    for i, lab in enumerate(("Patient name:", "MRN:", "Signature:")):
        pg.insert_text((60, 120 + i * 30), lab, fontsize=11, fontname="hebo")
        _pen(pg, 170, 118 + i * 30, colour=BLACK, seed=i + 1)
    return {"expect": "no chromatic signal; label fallback must still fire",
            "values": []}


def empty_page(doc):
    doc.new_page(width=595, height=842)
    return {"expect": "an empty page produces no boxes and does not crash",
            "values": []}


def blank_form(doc):
    """Printed labels, nothing filled in. Redacting here removes nothing and
    destroys scaffolding -- a real CKYC form was 36% black for zero PII."""
    pg = doc.new_page(width=595, height=842)
    _hdr(pg, "UNFILLED APPLICATION FORM")
    for i, lab in enumerate(("Full name", "Address", "Date of birth",
                             "Phone no", "PAN", "Account number")):
        y = 110 + i * 26
        pg.insert_text((60, y), lab + ":", fontsize=10, fontname="hebo")
        pg.draw_rect(fitz.Rect(180, y - 11, 520, y + 4), color=(0.6, 0.6, 0.6),
                     width=0.6)
    return {"expect": "ZERO boxes: every field is empty",
            "values": [], "max_boxes": 0}


def guidance_page(doc):
    pg = doc.new_page(width=595, height=842)
    _hdr(pg, "GUIDANCE FOR FILLING CLAIM FORM")
    rows = [("Policy No.", "Enter the policy number as printed on the schedule"),
            ("Name", "Enter the full name of the primary insured"),
            ("PAN", "Enter the PAN of the account holder in block letters"),
            ("Account Number", "Enter the account number without spaces")]
    pg.insert_text((60, 100), "DATA ELEMENT", fontsize=10, fontname="hebo")
    pg.insert_text((220, 100), "DESCRIPTION", fontsize=10, fontname="hebo")
    for i, (a, b) in enumerate(rows):
        y = 125 + i * 24
        pg.insert_text((60, y), a, fontsize=10)
        pg.insert_text((220, y), b, fontsize=9)
    return {"expect": "ZERO boxes: an instruction sheet holds no patient data",
            "values": [], "max_boxes": 0}


def numeric_table(doc):
    """A bill. The figures are the agent's working data and must all survive."""
    pg = doc.new_page(width=595, height=842)
    _hdr(pg, "BILL BREAK-UP")
    pg.insert_text((60, 100), f"Patient: {NAME}", fontsize=10)
    cols = (60, 300, 380, 460)
    for x, h in zip(cols, ("Particulars", "Qty", "Rate", "Amount")):
        pg.insert_text((x, 130), h, fontsize=10, fontname="hebo")
    amounts = []
    for i, (d, q, r, amt) in enumerate([("Surgeon", "1", "12000", "12000"),
                                        ("GST", "-", "18%", "2160"),
                                        ("Patient gown", "2", "75", "150"),
                                        ("Package discount", "-", "-", "-500"),
                                        ("Other", "1", "450", "450")]):
        y = 152 + i * 20
        for x, v in zip(cols, (d, q, r, amt)):
            pg.insert_text((x, y), v, fontsize=10)
        amounts.append(amt)
    pg.insert_text((380, 152 + 5 * 20 + 6), "TOTAL", fontsize=10, fontname="hebo")
    pg.insert_text((460, 152 + 5 * 20 + 6), "14260", fontsize=10, fontname="hebo")
    # A pack contract (every bill particular is the agent's data), so the case
    # names its pack: under generic_document a billed health service is G21
    # and is removed by design; regress.py reads this key.
    return {"expect": "the patient name goes; EVERY figure and particular stays",
            "taxonomy": "insurance_medical",
            "values": [NAME], "must_survive": amounts + ["14260", "Surgeon",
                                                         "GST", "Patient gown",
                                                         "Package discount"]}


def watermark(doc):
    pg = doc.new_page(width=595, height=842)
    _hdr(pg, "PAGE WITH DIAGONAL WATERMARK")
    pg.insert_text((60, 120), f"MRN: {MRN}", fontsize=11)
    # a true diagonal: insert_text's rotate= only accepts 0/90/180/270, so the
    # 45-degree sweep of a real watermark needs a morph matrix about a pivot
    pivot = fitz.Point(150, 500)
    pg.insert_text(pivot, MRN, fontsize=54, color=(0.86, 0.86, 0.86),
                   morph=(pivot, fitz.Matrix(45)))
    return {"expect": "KNOWN GAP -- the inline MRN is removed, the watermark "
                      "is not; this case exists to keep that measured",
            "values": [MRN], "known_gap": True}


def tall_page(doc):
    pg = doc.new_page(width=595, height=3400)
    _hdr(pg, "VERY TALL PAGE")
    for i in range(12):
        pg.insert_text((60, 120 + i * 260), f"Patient name: {NAME}", fontsize=11)
    return {"expect": "all 12 repetitions removed on an unusually tall page",
            "values": [NAME]}


CASES = {
    "rotated_90": lambda d: rotated(d, 90),
    "rotated_270": lambda d: rotated(d, 270),
    "invisible_ocr": invisible_ocr,
    "bitonal_scan": bitonal_scan,
    "black_pen": black_pen,
    "empty_page": empty_page,
    "blank_form": blank_form,
    "guidance_page": guidance_page,
    "numeric_table": numeric_table,
    "watermark": watermark,
    "tall_page": tall_page,
}


def build(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, fn in CASES.items():
        doc = fitz.open()
        meta = fn(doc)
        path = out / f"{name}.pdf"
        doc.save(path, garbage=4, deflate=True)
        doc.close()
        manifest.append({"case": name, "file": path.name, **meta})
        print(f"  {name:<18}{path.name:<22}{meta['expect'][:64]}")
    # a text file and an image, to exercise non-PDF modalities
    (out / "notes.txt").write_text(
        f"Handover note\nPatient name: {NAME}\nMRN: {MRN}\n"
        "Discussed discharge planning with the ward team.\n")
    tmp = fitz.open()
    t = tmp.new_page(width=400, height=250)
    t.insert_text((30, 60), f"Patient name: {NAME}", fontsize=12)
    t.insert_text((30, 90), f"MRN: {MRN}", fontsize=12)
    t.get_pixmap(dpi=120).save(out / "scan.png")
    tmp.close()
    manifest.append({"case": "text_file", "file": "notes.txt",
                     "expect": "routes to the document arm as modality=text",
                     "values": [NAME, MRN]})
    manifest.append({"case": "standalone_image", "file": "scan.png",
                     "expect": "routes as modality=image", "values": [NAME, MRN]})
    print(f"  {'text_file':<18}notes.txt")
    print(f"  {'standalone_image':<18}scan.png")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} edge cases -> {out}/manifest.json")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/edge")
    build(ap.parse_args().out)
