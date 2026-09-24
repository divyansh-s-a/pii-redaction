#!/usr/bin/env python3
"""Non-insurance documents that grade BOTH failure directions.

The existing synthetic corpus places PII and measures whether it is removed.
It places nothing that must SURVIVE, so it cannot see over-redaction at all --
and over-redaction, not leakage, is this pipeline's observed failure mode. The
generic pack is 41 rows of which ELEVEN are KEEP; a corpus with no keep-side
ground truth grades 27% of it and calls that a pass.

Each document here therefore carries two lists:

    values        must be ABSENT from the redacted output   (leak if present)
    must_survive  must be PRESENT in the redacted output    (destroyed if not)

and the genres are chosen so the two lists sit next to each other on the page,
which is where the pipeline actually has to make a decision. A letterhead email
one line above a personal email; an IFSC beside an account number; a numeric
column beside a salary; guidance text beside the field it describes.

    python3 synth/generic_cases.py --out out/generic
"""
import argparse
import json
import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "document_pipeline"))

BLUE = (0.12, 0.18, 0.62)


def _pen(page, x, y, w=90, seed=3):
    rng = np.random.default_rng(seed)
    pts = [(x + i * w / 12.0, y + float(rng.normal(0, 3.0))) for i in range(13)]
    for a, b in zip(pts, pts[1:]):
        page.draw_line(fitz.Point(*a), fitz.Point(*b), color=BLUE, width=1.3)


def _put(p, lines, x=60, y=100, size=9.5, lead=13.5, bold=None):
    for i, t in enumerate(lines):
        p.insert_text((x, y + i * lead), t, fontsize=size,
                      fontname="hebo" if bold and i in bold else "helv")
    return y + len(lines) * lead


# ------------------------------------------------------------------- genres
def hr_letter(d):
    """Letterhead vs recipient block -- the G30/G04 contest, side by side."""
    p = d.new_page()
    _put(p, ["Meridian Textiles Private Limited",
             "Plot 44, Industrial Estate, Coimbatore 641021",
             "info@meridiantextiles.example   |   www.meridiantextiles.example",
             "CIN U17111TZ1998PTC008512"], y=60, bold={0})
    p.draw_line(fitz.Point(60, 118), fitz.Point(535, 118))
    _put(p, ["Ananya Krishnan",
             "27 Race Course Road, Gandhipuram",
             "Coimbatore 641018"], y=150)
    _put(p, ["Dear Ananya,", "",
             "We are pleased to confirm your appointment as Production",
             "Supervisor with effect from 01 April 2026.", "",
             "Employee ID        MTX-2291",
             "Date of Birth      14 March 1996",
             "Gross Salary       Rs 68,400 per month",
             "Bank Account       50100294412876",
             "IFSC               HDFC0000523",
             "Personal Email     ananya.krishnan@gmail.example",
             "Mobile             9845012233", "",
             "Please sign and return the enclosed copy.", "",
             "For Meridian Textiles Private Limited"], y=210)
    _pen(p, 60, 445, seed=11)
    _put(p, ["Authorised Signatory"], y=470)
    return {
        "expect": "issuer letterhead survives; recipient block, IDs, salary, "
                  "personal contact and DOB go; IFSC survives beside the "
                  "account number it sits under",
        "values": ["Ananya Krishnan", "MTX-2291", "50100294412876",
                   "ananya.krishnan@gmail.example", "9845012233",
                   "27 Race Course Road", "68,400"],
        "must_survive": ["Meridian Textiles", "info@meridiantextiles.example",
                         "HDFC0000523", "U17111TZ1998PTC008512",
                         "Production Supervisor", "01 April 2026"],
    }


def bank_statement(d):
    """A numeric column that must survive, beside values that must not."""
    p = d.new_page()
    _put(p, ["Canara Bank", "Jayanagar 4th Block Branch, Bengaluru",
             "IFSC CNRB0001234    MICR 560015012"], y=60, bold={0})
    _put(p, ["Statement of Account", "",
             "Account Holder   Rohit Deshpande",
             "Account Number   0912101034567",
             "Customer ID      884512290",
             "Address          12 Shanti Nagar, Bengaluru 560027",
             "PAN              AAAPD1234C",
             "Closing Balance  Rs 2,41,882.50"], y=120, bold={0})
    rows = [("01-02-2026", "NEFT transfer", "12,500.00", "2,29,382.50"),
            ("04-02-2026", "Card purchase", "3,240.00", "2,26,142.50"),
            ("09-02-2026", "Salary credit", "68,400.00", "2,94,542.50"),
            ("15-02-2026", "Utility payment", "1,860.00", "2,92,682.50"),
            ("22-02-2026", "Cash withdrawal", "50,800.00", "2,41,882.50")]
    y = 270
    _put(p, ["Date          Particulars            Amount       Balance"],
         y=y, bold={0})
    for i, r in enumerate(rows):
        p.insert_text((60, y + 20 + i * 15), r[0], fontsize=9)
        p.insert_text((150, y + 20 + i * 15), r[1], fontsize=9)
        p.insert_text((300, y + 20 + i * 15), r[2], fontsize=9)
        p.insert_text((400, y + 20 + i * 15), r[3], fontsize=9)
    return {
        "expect": "per-customer values go; branch infrastructure and the whole "
                  "transaction table survive",
        "values": ["Rohit Deshpande", "0912101034567", "884512290",
                   "12 Shanti Nagar", "AAAPD1234C"],
        "must_survive": ["Canara Bank", "CNRB0001234", "Jayanagar",
                         "12,500.00", "68,400.00", "2,41,882.50",
                         "NEFT transfer", "Salary credit", "01-02-2026"],
    }


def clinic_letter(d):
    """THE DIVERGENCE. Under insurance this clinical text is kept; here it
    is the thing that must go."""
    p = d.new_page()
    _put(p, ["Lakeside Family Practice",
             "18 Station Road, Pune 411001    reception@lakeside.example"],
         y=60, bold={0})
    _put(p, ["Referral Letter", "",
             "Patient      Farida Sheikh",
             "NHS Number   943 476 5919",
             "Date of Birth 02 September 1981", "",
             "Dear Colleague,", "",
             "I am referring Mrs Sheikh who has a six-month history of",
             "rheumatoid arthritis with worsening morning stiffness. She is",
             "currently taking methotrexate 15mg weekly and folic acid.",
             "Her ESR was 48 mm/hr and CRP 22 mg/L at last review. She also",
             "has type 2 diabetes managed with metformin.", "",
             "Appointment requested for 12 May 2026.", "",
             "Dr Imran Qureshi", "GMC 6142288"], y=110)
    return {
        "expect": "health content is REDACTED here (opposite of insurance); "
                  "practice identity and the appointment date survive",
        "values": ["Farida Sheikh", "943 476 5919", "rheumatoid arthritis",
                   "methotrexate", "diabetes", "Imran Qureshi",
                   "02 September 1981"],
        "must_survive": ["Lakeside Family Practice", "Station Road",
                         "reception@lakeside.example", "12 May 2026"],
    }


def school_form(d):
    """Guidance text and empty fields beside filled ones."""
    p = d.new_page()
    _put(p, ["St Aloysius Higher Secondary School",
             "Admission Form 2026-27      Form No. ADM/2026/0418"],
         y=60, bold={0})
    _put(p, ["INSTRUCTIONS FOR FILLING THIS FORM",
             "Please write in block letters using black ink only. Enter the",
             "student's name exactly as it appears on the birth certificate.",
             "For example: RAJESH KUMAR SHARMA. Do not use abbreviations.",
             "Fields marked with an asterisk are mandatory."], y=110)
    _put(p, ["Student Name *     Meera Pillai",
             "Date of Birth *    08 June 2014",
             "Class              VI-B",
             "Roll Number        2026118",
             "Father's Name      Sanjay Pillai",
             "Religion           Hindu",
             "Address            9 Beach Road, Kochi 682001",
             "Guardian Mobile    9847112200",
             "Previous School    ",
             "Remarks            "], y=210)
    return {
        "expect": "child identity, religion and guardian contact go; the "
                  "instruction block including its EXAMPLE name survives, and "
                  "empty fields are not blacked out",
        "values": ["Meera Pillai", "08 June 2014", "2026118", "Sanjay Pillai",
                   "Hindu", "9 Beach Road", "9847112200"],
        "must_survive": ["INSTRUCTIONS FOR FILLING", "block letters",
                         "RAJESH KUMAR SHARMA", "birth certificate",
                         "St Aloysius", "ADM/2026/0418", "Previous School",
                         "Remarks"],
    }


def invoice(d):
    """Line items and totals must survive; the buyer must not."""
    p = d.new_page()
    _put(p, ["Northwind Supplies LLP", "GSTIN 29AABCT3518Q1ZV",
             "billing@northwind.example"], y=60, bold={0})
    _put(p, ["Tax Invoice   INV-2026-4471      Date 18 March 2026", "",
             "Bill To:", "Kavitha Raman", "Flat 7C, Palm Grove Apartments",
             "Chennai 600041", "GSTIN 33AAGCK9921H1Z8",
             "Contact kavitha.raman@outlook.example"], y=110, bold={0})
    rows = [("Copier paper A4", "40", "260.00", "10,400.00"),
            ("Toner cartridge", "6", "4,150.00", "24,900.00"),
            ("Stapler heavy duty", "12", "385.00", "4,620.00")]
    y = 250
    _put(p, ["Description              Qty     Rate        Amount"], y=y,
         bold={0})
    for i, r in enumerate(rows):
        p.insert_text((60, y + 20 + i * 15), r[0], fontsize=9)
        p.insert_text((250, y + 20 + i * 15), r[1], fontsize=9)
        p.insert_text((310, y + 20 + i * 15), r[2], fontsize=9)
        p.insert_text((410, y + 20 + i * 15), r[3], fontsize=9)
    _put(p, ["Subtotal   39,920.00", "CGST 9%     3,592.80",
             "SGST 9%     3,592.80", "Total      47,105.60"], x=310, y=330)
    return {
        "expect": "the buyer goes, the goods and every figure stay",
        "values": ["Kavitha Raman", "Palm Grove Apartments",
                   "kavitha.raman@outlook.example"],
        "must_survive": ["Northwind Supplies", "29AABCT3518Q1ZV",
                         "billing@northwind.example", "INV-2026-4471",
                         "Copier paper A4", "10,400.00", "47,105.60",
                         "18 March 2026", "Toner cartridge"],
    }


def handwritten_log(d):
    """Ink zoning: identity ink goes, content ink stays."""
    p = d.new_page()
    _put(p, ["Goods Inward Register", "Warehouse 3, Bay 2"], y=60, bold={0})
    _put(p, ["Received By (Name):", "Signature:", "",
             "Pallets:", "Temperature (C):", "Time:"], y=120)
    _pen(p, 200, 122, w=110, seed=5)     # name field  -> identity ink
    _pen(p, 200, 148, w=120, seed=9)     # signature   -> identity ink
    p.insert_text((200, 190), "14", fontsize=11, color=BLUE)
    p.insert_text((200, 216), "4.2", fontsize=11, color=BLUE)
    p.insert_text((200, 242), "06:40", fontsize=11, color=BLUE)
    return {
        "expect": "ink in the name and signature fields is destroyed; the "
                  "handwritten readings are kept",
        "values": [],
        "must_survive": ["Goods Inward Register", "Pallets", "Temperature",
                         "14", "4.2", "06:40"],
    }


def legal_filing(d):
    """Case numbers and court identity survive; parties do not."""
    p = d.new_page()
    _put(p, ["IN THE CITY CIVIL COURT AT BENGALURU",
             "Original Suit No. 4412 of 2026"], y=60, bold={0, 1})
    _put(p, ["BETWEEN:", "",
             "Vikram Sethi, aged 47 years,",
             "residing at 88 Cunningham Road, Bengaluru 560052",
             "                                        ... Plaintiff", "",
             "AND", "",
             "Priyanka Nair, aged 39 years,",
             "residing at 3 Lavelle Road, Bengaluru 560001",
             "                                        ... Defendant", "",
             "The suit is listed for hearing on 04 August 2026 before the",
             "Hon'ble XII Additional City Civil Judge."], y=120)
    return {
        "expect": "both parties go regardless of role; the court, the case "
                  "number and the hearing date survive",
        "values": ["Vikram Sethi", "Priyanka Nair", "88 Cunningham Road",
                   "3 Lavelle Road"],
        "must_survive": ["CITY CIVIL COURT", "4412 of 2026", "04 August 2026",
                         "Plaintiff", "Defendant"],
    }


def credentials_sheet(d):
    """Rows that exist in the pack but nothing has ever exercised."""
    p = d.new_page()
    _put(p, ["Internal IT Handover", "Asset and Access Record"], y=60, bold={0})
    _put(p, ["Issued To       Deepak Ranganathan",
             "Username        d.ranganathan",
             "Password        Tr0ub4dor&3",
             "IP Address      10.44.12.208",
             "MAC Address     3C:22:FB:91:04:AE",
             "Vehicle Number  KA 05 MJ 4471",
             "Asset Tag       LAP-2026-0881",
             "Issue Date      12 January 2026",
             "Department      Infrastructure"], y=120)
    return {
        "expect": "credentials, device and vehicle identifiers go; the asset "
                  "tag, issue date and department survive",
        "values": ["Deepak Ranganathan", "d.ranganathan", "Tr0ub4dor&3",
                   "10.44.12.208", "3C:22:FB:91:04:AE", "KA 05 MJ 4471"],
        "must_survive": ["LAP-2026-0881", "12 January 2026",
                         "Infrastructure", "Asset and Access Record"],
    }


CASES = [("hr_letter", hr_letter), ("bank_statement", bank_statement),
         ("clinic_letter", clinic_letter), ("school_form", school_form),
         ("invoice", invoice), ("handwritten_log", handwritten_log),
         ("legal_filing", legal_filing),
         ("credentials_sheet", credentials_sheet)]


def degrade(src, dst, dpi=105):
    """Print->scan, at the real corpus's median resolution."""
    s = fitz.open(src)
    out = fitz.open()
    for i in range(s.page_count):
        pm = s[i].get_pixmap(dpi=dpi)
        rng = np.random.default_rng(7 + i)
        a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n).copy()
        a = np.clip(a.astype(np.int16) + rng.normal(0, 9, a.shape), 0, 255)
        pm2 = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, pm.w, pm.h), False)
        pm2.samples_mv[:] = a.astype(np.uint8)[..., :3].tobytes()
        pg = out.new_page(width=s[i].rect.width, height=s[i].rect.height)
        pg.insert_image(pg.rect, pixmap=pm2)
    out.save(dst, garbage=4, deflate=True)
    out.close(); s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/generic")
    ap.add_argument("--scanned", action="store_true",
                    help="also emit a degraded scan of every case")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = []
    for name, fn in CASES:
        d = fitz.open()
        meta = fn(d)
        path = out / f"{name}.pdf"
        d.save(path, garbage=4, deflate=True); d.close()
        man.append({"case": name, "file": path.name, **meta})
        if a.scanned:
            sp = out / f"{name}_scan.pdf"
            degrade(path, sp)
            man.append({"case": name + "_scan", "file": sp.name,
                        **{k: v for k, v in meta.items()}})
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    nv = sum(len(m["values"]) for m in man)
    nk = sum(len(m.get("must_survive", [])) for m in man)
    print(f"  {len(man)} documents -> {out}")
    print(f"  {nv} values that must be REMOVED")
    print(f"  {nk} values that must SURVIVE  (the half the old corpus missed)")


if __name__ == "__main__":
    main()
