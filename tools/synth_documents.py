#!/usr/bin/env python3
"""Synthetic market-research study with known answers, for the dataset route.

    python3 tools/synth_documents.py OUT_DIR

Writes an invented two-study dataset (every name, brand, number and address
is made up) across the formats the route handles — docx (split runs, a
comment, a tracked deletion, header, properties), xlsx (identity columns,
numbers stored as numbers, a cell comment, a sheet named after the brand),
pptx (slide text, notes, a chart whose labels are brands), csv (the same
respondents, so codes must agree across files), fixed-width .asc, a text PDF,
a scanned PDF (text only as pixels) and file/folder names carrying the brand
— plus OUT_DIR/ground_truth.json: every value that must be removed or
replaced, its kind, and the values that must survive.
"""
import csv
import io
import json
import sys
from pathlib import Path

PEOPLE = {"moderator": "Meera Iyer", "client": "Vikram Rao",
          "interviewer": "Pooja Nair",
          "resp": ["Ravi Kulkarni", "Anjali Deshmukh", "Sanjay Patil"]}
BRANDS = {"client": "Zorvex", "comp1": "Kantoli", "comp2": "Brivana",
          "model": "Zorvex Vision 55Q", "dealer": "Shubham Electronics Hub",
          "agency": "Brightpath Research", "slogan": "See more, live more"}
PHONES = ["9822415736", "9765012345", "9011223344"]
EMAILS = ["ravi.k@examplemail.in", "anjali.d@examplemail.in",
          "sanjay.p@examplemail.in"]
ADDRESSES = ["Flat 12, Shanti Nagar, Kothrud", "B-4 Lotus Residency, Aundh",
             "22 Gandhi Path, Baner"]
RIDS = ["RSP-1041", "RSP-1042", "RSP-1043"]
LOGIN = "int_pooja22"
KEEP = ["Pune", "34", "41", "29", "television", "awareness", "QuickKart",
        "Female", "Male"]


def docx_file(path):
    import docx
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = docx.Document()
    d.core_properties.author = PEOPLE["moderator"]
    d.core_properties.title = f"{BRANDS['client']} TV study questionnaire"
    d.sections[0].header.paragraphs[0].text = BRANDS["agency"]
    d.add_heading(f"{BRANDS['client']} television study", 1)
    p = d.add_paragraph("Moderator: ")
    p.add_run(PEOPLE["moderator"])
    p = d.add_paragraph("Q1. Which brand of television do you own? Show card: ")
    p.add_run("Zor")
    p.add_run(f"vex, {BRANDS['comp1']}, {BRANDS['comp2']}")
    d.add_paragraph(f"Q2. Have you seen the {BRANDS['model']} at "
                    f"{BRANDS['dealer']}? Probe the line '{BRANDS['slogan']}'.")
    d.add_paragraph(f"Field contact: {PEOPLE['interviewer']}, "
                    f"{PHONES[2]}. Fieldwork city: Pune.")
    d.add_paragraph("Respondents order groceries on QuickKart and watch "
                    "television daily; brand awareness is the main measure.")
    # tracked deletion that still holds a competitor
    para = d.add_paragraph("Q3. Rate the picture quality.")
    dele = OxmlElement("w:del")
    dele.set(qn("w:id"), "1")
    dele.set(qn("w:author"), PEOPLE["client"])
    r = OxmlElement("w:r")
    dt = OxmlElement("w:delText")
    dt.text = f" Compare with {BRANDS['comp1']}."
    r.append(dt)
    dele.append(r)
    para._p.append(dele)
    try:
        para.runs[0].add_comment if False else None
        c = d.add_comment(para.runs[0], text=f"Ask about {BRANDS['client']} "
                          "remote", author=PEOPLE["client"], initials="VR")
    except Exception:                         # older python-docx
        pass
    d.save(path)


def xlsx_file(path):
    import openpyxl
    from openpyxl.comments import Comment
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{BRANDS['client']} data"
    ws.append(["SERIAL", "Respondent ID", "Name", "Phone Number", "Email",
               "Address", "City", "Age", "Gender", "Interviewer",
               "Q1 Brand owned", "Q6 Why (open)"])
    ages, genders = ["34", "41", "29"], ["Female", "Female", "Male"]
    whys = [f"My brother {PEOPLE['resp'][2].split()[0]} suggested "
            f"{BRANDS['client']}",
            f"Bought it at {BRANDS['dealer']}, staff were helpful",
            f"Cheaper than {BRANDS['comp2']}"]
    for k in range(3):
        ws.append([10001 + k, RIDS[k], PEOPLE["resp"][k], int(PHONES[k]),
                   EMAILS[k], ADDRESSES[k], "Pune", int(ages[k]), genders[k],
                   LOGIN, [BRANDS["client"], BRANDS["comp1"],
                           BRANDS["comp2"]][k], whys[k]])
    ws["L2"].comment = Comment(f"Check with {PEOPLE['client']}",
                               PEOPLE["interviewer"])
    wb.properties.creator = PEOPLE["interviewer"]
    wb.save(path)


def pptx_file(path):
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches
    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = f"{BRANDS['client']} leads television awareness"
    s.notes_slide.notes_text_frame.text = (f"Discussed with {PEOPLE['client']}"
                                           " on the call")
    cd = CategoryChartData()
    cd.categories = [BRANDS["client"], BRANDS["comp1"], BRANDS["comp2"]]
    cd.add_series("Awareness %", (62, 48, 31))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1.5),
                       Inches(6), Inches(4), cd)
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Respondent voice"
    s2.placeholders[1].text = (f"\"{BRANDS['slogan']}\" stuck with me — "
                               f"{PEOPLE['resp'][0]}, 34, Pune")
    prs.core_properties.author = PEOPLE["moderator"]
    prs.save(path)


def csv_file(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Respondent Id", "Name", "Mobile", "City", "Comment"])
        for k in range(3):
            w.writerow([RIDS[k], PEOPLE["resp"][k], PHONES[k], "Pune",
                        f"Prefer {BRANDS['client']} for television"])


def asc_file(path):
    lines = [f"{10001 + k:05d}{PHONES[k]}{'PUNE':<6}{k + 1:02d}"
             for k in range(3)]
    Path(path).write_text("\r\n".join(lines) + "\r\n")


def pdf_text(path):
    import fitz
    doc = fitz.open()
    pg = doc.new_page()
    y = 80
    for line in [f"Proposal: {BRANDS['client']} brand health study",
                 f"Prepared by {BRANDS['agency']} for {PEOPLE['client']}",
                 f"Contact: {EMAILS[0]} / {PHONES[0]}",
                 "Fieldwork in Pune across television owners."]:
        pg.insert_text((60, y), line, fontsize=11)
        y += 22
    doc.set_metadata({"author": PEOPLE["moderator"]})
    doc.save(path)


def pdf_scan(path):
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        font = ImageFont.load_default()
    y = 160
    for line in ["Recruitment screener (signed copy)",
                 f"Respondent: {PEOPLE['resp'][1]}",
                 f"Mobile: {PHONES[1]}",
                 f"Address: {ADDRESSES[1]}",
                 f"Owns: {BRANDS['comp1']} television", "City: Pune"]:
        d.text((120, y), line, fill="black", font=font)
        y += 70
    buf = io.BytesIO()
    im.save(buf, "PNG")
    doc = fitz.open()
    pg = doc.new_page(width=595, height=842)
    pg.insert_image(pg.rect, stream=buf.getvalue())
    doc.save(path)


def main():
    out = Path(sys.argv[1])
    s1 = out / "dataset" / f"IN 901 {BRANDS['client']} TV Study"
    s2 = out / "dataset" / f"IN 902 {BRANDS['comp1']} Follow-up"
    for d in (s1 / "Questionnaire", s1 / "Data", s1 / "Report", s2):
        d.mkdir(parents=True, exist_ok=True)
    docx_file(s1 / "Questionnaire" / f"{BRANDS['client']} QRE.docx")
    xlsx_file(s1 / "Data" / "Raw data.xlsx")
    csv_file(s1 / "Data" / "responses.csv")
    asc_file(s1 / "Data" / "data.asc")
    pptx_file(s1 / "Report" / f"{BRANDS['client']} Deck_{PEOPLE['moderator'].split()[0]}.pptx")
    pdf_text(s1 / "Report" / "Proposal.pdf")
    pdf_scan(s2 / "Screener scan.pdf")
    gt = {"remove": [], "keep": KEEP, "codes": RIDS + [LOGIN]}
    for k, v in PEOPLE.items():
        for name in (v if isinstance(v, list) else [v]):
            gt["remove"].append({"value": name, "kind": "person"})
    for k, v in BRANDS.items():
        gt["remove"].append({"value": v, "kind": "brand"})
    for v in PHONES:
        gt["remove"].append({"value": v, "kind": "phone"})
    for v in EMAILS:
        gt["remove"].append({"value": v, "kind": "email"})
    for v in ADDRESSES:
        gt["remove"].append({"value": v, "kind": "address"})
    for v in RIDS + [LOGIN]:
        gt["remove"].append({"value": v, "kind": "id"})
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    print(f"synthetic dataset -> {out / 'dataset'}")


if __name__ == "__main__":
    main()
