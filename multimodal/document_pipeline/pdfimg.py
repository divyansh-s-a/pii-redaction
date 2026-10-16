#!/usr/bin/env python3
"""PDFs, scans and images for the dataset route — read, redact or replace.

Reading: a PDF page's text layer gives words with boxes; a page with no text
layer (a scan) is rendered and read by Apple Vision OCR (local), which also
finds faces. A standalone image is read the same way. Words are grouped into
line segments with per-word boxes, so a detected span maps back to the exact
rectangles it covers.

Rendering (true removal, PyMuPDF redaction annotations — the text and the
image pixels under a box are deleted, not hidden):
  Redact mode    black box over every removed span
  Replace mode   the span's text removed and the invented name written in
                 the same box (white fill, fitted font); removed categories
                 (contact, address, ids) stay black boxes
  Both modes     images showing a brand, a product, a screenshot or a person
                 (classified once per image content) and faces on scans are
                 covered with neutral grey; document metadata is blanked.
A standalone image whose whole content is a brand image or a person photo is
greyed entirely (parity with the earlier study), otherwise boxed like a scan.
"""
import io
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "video_pipeline"))

BAR_CHARS = set("█X ")
GREY = (0.78, 0.78, 0.78)
SCAN_DPI = 200


def _ocr(png_paths):
    import frame_detect
    return frame_detect.read_frames(png_paths)


def _line_segments(words, part, page):
    """words: [(x0, y0, x1, y1, text, line_key)] -> line segments."""
    segs, cur, key = [], [], None
    for w in words:
        if cur and w[5] != key:
            segs.append(_seg(cur, part, page))
            cur = []
        cur.append(w)
        key = w[5]
    if cur:
        segs.append(_seg(cur, part, page))
    return segs


def _seg(ws, part, page):
    text, boxes = "", []
    for w in ws:
        if text:
            text += " "
        a = len(text)
        text += w[4]
        boxes.append((a, len(text), (w[0], w[1], w[2], w[3])))
    return {"part": part, "kind": "line", "page": page, "pieces": [],
            "text": text, "boxes": boxes}


def read_pdf(path):
    """-> (segments, info). info['scan_pages'] lists OCR'd pages; faces on
    scanned pages are returned in info['faces'] as (page, rect)."""
    import fitz
    doc = fitz.open(str(path))
    segs, scans, faces = [], [], []
    for pno, page in enumerate(doc):
        words = page.get_text("words")
        if len(words) >= 3:
            segs += _char_segments(page, pno)
            continue
        if not page.get_images():
            continue
        scans.append(pno)
        pix = page.get_pixmap(dpi=SCAN_DPI)
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "p.png"
            pix.save(str(png))
            fr = _ocr([png])[0]
        W, H = page.rect.width, page.rect.height
        ws = [(t["box"][0] * W, t["box"][1] * H, t["box"][2] * W,
               t["box"][3] * H, t["text"], t["line"]) for t in fr["tokens"]]
        segs += _line_segments(ws, f"page{pno}", pno)
        faces += [(pno, (f["box"][0] * W, f["box"][1] * H, f["box"][2] * W,
                         f["box"][3] * H)) for f in fr["faces"]]
    doc.close()
    return segs, {"scan_pages": scans, "faces": faces}


def _char_segments(page, pno):
    """Line segments with a box per CHARACTER (rawdict): a span maps to
    exactly its own characters, even inside words a PDF glued together
    ("HierarchyZorvex")."""
    segs = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            text, boxes, last_x1 = "", [], None
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if not c:
                        continue
                    x0, y0, x1, y1 = ch["bbox"]
                    # a visible gap between characters reads as a space
                    if text and last_x1 is not None and c != " " and \
                            x0 - last_x1 > 0.25 * (y1 - y0) and \
                            not text.endswith(" "):
                        text += " "
                    a = len(text)
                    text += c
                    if c.strip():
                        boxes.append((a, a + 1, (x0, y0, x1, y1)))
                    last_x1 = x1
            if text.strip():
                segs.append({"part": f"page{pno}", "kind": "line",
                             "page": pno, "pieces": [], "text": text,
                             "boxes": boxes})
    return segs


def read_image(path):
    from PIL import Image
    im = Image.open(str(path))
    W, H = im.size
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "i.png"
        im.convert("RGB").save(png)
        fr = _ocr([png])[0]
    ws = [(t["box"][0] * W, t["box"][1] * H, t["box"][2] * W, t["box"][3] * H,
           t["text"], t["line"]) for t in fr["tokens"]]
    segs = _line_segments(ws, "image", 0)
    faces = [(0, (f["box"][0] * W, f["box"][1] * H, f["box"][2] * W,
                  f["box"][3] * H)) for f in fr["faces"]]
    return segs, {"faces": faces, "size": (W, H)}


def span_rect(seg, a, b):
    """Union rectangle of the words a span covers (one line)."""
    rs = [r for (wa, wb, r) in seg["boxes"] if wa < b and wb > a]
    if not rs:
        return None
    return (min(r[0] for r in rs), min(r[1] for r in rs),
            max(r[2] for r in rs), max(r[3] for r in rs))


def _is_bar(new):
    return not new or set(new) <= BAR_CHARS


def _fit(new, rect):
    import fitz
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    fs = max(4.0, h * 0.8)
    tl = fitz.get_text_length(new, fontname="helv", fontsize=fs)
    if tl > w and tl > 0:
        fs = max(4.0, fs * w / tl)
    return fs


def render_pdf(path, segs, spans, replacer, info, classify):
    """-> (pdf bytes, report)."""
    import fitz
    doc = fitz.open(str(path))
    boxed = written = greyed = 0
    unmapped0 = len(replacer.unmapped)
    pending = []          # every box, kept for re-rasterising scanned pages
    for sp in spans:
        seg = segs[sp["seg"]]
        new = replacer(sp)
        if new is None:
            continue
        r = span_rect(seg, sp["a"], sp["b"])
        if r is None:
            continue
        page = doc[seg["page"]]
        rect = fitz.Rect(*r)
        pending.append({"page": seg["page"], "rect": tuple(r),
                        "text": "" if _is_bar(new) else new})
        if _is_bar(new):
            page.add_redact_annot(rect, fill=(0, 0, 0))
            boxed += 1
        else:
            # widen a little: the invented name may be longer than the real
            h = rect.height
            rect = fitz.Rect(rect.x0, rect.y0 - 0.15 * h, rect.x1 + max(
                0, len(new) - len(sp["text"])) * h * 0.25,
                rect.y1 + 0.35 * h)
            fs = min(_fit(new, (rect.x0, rect.y0, rect.x1, rect.y1)),
                     h * 0.72)
            page.add_redact_annot(rect, text=new, fontname="helv",
                                  fontsize=fs,
                                  fill=(1, 1, 1), text_color=(0, 0, 0),
                                  align=fitz.TEXT_ALIGN_LEFT)
            written += 1
    # images: brand / product / screenshot / person -> grey
    for pno, page in enumerate(doc):
        for im in page.get_image_info(xrefs=True):
            x = im.get("xref")
            bbox = fitz.Rect(im["bbox"])
            if not x or bbox.width < 12 or bbox.height < 12:
                continue
            if pno in info.get("scan_pages", []) and \
                    bbox.width * bbox.height > 0.6 * page.rect.width * \
                    page.rect.height:
                continue        # the scan itself: handled by text boxes
            try:
                data = doc.extract_image(x)
            except Exception:
                continue
            grey, _kind = classify(data["image"], data.get("ext", "png"))
            if grey:
                page.add_redact_annot(bbox, fill=GREY)
                greyed += 1
    for pno, r in info.get("faces", []):
        doc[pno].add_redact_annot(fitz.Rect(*r), fill=GREY)
        pending.append({"page": pno, "rect": tuple(r), "text": "",
                        "grey": True})
        greyed += 1
    scan_edits = {}
    for page in doc:
        if page.number in info.get("scan_pages", []):
            # collect this page's boxes; the page is re-rasterised below
            scan_edits[page.number] = [
                (a.rect, a.info.get("content", ""))
                for a in page.annots(types=[fitz.PDF_ANNOT_REDACT])]
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
    for pno, _edits in scan_edits.items():
        _rasterise(doc, pno, [e for e in pending if e["page"] == pno])
    doc.set_metadata({k: "" for k in ("author", "title", "subject",
                                      "keywords", "creator", "producer")})
    try:
        doc.del_xml_metadata()
    except Exception:
        pass
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out, {"boxes": boxed, "written": written, "greyed": greyed,
                 "unmapped_pseudonyms": len(replacer.unmapped) - unmapped0}


def render_image(path, segs, spans, replacer, info, classify):
    from PIL import Image, ImageDraw, ImageFont
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower().lstrip(".")
    im = Image.open(io.BytesIO(data)).convert("RGB")
    grey, kind = classify(data, ext)
    rep = {"boxes": 0, "written": 0, "greyed": 0, "kind": kind}
    if grey and kind in ("brand_logo", "brand_product_or_store",
                         "person_photo", "screenshot_or_ad"):
        import office
        rep["greyed"] = 1
        return office.grey_image(data, ext), rep
    d = ImageDraw.Draw(im)
    for sp in spans:
        new = replacer(sp)
        if new is None:
            continue
        r = span_rect(segs[sp["seg"]], sp["a"], sp["b"])
        if r is None:
            continue
        if _is_bar(new):
            d.rectangle(r, fill=(0, 0, 0))
            rep["boxes"] += 1
        else:
            d.rectangle(r, fill=(255, 255, 255))
            h = max(8, int((r[3] - r[1]) * 0.85))
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",
                                          h)
            except OSError:
                font = ImageFont.load_default()
            d.text((r[0], r[1]), new, fill=(0, 0, 0), font=font)
            rep["written"] += 1
    for _p, r in info.get("faces", []):
        d.rectangle(r, fill=(200, 200, 200))
        rep["greyed"] += 1
    buf = io.BytesIO()
    fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}.get(ext, "PNG")
    im.save(buf, fmt)
    return buf.getvalue(), rep


def _rasterise(doc, pno, edits):
    """A scanned page becomes a new image with its boxes drawn in (pixel
    removal inside some scan encodings is not reliable)."""
    import fitz
    from PIL import Image, ImageDraw
    page = doc[pno]
    zoom = SCAN_DPI / 72.0
    pix = page.get_pixmap(dpi=SCAN_DPI)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    d = ImageDraw.Draw(im)
    for e in edits:
        x0, y0, x1, y1 = [v * zoom for v in e["rect"]]
        if e.get("grey"):
            d.rectangle((x0, y0, x1, y1), fill=(200, 200, 200))
        elif not e["text"]:
            d.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
        # an invented name is already drawn on the page by the applied
        # redaction: the render below flattens it into the new image
    buf = io.BytesIO()
    im.save(buf, "PNG")
    rect = page.rect
    page.add_redact_annot(rect)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
    page.insert_image(rect, stream=buf.getvalue())


def flatten_pages_with(blob, finder):
    """Pages whose text layer still holds a removable form (text inside an
    object redaction cannot edit) are flattened into images: the boxes
    already drawn cover what was removed, the hidden text is gone."""
    import fitz
    doc = fitz.open(stream=blob, filetype="pdf")
    hit = [p.number for p in doc if finder(p.get_text())]
    for pno in hit:
        _rasterise(doc, pno, [])
    out = doc.tobytes(garbage=3, deflate=True) if hit else blob
    doc.close()
    return out


def output_text_pdf(blob):
    """Text of a rendered PDF for verification (text layer; scanned pages
    re-read by OCR)."""
    import fitz
    doc = fitz.open(stream=blob, filetype="pdf")
    out = []
    for page in doc:
        t = page.get_text()
        if len(t.split()) < 3 and page.get_images():
            pix = page.get_pixmap(dpi=SCAN_DPI)
            with tempfile.TemporaryDirectory() as td:
                png = Path(td) / "p.png"
                pix.save(str(png))
                fr = _ocr([png])[0]
            t = "\n".join(ln["text"] for ln in fr["lines"])
        out.append(t)
    meta = " ".join(str(v) for v in (doc.metadata or {}).values() if v)
    doc.close()
    return "\n".join(out) + "\n" + meta


def output_text_image(blob):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.png"
        p.write_bytes(blob)
        fr = _ocr([p])[0]
    return "\n".join(ln["text"] for ln in fr["lines"])


def _self_test():
    import fitz
    import office
    sys.path.insert(0, str(HERE.parent / "taxonomies"))
    import base
    pack = base.load(HERE.parent / "taxonomies" / "market_research")
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "t.pdf"
        doc = fitz.open()
        pg = doc.new_page()
        pg.insert_text((72, 100), "Report for Zorvex by Meera Iyer",
                       fontsize=12)
        pg.insert_text((72, 130), "Call 98224 15736 today", fontsize=12)
        doc.save(str(pdf))
        segs, info = read_pdf(pdf)
        t0 = segs[0]["text"]
        spans = [{"seg": 0, "a": t0.index("Zorvex"), "b": t0.index("Zorvex")
                  + 6, "text": "Zorvex", "categories": ["BRAND"]},
                 {"seg": 0, "a": t0.index("Meera"), "b": len(t0),
                  "text": "Meera Iyer", "categories": ["FULL_NAME"]}]
        spans += office.deterministic(segs)

        class EM:
            by_id = {}

            def match_tokens(self, toks):
                out = []
                for i, w in enumerate(toks):
                    if w == "Zorvex":
                        out.append((i, i, {"replace": "Kelmora"}, "regex"))
                    if w == "Meera":
                        out.append((i, i + 1, {"replace": "Anita Rao"},
                                    "regex"))
                return out
        no_img = lambda d, e: (False, "none")       # noqa: E731
        red, r1 = render_pdf(pdf, segs, spans, office.Replacer("redact", pack),
                             info, no_img)
        txt = output_text_pdf(red)
        assert "Zorvex" not in txt and "Meera" not in txt and \
            "98224" not in txt and "Report for" in txt, txt
        rep, r2 = render_pdf(pdf, segs, spans,
                             office.Replacer("replace", pack, EM()), info,
                             no_img)
        txt2 = output_text_pdf(rep)
        assert "Kelmora" in txt2 and "Anita Rao" in txt2, txt2
        assert "Zorvex" not in txt2 and "98224" not in txt2, txt2
    print("pdfimg self-test: OK")


if __name__ == "__main__":
    _self_test()
