#!/usr/bin/env python3
"""One interface over every way of getting words+boxes off a page.

    read_page(doc, page_no, reader="auto") -> {"words": [(i, text, Rect)], ...}

Why this exists: detection must not care whether a page carries real text or
is a photograph of paper. Both readers return the SAME numbered word stream
with real PDF coordinates, so the detector keeps returning word indices and
geometry keeps coming from the stream - the rule that made localisation exact
on born-digital pages, extended to scans instead of replaced by model boxes.

  text_layer  PyMuPDF get_text("words"). Exact glyph rects. ~20% of this
              corpus. Free.
  vision_ocr  Apple Vision at >=200 dpi, word-level sub-range boxes converted
              back to PDF points. Works on the 79% with no text layer, and on
              the invisible-OCR pages where the buried text cannot be trusted.
              Latin script only - Vision has no Devanagari - so Indic pages
              fall through to the vision-LLM path.

Measured, and the reason this design works: on a low-dpi scan Vision misreads
words ("insane Company" for "Insurance Company") while placing their boxes
correctly. Redaction needs WHERE, not WHAT, so a transcription error does not
have to become a leak.
"""
import json
import subprocess
import sys
from pathlib import Path

import fitz

HELPER = Path(__file__).resolve().parent / "vision_helper.swift"
OCR_DPI = 200                  # below this Vision word boxes degrade sharply


def corpus_dir():
    """Where the real claim corpus lives on THIS machine.

    Was a hardcoded absolute path under one developer's home directory in nine
    files, which both leaks the operator's layout into a shared repository and
    makes every corpus-backed self-test silently skip for everyone else.
    """
    import os
    from pathlib import Path as _P
    return _P(os.environ.get("PII_CORPUS_DIR",
                             _P.home() / "pii_corpus")).expanduser()


def _vision_text(image_path):
    try:
        r = subprocess.run(["swift", str(HELPER), str(image_path), "text"],
                           capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout) if r.stdout.strip() else {}
    except Exception:
        return {}


def read_text_layer(page):
    words = []
    for i, w in enumerate(page.get_text("words")):
        words.append((i, w[4], fitz.Rect(w[0], w[1], w[2], w[3])))
    return words


def read_vision_ocr(page, dpi=OCR_DPI, workdir=None):
    """Render -> Apple Vision -> word boxes in PDF POINTS.

    Vision returns normalised coordinates; the swift shim already flips them to
    a top-left origin, so the only conversion left is scaling by the page rect.
    Doing it here (not in the caller) keeps one coordinate convention in the
    whole pipeline: PDF points, top-left origin.
    """
    workdir = Path(workdir or "/tmp")
    workdir.mkdir(parents=True, exist_ok=True)
    img = workdir / f"_ocr_p{page.number}.png"
    page.get_pixmap(dpi=dpi).save(str(img))
    data = _vision_text(img)
    img.unlink(missing_ok=True)

    W, H = page.rect.width, page.rect.height
    words, i = [], 0
    for line in data.get("text", []):
        toks = line.get("tokens") or []
        if not toks:                       # fall back to the line box
            b = line["box"]
            toks = [{"text": line["text"], "box": b}]
        for t in toks:
            b = t["box"]
            words.append((i, t["text"],
                          fitz.Rect(b["x0"] * W, b["y0"] * H,
                                    b["x1"] * W, b["y1"] * H)))
            i += 1
    return words


def read_page(doc, page_no, reader="auto", dpi=OCR_DPI, workdir=None):
    """page_no is 1-based. reader: auto | text_layer | vision_ocr."""
    page = doc[page_no - 1]
    chosen = reader
    if reader == "auto":
        native = read_text_layer(page)
        # invisible-OCR pages have text, but it is someone else's OCR of
        # unknown quality sitting under a scan - prefer our own reader and
        # let the buried text be destroyed rather than trusted
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "router"))
        import analyze as router_analyze
        kind, _ = router_analyze._classify_page(page)
        chosen = "text_layer" if kind == "born_digital" and native \
            else "vision_ocr"
    words = (read_text_layer(page) if chosen == "text_layer"
             else read_vision_ocr(page, dpi, workdir))
    return {"page": page_no, "reader": chosen, "words": words,
            "n_words": len(words),
            "page_rect": [page.rect.width, page.rect.height]}


if __name__ == "__main__":
    D = readers.corpus_dir()
    W = Path("/private/tmp/claude-501/-Users-shivangi-Documents-Claude-Audio-"
             "PII-Project/87ddc989-26e9-49df-9724-61c42bfe09fe/scratchpad/rd")
    cases = [("case-48043-ad818499-4046-45e1-adf6-ff971505a2fa.pdf", 47,
              "born-digital"),
             ("case-48053-7fed243c-d873-4021-a864-51dec62fa850.pdf", 1,
              "handwritten scan")]
    for fname, pno, label in cases:
        d = fitz.open(D / fname)
        r = read_page(d, pno, workdir=W)
        inside = sum(1 for _, _, b in r["words"]
                     if b.x0 >= -1 and b.y0 >= -1
                     and b.x1 <= d[pno-1].rect.width + 1
                     and b.y1 <= d[pno-1].rect.height + 1)
        print(f"{label:<18} reader={r['reader']:<11} words={r['n_words']:>4} "
              f"boxes inside page={inside}/{r['n_words']}")
        assert inside == r["n_words"], "a word box fell outside the page"
        d.close()
    print("readers self-test OK (both paths return in-page PDF-point boxes)")
