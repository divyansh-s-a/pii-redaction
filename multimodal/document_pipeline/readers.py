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
import time
import sys
from pathlib import Path

import fitz

import vision_exec

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


def _vision_text(image_path, tries=3, mode="text"):
    """Apple Vision OCR for one rendered page.

    `swift` COMPILES the helper on every invocation, so this is not a cheap
    call and it is not a reliable one under load: with several pages in
    flight plus the face-tiling batches, a spawn can fail or be killed and
    the process then writes nothing to stdout. The original code returned {}
    for that case -- indistinguishable from a page that genuinely has no
    text. Downstream, zero words makes detect_page return before it calls a
    model AND before its deterministic regex pass, so the page is redacted
    on faces and QR codes alone, with no error anywhere and the reader still
    reported as vision_ocr. That is a silent, total loss of text coverage on
    an arbitrary page, so every failure mode here is retried and the last one
    is reported rather than swallowed.
    """
    last = "no attempt"
    for attempt in range(tries):
        try:
            r = subprocess.run(
                vision_exec.argv([str(image_path), mode]),
                capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                last = (f"swift exit {r.returncode}: "
                        f"{(r.stderr or '').strip()[:120]}")
            elif not r.stdout.strip():
                last = "swift wrote nothing to stdout"
            else:
                return json.loads(r.stdout)
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:120]}"
        if attempt < tries - 1:
            time.sleep(1.5 * (attempt + 1))     # back off; the usual cause
    return {"_failed": last}                    # is contention, not the image


def read_text_layer(page):
    words = []
    for i, w in enumerate(page.get_text("words")):
        words.append((i, w[4], fitz.Rect(w[0], w[1], w[2], w[3])))
    return words


ORIENT_MIN_LINES = 5      # enough multi-word lines to judge direction
ORIENT_FRAC = 0.80        # share of them running right-to-left
ORIENT_KEEP = 1.40        # the re-read must be this much stronger to win


def _reads_backwards(data):
    """True when most multi-token lines have their tokens right-to-left."""
    multi = back = 0
    for line in data.get("text", []):
        xs = [t["box"]["x0"] for t in (line.get("tokens") or []) if t.get("box")]
        if len(xs) < 2:
            continue
        multi += 1
        if all(b < a for a, b in zip(xs, xs[1:])):
            back += 1
    return multi >= ORIENT_MIN_LINES and back / multi >= ORIENT_FRAC


def _strength(data):
    """How confidently a reading was made: sum of confidence x token count."""
    return sum(float(l.get("confidence", 0) or 0) * max(1, len(l.get("tokens") or []))
               for l in data.get("text", []))


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
    # ORIENTATION GUARD. Vision's current text recogniser can decide an
    # upright page is upside down -- it did so on every dark ultrasound
    # printout of one bundle -- and transcribe each glyph flipped while
    # keeping the boxes where they are: a 9-digit patient ID became a run of letter look-alikes. Both
    # detectors were then handed garbage and correctly returned nothing, so a
    # patient ID printed six times on the page survived five times. The tell
    # is unambiguous: words on a line come back RIGHT-TO-LEFT. When most
    # multi-word lines do, re-read the same render with the recogniser that
    # never re-orients, and keep that reading only if it is clearly stronger
    # (measured: misread pages 1.9-3.6x, genuinely rotated pages 0.55-0.99x).
    if "_failed" not in data and _reads_backwards(data):
        alt = _vision_text(img, mode="text_upright")
        if "_failed" not in alt and _strength(alt) >= ORIENT_KEEP * _strength(data):
            data = alt
            data["_orientation"] = "upright_reread"
    img.unlink(missing_ok=True)
    if "_failed" in data:
        return [], data["_failed"]

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
    return words, None


RASTER_DPI = 300      # embedded screenshots carry small type; the page's own
                      # OCR_DPI is chosen for a full A4 sheet, not a 227pt crop
RASTER_MIN_PT = 40    # smaller than this is a logo or an icon, not a document


def read_embedded_rasters(page, words, dpi=RASTER_DPI, workdir=None):
    """OCR embedded images that the page's own text layer does not cover.

    A born-digital page can paste in a raster that itself contains rendered
    text -- a device screenshot, a photographed document, a scanned
    attachment. read_text_layer() sees only the PDF's own glyphs, so that
    text is never read by anything: on page 14 of the reference bundle an
    A-scan biometry screenshot carries a patient name and a hospital ID, and
    the page reports 344 words, NONE of them inside the image.

    This is the same failure that hid every face in the corpus -- content
    below the resolution at which the chosen reader can see it -- and it gets
    the same answer: render the region on its own terms and read it there.

    Geometry comes from page.get_image_rects(), the placement rectangle the
    PDF's own content stream declares, so a box derived here is clipped to
    the image by construction and cannot escape onto the page around it.
    """
    out = []
    try:
        images = page.get_images(full=True)
    except Exception:
        return out
    covered = [r for _, _, r in words]
    for img in images:
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if rect.width < RASTER_MIN_PT or rect.height < RASTER_MIN_PT:
                continue
            # If the text layer already describes this area, the text is
            # being read the normal way and re-reading it would duplicate
            # every token. Only an image NO reader has covered qualifies.
            if any(not (w & rect).is_empty for w in covered):
                continue
            wd = Path(workdir or "/tmp")
            wd.mkdir(parents=True, exist_ok=True)
            img_path = wd / f"_raster_p{page.number}_{xref}.png"
            try:
                page.get_pixmap(dpi=dpi, clip=rect).save(str(img_path))
                data = _vision_text(img_path)
            except Exception:
                data = {"_failed": "render or OCR raised"}
            finally:
                img_path.unlink(missing_ok=True)
            if "_failed" in data:
                continue
            for line in data.get("text", []):
                for t in (line.get("tokens")
                          or [{"text": line.get("text", ""),
                               "box": line.get("box")}]):
                    b = t.get("box")
                    if not b or not str(t.get("text", "")).strip():
                        continue
                    # Vision's box is normalised to the CLIP it was handed,
                    # so it maps back through the clip's own origin and size.
                    out.append((str(t["text"]), fitz.Rect(
                        rect.x0 + b["x0"] * rect.width,
                        rect.y0 + b["y0"] * rect.height,
                        rect.x0 + b["x1"] * rect.width,
                        rect.y0 + b["y1"] * rect.height)))

    # PHASE 2: images phase 1 skipped because text sits ON them. A lab
    # report's letterhead and footer -- hospital name, web domain, a
    # "Services :" line naming the hospital -- are printed INTO a full-page
    # background image; the body text is the real text layer on top. Phase 1
    # skips any image a text word touches, and a page-sized background always
    # touches the body, so the letterhead was never read and nothing could
    # redact it. Read those images too, smallest first, and keep only tokens
    # that nothing else already accounts for (<=30% covered by text-layer
    # words or earlier tokens). Phase 1 is unchanged, so today's stream is an
    # exact prefix and every existing index keeps its meaning.
    skipped = []
    for img in images:
        try:
            rects = page.get_image_rects(img[0])
        except Exception:
            continue
        for rect in rects:
            if rect.width < RASTER_MIN_PT or rect.height < RASTER_MIN_PT:
                continue
            if any(not (w & rect).is_empty for w in covered):
                skipped.append((img[0], rect))
    skipped.sort(key=lambda xr: xr[1].width * xr[1].height)
    known = list(covered) + [r for _t, r in out]

    def _mostly_known(r):
        area = max(r.width * r.height, 1e-6)
        seen = sum((r & k).width * (r & k).height for k in known
                   if not (r & k).is_empty)
        return seen / area > 0.30

    for xref, rect in skipped:
        wd = Path(workdir or "/tmp")
        wd.mkdir(parents=True, exist_ok=True)
        img_path = wd / f"_raster2_p{page.number}_{xref}_{int(rect.x0)}.png"
        try:
            page.get_pixmap(dpi=dpi, clip=rect).save(str(img_path))
            data = _vision_text(img_path)
        except Exception:
            data = {"_failed": "render or OCR raised"}
        finally:
            img_path.unlink(missing_ok=True)
        if "_failed" in data:
            continue
        for line in data.get("text", []):
            for t in (line.get("tokens")
                      or [{"text": line.get("text", ""), "box": line.get("box")}]):
                b = t.get("box")
                if not b or not str(t.get("text", "")).strip():
                    continue
                r = fitz.Rect(rect.x0 + b["x0"] * rect.width,
                              rect.y0 + b["y0"] * rect.height,
                              rect.x0 + b["x1"] * rect.width,
                              rect.y0 + b["y1"] * rect.height)
                if _mostly_known(r):
                    continue
                known.append(r)
                out.append((str(t["text"]), r))
    return out


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
    err = None
    if chosen == "text_layer":
        words = read_text_layer(page)
    else:
        words, err = read_vision_ocr(page, dpi, workdir)
    # Reporting "vision_ocr" for a page whose OCR returned nothing is what
    # made this failure invisible: the run log looked identical to a page
    # that was read successfully and simply had little on it.
    if chosen == "vision_ocr" and not words:
        chosen = "ocr_empty"
    # Text inside an embedded raster on a born-digital page. APPENDED, so
    # indices 0..n-1 keep the meaning every caller already relies on and the
    # image tokens simply extend the stream. Everything downstream -- the
    # detectors, the label pass, the regex pass, propagation -- then sees
    # this text for free, which is the whole reason to merge it into the
    # stream rather than raise findings off to one side.
    n_raster = 0
    if chosen == "text_layer":
        try:
            extra = read_embedded_rasters(page, words, workdir=workdir)
        except Exception:
            extra = []
        for text, rect in extra:
            words.append((len(words), text, rect))
            n_raster += 1
    return {"page": page_no, "reader": chosen, "words": words,
            "n_words": len(words), "ocr_error": err,
            "n_raster_words": n_raster,
            "page_rect": [page.rect.width, page.rect.height]}


if __name__ == "__main__":
    D = corpus_dir()
    W = Path("/private/tmp/claude-501/-Users-shivangi-Documents-Claude-Audio-"
             "PII-Project/87ddc989-26e9-49df-9724-61c42bfe09fe/scratchpad/rd")
    cases = [("case-48043-ad818499-4046-45e1-adf6-ff971505a2fa.pdf", 47,
              "born-digital"),
             ("case-48053-7fed243c-d873-4021-a864-51dec62fa850.pdf", 1,
              "handwritten scan")]
    ran = 0
    for fname, pno, label in cases:
        # The corpus is not in the repository and PII_CORPUS_DIR may be
        # unset, so these assertions are skipped rather than fatal: a
        # self-test that cannot find its fixtures has nothing to say, and
        # crashing makes the module look broken when it is not.
        if not (D / fname).exists():
            print(f"{label:<18} SKIP (no corpus at {D})")
            continue
        ran += 1
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
    if ran:
        print("readers self-test OK "
              "(both paths return in-page PDF-point boxes)")
    else:
        print(f"readers self-test SKIPPED: no corpus at {D}. "
              f"Set PII_CORPUS_DIR to run the corpus-backed assertions.")
