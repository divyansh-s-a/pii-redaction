#!/usr/bin/env python3
"""Small embedded images that carry no text: signatures and logos.

A born-digital page can place a signature or a logo as its own small raster
XObject. It has no glyphs, so no text mechanism can see it, and it is often
black or near-black, so the pigment-based carrier pass cannot see it either.
On the review bundle case-47999 that left the signatory's signature on the
promotion letter, the card holder's signature on all three insurance cards,
and the TPA's logo on all three cards readable in both packs.

WHAT THIS POPULATION ACTUALLY IS. Measured, then inspected by eye, over every
born-digital page of the 24-bundle corpus (447 pages): 307 placements of 83
distinct small no-text images. Every one was a signature, an insurer/TPA/
provider/employer/bank logo, a QR code, an ID photo, or a boilerplate
instruction block rendered as an image. NONE was clinical -- no chart, no
scan thumbnail, no lab graph. That census is the evidence for the rule
below, and the reason there is no photographic veto: the only continuous-
tone images in the population were ID photos, which must go anyway.

Two signature images in that census include the doctor's printed name and
designation as part of the picture ("Dr. Ruchismita Pal, Consultant
Pathologist"), invisible to text detection. They are covered here too.

THE ONE DISTINCTION THAT MATTERS: a BANK's mark is institution-level, like its
IFSC, and both packs keep it; an insurer's, TPA's, provider's or employer's
mark is identity, and both packs destroy it. Bank marks are the only logos in
the census whose own text says BANK, so the mark is OCR'd and a bank reading
is emitted as sem:bank_common -- the pack decides what that means, exactly as
it does for an IFSC. No provider is named anywhere in this file.

Detection states WHAT a mark is; the pack states what to do with it.
"""
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))

MARK_MAX_SIDE = 220.0     # larger than this is a banner, a chart or a scan
MARK_MIN_SIDE = 8.0       # smaller than this is a rule, a bullet or a dot
MARK_OCR_DPI = 300
BANK = re.compile(r"\bbank\b", re.I)


def _ocr_text(page, rect, workdir):
    import objective
    wd = Path(workdir or "/tmp")
    wd.mkdir(parents=True, exist_ok=True)
    f = wd / f"_mark_p{page.number}_{int(rect.x0)}_{int(rect.y0)}.png"
    try:
        page.get_pixmap(dpi=MARK_OCR_DPI, clip=rect).save(str(f))
        got = objective.vision(f, "text") or {}
    except Exception:
        return ""
    finally:
        f.unlink(missing_ok=True)
    return " ".join(str(l.get("text", "")) for l in got.get("text", []))


def findings(page, pack=None, workdir=None):
    """-> findings for every small no-text image placement on a text page."""
    if not page.get_text().strip():
        return []           # a scan's marks are part of its raster, not objects
    words = [fitz.Rect(w[:4]) for w in page.get_text("words")]
    out, seen, verdict = [], set(), {}
    try:
        images = page.get_images(full=True)
    except Exception:
        return []
    for img in images:
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            key = (xref, round(r.x0), round(r.y0))
            if key in seen:
                continue
            seen.add(key)
            if max(r.width, r.height) > MARK_MAX_SIDE:
                continue
            if min(r.width, r.height) < MARK_MIN_SIDE:
                continue
            # Text the PDF itself carries over the image is read and judged
            # the normal way; only an image nothing else describes qualifies.
            if any(not (w & r).is_empty for w in words):
                continue
            # The same xref is usually the same mark (a logo on every card),
            # so read it once and reuse the reading for each placement.
            if xref not in verdict:
                text = _ocr_text(page, r, workdir)
                verdict[xref] = ("sem:bank_common" if BANK.search(text)
                                 else "sem:visual_mark", text[:60])
            cat, text = verdict[xref]
            action = (pack.action_for([cat]) if pack is not None
                      else ("KEEP" if cat == "sem:bank_common" else "REDACT"))
            out.append({
                "categories": [cat],
                "text": "[bank mark]" if cat == "sem:bank_common"
                        else "[signature or logo]",
                "bbox": [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)],
                "all_boxes": [[round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]],
                "n_boxes": 1,
                "source": "embedded_mark", "locator": "image_rect",
                "action": action,
                "reason": (f"small no-text image xref={xref} "
                           f"{r.width:.0f}x{r.height:.0f}pt; read as "
                           f"{text!r}" if text else
                           f"small no-text image xref={xref} "
                           f"{r.width:.0f}x{r.height:.0f}pt; no readable text"),
            })
    return out
