#!/usr/bin/env python3
"""P8 — the gate. Re-read the OUTPUT and prove each value is actually gone.

    python3 document_pipeline/verify_removal.py --findings out/findings.jsonl \
            --redacted out/redacted --label true_redaction

"Detected" is not "removed". This checks the artifact, not the intention:

  text        the value no longer appears in any page's extracted text
  raw stream  the value no longer appears in ANY decompressed object stream -
              catches content living in an object the page tree no longer
              references, which page-level extraction cannot see
  pixels      the redacted rect renders as uniform fill (variance ~ 0) -
              catches "annotation drawn, pixels untouched"
  objects     metadata, XMP, bookmarks, annotations and form fields are clean

A value is REMOVED only if every applicable check passes. The naive cosmetic
redactor must fail `text`; if it does not, the verifier is not measuring
anything and no result from it should be believed.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz
import numpy as np

MIN_LEN = 4               # ignore 1-3 char values: unsearchable, huge FP rate


def xobject_ink(page, box):
    """Is this region dark INSIDE THE EMBEDDED IMAGE, not just on the render?

    THE control test. A cosmetic redactor paints an opaque rectangle over the
    page; the composited render is then black in exactly the same way a real
    redaction is, so a render-based pixel check cannot tell them apart -- and
    on this corpus 79% of pages are image-only, which is where the text check
    has nothing to examine either. Measured: cosmetic and true redaction both
    scored residual_text=0, residual_pixels=0, so the negative control PASSED
    and the verifier was proving nothing on the majority of pages.

    Decoding the XObject sees through the overlay: apply_redactions rewrites
    the image's pixels to a uniform fill, draw_rect leaves them untouched. The
    discriminator is therefore INK VARIANCE inside the image, not darkness --
    measured on an image-only page, a cosmetic cover-up still carries 5.4%
    dark ink where a true redaction carries 0.0% and zero variance.
    -> (std_inside_image, checked_any_image)
    """
    best, seen = 0.0, False
    r = fitz.Rect(box)
    for im in page.get_images(full=True):
        xref = im[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for pr in rects:
            inter = r & pr
            if inter.is_empty or inter.width < 1 or inter.height < 1:
                continue
            try:
                pix = fitz.Pixmap(page.parent, xref)
                if pix.n > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                a = np.frombuffer(pix.samples, np.uint8).reshape(
                    pix.h, pix.w, pix.n)[..., :3]
            except Exception:
                continue
            # map the page-space box into the image's own pixel grid
            fx0 = (inter.x0 - pr.x0) / max(pr.width, 1e-6)
            fx1 = (inter.x1 - pr.x0) / max(pr.width, 1e-6)
            fy0 = (inter.y0 - pr.y0) / max(pr.height, 1e-6)
            fy1 = (inter.y1 - pr.y0) / max(pr.height, 1e-6)
            x0, x1 = int(fx0 * pix.w), max(int(fx1 * pix.w), int(fx0 * pix.w) + 1)
            y0, y1 = int(fy0 * pix.h), max(int(fy1 * pix.h), int(fy0 * pix.h) + 1)
            sub = a[max(y0, 0):y1, max(x0, 0):x1]
            if sub.size == 0:
                continue
            seen = True
            best = max(best, float(sub.std()))
    return best, seen


def text_in_box(page, box, pad=0.5):
    """Text still living INSIDE the redacted rectangle.

    This is the authoritative mechanism test, and substring search is not.
    Measured case: a hospital-bill page contains "0.00" eighteen times; four
    were detected and correctly redacted, yet a document-wide search still
    finds the string and reports a failure that did not happen. Position is
    unambiguous where the value is not.
    """
    import fitz
    r = fitz.Rect(box)
    r = fitz.Rect(r.x0 + pad, r.y0 + pad, r.x1 - pad, r.y1 - pad)
    if r.is_empty or r.width <= 0 or r.height <= 0:
        return ""
    try:
        return page.get_text(clip=r).strip()
    except Exception:
        return ""


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def survivors(redacted_path, findings_by_page):
    """Which DESTRUCTIVE values are still present in the finished document?

    The programmatic half of this module. `main()` prints a report for a human;
    this returns the values themselves, which is what a correction pass needs:
    not "3 failures" but "this string, on this page, in this box".

    -> [{"page", "text", "bbox", "where"}]
    """
    doc = fitz.open(redacted_path)
    out = []
    for pno, finds in (findings_by_page or {}).items():
        if pno > doc.page_count:
            continue
        page = doc[pno - 1]
        for f in finds:
            if f.get("action") in ("annotate_only", "KEEP"):
                continue
            t = str(f.get("text", "")).strip()
            if len(norm(t)) < MIN_LEN or t.startswith("["):
                continue
            for box in (f.get("all_boxes") or
                        ([f.get("bbox")] if f.get("bbox") else [])):
                if not box:
                    continue
                if norm(t) in norm(text_in_box(page, box)):
                    out.append({"page": pno, "text": t, "bbox": list(box),
                                "where": "inside its own box",
                                "categories": f.get("categories") or [],
                                "action": f.get("action")})
                    break
    doc.close()
    return out


def verify_document(redacted_path, findings_by_page):
    doc = fitz.open(redacted_path)
    wanted = []
    for pno, finds in findings_by_page.items():
        for f in finds:
            # Only DESTRUCTIVE findings are the gate's business. A five-action
            # pack deliberately KEEPs clinical content, dates and amounts, and
            # checking those as if they should have been destroyed made the
            # removal gate permanently red -- on one 45-page run, 117 KEEP
            # values would each have been scored as a surviving leak. A gate
            # that always fails cannot signal a real regression.
            if f.get("action") in ("annotate_only", "KEEP"):
                continue
            t = str(f.get("text", "")).strip()
            if len(norm(t)) >= MIN_LEN:
                wanted.append((pno, t, f))

    # TWO different questions, never conflated:
    #   page_text  - did the redaction mechanism work on the page we redacted?
    #   doc_text   - does the value still live elsewhere in the document?
    # A value redacted on p12 and untouched on p40 is still a leak, but that is
    # a COVERAGE failure (we never looked at p40), not a mechanism failure.
    page_text = {i + 1: norm(doc[i].get_text()) for i in range(doc.page_count)}
    all_text = norm("".join(page_text.values()))

    # raw object streams: text can survive in an unreferenced object
    raw_blob = ""
    try:
        for x in range(1, doc.xref_length()):
            try:
                s = doc.xref_stream(x)
            except Exception:
                continue
            if s and len(s) < 4_000_000:
                raw_blob += s.decode("latin-1", "ignore")
    except Exception:
        pass
    raw_blob = norm(raw_blob)

    res = {"checked": 0, "residual_text": 0, "residual_raw": 0,
           "residual_pixels": 0, "elsewhere_in_doc": 0, "examples": []}
    for pno, text, f in wanted:
        res["checked"] += 1
        n = norm(text)
        # authoritative: is anything still inside the box we redacted?
        first_box = (f.get("all_boxes") or [f.get("bbox")])[0]
        in_box = bool(first_box) and bool(
            text_in_box(doc[pno - 1], first_box)) if pno <= doc.page_count \
            else False
        in_text = in_box                               # the redacted page
        in_raw = n in raw_blob
        if n in all_text and not in_text:
            res["elsewhere_in_doc"] += 1               # coverage gap, not
        if in_text:                                    # a mechanism failure
            res["residual_text"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append({"page": pno, "len": len(text),
                                        "where": "page text"})
        if in_raw and not in_text:
            res["residual_raw"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append({"page": pno, "len": len(text),
                                        "where": "raw object stream"})
        # Pixel check, on EVERY placement and INSIDE the image object.
        # Checking only the first box exempted every extra occurrence -- the
        # ones the OCR-tolerant matcher and propagation placed, which are
        # exactly the ones most likely to be wrong. Checking the composited
        # render instead of the XObject let a painted-over rectangle pass.
        for box in (f.get("all_boxes") or ([f.get("bbox")] if f.get("bbox") else [])):
            if not box or pno > doc.page_count:
                continue
            page = doc[pno - 1]
            res["boxes_checked"] = res.get("boxes_checked", 0) + 1
            try:
                xstd, saw_image = xobject_ink(page, box)
            except Exception:
                xstd, saw_image = 0.0, False
            if saw_image:
                if xstd > 12:                 # ink survives inside the image
                    res["residual_pixels"] += 1
                continue
            try:
                pix = page.get_pixmap(clip=fitz.Rect(box), dpi=150,
                                      colorspace=fitz.csGRAY)
                if pix.width > 1 and pix.height > 1:
                    arr = np.frombuffer(pix.samples, dtype=np.uint8)
                    if float(arr.std()) > 12:
                        res["residual_pixels"] += 1
            except Exception:
                pass

    obj = {}
    meta = {k: v for k, v in (doc.metadata or {}).items()
            if v and k in ("title", "author", "subject", "keywords")}
    if meta:
        obj["metadata"] = meta
    if doc.get_toc():
        obj["bookmarks"] = len(doc.get_toc())
    try:
        if doc.xref_xml_metadata():
            obj["xmp"] = True
    except Exception:
        pass
    n_annot = sum(len(list(doc[i].annots() or []))
                  for i in range(min(doc.page_count, 40)))
    if n_annot:
        obj["annotations"] = n_annot
    res["note"] = ("residual_text is measured POSITIONALLY (text still "
                   "inside the redacted rect), not by substring search")
    res["object_level"] = obj
    doc.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--findings", default="out/findings.jsonl")
    ap.add_argument("--redacted", default="out/redacted")
    ap.add_argument("--suffix", default="_redacted")
    ap.add_argument("--label", default="true_redaction")
    args = ap.parse_args()

    by_doc = defaultdict(lambda: defaultdict(list))
    for line in Path(args.findings).open():
        r = json.loads(line)
        by_doc[r["file"]][r["page"]].extend(r["findings"])

    tot = defaultdict(int)
    print(f"=== verifying: {args.label} ===")
    for fname, pf in by_doc.items():
        path = Path(args.redacted) / (Path(fname).stem + args.suffix + ".pdf")
        if not path.exists():
            print(f"  MISSING {path.name}")
            continue
        r = verify_document(path, pf)
        for k in ("checked", "residual_text", "residual_raw",
                  "residual_pixels", "elsewhere_in_doc"):
            tot[k] += r[k]
        obj = r["object_level"]
        print(f"  {fname[:30]:<32}checked {r['checked']:>3} | "
              f"text {r['residual_text']:>3} | raw {r['residual_raw']:>3} | "
              f"pixels {r['residual_pixels']:>3}"
              + (f" | OBJECT LEAKS: {','.join(obj)}" if obj else ""))

    c = max(1, tot["checked"])
    print(f"\n  values checked:            {tot['checked']}")
    print(f"  residual in page text:     {tot['residual_text']}"
          f"  ({tot['residual_text']/c:.0%})")
    print(f"  residual in raw streams:   {tot['residual_raw']}")
    print(f"  boxes not uniformly filled:{tot['residual_pixels']}")
    print(f"  same value on OTHER pages: {tot['elsewhere_in_doc']}"
          f"  (coverage gap - those pages were never in the sample)")
    # Two independent verdicts. Merging them hides which one failed, and they
    # need opposite fixes: a mechanism failure means the redactor is wrong; a
    # coverage failure means we never looked at the page where the value also
    # appears. Raw-stream residual is reported document-wide and is therefore
    # only meaningful when every page of the document was redacted.
    mech_ok = tot["residual_text"] == 0 and tot["residual_pixels"] == 0
    if tot["checked"] == 0:
        # Zero residual out of zero checks is not a pass, it is an absence of
        # evidence -- and it prints identically to a real pass. On this corpus
        # that is the common case, not a corner: 79% of pages are image-only,
        # so a text-residual check there has nothing to examine and a broken
        # redactor would have sailed through.
        print(f"\n  MECHANISM [{args.label}]: INCONCLUSIVE - 0 values were "
              f"checked, so nothing was proved. A text-residual check needs a "
              f"text layer; on image-only pages rely on the pixel check.")
    else:
        print(f"\n  MECHANISM [{args.label}]: "
              + ("PASS - nothing survives inside any redacted box"
                 if mech_ok else
                 f"FAIL - {tot['residual_text']} values still inside their box"))
    if tot["elsewhere_in_doc"]:
        print(f"  COVERAGE: INCOMPLETE - {tot['elsewhere_in_doc']} value(s) "
              f"also appear on pages that were never redacted")
    else:
        print("  COVERAGE: every checked value appears only where redacted")
    # EXIT CODE, so a script, a CI job or a batch over 500 documents can see
    # what a human reading the output sees. Printing FAIL and returning 0 made
    # this a report, not a gate.
    return 0 if mech_ok else 1


if __name__ == "__main__":
    sys.exit(main())
