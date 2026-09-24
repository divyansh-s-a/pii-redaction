#!/usr/bin/env python3
"""P8 — apply redactions to a PDF, for real or deliberately cosmetically.

    python3 document_pipeline/redact.py --findings out/findings.jsonl \
            --outdir out/redacted [--cosmetic]

--cosmetic is the NEGATIVE CONTROL, not an option anyone should ship. It draws
an opaque rectangle over the value and changes nothing underneath - which is
what a naive implementation does, and what looks correct in every viewer. The
verifier must catch it. If a run of --cosmetic passes verification, the
verifier is broken and no result from it means anything.

Real mode destroys content three ways, because this corpus leaks all three:
  text    PDF_REDACT_TEXT_REMOVE   deletes the glyphs (incl. an invisible OCR
                                   layer sitting under a scan)
  pixels  PDF_REDACT_IMAGE_PIXELS  overwrites pixels INSIDE the rect of an
                                   embedded image and re-encodes it, so a
                                   burned-in watermark or a scanned value is
                                   gone from the file, not covered
  objects scrub() + set_toc([])    metadata, XMP, annotations, form fields,
                                   embedded files, JavaScript, and the outline
                                   - where all four already-"redacted" files
                                   in this corpus actually leak

save(garbage=4, clean=True) is mandatory: an incremental save leaves the entire
pre-redaction object graph in the file.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import actions as A

# Findings produced before the taxonomy packs existed carry the audio
# pipeline's two-valued action. Map them rather than crashing on old artifacts.
LEGACY = {"beep": "REDACT", "annotate_only": "KEEP"}


def boxes_for(finding):
    """Every placement of the value, not just the first: a name printed three
    times on a page needs three boxes."""
    out = []
    if finding.get("all_boxes"):
        out.extend(finding["all_boxes"])
    elif finding.get("bbox"):
        out.append(finding["bbox"])
    return [fitz.Rect(b) for b in out if b]


def resolve_action(finding, pack):
    """The action for one finding, and the text to write back if any.

    Order: an explicit action on the finding wins (the detector already applied
    the pack), then the pack's own matrix, then the legacy two-valued action,
    then REDACT. The last default matters -- an unrecognised finding must
    over-redact, which is visible, rather than fall through to KEEP, which is
    a silent leak.
    """
    act = finding.get("action")
    if act in A.DESTRUCTIVE or act == "KEEP":
        return act
    if act in LEGACY:
        return LEGACY[act]
    if pack is not None:
        return pack.action_for(finding.get("categories") or ())
    return "REDACT"


def replacement_text(finding, action, pack, tokens):
    """What goes back into the vacated box, or None.

    NOTHING IS EVER WRITTEN, so this only ever feeds the audit's record of
    what was declined. PSEUDONYMIZE used to mint a stable `[PERSON_1]` here
    and write it into the vacated box; that write-back was removed on the
    owner's instruction, and apply_action now accepts `text` and discards it.

    The minting outlived the feature. It produced a `*.tokens.json` mapping
    every token to the REAL name -- a re-identification key -- written by
    default beside the output, which under dispatch.py's default lands in the
    source corpus folder. Nothing read it back: not the PDF, not the audit
    (591/591 rows carried wrote=None and no real values), not verification,
    not propagation, which matches by value rather than by token. A file of
    patient names generated for no consumer is a liability with no benefit,
    so it is no longer produced.
    """
    if action == "PSEUDONYMIZE":
        return None
    if action == "GENERALIZE":
        return A.generalize(finding.get("text"), finding.get("categories") or ())
    return None


def neutralise_inherited_redactions(page):
    """Strip overlay TEXT from redaction marks the source already carried.

    A partner-supplied "already redacted" PDF often carries Acrobat redaction
    annotations that were marked but never applied, each with an overlay label
    like "REDACTED BY ACME LLP". apply_redactions() BURNS that label into the
    page -- so our output would contain text no one here authored, on pages
    this pipeline found nothing on. Measured: a page reading "Hospital bill
    total 4520.00" came out reading "... CONFIDENTIAL XKPQ", and
    added_text() was non-empty. That breaks the one invariant this module
    exists to hold.

    The region is still destroyed -- more redaction is the safe direction, and
    leaving an unapplied mark in the output would cover without destroying.
    Only the text is dropped: the annotation is deleted and re-added with the
    same rect and no overlay.
    -> number of inherited marks neutralised
    """
    try:
        annots = list(page.annots() or [])
    except Exception:
        return 0
    rects, n = [], 0
    for a in annots:
        try:
            if a.type[0] != fitz.PDF_ANNOT_REDACT:
                continue
            rects.append(fitz.Rect(a.rect))
            page.delete_annot(a)
            n += 1
        except Exception:
            continue
    for r in rects:
        try:
            page.add_redact_annot(r, fill=BLACK)      # no `text=`
        except Exception:
            pass
    return n


def redact_document(src, page_findings, dest, cosmetic=False, pad=1.0,
                    jpeg_quality=85, pack=None, tokens=None, audit=None):
    """page_findings: {page_no(1-based): [finding, ...]} -> stats

    Every destructive action uses the same removal primitive, so PSEUDONYMIZE
    and GENERALIZE destroy the original exactly as hard as REDACT does; only
    what is written back differs.
    """
    doc = fitz.open(src)
    n_boxes = n_pages = 0
    tally, degraded = Counter(), Counter()
    for pno, finds in sorted(page_findings.items()):
        page = doc[pno - 1]
        # BEFORE we add any of our own marks. Run afterwards it cannot tell an
        # inherited annotation from one apply_action just created, so it
        # deletes and re-adds ours and the application silently stops removing
        # text -- measured: a re-redaction pass reported 1 box drawn and left
        # the value readable.
        inherited = 0 if cosmetic else neutralise_inherited_redactions(page)
        if inherited:
            degraded["inherited_redaction_marks_stripped"] += inherited
        applied = 0
        for f in finds:
            action = resolve_action(f, pack)
            if action == "KEEP":
                tally["KEEP"] += 1
                continue
            text = replacement_text(f, action, pack, tokens)
            for r in boxes_for(f):
                info = A.apply_action(page, r, action, text=text, pad=pad,
                                      cosmetic=cosmetic)
                tally[info["action"]] += 1
                if info["degraded"]:
                    degraded[info["degraded"]] += 1
                if audit is not None:
                    audit.append({"page": pno, "action": info["action"],
                                  "requested": action, "wrote": info["wrote"],
                                  "degraded": info["degraded"],
                                  "categories": f.get("categories"),
                                  "source": f.get("source"),
                                  "bbox": [round(v, 2) for v in r]})
                applied += 1
        if (applied or inherited) and not cosmetic:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS,
                                  graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                                  text=fitz.PDF_REDACT_TEXT_REMOVE)
        n_boxes += applied
        n_pages += 1 if applied else 0

    if not cosmetic and jpeg_quality:
        # apply_redactions() decodes each touched page image, paints the
        # blacked-out pixels in, and re-encodes it as PNG/Flate -- so a
        # JPEG-scanned bundle balloons (measured 10.4 MB -> 55.6 MB, 33 of 65
        # images converted). Put them back to JPEG. This runs on the ALREADY
        # blackened pixels, so it cannot resurrect anything; the cost is a
        # second-generation JPEG on pages that were touched.
        doc.rewrite_images(lossy=True, lossless=True, quality=jpeg_quality,
                           color=True, gray=True)

    if not cosmetic:
        # Delete form widgets outright rather than relying on scrub's
        # reset_fields. A reset leaves the field in place and MuPDF then writes
        # a DEFAULT APPEARANCE for it -- on the Aadhaar page an empty signature
        # widget reappeared as the text 'SIGN', which the document never
        # contained. Under a redaction-only rule the output may not gain a
        # character, including one a library helpfully supplies.
        for page in doc:
            for w in list(page.widgets()):
                try:
                    page.delete_widget(w)
                except Exception:
                    pass

        # object-level hygiene: this is where patient-*.pdf lost the game
        doc.scrub(metadata=True, xml_metadata=True, embedded_files=True,
                  hidden_text=False, javascript=True, reset_fields=True,
                  thumbnails=True, remove_links=False)
        doc.set_toc([])                      # scrub() does NOT clear bookmarks
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dest, garbage=4, deflate=True, clean=True)
    doc.close()
    return {"pages_touched": n_pages, "boxes": n_boxes,
            "by_action": dict(tally), "degraded": dict(degraded),
            "mode": "cosmetic" if cosmetic else "true_redaction"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--findings", default="out/findings.jsonl")
    ap.add_argument("--outdir", default="out/redacted")
    ap.add_argument("--corpus", default=str(readers.corpus_dir()))
    ap.add_argument("--cosmetic", action="store_true",
                    help="NEGATIVE CONTROL: draw a rectangle, change nothing")
    args = ap.parse_args()

    by_doc = defaultdict(lambda: defaultdict(list))
    for line in Path(args.findings).open():
        r = json.loads(line)
        by_doc[r["file"]][r["page"]].extend(r["findings"])

    outdir = Path(args.outdir)
    tot = {"pages_touched": 0, "boxes": 0}
    for fname, pf in by_doc.items():
        suffix = "_cosmetic" if args.cosmetic else "_redacted"
        dest = outdir / (Path(fname).stem + suffix + ".pdf")
        st = redact_document(Path(args.corpus) / fname, pf, dest,
                             args.cosmetic)
        tot["pages_touched"] += st["pages_touched"]
        tot["boxes"] += st["boxes"]
        print(f"  {fname[:34]:<36}{st['boxes']:>4} boxes on "
              f"{st['pages_touched']:>3} pages -> {dest.name}")
    print(f"\nmode={'COSMETIC (control)' if args.cosmetic else 'true redaction'}"
          f" | {tot['boxes']} boxes, {tot['pages_touched']} pages")
    print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
