#!/usr/bin/env python3
"""P2 — deterministic census of a document corpus, plus the object-level leak
audit. No API calls, no model, no cost.

    python3 benchmark/census.py CORPUS_DIR [--pixels] [--out out/]

Two passes:
  structural  (always)  per page: kind, text size, images, native dpi,
                        rotation, fonts, layout fingerprint
  pixel       (--pixels) rasterises each page once at low dpi: ink fraction
                        and SOLID BLACK RECTANGLES — the latter recovers a
                        previous redactor's decisions for free, which is what
                        makes the already-redacted files usable as a
                        validation set.

The leak audit looks where page content is not: metadata, XMP, bookmarks,
annotations, form fields, embedded files. That is where this corpus actually
leaks — four already-"redacted" files still carry the patient's given name in
/Title.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
import analyze as router_analyze

import fitz
import numpy as np
from scipy import ndimage

PIXEL_DPI = 100          # enough for ink fraction and black-box geometry
BLACK_MAX = 60           # 0-255; below this counts as "solid dark"
MIN_BOX_PX = 12          # ignore specks
MIN_FILL = 0.92          # a redaction box is essentially solid


def solid_dark_rects(page, dpi=PIXEL_DPI):
    """Find burned-in solid rectangles (a previous redactor's boxes).

    Deliberately geometric, not semantic: label connected dark regions and keep
    those that fill their own bounding box. Text ink never does - glyphs leave
    the box mostly empty.  Returns rects in PDF points, plus the ink fraction.
    """
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,
                                                             pix.width)
    dark = arr < BLACK_MAX
    ink = float(dark.mean())
    rects = []
    if dark.any():
        lab, n = ndimage.label(dark)
        for sl in ndimage.find_objects(lab)[:4000]:
            ys, xs = sl
            h, w = ys.stop - ys.start, xs.stop - xs.start
            if h < MIN_BOX_PX or w < MIN_BOX_PX:
                continue
            if dark[sl].mean() < MIN_FILL:
                continue
            s = 72.0 / dpi
            rects.append([round(xs.start * s, 1), round(ys.start * s, 1),
                          round(xs.stop * s, 1), round(ys.stop * s, 1)])
    return rects, ink


def _safe(obj):
    """Strip unpaired surrogates before JSON.

    Real case: the leaked /Title values in this corpus end in lone surrogates
    (b'\xc0\x80' decoded loosely), which json.dumps cannot encode. Losing the
    report because the leak itself is malformed would be the wrong failure.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8", "replace")
    if isinstance(obj, dict):
        return {_safe(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(v) for v in obj]
    return obj


def layout_fingerprint(page):
    """Cheap signature used later to group pages into runs of one document, so
    a genre label can be inferred once per run instead of per page."""
    sizes = sorted((im[2], im[3]) for im in page.get_images(full=True))
    r = page.rect
    return f"{int(r.width)}x{int(r.height)}|{len(sizes)}|" + \
           ",".join(f"{w}x{h}" for w, h in sizes[:4])


def census_document(path, do_pixels=False):
    doc = fitz.open(path)
    rows = []
    for i in range(doc.page_count):
        page = doc[i]
        kind, dpi = router_analyze._classify_page(page)
        text = page.get_text()
        row = {
            "file": Path(path).name, "page": i + 1, "kind": kind,
            "chars": len(text.strip()),
            "words": len(page.get_text("words")),
            "images": len(page.get_images(full=True)),
            "native_dpi": round(dpi) if dpi else None,
            "rotation": page.rotation,
            "drawings": len(page.get_drawings()),
            "scripts": router_analyze._scripts(text[:4000]),
            "fingerprint": layout_fingerprint(page),
        }
        if do_pixels:
            try:
                rects, ink = solid_dark_rects(page)
                row["ink_fraction"] = round(ink, 4)
                row["solid_dark_rects"] = rects
                row["n_solid_dark_rects"] = len(rects)
            except Exception as exc:
                row["pixel_error"] = str(exc)[:80]
        rows.append(row)
    leaks = router_analyze._object_leaks(doc)
    doc.close()
    return rows, leaks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus")
    ap.add_argument("--pixels", action="store_true",
                    help="also rasterise each page (ink fraction + solid "
                         "dark rectangles); slower but finds prior redactions")
    ap.add_argument("--out", default="out")
    ap.add_argument("--glob", default="*.pdf")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.corpus).glob(args.glob))
    all_rows, all_leaks = [], {}
    for f in files:
        rows, leaks = census_document(f, args.pixels)
        all_rows.extend(rows)
        if leaks:
            all_leaks[f.name] = leaks
        boxes = sum(r.get("n_solid_dark_rects", 0) for r in rows)
        print(f"  {f.name[:44]:<46}{len(rows):>4}p"
              + (f"  {boxes:>4} solid-dark rects" if args.pixels else "")
              + (f"  LEAKS: {','.join(leaks)}" if leaks else ""))

    with (out / "census.jsonl").open("w") as fh:
        for r in all_rows:
            fh.write(json.dumps(_safe(r), ensure_ascii=False) + "\n")
    (out / "leak_report.json").write_text(
        json.dumps(_safe(all_leaks), indent=1, ensure_ascii=False))

    kinds = Counter(r["kind"] for r in all_rows)
    dpis = [r["native_dpi"] for r in all_rows if r["native_dpi"]]
    scripts = Counter(s for r in all_rows for s in r["scripts"])
    print(f"\n=== census: {len(files)} documents, {len(all_rows)} pages ===")
    for k, v in kinds.most_common():
        print(f"  {k:<16}{v:>6}  ({v/len(all_rows):.0%})")
    if dpis:
        print(f"  scan dpi: median {statistics.median(dpis):.0f}, "
              f"p10 {sorted(dpis)[len(dpis)//10]:.0f}, "
              f"p90 {sorted(dpis)[9*len(dpis)//10]:.0f}")
    if scripts:
        print(f"  non-latin scripts: {dict(scripts)}")
    print(f"  documents with object-level leaks: {len(all_leaks)}/{len(files)}")
    print(f"\n  -> {out/'census.jsonl'}\n  -> {out/'leak_report.json'}")


if __name__ == "__main__":
    main()
