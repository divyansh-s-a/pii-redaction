#!/usr/bin/env python3
"""Score a run against the synthetic ground truth.

TRUE recall and precision, not judge-estimated: every item's category and box
were placed by the generator, so a miss is a miss and a false positive is a
false positive. Reported separately for the born-digital and degraded-scan
renditions of the SAME documents, because the gap between them is the honest
measure of how much the scan path costs.

    python3 synth/score.py --dir out/synth
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import fitz

COVER = 0.80        # a ground-truth item counts as removed at this coverage
BLACK = 0.80        # fraction of the item's pixels that must be destroyed


def removed(doc, item):
    """Is this value ACTUALLY gone from the redacted artifact?

    Measured on the output, not on the findings. Counting boxes only proves the
    pipeline INTENDED to redact -- it cannot catch a box that was dropped, drawn
    on the wrong page, or silently downgraded. Two independent checks, because
    either alone is vacuous somewhere: text absence says nothing on an
    image-only page, and pixel darkness says nothing about an invisible OCR
    layer underneath.
    """
    page = doc[item["page"] - 1]
    r = fitz.Rect(item["bbox"])
    if r.width < 1 or r.height < 1:
        r = fitz.Rect(r.x0, r.y0, r.x0 + 2, r.y0 + 2)
    pm = page.get_pixmap(dpi=150, clip=r)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3]
    dark = float((a.max(axis=2) < 50).mean()) if a.size else 0.0
    val = str(item.get("value") or "")
    text_gone = val.startswith("[") or val not in page.get_text()
    return dark >= BLACK, text_gone


def _area(r):
    return max(r.width, 0) * max(r.height, 0)


def cover(item, boxes):
    a = fitz.Rect(item)
    if _area(a) <= 0:
        a = fitz.Rect(a.x0, a.y0, a.x1 + 1, a.y1 + 1)
    tot = 0.0
    for b in boxes:
        inter = a & fitz.Rect(b)
        if not inter.is_empty:
            tot += _area(inter)
    return min(1.0, tot / max(_area(a), 1e-6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="out/synth")
    a = ap.parse_args()
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text())
    agg = defaultdict(lambda: dict(n=0, hit=0, boxes=0, boxes_hit=0))
    percat = defaultdict(lambda: [0, 0])
    print(f"{'document':<34}{'variant':<10}{'items':>6}{'found':>7}{'recall':>9}"
          f"{'boxes':>7}{'on-target':>11}")
    for m in man:
        red = d / f"red_{Path(m['file']).stem}.jsonl"
        boxes_by_page = defaultdict(list)
        if red.exists():
            for line in red.open():
                r = json.loads(line)
                for f in r["findings"]:
                    for b in (f.get("all_boxes")
                              or ([f["bbox"]] if f.get("bbox") else [])):
                        boxes_by_page[r["page"]].append(b)
        red_pdf = d / f"red_{Path(m['file']).stem}.pdf"
        rdoc = fitz.open(red_pdf) if red_pdf.exists() else None
        hit = 0
        for it in m["items"]:
            if rdoc is not None:
                pix_gone, text_gone = removed(rdoc, it)
                ok = pix_gone and text_gone
            else:
                ok = cover(it["bbox"], boxes_by_page.get(it["page"], [])) >= COVER
            hit += ok
            percat[it["category"]][0] += ok
            percat[it["category"]][1] += 1
        if rdoc is not None:
            rdoc.close()
        nb = sum(len(v) for v in boxes_by_page.values())
        on = 0
        for pg, bs in boxes_by_page.items():
            for b in bs:
                if any(it["page"] == pg and not (fitz.Rect(b) & fitz.Rect(it["bbox"])).is_empty
                       for it in m["items"]):
                    on += 1
        v = agg[m["variant"]]
        v["n"] += len(m["items"]); v["hit"] += hit
        v["boxes"] += nb; v["boxes_hit"] += on
        print(f"{m['file'][:33]:<34}{m['variant']:<10}{len(m['items']):>6}{hit:>7}"
              f"{hit/max(len(m['items']),1):>8.0%}{nb:>7}{on/max(nb,1):>10.0%}")
    print()
    for var, v in agg.items():
        print(f"{var:<10} TRUE recall {v['hit']}/{v['n']} = {v['hit']/max(v['n'],1):>5.0%}"
              f"   |  boxes landing on a real item "
              f"{v['boxes_hit']}/{v['boxes']} = {v['boxes_hit']/max(v['boxes'],1):>4.0%}")
    print("\nrecall by category (both variants pooled):")
    for c, (h, n) in sorted(percat.items(), key=lambda kv: -kv[1][1]):
        print(f"   {c:<28}{h}/{n} = {h/max(n,1):>4.0%}")
    print("\nRecall is measured on the REDACTED ARTIFACT: a value counts as "
          "removed only if its\npixels are destroyed AND the string is absent "
          "from the page text. Counting boxes\nwould instead measure intent.")
    print("\nNOTE: 'boxes landing on a real item' is not precision in the usual "
          "sense -- a box over a company name or a salary is correct behaviour "
          "for a pack that cannot tell domains apart, and counts as off-target "
          "here. It measures over-redaction, and for generic_document a low "
          "number is the DESIGNED trade, not necessarily a defect.")


if __name__ == "__main__":
    main()
