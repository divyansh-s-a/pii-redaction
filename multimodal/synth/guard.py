#!/usr/bin/env python3
"""Did a shared-engine change move a pack it was not aimed at?

The document arm is one engine with pluggable packs, so an edit made to serve
one pack runs under every other. This snapshots what the DETERMINISTIC
mechanisms -- labels, unlabelled blocks, ink zoning, the objective regex pass
-- produce on a real document under a given pack, and diffs a later run
against it. No model is called, so the comparison is exact and repeatable;
model variance would make it meaningless.

    # before a change
    python3 synth/guard.py BUNDLE.pdf --taxonomy insurance_medical \
            --save out/guard/insurance.json

    # after
    python3 synth/guard.py BUNDLE.pdf --taxonomy insurance_medical \
            --baseline out/guard/insurance.json

Findings are compared by EFFECT -- (page, source, action, text) -- not by
category name, because renaming a semantic key while preserving its action is
a refactor, and a guard that reports it as a regression trains you to ignore
the guard. Exit code is non-zero when anything is lost or added.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "document_pipeline"))
sys.path.insert(0, str(ROOT / "taxonomies"))

import fitz                       # noqa: E402
import base                       # noqa: E402
import formfields                 # noqa: E402
import handwriting                # noqa: E402
import objective                  # noqa: E402


def snapshot(pdf, pack):
    doc = fitz.open(pdf)
    rows = []
    for pno in range(doc.page_count):
        page = doc[pno]
        words = [(i, w[4], fitz.Rect(w[:4]))
                 for i, w in enumerate(page.get_text("words"))]
        try:
            ink, st = handwriting.ink_boxes(page)
        except Exception:
            ink, st = [], {"blue_share": 0.0}
        try:
            labs = formfields.value_regions(page, words, ink=ink, pack=pack)
        except Exception:
            labs = []
        try:
            blocks = formfields.unlabelled_identity_blocks(
                page, words, labels=labs, pack=pack)
        except Exception:
            blocks = []
        try:
            hw, _ = handwriting.regions(page, labels=labs, boxes=ink, stats=st)
        except Exception:
            hw = []
        obj = [{"categories": [h["category"]], "text": h["value"], "bbox": None}
               for h in objective.scan_text(" ".join(w[1] for w in words))]
        for src, items in (("label", labs), ("block", blocks),
                           ("ink", hw), ("objective", obj)):
            for f in items:
                cats = f.get("categories") or []
                rows.append({
                    "page": pno + 1, "src": src,
                    "action": pack.action_for(cats),
                    "cats": sorted(cats),
                    "bbox": ([round(v, 1) for v in f["bbox"]]
                             if f.get("bbox") else None),
                    "text": str(f.get("text", ""))[:60]})
    doc.close()
    return rows


def effect(r):
    return (r["page"], r["src"], r["action"], r["text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("--taxonomy", default="insurance_medical")
    ap.add_argument("--save")
    ap.add_argument("--baseline")
    a = ap.parse_args()

    pack = base.load(ROOT / "taxonomies" / a.taxonomy)
    rows = snapshot(a.pdf, pack)
    acts = Counter(r["action"] for r in rows)
    area = sum((r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])
               for r in rows if r["bbox"] and r["action"] != "KEEP")
    print(f"  {Path(a.pdf).name} under {pack.id}: {len(rows)} findings")
    print(f"  actions: {dict(acts)}")
    print(f"  destructive box area: {area:,.0f} pt^2")

    if a.save:
        p = Path(a.save); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pack": pack.id, "pdf": Path(a.pdf).name,
                                 "rows": rows, "area": area}, indent=1))
        print(f"  baseline written: {p}")
        return 0

    if not a.baseline:
        return 0
    old = json.loads(Path(a.baseline).read_text())
    O, N = Counter(map(effect, old["rows"])), Counter(map(effect, rows))
    lost, added = O - N, N - O
    same = sum((O & N).values())
    print(f"\n  baseline {len(old['rows'])} -> now {len(rows)}"
          f"   effect-identical {same}")
    print(f"  lost {sum(lost.values())}   added {sum(added.values())}")
    da = (area / old["area"] - 1) * 100 if old.get("area") else 0.0
    print(f"  destructive area {old.get('area', 0):,.0f} -> {area:,.0f} "
          f"({da:+.1f}%)")
    for r in sorted(lost):
        print(f"     - p{r[0]:<4}{r[2]:<14}{r[3][:44]!r}")
    for r in sorted(added):
        print(f"     + p{r[0]:<4}{r[2]:<14}{r[3][:44]!r}")
    if lost or added:
        print("\n  REVIEW EACH. A loss is only acceptable once you have shown "
              "the finding was a phantom; an addition, that it is real.")
        return 1
    print("\n  GUARD: unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
