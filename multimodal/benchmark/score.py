#!/usr/bin/env python3
"""Score a detector against the reference-free reference.

    python3 benchmark/score.py --reference out/reference.jsonl \
            [--redacted out/redacted] [--label gpt]

Metrics, defined exactly as in the audio project so the two are comparable:

  DETECTION RECALL     reference items matched by a detector finding
                       / all reference items
  DETECTION PRECISION  detector findings matching a reference item
                       / all detector findings
  LEAK RATE            must-redact reference items NOT >=98% covered by the
                       rendered redaction boxes / all must-redact items.
                       The headline safety number. 1 - redaction recall.
  OVER-REDACTION       redacted area not overlapping any reference item,
                       as a share of redacted area - without it, "redact the
                       whole page" would win every safety metric.

Match rule = the clustering rule (bbox IoU >= 0.3 or cover >= 0.7).

Reported per EVIDENCE TIER and never pooled: T0 is proof, T2 is opinion.
A page-set hash guard refuses to compare runs scored over different pages -
the audio project published a table once where two configurations had been
scored over different file sets, and the comparison was meaningless.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "document_pipeline"))
import policy
import geometry

PROFILE = "strict"
COVER_MIN = 0.98


def matches(a_box, b_box):
    return (geometry.iou(a_box, b_box) >= 0.3
            or geometry.cover(a_box, b_box) >= 0.7
            or geometry.cover(b_box, a_box) >= 0.7)


def page_set_hash(rows):
    ids = sorted(f"{r['file']}#{r['page']}" for r in rows)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def score(rows, redaction_boxes=None):
    """redaction_boxes: {(file, page): [box, ...]} - what was actually drawn."""
    tiers = ["T0_objective", "T1_corroborated", "T2_judge_only"]
    stat = {t: Counter() for t in tiers + ["ALL"]}
    cat_stat = defaultdict(Counter)
    det_tot = det_hit = 0
    leak_tot = leak_miss = 0
    over_num = over_den = 0.0

    for r in rows:
        ref, det = r["reference"], r["detector"]
        for item in ref:
            t = item.get("tier", "T2_judge_only")
            hit = any(matches(item["bbox"], d["bbox"]) for d in det
                      if d.get("bbox"))
            for bucket in (t, "ALL"):
                stat[bucket]["ref"] += 1
                stat[bucket]["ref_matched"] += 1 if hit else 0
            for c in item["categories"]:
                cat_stat[c]["ref"] += 1
                cat_stat[c]["ref_matched"] += 1 if hit else 0
            # leak: must-redact items not covered by what was actually drawn
            act = policy.action_for(item["categories"],
                                    item.get("public_figure", False), PROFILE)
            if act == "beep" and redaction_boxes is not None:
                boxes = redaction_boxes.get((r["file"], r["page"]), [])
                leak_tot += 1
                if not geometry.covered_by_any(item["bbox"], boxes, COVER_MIN):
                    leak_miss += 1
        for d in det:
            if not d.get("bbox"):
                continue
            det_tot += 1
            if any(matches(d["bbox"], item["bbox"]) for item in ref):
                det_hit += 1
        if redaction_boxes is not None:
            boxes = redaction_boxes.get((r["file"], r["page"]), [])
            for b in boxes:
                a = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
                over_den += a
                if not any(geometry.intersect(b, item["bbox"]) for item in ref):
                    over_num += a
    return {"tiers": stat, "cats": cat_stat,
            "det_tot": det_tot, "det_hit": det_hit,
            "leak_tot": leak_tot, "leak_miss": leak_miss,
            "over_num": over_num, "over_den": over_den}


def pct(n, d):
    return f"{n}/{d} = {n/d:.1%}" if d else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="out/reference.jsonl")
    ap.add_argument("--label", default="detector")
    ap.add_argument("--expect-hash", default=None,
                    help="refuse to score a different page set")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.reference).open()]
    h = page_set_hash(rows)
    if args.expect_hash and args.expect_hash != h:
        sys.exit(f"REFUSING: page set {h} != expected {args.expect_hash}. "
                 "Comparing configurations over different pages is invalid.")

    # what was actually redacted = the detector's must-redact boxes
    rb = defaultdict(list)
    for r in rows:
        for d in r["detector"]:
            if not d.get("bbox"):
                continue
            if policy.action_for(d["categories"],
                                 d.get("public_figure", False),
                                 PROFILE) == "beep":
                rb[(r["file"], r["page"])].extend(
                    d.get("all_boxes") or [d["bbox"]])

    s = score(rows, rb)
    print(f"=== {args.label} — {len(rows)} pages, page_set_hash {h} ===\n")
    print(f"DETECTION PRECISION  {pct(s['det_hit'], s['det_tot'])}")
    print("DETECTION RECALL by evidence tier "
          "(T0 = proof, T2 = judge opinion; never pooled):")
    for t in ("T0_objective", "T1_corroborated", "T2_judge_only", "ALL"):
        c = s["tiers"][t]
        if c["ref"]:
            print(f"   {t:<18}{pct(c['ref_matched'], c['ref'])}")
    print(f"\nLEAK RATE (must-redact not >={COVER_MIN:.0%} covered)  "
          f"{pct(s['leak_miss'], s['leak_tot'])}")
    if s["over_den"]:
        print(f"OVER-REDACTION (area not on any reference item)  "
              f"{s['over_num']/s['over_den']:.1%}")
    print("\nper category (reference matched / reference):")
    for c, v in sorted(s["cats"].items(), key=lambda kv: -kv[1]["ref"])[:10]:
        print(f"   {c:<28}{pct(v['ref_matched'], v['ref'])}")
    print("\nCAVEAT: the reference is the union of what our readers surfaced. "
          "Recall is an upper bound on MEASURED recall and a lower bound on "
          "true miss rate; PII no reader and no judge saw is invisible here.")


if __name__ == "__main__":
    main()
