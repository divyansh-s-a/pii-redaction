#!/usr/bin/env python3
"""Build a reference set WITHOUT human annotation, then score against it.

    python3 benchmark/judge_pdf_ref.py --findings out/findings.jsonl \
            --out out/reference.jsonl

Method, ported from tools/judge_pseudo_gt.py + judge_precision_recall.py:

  1. An INDEPENDENT JUDGE reads the same page word-stream under the same
     Taxonomy v0.2 policy. Judge is Gemini; the detector is GPT - different
     families, so the reference is not grading its own work.
  2. Every judge finding passes the ADVERSARIAL VERIFIER (cite an include rule
     or be rejected), reused unchanged from pipeline/judge_client.py.
  3. A HALLUCINATION GUARD drops any judge finding whose quoted text cannot be
     located in the page's own word stream. In audio the judge over-tagged on
     weak languages; here it will over-tag on low-dpi scans, and a finding no
     other channel can corroborate must not sit in the headline denominator.
  4. Objective hits (checksum-verified IDs, decoded QR) enter as tier T0 with
     no model in the loop at all.

Evidence tiers are reported separately and never pooled:
  T0 objective   checksum/QR/face - proof, no opinion
  T1 corroborated judge + detector agree, or judge + a locatable quote
  T2 judge-only  a lower bound of unknown tightness, always labelled
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "document_pipeline"))
import common
import judge_client
import policy
import geometry
import readers
import fitz

SUPPLEMENT = (Path(__file__).resolve().parent.parent / "prompts"
              / "pdf_detection_supplement_v1.txt")

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"spans": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "start_word": {"type": "integer"},
            "end_word": {"type": "integer"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "spoken_text": {"type": "string"},
            "public_figure": {"type": "boolean"},
            "confidence": {"type": "string",
                           "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"}},
        "required": ["start_word", "end_word", "categories", "spoken_text",
                     "public_figure", "confidence", "reason"]}}},
    "required": ["spans"]}


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def judge_page(words, model=None):
    """Independent read of the same word stream by the judge family."""
    stream = "\n".join(f"{i}\t{t}" for i, t, _ in words)
    prompt = (policy.DETECTION_POLICY + "\n\n" + SUPPLEMENT.read_text()
              + "\n\nPAGE WORDS:\n" + stream)
    got = judge_client.judge_call(prompt, JUDGE_SCHEMA,
                                  model=model or judge_client.DOC_JUDGE,
                                  native_gemini=True,
                                  log_engine="judge_pdf")
    return got.get("spans", [])


def spans_to_items(spans, words, source):
    out = []
    for s in spans:
        try:
            a, b = int(s["start_word"]), int(s["end_word"])
        except (KeyError, TypeError, ValueError):
            continue
        sel = [w for w in words if a <= w[0] <= b]
        if not sel:
            continue                       # hallucination guard: not locatable
        box = sel[0][2]
        for _, _, r in sel[1:]:
            box = box | r
        cats = [c for c in s.get("categories", []) if c in policy.CATEGORIES]
        if not cats:
            continue
        out.append({"categories": cats,
                    "text": " ".join(t for _, t, _ in sel),
                    "bbox": [round(v, 2) for v in
                             (box.x0, box.y0, box.x1, box.y1)],
                    "source": source,
                    "confidence": s.get("confidence", "medium"),
                    "public_figure": bool(s.get("public_figure", False))})
    return out


def build_reference(page_row, doc, verify=True):
    """-> (reference_items, detector_items) for one page."""
    rd = readers.read_page(doc, page_row["page"])
    words = rd["words"]
    det_items = [f for f in page_row["findings"] if f.get("bbox")]

    common.set_usage_context(f"judgeref:{page_row['file']}:p{page_row['page']}")
    try:
        jspans = judge_page(words)
    finally:
        common.set_usage_context("")
    jitems = spans_to_items(jspans, words, "judge")

    if verify and jitems:
        payload = [{"quoted_text": j["text"], "categories": j["categories"],
                    "reason": ""} for j in jitems]
        try:
            # returns a list of booleans PARALLEL to the input, and fails open
            # (a malformed verdict keeps the finding) - verification may only
            # remove, never silently invent. Verifier must be the JUDGE family,
            # not the detector's, or the filter grades its own detector.
            keep = judge_client.verify_findings(
                payload, policy.DETECTION_POLICY,
                model=judge_client.DOC_VERIFIER,
                native_gemini=True)
            n_before = len(jitems)
            jitems = [j for j, k in zip(jitems, keep) if k]
            if n_before != len(jitems):
                print(f"     verifier rejected {n_before - len(jitems)}"
                      f"/{n_before} judge findings as over-reach", flush=True)
        except Exception as exc:
            print(f"     verifier unavailable ({str(exc)[:60]}) - keeping all",
                  flush=True)

    # objective items already carry proof; mark tier and merge
    obj = [f for f in page_row["findings"]
           if f.get("source") == "objective_regex" and f.get("checksum_ok")]
    reference = []
    for it in jitems:
        it["tier"] = "T2_judge_only"
        reference.append(it)
    for o in obj:
        if o.get("bbox"):
            reference.append({**o, "tier": "T0_objective", "source": "objective"})

    # promote judge items the detector independently found -> T1
    for ref in reference:
        if ref["tier"] != "T2_judge_only":
            continue
        for d in det_items:
            if geometry.iou(ref["bbox"], d["bbox"]) >= 0.3 \
                    or geometry.cover(ref["bbox"], d["bbox"]) >= 0.7:
                ref["tier"] = "T1_corroborated"
                break
    return reference, det_items, rd["reader"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--findings", default="out/findings.jsonl")
    ap.add_argument("--out", default="out/reference.jsonl")
    ap.add_argument("--corpus", default=str(readers.corpus_dir()))
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.findings).open()]
    rows = rows[:args.limit or None]
    opened = {}
    tiers = Counter()
    with Path(args.out).open("w") as fh:
        for i, r in enumerate(rows, 1):
            doc = opened.get(r["file"]) or opened.setdefault(
                r["file"], fitz.open(Path(args.corpus) / r["file"]))
            try:
                ref, det, reader = build_reference(r, doc,
                                                   verify=not args.no_verify)
            except Exception as exc:
                print(f"  !! {r['file'][:22]} p{r['page']}: {str(exc)[:80]}")
                continue
            tiers.update(x["tier"] for x in ref)
            fh.write(json.dumps({"file": r["file"], "page": r["page"],
                                 "reader": reader, "reference": ref,
                                 "detector": det}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(rows)}] {r['file'][:24]:<26}p{r['page']:<5}"
                  f"ref={len(ref):>3} det={len(det):>3}", flush=True)
    for d in opened.values():
        d.close()
    print(f"\nreference tiers: {dict(tiers)}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
