#!/usr/bin/env python3
"""P4 — seeded, stratified page sampler.

    python3 benchmark/sample.py --kind born_digital --n 120 --out out/sample_born.json

Two properties that matter more than the sampling maths:

1. REPRODUCIBLE AND EXTENDABLE. Selection order is sha1(file|page|seed), so the
   same seed always yields the same order and enlarging n keeps every page the
   smaller sample had. No redraw, no silent set change between runs.

2. THE PAGE SET IS HASHED. Comparing two configurations scored over different
   page sets produces meaningless deltas - that exact bug (a 24- vs 25-file
   Bengali comparison) survived into a published table in the audio project
   until it was caught. Here the set hash travels with the sample and the
   scorer refuses to cross it.

`patient-*` files are excluded by default: their PII is already destroyed by a
previous redactor, so pooling them with raw pages would corrupt every rate.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SEED = "multimodal_pdfs/v1"   # FROZEN: a hash input, not a path.
# Renaming the package must NOT change it -- the sample set and every
# page-set hash compared against it would shift.


def order_key(row, seed=SEED):
    h = hashlib.sha1(f"{row['file']}|{row['page']}|{seed}".encode())
    return h.hexdigest()


def page_set_hash(rows):
    """Identity of the exact page set, for the scorer's guard."""
    ids = sorted(f"{r['file']}#{r['page']}" for r in rows)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def sample(census, kind=None, n=120, per_file_cap=25, include_patient=False,
           seed=SEED):
    rows = [r for r in census
            if (include_patient or not r["file"].startswith("patient-"))
            and (kind is None or r["kind"] == kind)]
    rows.sort(key=lambda r: order_key(r, seed))
    picked, per_file = [], Counter()
    for r in rows:                      # cap first, so one 217-page bundle
        if per_file[r["file"]] >= per_file_cap:   # cannot dominate the sample
            continue
        picked.append(r)
        per_file[r["file"]] += 1
        if len(picked) >= n:
            break
    # design weights: each picked page stands for this many corpus pages
    strata = Counter(r["kind"] for r in rows)
    got = Counter(r["kind"] for r in picked)
    for r in picked:
        r["weight"] = round(strata[r["kind"]] / max(1, got[r["kind"]]), 3)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", default="out/census.jsonl")
    ap.add_argument("--kind", default=None,
                    help="born_digital | image_only | ocr_over_scan")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--per-file-cap", type=int, default=25)
    ap.add_argument("--include-patient", action="store_true")
    ap.add_argument("--out", default="out/sample_v1.json")
    args = ap.parse_args()

    census = [json.loads(l) for l in open(args.census) if l.strip()]
    picked = sample(census, args.kind, args.n, args.per_file_cap,
                    args.include_patient)
    doc = {
        "seed": SEED, "kind_filter": args.kind, "requested_n": args.n,
        "per_file_cap": args.per_file_cap,
        "n": len(picked), "page_set_hash": page_set_hash(picked),
        "pages": [{k: r[k] for k in ("file", "page", "kind", "chars",
                                     "words", "native_dpi", "weight")}
                  for r in picked],
    }
    Path(args.out).write_text(json.dumps(doc, indent=1))
    byfile = Counter(r["file"] for r in picked)
    print(f"sampled {len(picked)} pages from {len(byfile)} documents")
    print(f"  kinds: {dict(Counter(r['kind'] for r in picked))}")
    print(f"  pages/doc: min {min(byfile.values())} max {max(byfile.values())}")
    print(f"  page_set_hash: {doc['page_set_hash']}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
