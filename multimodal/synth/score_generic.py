#!/usr/bin/env python3
"""Grade the generic corpus on BOTH axes, and never average them.

    leak        a `values` string still readable in the output
    destroyed   a `must_survive` string no longer readable in the output

They are reported separately because they need opposite fixes and because a
single combined score lets one hide the other -- a pack that blacks out the
whole page scores perfectly on leaks.

Measured on the ARTIFACT: the text is re-extracted from the redacted PDF, so a
box drawn in the right place but painted over nothing counts as a leak.

    python3 synth/score_generic.py --dir out/generic --run
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def page_text(pdf):
    """Re-read the OUTPUT, falling back to OCR where there is no text layer.

    Without the fallback every scanned case scores 0% survived: the redacted
    scan is an image, get_text() returns nothing, and all 102 keep-side values
    look destroyed. That is a property of the measurement, not of the pack, and
    reporting it as destruction would have hidden whatever the real number is.
    """
    sys.path.insert(0, str(ROOT / "document_pipeline"))
    import readers
    d = fitz.open(pdf)
    parts = []
    for i in range(d.page_count):
        t = d[i].get_text().strip()
        if len(t) < 40:                       # image-only page
            try:
                # read_page is 1-BASED (it does doc[page_no - 1]); passing
                # the 0-based loop index read the WRONG page -- and for i=0,
                # doc[-1], the last page. Single-page corpora hid it because
                # doc[-1] is doc[0] there; on a real multi-page bundle every
                # probe was off by one.
                r = readers.read_page(d, i + 1, reader="vision_ocr")
                t = " ".join(str(w[1]) for w in r.get("words", []))
            except Exception:
                t = ""
        parts.append(t)
    d.close()
    return norm(" ".join(parts))


def run(src, out, taxonomy):
    cmd = [sys.executable, str(ROOT / "document_pipeline" / "run_document.py"),
           str(src), "--out", str(out), "--findings-out",
           str(Path(out).with_suffix(".jsonl")), "--taxonomy", taxonomy,
           "--workers", "3"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return p.returncode == 0, (p.stdout + p.stderr)[-400:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="out/generic")
    ap.add_argument("--outdir", default="out/generic/red")
    ap.add_argument("--taxonomy", default="generic_document")
    ap.add_argument("--run", action="store_true", help="run the pipeline first")
    ap.add_argument("--only", help="substring filter on case name")
    a = ap.parse_args()

    src_dir = ROOT / a.dir
    out_dir = ROOT / a.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((src_dir / "manifest.json").read_text())
    if a.only:
        man = [m for m in man if a.only in m["case"]]

    tot = {"leak": 0, "vals": 0, "destroyed": 0, "keeps": 0}
    rows = []
    for m in man:
        src = src_dir / m["file"]
        red = out_dir / f"red_{src.stem}.pdf"
        if a.run or not red.exists():
            ok, log = run(src, red, a.taxonomy)
            if not ok:
                print(f"  {m['case']:<24}RUN FAILED  {log.strip()[-90:]}")
                continue
        txt = page_text(red)
        leaks = [v for v in m["values"] if norm(v) in txt]
        keeps = m.get("must_survive", [])
        gone = [v for v in keeps if norm(v) not in txt]
        tot["leak"] += len(leaks); tot["vals"] += len(m["values"])
        tot["destroyed"] += len(gone); tot["keeps"] += len(keeps)
        rows.append((m["case"], len(m["values"]) - len(leaks), len(m["values"]),
                     len(keeps) - len(gone), len(keeps), leaks, gone))

    print(f"\n{'case':<26}{'removed':>10}{'survived':>11}   failures")
    for c, r, nv, k, nk, leaks, gone in rows:
        bad = ""
        if leaks:
            bad += "LEAK " + ", ".join(repr(x[:22]) for x in leaks[:3])
        if gone:
            bad += ("  " if bad else "") + "DESTROYED " + \
                   ", ".join(repr(x[:22]) for x in gone[:3])
        print(f"  {c:<24}{r:>5}/{nv:<5}{k:>6}/{nk:<5}   {bad}")

    v, k = max(1, tot["vals"]), max(1, tot["keeps"])
    print(f"\n  REMOVED  {tot['vals']-tot['leak']}/{tot['vals']} "
          f"= {1-tot['leak']/v:.0%}   ({tot['leak']} leaks)")
    print(f"  SURVIVED {tot['keeps']-tot['destroyed']}/{tot['keeps']} "
          f"= {1-tot['destroyed']/k:.0%}   ({tot['destroyed']} destroyed)")
    print("\n  The two are never averaged: a pack that blacks out every page "
          "scores 100% on the first line and 0% on the second.")


if __name__ == "__main__":
    main()
