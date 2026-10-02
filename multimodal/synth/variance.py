#!/usr/bin/env python3
"""Is the scan-path leak rate stable, and does a second detector stabilise it?

A single run of a model-based detector over a 105-dpi scan is a SAMPLE. Across
three runs of the same 16 documents the leak count moved 0 -> 7 -> 0 with no
code change between them, so any single number quoted from it is unfalsifiable.
This runs each configuration N times and reports the distribution, plus the
union of everything that leaked at least once -- which is the number that
actually bounds exposure.

    python3 synth/variance.py --dir out/generic --runs 3 --only _scan

Reports per configuration:
    per-run leaks     the spread a single run would have quoted
    ever-leaked       values that leaked in AT LEAST ONE run (the real bound)
    always-removed    values no run ever missed
    destroyed         keep-side failures, which should not move at all
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "document_pipeline"))


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def artifact_text(pdf):
    import readers
    d = fitz.open(pdf)
    parts = []
    for i in range(d.page_count):
        t = d[i].get_text().strip()
        if len(t) < 40:
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


def run_once(src, out, taxonomy, second):
    cmd = [sys.executable, str(ROOT / "document_pipeline" / "run_document.py"),
           str(src), "--out", str(out), "--findings-out",
           str(Path(out).with_suffix(".jsonl")), "--taxonomy", taxonomy,
           "--workers", "3"]
    if second:
        cmd += ["--second-detector", second]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="out/generic")
    ap.add_argument("--outdir", default="out/variance")
    ap.add_argument("--taxonomy", default="generic_document")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default="_scan")
    ap.add_argument("--configs", default="single,union")
    a = ap.parse_args()

    src_dir, out_dir = ROOT / a.dir, ROOT / a.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    man = [m for m in json.loads((src_dir / "manifest.json").read_text())
           if a.only in m["case"] or not a.only]

    CFG = {"single": None, "union": "gemini"}
    summary = {}
    for cfg in a.configs.split(","):
        second = CFG[cfg]
        leaks_per_run, destroyed_per_run = [], []
        ever_leaked, ever_destroyed = Counter(), Counter()
        total_v = sum(len(m["values"]) for m in man)
        total_k = sum(len(m.get("must_survive", [])) for m in man)
        for r in range(a.runs):
            nl = nd = 0
            for m in man:
                src = src_dir / m["file"]
                red = out_dir / f"{cfg}_r{r}_{src.stem}.pdf"
                if not run_once(src, red, a.taxonomy, second):
                    print(f"    RUN FAILED {m['case']}")
                    continue
                txt = artifact_text(red)
                for v in m["values"]:
                    if norm(v) in txt:
                        nl += 1; ever_leaked[(m["case"], v)] += 1
                for v in m.get("must_survive", []):
                    if norm(v) not in txt:
                        nd += 1; ever_destroyed[(m["case"], v)] += 1
            leaks_per_run.append(nl); destroyed_per_run.append(nd)
            print(f"  {cfg:<7} run {r+1}/{a.runs}: {nl} leaks, {nd} destroyed")
        summary[cfg] = dict(
            leaks=leaks_per_run, destroyed=destroyed_per_run,
            total_values=total_v, total_keeps=total_k,
            ever_leaked=sorted(f"{c}:{v}" for (c, v) in ever_leaked),
            always_leaked=sorted(f"{c}:{v}" for (c, v), n in ever_leaked.items()
                                 if n == a.runs),
            ever_destroyed=sorted(f"{c}:{v}" for (c, v) in ever_destroyed))

    print(f"\n{'config':<10}{'leaks per run':<22}{'spread':<10}"
          f"{'ever leaked':<14}{'always leaked':<15}destroyed")
    for cfg, s in summary.items():
        lk = s["leaks"]
        print(f"  {cfg:<8}{str(lk):<22}"
              f"{max(lk)-min(lk):<10}{len(s['ever_leaked']):<14}"
              f"{len(s['always_leaked']):<15}{sum(s['destroyed'])}")
    print(f"\n  values at risk: {summary[list(summary)[0]]['total_values']}   "
          f"keep-side values: {summary[list(summary)[0]]['total_keeps']}")
    print("\n  'ever leaked' is the number that bounds exposure; a single run's")
    print("  leak count is a sample and understates it whenever spread > 0.")
    (out_dir / "variance.json").write_text(json.dumps(summary, indent=1))
    for cfg, s in summary.items():
        if s["ever_leaked"]:
            print(f"\n  {cfg} ever-leaked:")
            for v in s["ever_leaked"]:
                print(f"     {v}")


if __name__ == "__main__":
    main()
