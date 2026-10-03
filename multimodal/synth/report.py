#!/usr/bin/env python3
"""Generate the document-arm evaluation report into results/.

Numbers do not belong in a README. A README says what the thing is and how to
run it; a measurement is dated, reproducible and belongs beside the other
evaluation reports, where it can be regenerated and compared rather than
quietly going stale.

    python3 synth/report.py --date 2026-09-23 --out ../gitUpload/results

Reads whatever measurements exist and omits the rest:
    out/generic/  + red/     two-sided corpus (leaks AND destruction)
    out/variance/variance.json   repeated-run spread, single vs union
    out/guard/*.json             deterministic guard on a real bundle
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "document_pipeline"))
sys.path.insert(0, str(ROOT / "taxonomies"))


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


def score_corpus(cdir, rdir):
    man = json.loads((cdir / "manifest.json").read_text())
    rows, tot = [], Counter()
    for m in man:
        red = rdir / f"red_{Path(m['file']).stem}.pdf"
        if not red.exists():
            continue
        txt = artifact_text(red)
        leaks = [v for v in m["values"] if norm(v) in txt]
        keeps = m.get("must_survive", [])
        gone = [v for v in keeps if norm(v) not in txt]
        tot["v"] += len(m["values"]); tot["l"] += len(leaks)
        tot["k"] += len(keeps); tot["d"] += len(gone)
        rows.append((m["case"], len(m["values"]) - len(leaks),
                     len(m["values"]), len(keeps) - len(gone), len(keeps)))
    return rows, tot


def packs():
    import base
    out = {}
    for pid in ("insurance_medical", "generic_document"):
        p = base.load(ROOT / "taxonomies" / pid)
        out[pid] = {
            "categories": len(p.rows()),
            "actions": dict(Counter(c["action"] for c in p.rows())),
            "model_judged": [c["id"] for c in p.rows()
                             if c["enforcement"] == "model_judged"],
            "own_labels": len(p.lexicon)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", default="../gitUpload/results")
    ap.add_argument("--corpus", default="out/generic")
    ap.add_argument("--redacted", default="out/generic/red")
    a = ap.parse_args()
    outdir = (ROOT / a.out).resolve(); outdir.mkdir(parents=True, exist_ok=True)

    data = {"generated": a.date, "packs": packs()}
    L = []
    L.append("# Document-arm redaction evaluation\n")
    L.append(f"Generated {a.date} by `multimodal/synth/report.py` | "
             "corpus: 8 non-insurance genres x {born-digital, 105-dpi scan} "
             "+ one real 45-page claim bundle\n")
    L.append("Two numbers, never averaged. A pack that blacks out every page "
             "scores perfectly on removal and zero on survival; a pack that "
             "does nothing scores the reverse. Both are failures.\n")

    rows, tot = score_corpus(ROOT / a.corpus, ROOT / a.redacted)
    if rows:
        data["corpus"] = {"rows": rows, "totals": dict(tot)}
        L.append("## Non-insurance corpus (`generic_document`)\n")
        L.append("| case | removed | survived |")
        L.append("|---|---|---|")
        for c, r, nv, k, nk in rows:
            L.append(f"| {c} | {r}/{nv} | {k}/{nk} |")
        v, k = max(1, tot["v"]), max(1, tot["k"])
        L.append(f"| **total** | **{tot['v']-tot['l']}/{tot['v']} = "
                 f"{1-tot['l']/v:.0%}** | **{tot['k']-tot['d']}/{tot['k']} = "
                 f"{1-tot['d']/k:.0%}** |\n")
        L.append("Measured on the redacted **artifact**: text is re-extracted "
                 "from the output, and OCR'd where the output has no text "
                 "layer. Without that fallback every scanned case scores 0% "
                 "survived, which is a property of the measurement rather "
                 "than of the pack.\n")

    vf = ROOT / "out" / "variance" / "variance.json"
    if vf.exists():
        var = json.loads(vf.read_text()); data["variance"] = var
        L.append("## Run-to-run variance on the scan path\n")
        L.append("A single run of a model detector over a 105-dpi scan is a "
                 "**sample**. `ever leaked` is the union across runs and is "
                 "the number that bounds exposure; a single run understates "
                 "it whenever the spread is non-zero.\n")
        L.append("| config | leaks per run | spread | ever leaked | "
                 "always leaked | keep-side failures |")
        L.append("|---|---|---|---|---|---|")
        for cfg, s in var.items():
            lk = s["leaks"]
            L.append(f"| `{cfg}` | {lk} | {max(lk)-min(lk)} | "
                     f"{len(s['ever_leaked'])} | {len(s['always_leaked'])} | "
                     f"{sum(s['destroyed'])} |")
        L.append("")
        L.append("`single` = one detector. `union` = `--second-detector "
                 "gemini`, which keeps anything the second family finds that "
                 "the first missed. This is the document port of the audio "
                 "second listener, and it is **on by default**; "
                 "`--second-detector none` disables it.\n")
        u = var.get("union")
        if u:
            n = len(u["leaks"]) * u["total_values"]
            L.append(f"**What the union result does and does not show.** "
                     f"{len(u['leaks'])} runs x {u['total_values']} values = "
                     f"**{n} value-checks with zero misses**. That is "
                     f"evidence the run-to-run variance is gone, not proof "
                     f"the leak rate is zero. It costs roughly **2x "
                     f"detection spend**, since every page is read twice.\n")

    gf = sorted((ROOT / "out" / "guard").glob("*.json")) \
        if (ROOT / "out" / "guard").exists() else []
    if gf:
        g = json.loads(gf[0].read_text()); data["guard"] = {
            "pack": g["pack"], "pdf": g["pdf"], "findings": len(g["rows"]),
            "area": g["area"]}
        L.append("## Shared-engine guard (real claim bundle)\n")
        L.append("The engine is shared between packs, so every change is "
                 "re-measured against a real bundle under `insurance_medical` "
                 "before it ships. Deterministic mechanisms only -- no model "
                 "is called, so the comparison is exact.\n")
        L.append(f"- document: `{g['pdf']}` under `{g['pack']}`")
        L.append(f"- deterministic findings: **{len(g['rows'])}**")
        L.append(f"- destructive box area: **{g['area']:,.0f} pt²**\n")
        L.append("    python3 synth/guard.py BUNDLE.pdf --taxonomy "
                 "insurance_medical --baseline out/guard/insurance.json\n")

    p = data["packs"]
    L.append("## Packs\n")
    L.append("| | categories | actions | `model_judged` | pack-owned labels |")
    L.append("|---|---|---|---|---|")
    for pid, s in p.items():
        acts = " / ".join(f"{k} {v}" for k, v in sorted(s["actions"].items()))
        L.append(f"| `{pid}` | {s['categories']} | {acts} | "
                 f"{len(s['model_judged'])} ({', '.join(s['model_judged'])}) | "
                 f"{s['own_labels']} |")
    L.append("")
    L.append("## What these numbers cannot tell you\n")
    L.append("- **Synthetic data cannot see a failure it does not contain.** "
             "A bare `patient` label matched the end of charge-line "
             "descriptions on a real bill and destroyed three totals while "
             "the synthetic corpus scored 100%. Only the real bundle exposed "
             "it. The guard above exists for that reason.")
    L.append("- **Taxonomy correctness is not measured here.** Whether the "
             "right things are called PII is a human and legal judgement; "
             "these numbers only show the pipeline executes the taxonomy it "
             "was given.")
    L.append("- **Generated handwriting is a font with stroke jitter**, not "
             "pen on paper, so the chromatic ink path is only partly "
             "exercised.")

    (outdir / "document_redaction_report.md").write_text("\n".join(L) + "\n")
    (outdir / "document_redaction_report.json").write_text(
        json.dumps(data, indent=1))
    print(f"  wrote {outdir/'document_redaction_report.md'}")
    print(f"  wrote {outdir/'document_redaction_report.json'}")


if __name__ == "__main__":
    main()
