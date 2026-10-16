#!/usr/bin/env python3
"""Score a run_dataset.py delivery against a known-answer dataset.

    python3 tools/score_dataset.py --gt GROUND_TRUTH.json --input DATASET \
        --out DELIVERY --work WORK --mode redact|replace

Every output is read back the way verification reads it (all XML parts,
PDF text layer, OCR of scans) and checked for:
  leak rate          ground-truth values still present (by kind)
  over-redaction     values that must survive (cities, ages, platforms,
                     ordinary words) that are gone
  Replace mode:
    mapped           person / brand values that got an entry in the mapping
    correct          files where the value was, whose output now shows the
                     mapping's invented name (and not the real one)
    codes agree      a linking id has ONE code, present in every file of
                     the study that carried it
  detection precision  ledger spans that match no ground-truth value
                     (listed for review)
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "multimodal" / "router"))
sys.path.insert(0, str(ROOT / "multimodal" / "document_pipeline"))
import run_dataset as rd           # noqa: E402


def texts_of(folder: Path):
    out = {}
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        rel = f.relative_to(folder).as_posix()
        route = rd.route_of(f)
        ext = f.suffix.lower().lstrip(".")
        try:
            if route == "zip":
                with zipfile.ZipFile(f) as z:
                    for n in z.namelist():
                        if not n.endswith("/"):
                            out[f"{rel}!/{n}"] = rd.output_text(
                                z.read(n), n.rsplit(".", 1)[-1].lower(),
                                rd.route_of(Path(n)))
            elif route in ("office", "text", "pdf", "image"):
                out[rel] = rd.output_text(f.read_bytes(), ext, route)
        except Exception as exc:              # noqa: BLE001
            out[rel] = f"<unreadable {exc}>"
        out[rel + " <name>"] = rel
    return out


def present(value, text):
    body = re.escape(value).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){body}(?!\w)", text, re.I) is not None


def score_ref(a):
    """Leaks against a reference mapping: every real-name pattern it lists
    must be absent from the delivery (file names included)."""
    import yaml
    ref = yaml.safe_load(Path(a.ref_mapping).read_text()) or {}
    outs = texts_of(Path(a.out))
    rows = []
    for sec, items in ref.items():
        if not isinstance(items, list):
            continue
        for i, it in enumerate(items):
            if not isinstance(it, dict) or "pattern" not in it:
                continue
            rx = re.compile(str(it["pattern"]),
                            0 if sec == "abbreviations" else re.I)
            files = [k for k, t in outs.items() if rx.search(t)]
            rows.append({"entry": f"{sec}:{i}", "files_with_real_form":
                         len(files), "examples": files[:3]})
    leaked = [r for r in rows if r["files_with_real_form"]]
    print(json.dumps({"reference_entries": len(rows),
                      "entries_leaking": len(leaked), "leaks": leaked,
                      "files_out": len([k for k in outs
                                        if not k.endswith("<name>")])},
                     indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=None)
    ap.add_argument("--ref-mapping", default=None,
                    help="score leaks against another project's mapping "
                         "(its real-name patterns are the answers)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", required=True)
    a = ap.parse_args()
    if a.ref_mapping:
        return score_ref(a)
    gt = json.loads(Path(a.gt).read_text())
    outs = texts_of(Path(a.out))
    ins = texts_of(Path(a.input))
    work = Path(a.work)
    inv = {i["id"]: i["rel"] for i in json.loads(
        (work / "inventory.json").read_text())}
    rep = json.loads((work / "report.json").read_text())
    out_of = {inv[d["id"]]: d["out"] for d in rep["delivered"]
              if d["id"] in inv}
    all_out = "\n".join(outs.values())
    rem = gt["remove"]
    leaks = [r for r in rem if present(r["value"], all_out)]
    by_kind = {}
    for r in rem:
        k = by_kind.setdefault(r["kind"], [0, 0])
        k[0] += 1
        k[1] += r in leaks
    gone = [k for k in gt["keep"] if not present(k, all_out)]
    res = {"mode": a.mode, "files_out": len([k for k in outs
                                             if not k.endswith("<name>")]),
           "values": len(rem), "leaked": len(leaks),
           "leak_rate": round(len(leaks) / max(1, len(rem)), 3),
           "leaks": [f"{r['kind']}: {r['value']}" for r in leaks],
           "by_kind_total_leaked": by_kind,
           "keep_total": len(gt["keep"]), "keep_lost": gone}
    if a.mode == "replace":
        sys.path.insert(0, str(ROOT / "pipeline"))
        import entity_map
        em = entity_map.load(work / "mapping.yaml")
        mapped, correct, wrong = 0, 0, []
        named = [r for r in rem if r["kind"] in ("person", "brand")]
        for r in named:
            hits = em.match_tokens(r["value"].split())
            full = [h for h in hits if h[0] == 0
                    and h[1] == len(r["value"].split()) - 1]
            e = (full or hits or [None])[0]
            if e is None:
                wrong.append(f"unmapped {r['kind']}: {r['value']}")
                continue
            mapped += 1
            inv_name = em.by_id[e[2]["id"]]["replace"] if e[2].get("id") \
                else e[2]["replace"]
            import office
            exp = office.Replacer("replace", None, em).lookup(
                {"text": r["value"]}) or inv_name
            files = [k for k, t in ins.items() if present(r["value"], t)
                     and not k.endswith("<name>")]
            ok, miss = 0, []
            for k in files:
                o = outs.get(out_of.get(k, "\0"), "")
                if o and present(exp, o) and not present(r["value"], o):
                    ok += 1
                else:
                    miss.append(Path(k).name if o else
                                Path(k).name + " (held)")
            if files and ok == len(files):
                correct += 1
            else:
                wrong.append(f"{r['kind']} {r['value']} -> {exp}: "
                             f"{ok}/{len(files)} files; missing in {miss}")
        res.update({"named_values": len(named), "mapped": mapped,
                    "replaced_correctly_everywhere": correct,
                    "replacement_problems": wrong})
        cb = json.loads((work / "codebook.json").read_text()) \
            if (work / "codebook.json").exists() else {}
        code_rows = []
        for study, d in cb.items():
            for kk, code in d.items():
                pre, val = json.loads(kk)
                if pre == "#":
                    continue
                if any(val == c.lower() for c in gt.get("codes", [])):
                    files = [k for k, t in outs.items() if present(code, t)]
                    code_rows.append({"value": val, "code": code,
                                      "files_with_code": len(files)})
        res["codes"] = code_rows
    # detection precision proxy: ledger spans not matching any GT value
    vals = [r["value"].lower() for r in rem]
    extra, total = [], 0
    for lf in (work / "ledger").glob("*.json"):
        led = json.loads(lf.read_text())
        spans = []
        if "subs" in led:
            for rec in led["subs"].values():
                spans += rec["spans"]
        elif "spans" in led:
            spans = led["spans"]
        for s in spans:
            total += 1
            t = s["text"].lower()
            if not any(t in v or v in t for v in vals):
                extra.append(f"{s['text'][:40]} {s['categories']}")
    res["ledger_spans"] = total
    res["ledger_spans_not_in_gt"] = len(extra)
    res["not_in_gt_examples"] = sorted(set(extra))[:40]
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
