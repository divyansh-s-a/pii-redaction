#!/usr/bin/env python3
"""Score a replacement-mode run on synthetic clips against their answer keys
(tools/synth_media.py writes <clip>.gt.json with exact entity times).

Metrics, per clip and pooled (time-overlap matching; a labelled span counts
as found when detected intervals cover >= 50% of it):

  detection recall     mapped entities (any detection / correct entry id),
                       personal data
  detection precision  detected intervals overlapping a labelled span, for
                       mapped and personal-data intervals separately
  removal              share of each mapped/personal span's audio removed by
                       the RENDER (beeps + voiced slots); fully removed means
                       >= 95% — the privacy number, independent of detection
  outcome (mapped)     voiced with the right invented name / beeped / missed
  kept                 spans the profile or keep list keeps that stayed audible
  render               voiced-slot fit rate, level difference, QC status
  listen check         real-name hits, suspected, invented-name heard rate

  python3 tools/score_replacement.py --media DIR --work WORK_ROOT
      --deliveries DIR [--out DIR]
"""
import argparse
import json
from pathlib import Path


def cover(span, ivs):
    """Fraction of span (s, e) covered by the union of intervals ivs."""
    s, e = span
    if e <= s:
        return 0.0
    segs = sorted((max(s, a), min(e, b)) for a, b in ivs if b > s and a < e)
    tot, cur_s, cur_e = 0, None, None
    for a, b in segs:
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot / (e - s)


def score_clip(gt, work, bundle):
    iv = json.loads((work / "intervals.json").read_text())["intervals"]
    ri = json.loads((work / "rendered_intervals.json").read_text())
    ma_path = work / "mapped_audit" / "mapped_audit.json"
    ma = json.loads(ma_path.read_text()) if ma_path.exists() else None
    man = json.loads((bundle / "manifest.json").read_text()) if bundle else {}
    beeps = [(r["start_ms"], r["end_ms"]) for r in ri["rendered_merged"]]
    voiced = ri.get("rendered_voiced") or []
    removed = beeps + [(v["start_ms"], v["end_ms"]) for v in voiced]
    voiced_ids = set(ri.get("voiced_ids") or [])
    kept_ids = set(ri.get("annotate_only_ids") or [])
    ents = gt["entities"]

    def dets(e, pred=lambda i: True):
        return [i for i in iv if pred(i)
                and i["start_ms"] < e["end_ms"] and e["start_ms"] < i["end_ms"]]

    out = {"clip": gt["clip"], "language": gt["language"],
           "qc": man.get("qc_status"),
           "verdict": "quarantine" if man.get("quarantined_reason")
           else "delivered", "quarantined_reason": man.get("quarantined_reason")}
    M = [e for e in ents if e["kind"] == "M"]
    P = [e for e in ents if e["kind"] == "P"]
    K = [e for e in ents if e["kind"] == "K"]
    # a mapped PERSON entry is personal data (always beeped), not a brand
    is_mapped = lambda i: "MAPPED_ENTITY" in i["categories"]
    found_any = [e for e in M if cover((e["start_ms"], e["end_ms"]), [
        (i["start_ms"], i["end_ms"]) for i in dets(e, is_mapped)]) >= 0.5]
    found_ok = [e for e in M if any(i.get("mapping_id") == e["label"]
                                    for i in dets(e, is_mapped))]
    out["mapped"] = {"n": len(M), "recall_any": _r(len(found_any), len(M)),
                     "recall_entity": _r(len(found_ok), len(M))}
    found_p = [e for e in P if cover((e["start_ms"], e["end_ms"]), [
        (i["start_ms"], i["end_ms"]) for i in dets(e)]) >= 0.5]
    out["personal"] = {"n": len(P), "recall": _r(len(found_p), len(P))}
    lab = [(e["start_ms"], e["end_ms"], e["kind"]) for e in ents]

    def prec(pred, kinds):
        ds = [i for i in iv if pred(i)]
        ok = [i for i in ds if any(i["start_ms"] < b and a < i["end_ms"]
                                   and k in kinds for a, b, k in lab)]
        return {"detected": len(ds), "precision": _r(len(ok), len(ds))}
    out["mapped"].update(prec(is_mapped, {"M", "H"}))
    out["personal"].update(prec(lambda i: not is_mapped(i), {"P", "K"}))
    # removal coverage (audio)
    for name, group in (("mapped", M), ("personal", P)):
        cov = [cover((e["start_ms"], e["end_ms"]), removed) for e in group]
        out[name]["fully_removed"] = _r(sum(c >= 0.95 for c in cov), len(cov))
        out[name]["mean_removed"] = (round(sum(cov) / len(cov), 3)
                                     if cov else None)
        out[name]["worst"] = [{"text": e["text"], "removed": round(c, 2)}
                              for e, c in zip(group, cov) if c < 0.95]
    # outcome per mapped entity
    vmap = {v["annotation_id"]: v for v in voiced}
    outcome = {"voiced_correct": 0, "voiced_wrong": 0, "beeped": 0,
               "missed": 0}
    for e in M:
        hits = dets(e)
        vs = [i for i in hits if i["annotation_id"] in voiced_ids]
        if vs:
            right = any(i.get("mapping_id") == e["label"] for i in vs)
            outcome["voiced_correct" if right else "voiced_wrong"] += 1
        elif cover((e["start_ms"], e["end_ms"]), removed) >= 0.5:
            outcome["beeped"] += 1
        else:
            outcome["missed"] += 1
    out["mapped"]["outcome"] = outcome
    # kept spans stay audible
    k_ok = [e for e in K if cover((e["start_ms"], e["end_ms"]), removed) < 0.2]
    out["kept"] = {"n": len(K), "kept_audible": _r(len(k_ok), len(K)),
                   "removed": [e["text"] for e in K if e not in k_ok]}
    rates = [v.get("fit_rate") for v in voiced if v.get("fit_rate")]
    lv = [abs(v["clip_db"] - v["target_db"]) for v in voiced
          if v.get("clip_db") is not None and v.get("target_db") is not None]
    out["render"] = {"voiced": len(voiced),
                     "voice_fallbacks": len(ri.get("voice_fallbacks") or []),
                     "fit_rate_range": [min(rates), max(rates)] if rates
                     else None,
                     "max_level_diff_db": round(max(lv), 2) if lv else None,
                     "tts": sorted({v.get("voice") for v in voiced})}
    lst = (ma or {}).get("listen") or {}
    out["listen"] = ({"hits": len(lst.get("hits", [])),
                      "suspected": len(lst.get("suspected", [])),
                      "heard_rate": lst.get("invented_name_heard_rate"),
                      "heard": lst.get("invented_name_heard"),
                      "voiced_slots": lst.get("voiced_slots")}
                     if lst else None)
    if ma:
        out["mapped_audit"] = {"transcript_hits": len(ma["transcript_hits"]),
                               "judge_confirmed": len(
                                   ma["judge"]["confirmed"]),
                               "leak_candidates": ma["leak_candidates"]}
    return out


def _r(a, b):
    return round(a / b, 3) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--deliveries", required=True)
    ap.add_argument("--quarantine", default=None)
    ap.add_argument("--engine-suffix", default="__elevenlabs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    media, work = Path(a.media), Path(a.work)
    rows = []
    for g in sorted(media.glob("*.gt.json")):
        gt = json.loads(g.read_text())
        wd = work / (gt["clip"] + a.engine_suffix)
        if not wd.exists():
            wd = work / gt["clip"]
        bundle = None
        for base in [Path(a.deliveries)] + ([Path(a.quarantine)]
                                            if a.quarantine else []):
            if (base / gt["clip"] / "manifest.json").exists():
                bundle = base / gt["clip"]
        rows.append(score_clip(gt, wd, bundle))
    pooled = {}
    for grp, keys in (("mapped", ("n",)), ("personal", ("n",))):
        n = sum(r[grp]["n"] for r in rows)
        pooled[grp] = {"n": n}
    tot = lambda f: sum(f(r) for r in rows)
    pooled["mapped"]["recall_entity"] = _r(
        tot(lambda r: round((r["mapped"]["recall_entity"] or 0)
                            * r["mapped"]["n"])), pooled["mapped"]["n"])
    pooled["personal"]["recall"] = _r(
        tot(lambda r: round((r["personal"]["recall"] or 0)
                            * r["personal"]["n"])), pooled["personal"]["n"])
    for grp in ("mapped", "personal"):
        d = tot(lambda r: r[grp]["detected"])
        ok = tot(lambda r: round((r[grp]["precision"] or 0)
                                 * r[grp]["detected"]))
        pooled[grp]["precision"] = _r(ok, d)
        pooled[grp]["fully_removed"] = _r(
            tot(lambda r: round((r[grp]["fully_removed"] or 0)
                                * r[grp]["n"])), pooled[grp]["n"])
    oc = {}
    for r in rows:
        for k, v in r["mapped"]["outcome"].items():
            oc[k] = oc.get(k, 0) + v
    pooled["mapped"]["outcome"] = oc
    res = {"clips": rows, "pooled": pooled}
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "score.json").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
