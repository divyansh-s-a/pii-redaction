#!/usr/bin/env python3
"""Judge-based DETECTION PRECISION / DETECTION RECALL scorer for pipeline
configurations, plus redaction recall / leak rate, speakers and languages.

REFERENCE SET (per file): the verified judge detections on BOTH listener
transcripts — elevenlabs and gemini — clustered into unique items. The judge
(default openai/gpt-5.6-sol) applies the same Taxonomy v0.2 detection policy
as stage C; every time-anchorable span passes the adversarial verifier
(default openai/gpt-5.6-luna: cite the include rule or REJECT) before
clustering. Clustering: temporal IoU >= 0.3 / 70% cover, OR same canonicalized
value within +/-2.5 s (cross-listener clock skew). All machinery and span
caches are shared with tools/judge_pseudo_gt.py — cached verified spans are
reused, not re-judged.

METRICS per configuration (per file, per category, micro-averaged):
  DETECTION RECALL     reference items matched by the config's intervals.json
                       / all reference items. Match = temporal cluster rule OR
                       same canonicalized value within +/-2.5 s.
  DETECTION PRECISION  intervals matching a reference item / all intervals —
                       the direct counterpart of detection recall, judged
                       against the same reference.
  REDACTION RECALL     must-beep reference items >= 95% covered by the
                       config's rendered beeps / all must-beep items
                       (channel-aware). LEAK RATE = 1 - redaction recall: the
                       headline safety number.
  POLICY-VERIFIED PRECISION (supplementary) intervals the adversarial
                       verifier confirms satisfy an include rule / all
                       intervals. Fairer to the detector when the reference
                       union missed a valid detection; reported alongside
                       detection precision, never instead of it.
Plus per file: speakers heard by each listener (count + agreement) and
languages (primary + all, cross-listener agreement).

SCOPE: the reference is a UNION of what two listeners surfaced — a shared
benchmark for fair comparison; human spot-checks complement it. For __gpt detector
configs the detector shares a model family with the judge (luna vs sol);
rule-verified precision especially may flatter them — the cross-family read
(Gemini detector, GPT judge) is the cleaner comparison. Human GT remains the
calibrator.

Usage:
  python3 tools/judge_precision_recall.py
      [--configs elevenlabs,gemini,openai,elevenlabs__gpt,openai__gpt]
      [--files <fid>,...] [--pred-root output]
      [--out output/judge/precision_recall]
      [--span-cache output/judge/pseudo_gt]
      [--judge-model openai/gpt-5.6-sol] [--verify-model openai/gpt-5.6-luna]
      [--profile strict|medical_consult] [--no-pverify] [--force]
  python3 tools/judge_precision_recall.py --self-test
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
import detect_pii
import policy
import judge_client
import score_run
import judge_pseudo_gt as pgt

LISTENERS = ["elevenlabs", "gemini"]     # the two ears that build the reference
PVERIFY_CONTEXT_PAD = 8

CONFIG_LABELS = {
    "elevenlabs": "ElevenLabs + Gemini (production)",
    "gemini": "Gemini end-to-end",
    "openai": "Whisper + Gemini",
    "elevenlabs__gpt": "ElevenLabs + GPT detector",
    "openai__gpt": "Whisper + GPT detector",
}

CAVEATS = [
    "The reference is a UNION of what the two listener transcripts "
    "(elevenlabs, gemini) surfaced to the judge — a shared benchmark for "
    "comparing configurations fairly; values neither listener transcribed "
    "sit outside its scope, so human spot-checks complement these numbers.",
    "The judge and verifier are LLMs applying the written policy; their "
    "reference is unvalidated. Human ground truth remains the calibrator "
    "and will later score the judge itself.",
    "Configs with a __gpt detector share a model family with the judge "
    "(detector luna, judge sol). Rule-verified precision especially may "
    "flatter them; the cross-family read (Gemini detector, GPT judge) is "
    "the cleaner comparison.",
    "Value-based matching requires a shared category between reference item "
    "and interval; purely temporal matches do not.",
]

METRIC_DEFS = [
    ("DETECTION RECALL", "reference items matched by the config's "
     "intervals.json / all reference items (match = temporal IoU >= 0.3 or "
     "70% cover, OR same canonicalized value within +/-2.5 s)"),
    ("DETECTION PRECISION", "intervals matching a reference item / all "
     "intervals in intervals.json (same match rule, same reference)"),
    ("REDACTION RECALL", "must-beep reference items covered by the rendered "
     "beeps / all must-beep items. An item counts as covered when a config "
     "interval matching it (detection match rule) is >= 95% beep-covered on "
     "the config's own clock, or when the reference spans themselves are "
     ">= 95% covered (channel-aware). The own-clock rule keeps cross-engine "
     "timestamp skew between the reference listeners and the config's "
     "engine from counting as leaks; items the config never detected are "
     "judged on reference-span coverage alone."),
    ("LEAK RATE", "1 - redaction recall — the headline safety number"),
    ("POLICY-VERIFIED PRECISION (supplementary)", "intervals the adversarial "
     "verifier confirms satisfy a policy include rule / all intervals; "
     "reported alongside detection precision, never instead of it"),
]


def config_dir(pred_root, fid, config):
    return Path(pred_root) / (fid if config == "gemini"
                              else "%s__%s" % (fid, config))


def _load_optional(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------- reference set

def build_reference(fid, pred_root, span_cache, judge_model, verify_model,
                    force):
    """Verified judge spans on both listener transcripts -> clustered
    reference items. Returns (items, notes, rejected_by_listener)."""
    all_spans, notes, rejected = [], [], {}
    for eng in LISTENERS:
        tpath = pgt.engine_dir(pred_root, fid, eng) / "transcript.json"
        transcript = _load_optional(tpath)
        if transcript is None:
            notes.append("%s: transcript.json missing — reference is "
                         "single-listener for this file" % eng)
            continue
        words = detect_pii.flatten_words(transcript)
        if not words:
            notes.append("%s: empty transcript" % eng)
            continue
        spans, ver = pgt.judge_spans_cached(fid, eng, words, span_cache,
                                            judge_model, force, verify=True,
                                            verify_model=verify_model)
        rejected[eng] = ver.get("rejected", 0)
        anchored = [x for x in (pgt.anchor_span(words, s, eng) for s in spans)
                    if x]
        if len(anchored) < len(spans):
            notes.append("%s: %d judge span(s) dropped (invalid indices)"
                         % (eng, len(spans) - len(anchored)))
        all_spans.extend(anchored)
    items = pgt.cluster_spans(all_spans)
    return items, notes, rejected


# ----------------------------------------------------------------- matching

def _iv_as_span(iv):
    return {"start_ms": iv["start_ms"], "end_ms": iv["end_ms"],
            "categories": list(iv.get("categories", [])),
            "spoken_text": iv.get("spoken_text", ""), "engine": "pipeline"}


def match_item_interval(item, iv):
    """Reference item <-> pipeline interval match: the temporal cluster rule,
    OR the same canonicalized value within +/-2.5 s (bridges engine clock
    skew; requires a shared category)."""
    if pgt.overlap_pair(item, iv):
        return True
    return pgt.same_value_nearby(dict(item, engine="reference"),
                                 _iv_as_span(iv))


# ------------------------------------------------------------ config scoring

def _cat_bucket(percat, c):
    return percat.setdefault(c, {"ref": 0, "ref_matched": 0,
                                 "iv": 0, "iv_matched": 0})


def _redaction_counts(items, ivs, rendered, profile):
    """Own-clock redaction coverage. Reference items carry the LISTENERS'
    timestamps; a config renders beeps on its OWN engine's clock, so raw
    span-coverage charges cross-engine clock skew as leaks (a config whose
    engine is not a reference listener has no same-clock span to anchor
    item_coverage's sibling bridge). A must-beep item therefore counts as
    covered when EITHER: (a) a config interval MATCHING the item (the
    detection match rule) is >= 95% beep-covered — the engine heard the
    value and beeped its own placement of it — or (b) the reference spans
    themselves are covered per item_coverage (coincidental or cross-clock
    coverage; also the only path for items the config never detected)."""
    counts = {"must_beep": 0, "annotate_only": 0, "covered": 0, "leaked": 0}
    leaked_idx = []
    for k, it in enumerate(items):
        action = policy.action_for(it["categories"], it["public_figure"],
                                   profile)
        if action != "beep":
            counts["annotate_only"] += 1
            continue
        counts["must_beep"] += 1
        own_cov = any(
            score_run.coverage_of(
                {"start_ms": iv["start_ms"], "end_ms": iv["end_ms"],
                 "channel": iv.get("channel")}, rendered)
            >= pgt.COVERAGE_MIN
            for iv in ivs if match_item_interval(it, iv))
        if own_cov or pgt.item_coverage(it, rendered) >= pgt.COVERAGE_MIN:
            counts["covered"] += 1
        else:
            counts["leaked"] += 1
            leaked_idx.append(k)
    return counts, leaked_idx


def score_config_file(items, cdir, profile):
    """Score one config's outputs for one file against the reference items.
    Returns None when the config has no intervals.json for this file."""
    int_doc = _load_optional(Path(cdir) / "intervals.json")
    if int_doc is None:
        return None
    ivs = int_doc.get("intervals", [])
    rend_doc = _load_optional(Path(cdir) / "rendered_intervals.json")
    rendered = (rend_doc or {}).get("rendered_merged", [])

    matched_items = [any(match_item_interval(it, iv) for iv in ivs)
                     for it in items]
    matched_ivs = [any(match_item_interval(it, iv) for it in items)
                   for iv in ivs]
    counts, leaked_idx = _redaction_counts(items, ivs, rendered, profile)
    percat = {}
    for it, m in zip(items, matched_items):
        for c in it["categories"]:
            b = _cat_bucket(percat, c)
            b["ref"] += 1
            b["ref_matched"] += 1 if m else 0
    for iv, m in zip(ivs, matched_ivs):
        for c in iv.get("categories", []):
            b = _cat_bucket(percat, c)
            b["iv"] += 1
            b["iv_matched"] += 1 if m else 0

    missed = [it["item_id"] for it, m in zip(items, matched_items) if not m]
    return {
        "ref_items": len(items),
        "ref_matched": sum(matched_items),
        "intervals": len(ivs),
        "intervals_matched": sum(matched_ivs),
        "must_beep": counts["must_beep"],
        "covered": counts["covered"],
        "leaked": counts["leaked"],
        "annotate_only": counts["annotate_only"],
        "rendered_missing": rend_doc is None,
        "missed_item_ids": missed,
        "unmatched_intervals": [
            {"start_ms": iv["start_ms"], "end_ms": iv["end_ms"],
             "categories": iv.get("categories", []),
             "spoken_text": str(iv.get("spoken_text", ""))[:60]}
            for iv, m in zip(ivs, matched_ivs) if not m],
        "per_category": percat,
    }


# ----------------------------------------- policy-verified precision (suppl.)

def _context_for_interval(words, iv, pad=PVERIFY_CONTEXT_PAD):
    idxs = [k for k, w in enumerate(words)
            if w["end_ms"] >= iv["start_ms"] and w["start_ms"] <= iv["end_ms"]]
    if not idxs:
        return str(iv.get("spoken_text", ""))
    a, b = max(0, idxs[0] - pad), min(len(words), idxs[-1] + pad + 1)
    return " ".join(w.get("w", "") for w in words[a:b])


def _pverify_key(ivs, model):
    basis = json.dumps([[str(iv.get("spoken_text", "")),
                         sorted(iv.get("categories", []))] for iv in ivs],
                       ensure_ascii=False, sort_keys=True) + model
    return hashlib.sha256(basis.encode()).hexdigest()


def policy_verified(fid, config, cdir, out_dir, verify_model, force):
    """Luna-verify the config's own intervals (ONE batched call, cached).
    Returns (kept, total) or None when intervals.json is missing."""
    int_doc = _load_optional(Path(cdir) / "intervals.json")
    if int_doc is None:
        return None
    ivs = int_doc.get("intervals", [])
    if not ivs:
        return (0, 0)
    key = _pverify_key(ivs, verify_model)
    cache = Path(out_dir) / "pverify" / ("%s__%s.json" % (fid, config))
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if cached.get("key") == key:
            return (cached["kept"], cached["total"])
    transcript = _load_optional(Path(cdir) / "transcript.json")
    words = detect_pii.flatten_words(transcript) if transcript else []
    import common
    common.set_usage_context("pverify:%s:%s" % (fid, config))
    try:
        keep = judge_client.verify_findings(
            ivs, policy.DETECTION_POLICY, model=verify_model,
            context_texts=[_context_for_interval(words, iv) for iv in ivs])
    finally:
        common.set_usage_context("")
    kept = sum(1 for k in keep if k)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "key": key, "kept": kept, "total": len(ivs), "model": verify_model,
        "verified_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rejected": [{"spoken_text": str(iv.get("spoken_text", ""))[:60],
                      "categories": iv.get("categories", [])}
                     for iv, k in zip(ivs, keep) if not k]},
        indent=2, ensure_ascii=False))
    return (kept, len(ivs))


# -------------------------------------------------- cross-listener timing

def timing_offsets(items):
    """|start offset| in ms between the two listeners' spans of the SAME
    value, one measurement per ElevenLabs span (nearest Gemini span of the
    same item). Quantifies word-timestamp agreement between the ears —
    the engines' relative clock skew — without needing human-aligned times."""
    offs = []
    for it in items:
        spans = it.get("source_spans") or []
        first = [s for s in spans if s.get("engine") == LISTENERS[0]]
        second = [s for s in spans if s.get("engine") == LISTENERS[-1]]
        if not second:
            continue
        for a in first:
            offs.append(min(abs(a["start_ms"] - b["start_ms"])
                            for b in second))
    return offs


def _pctl(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


# ------------------------------------------------------ speakers + languages

def listener_summary(fid, pred_root):
    """Speakers + languages as each listener heard them, with agreement."""
    per = {}
    for eng in LISTENERS:
        t = _load_optional(pgt.engine_dir(pred_root, fid, eng)
                           / "transcript.json")
        if t is None:
            continue
        speakers = {u.get("speaker") for u in t.get("utterances", [])}
        langs, seen = [], set()
        for l in t.get("languages", []):
            n = score_run.norm_lang(l.get("lang"))
            if n not in seen:
                seen.add(n)
                langs.append(n)
        per[eng] = {"speakers": len(speakers),
                    "primary_language":
                        score_run.norm_lang(t.get("primary_language")),
                    "languages": langs}
    both = all(e in per for e in LISTENERS)
    return {
        "per_listener": per,
        "speakers_agree": (len({per[e]["speakers"] for e in LISTENERS}) == 1
                           if both else None),
        "primary_language_agree":
            (len({per[e]["primary_language"] for e in LISTENERS}) == 1
             if both else None),
    }


# --------------------------------------------------------------- aggregation

def _quarantined(fid):
    """Gate status from the production run: a file under quarantine/ was
    withheld from delivery, so its uncovered items never shipped. Files with
    no gate output count as delivered (the conservative direction)."""
    return (ROOT / "quarantine" / fid).exists()


def _ratio(num, den):
    return (num / float(den)) if den else None


def _pct(num, den):
    if not den:
        return "n/a"
    return "%d/%d = %.1f%%" % (num, den, 100.0 * num / den)


AGG_KEYS = ("ref_items", "ref_matched", "intervals", "intervals_matched",
            "must_beep", "covered", "leaked", "annotate_only")


def aggregate(file_results, configs):
    agg = {}
    for cfg in configs:
        a = {k: 0 for k in AGG_KEYS}
        a["files"] = 0
        a["pverify_kept"] = 0
        a["pverify_total"] = 0
        a["pverify_files"] = 0
        percat = {}
        for f in file_results:
            r = f["configs"].get(cfg)
            if r is None:
                continue
            a["files"] += 1
            for k in AGG_KEYS:
                a[k] += r[k]
            for c, b in r["per_category"].items():
                d = _cat_bucket(percat, c)
                for k in b:
                    d[k] += b[k]
            pv = f["pverify"].get(cfg)
            if pv is not None:
                a["pverify_files"] += 1
                a["pverify_kept"] += pv[0]
                a["pverify_total"] += pv[1]
        # delivered-only leak split: items on quarantined files never ship,
        # so the headline leak rate is also reported for delivered files alone
        a["must_beep_delivered"] = sum(
            f["configs"][cfg]["must_beep"] for f in file_results
            if f["configs"].get(cfg) is not None
            and not _quarantined(f["file_id"]))
        a["leaked_delivered"] = sum(
            f["configs"][cfg]["leaked"] for f in file_results
            if f["configs"].get(cfg) is not None
            and not _quarantined(f["file_id"]))
        a["detection_recall"] = _ratio(a["ref_matched"], a["ref_items"])
        a["detection_precision"] = _ratio(a["intervals_matched"],
                                          a["intervals"])
        a["redaction_recall"] = _ratio(a["covered"], a["must_beep"])
        a["leak_rate"] = (1 - a["redaction_recall"]
                          if a["redaction_recall"] is not None else None)
        a["policy_verified_precision"] = _ratio(a["pverify_kept"],
                                                a["pverify_total"])
        a["per_category"] = percat
        agg[cfg] = a
    return agg


# ----------------------------------------------------------------- reporting

def build_report(cfg_meta, file_results, agg, configs):
    L = ["# Detection precision/recall report — judge `%s`, verifier `%s`, "
         "profile `%s`" % (cfg_meta["judge_model"], cfg_meta["verify_model"],
                           cfg_meta["profile"]),
         "",
         "Generated %s by `tools/judge_precision_recall.py` | reference "
         "listeners: %s | %d file(s) | predictions: `%s`"
         % (cfg_meta["generated_at"], ", ".join(LISTENERS),
            len(file_results), cfg_meta["pred_root"]),
         "", "## Metric definitions", ""]
    for name, d in METRIC_DEFS:
        L.append("- **%s** — %s" % (name, d))
    L += ["", "## Scope of these numbers", ""]
    for c in CAVEATS:
        L.append("> **%s**" % c)
        L.append(">")
    L[-1] = ""
    L += ["## Aggregate per configuration (micro-averaged)", "",
          "| config | files | detection recall | detection precision | "
          "redaction recall | leak rate (all files) | leak rate (delivered "
          "files) | policy-verified precision (suppl.) |",
          "|---|---|---|---|---|---|---|---|"]
    for c in configs:
        a = agg[c]
        leak = ("%s" % _pct(a["must_beep"] - a["covered"], a["must_beep"]))
        leak_del = _pct(a["leaked_delivered"], a["must_beep_delivered"])
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
            CONFIG_LABELS.get(c, c), a["files"],
            _pct(a["ref_matched"], a["ref_items"]),
            _pct(a["intervals_matched"], a["intervals"]),
            _pct(a["covered"], a["must_beep"]), leak, leak_del,
            _pct(a["pverify_kept"], a["pverify_total"])))
    L += ["",
          "Leak rate (all files) counts every corpus file including "
          "quarantined ones; the production gates withhold quarantined "
          "files from delivery, so leak rate (delivered files) is the "
          "shipping-relevant view. A file with no gate output counts as "
          "delivered."]
    L += ["", "## Per file", ""]
    for c in configs:
        L += ["### %s" % CONFIG_LABELS.get(c, c), "",
              "| file_id | ref items | detection recall | detection "
              "precision | must-beep | leaked | notes |",
              "|---|---|---|---|---|---|---|"]
        for f in file_results:
            r = f["configs"].get(c)
            if r is None:
                L.append("| %s | %d | n/a | n/a | n/a | n/a | no output for "
                         "this config |" % (f["file_id"], len(f["items"])))
                continue
            notes = []
            if _quarantined(f["file_id"]):
                notes.append("QUARANTINED — withheld from delivery")
            if r["rendered_missing"]:
                notes.append("rendered_intervals.json missing — all "
                             "must-beep count leaked")
            if r["missed_item_ids"]:
                notes.append("missed: " + ",".join(r["missed_item_ids"]))
            L.append("| %s | %d | %s | %s | %d | %d | %s |" % (
                f["file_id"], r["ref_items"],
                _pct(r["ref_matched"], r["ref_items"]),
                _pct(r["intervals_matched"], r["intervals"]),
                r["must_beep"], r["leaked"], "; ".join(notes)))
        L.append("")
    L += ["## Per category (aggregate)", ""]
    for c in configs:
        a = agg[c]
        if not a["per_category"]:
            continue
        L += ["### %s" % CONFIG_LABELS.get(c, c), "",
              "| category | detection recall | detection precision |",
              "|---|---|---|"]
        for cat in sorted(a["per_category"]):
            b = a["per_category"][cat]
            L.append("| %s | %s | %s |" % (
                cat, _pct(b["ref_matched"], b["ref"]),
                _pct(b["iv_matched"], b["iv"])))
        L.append("")
    l1, l2 = LISTENERS[0], LISTENERS[-1]
    L += ["## Speakers and languages (per file, per listener)", "",
          "| file_id | %s speakers | %s speakers | agree | %s primary "
          "lang | %s primary lang | agree | all languages (union) |"
          % (l1, l2, l1, l2),
          "|---|---|---|---|---|---|---|---|"]
    for f in file_results:
        s = f["speakers_languages"]
        pe = s["per_listener"].get(l1, {})
        pg = s["per_listener"].get(l2, {})
        langs = []
        for src in (pe.get("languages", []), pg.get("languages", [])):
            for l in src:
                if l not in langs:
                    langs.append(l)
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            f["file_id"], pe.get("speakers", "n/a"), pg.get("speakers", "n/a"),
            s["speakers_agree"], pe.get("primary_language", "n/a"),
            pg.get("primary_language", "n/a"), s["primary_language_agree"],
            ", ".join(langs)))
    L += ["", "## Cross-listener timestamp agreement", "",
          "Start-time offset between the two listeners' spans of the SAME "
          "value (their relative clock skew — the reason beep coverage is "
          "scored skew-aware and value clustering allows +/-2.5 s).", "",
          "| file_id | value pairs | median offset ms | p90 offset ms |",
          "|---|---|---|---|"]
    all_offs = []
    for f in file_results:
        tm = f.get("timing_offsets_ms") or {}
        all_offs.extend(f.get("timing_offsets_raw") or [])
        L.append("| %s | %s | %s | %s |" % (
            f["file_id"], tm.get("n", 0),
            tm.get("median") if tm.get("median") is not None else "n/a",
            tm.get("p90") if tm.get("p90") is not None else "n/a"))
    if all_offs:
        L.append("| **all files** | %d | %s | %s |"
                 % (len(all_offs), _pctl(all_offs, 0.5), _pctl(all_offs, 0.9)))
    L += ["", "## Reference items per file", ""]
    for f in file_results:
        L += ["### %s — %d item(s)" % (f["file_id"], len(f["items"])), "",
              "| item | interval ms | categories | heard by | action | "
              "spoken_text |", "|---|---|---|---|---|---|"]
        for it in f["items"]:
            L.append("| %s | %d-%d | %s | %s | %s | %s |" % (
                it["item_id"], it["start_ms"], it["end_ms"],
                ",".join(it["categories"]), ",".join(it["engines"]),
                it["action"], (it["spoken_text"] or "")[:60].replace("|", "/")))
        L.append("")
    return "\n".join(L) + "\n"


def write_reports(out_dir, cfg_meta, file_results, agg, configs):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "precision_recall_report.md"
    md.write_text(build_report(cfg_meta, file_results, agg, configs))
    js = out / "precision_recall_report.json"
    js.write_text(json.dumps({
        "config": cfg_meta, "caveats": CAVEATS,
        "metric_definitions": {k: v for k, v in METRIC_DEFS},
        "thresholds": {"cluster_iou": pgt.CLUSTER_IOU,
                       "cluster_cover": pgt.CLUSTER_COVER,
                       "value_match_slop_ms": 2500,
                       "coverage_min": pgt.COVERAGE_MIN},
        "aggregate": agg, "files": file_results},
        indent=2, ensure_ascii=False, default=str))
    return md, js


# ----------------------------------------------------------------------- run

def resolve_profile(fid, pred_root, configs, override):
    if override:
        return override
    for c in ["gemini"] + [c for c in configs if c != "gemini"]:
        ri = _load_optional(config_dir(pred_root, fid, c)
                            / "rendered_intervals.json")
        if ri and ri.get("profile"):
            return ri["profile"]
    import common
    return common.CONFIG.get("profile", "strict")


def process_file(fid, configs, pred_root, out_dir, span_cache, judge_model,
                 verify_model, profile_override, force, pverify=True):
    items, notes, rejected = build_reference(fid, pred_root, span_cache,
                                             judge_model, verify_model, force)
    profile = resolve_profile(fid, pred_root, configs, profile_override)
    for k, it in enumerate(items):
        it["item_id"] = "%s_R%03d" % (fid, k + 1)
        it["action"] = policy.action_for(it["categories"],
                                         it["public_figure"], profile)
    offs = timing_offsets(items)
    res = {"file_id": fid, "profile": profile, "items": items,
           "notes": notes, "verifier_rejected": rejected,
           "configs": {}, "pverify": {},
           "timing_offsets_ms": {"n": len(offs),
                                 "median": _pctl(offs, 0.5),
                                 "p90": _pctl(offs, 0.9)},
           "timing_offsets_raw": offs,
           "speakers_languages": listener_summary(fid, pred_root)}
    for c in configs:
        cdir = config_dir(pred_root, fid, c)
        res["configs"][c] = score_config_file(items, cdir, profile)
        res["pverify"][c] = (policy_verified(fid, c, cdir, out_dir,
                                             verify_model, force)
                             if pverify and res["configs"][c] is not None
                             else None)
    return res


def run(args):
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    fids = ([f.strip() for f in args.files.split(",") if f.strip()]
            if args.files else pgt.discover_files(args.pred_root, configs))
    if not fids:
        print("no files found under %s" % args.pred_root)
        return 0
    file_results = []
    for fid in fids:
        print("scoring %s ..." % fid)
        res = process_file(fid, configs, args.pred_root, args.out,
                           args.span_cache, args.judge_model,
                           args.verify_model, args.profile, args.force,
                           pverify=not args.no_pverify)
        print("  %d reference item(s) | %s" % (
            len(res["items"]),
            " | ".join("%s: R %s P %s" % (
                c, _pct(r["ref_matched"], r["ref_items"]),
                _pct(r["intervals_matched"], r["intervals"]))
                for c, r in res["configs"].items() if r)))
        file_results.append(res)
    agg = aggregate(file_results, configs)
    cfg_meta = {"generated_at": datetime.date.today().isoformat(),
                "judge_model": args.judge_model,
                "verify_model": args.verify_model,
                "profile": file_results[0]["profile"] if file_results else None,
                "configs": configs, "pred_root": str(args.pred_root),
                "span_cache": str(args.span_cache), "out": str(args.out)}
    md, js = write_reports(args.out, cfg_meta, file_results, agg, configs)
    for c in configs:
        a = agg[c]
        print("%-28s det recall %-16s det precision %-16s leak %-12s "
              "pverified %s"
              % (CONFIG_LABELS.get(c, c),
                 _pct(a["ref_matched"], a["ref_items"]),
                 _pct(a["intervals_matched"], a["intervals"]),
                 _pct(a["must_beep"] - a["covered"], a["must_beep"]),
                 _pct(a["pverify_kept"], a["pverify_total"])))
    print("report -> %s\njson   -> %s" % (md, js))
    return 0


# ------------------------------------------------------------------ self-test

def self_test():
    import tempfile
    LISTENERS[:] = ["elevenlabs", "gemini"]   # fixtures build these two ears
    cases = []

    def case(name, ok, detail=""):
        cases.append((name, ok))
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  [%s]" % detail) if (detail and not ok) else ""))

    counter = {"calls": 0, "verify": 0}
    old_call = judge_client.judge_call
    judge_client.judge_call = pgt._fake_judge_factory(counter)
    try:
        tmp = Path(tempfile.mkdtemp(prefix="pr_selftest_"))
        pred, out, span_cache = tmp / "output", tmp / "pr", tmp / "spans"
        fid = "prsynth"
        DIGITS = pgt.DIGITS

        def digits_at(t0):
            return [(DIGITS[k], t0 + k * 500, t0 + k * 500 + 400)
                    for k in range(10)]

        fill = [("hello%d" % k, k * 700, k * 700 + 300) for k in range(8)]
        elv = (fill + digits_at(20100)
               + [("Ramesh", 30000, 30400), ("Kumar", 30500, 30900),
                  ("diabetes", 45050, 45450),
                  ("paid", 50000, 50300), ("the", 50400, 50600),
                  ("bill", 50700, 51000)])
        gem = (fill + [("Vidya", 13000, 13400)] + digits_at(20000)
               + [("diabetes", 45000, 45400),
                  ("thirty", 60000, 60300), ("thousand", 60400, 60800)])
        for eng, wtrip in (("elevenlabs", elv), ("gemini", gem)):
            d = pgt.engine_dir(pred, fid, eng)
            d.mkdir(parents=True)
            (d / "transcript.json").write_text(
                json.dumps(pgt._mk_transcript(wtrip)))

        # the config under test: phone interval deliberately NON-overlapping
        # with the reference hull but within 2.5 s and same value (tests the
        # value-match bridge); ramesh temporal match; thirty-thousand match;
        # "paid the bill" is the planted false positive (the verifier
        # rejected it from the reference, and rejects it again here)
        cdir = pred / (fid + "__cfgtest")
        cdir.mkdir(parents=True)
        ivs = [
            {"start_ms": 27000, "end_ms": 29000, "categories": ["PHONE"],
             "spoken_text": " ".join(DIGITS)},
            {"start_ms": 30000, "end_ms": 30900,
             "categories": ["FULL_NAME"], "spoken_text": "Ramesh Kumar"},
            {"start_ms": 60000, "end_ms": 60800,
             "categories": ["FINANCIAL_INFO"],
             "spoken_text": "thirty thousand"},
            {"start_ms": 50000, "end_ms": 51000,
             "categories": ["FINANCIAL_INFO"], "spoken_text": "paid the bill"},
        ]
        (cdir / "intervals.json").write_text(
            json.dumps({"file_id": fid, "intervals": ivs}))
        (cdir / "rendered_intervals.json").write_text(json.dumps({
            "profile": "medical_consult",
            "rendered_merged": [
                {"start_ms": 19800, "end_ms": 25200, "channel": None},
                {"start_ms": 29800, "end_ms": 31100, "channel": None}]}))
        (cdir / "transcript.json").write_text(
            json.dumps(pgt._mk_transcript(elv)))

        res = process_file(fid, ["cfgtest"], pred, out, span_cache, "fake",
                           "fake-verifier", None, False, pverify=True)
        texts = sorted(it["spoken_text"] for it in res["items"])
        case("reference: 5 items, paid-the-bill rejected by verifier",
             len(res["items"]) == 5 and "paid the bill" not in texts
             and "Vidya" in texts and "Ramesh Kumar" in texts
             and "thirty thousand" in texts, str(texts))

        r = res["configs"]["cfgtest"]
        case("detection recall 3/5 (phone via value-match bridge, ramesh, "
             "thirty-thousand; vidya + diabetes missed)",
             r["ref_matched"] == 3 and r["ref_items"] == 5
             and len(r["missed_item_ids"]) == 2, str(r))
        phone_item = next(it for it in res["items"]
                          if "PHONE" in it["categories"])
        case("value-match bridges clock skew (no temporal overlap)",
             not pgt.overlap_pair(phone_item, ivs[0])
             and match_item_interval(phone_item, ivs[0]),
             str((phone_item["start_ms"], phone_item["end_ms"])))
        case("detection precision 3/4 (paid-the-bill unmatched)",
             r["intervals_matched"] == 3 and r["intervals"] == 4
             and r["unmatched_intervals"][0]["spoken_text"] == "paid the bill",
             str(r["unmatched_intervals"]))
        case("profile exclusion: diabetes annotate-only; redaction 2/4 "
             "covered, 2 leaked",
             r["must_beep"] == 4 and r["annotate_only"] == 1
             and r["covered"] == 2 and r["leaked"] == 2, str(r))
        pc = r["per_category"]
        case("per-category rollup: FULL_NAME recall 1/2, PHONE 1/1",
             pc["FULL_NAME"]["ref"] == 2 and pc["FULL_NAME"]["ref_matched"] == 1
             and pc["PHONE"]["ref"] == 1 and pc["PHONE"]["ref_matched"] == 1,
             str(pc))
        case("policy-verified precision 3/4 (verifier rejects the plant)",
             res["pverify"]["cfgtest"] == (3, 4),
             str(res["pverify"]))
        tm = res["timing_offsets_ms"]
        case("cross-listener timing offsets measured (phone 100ms, "
             "diabetes 50ms)",
             tm["n"] == 2 and tm["median"] in (50, 100)
             and sorted(res["timing_offsets_raw"]) == [50, 100], str(tm))
        s = res["speakers_languages"]
        case("speakers/languages summary present with agreement flags",
             s["per_listener"]["elevenlabs"]["speakers"] == 1
             and s["speakers_agree"] is True
             and s["primary_language_agree"] is True, str(s))

        # caching: a re-run must add no judge calls (spans + pverify cached)
        n_calls, n_verify = counter["calls"], counter["verify"]
        process_file(fid, ["cfgtest"], pred, out, span_cache, "fake",
                     "fake-verifier", None, False, pverify=True)
        case("caches: re-run adds no judge or verifier calls",
             counter["calls"] == n_calls and counter["verify"] == n_verify,
             str(counter))

        agg = aggregate([res], ["cfgtest"])
        a = agg["cfgtest"]
        case("aggregate rates",
             abs(a["detection_recall"] - 0.6) < 1e-9
             and abs(a["detection_precision"] - 0.75) < 1e-9
             and abs(a["redaction_recall"] - 0.5) < 1e-9
             and abs(a["leak_rate"] - 0.5) < 1e-9
             and abs(a["policy_verified_precision"] - 0.75) < 1e-9, str(a))
        cfg_meta = {"generated_at": "test", "judge_model": "fake",
                    "verify_model": "fake-verifier",
                    "profile": "medical_consult", "configs": ["cfgtest"],
                    "pred_root": str(pred), "span_cache": str(span_cache),
                    "out": str(out)}
        md, js = write_reports(out, cfg_meta, [res], agg, ["cfgtest"])
        txt = md.read_text()
        case("report written: definitions, caveats, detection naming",
             md.exists() and js.exists()
             and "DETECTION RECALL" in txt and "DETECTION PRECISION" in txt
             and "shared benchmark" in txt and "model family" in txt
             and "Speakers and languages" in txt
             and "Cross-listener timestamp agreement" in txt, txt[:200])

        # a config with no output for this file reports as absent, not zero
        res2 = process_file(fid, ["cfgtest", "ghost"], pred, out, span_cache,
                            "fake", "fake-verifier", None, False,
                            pverify=True)
        case("missing config output -> None (absent), not zeros",
             res2["configs"]["ghost"] is None
             and res2["pverify"]["ghost"] is None
             and aggregate([res2], ["ghost"])["ghost"]["files"] == 0)
    finally:
        judge_client.judge_call = old_call

    failed = [n for n, ok in cases if not ok]
    print("\nself-test: %d/%d passed" % (len(cases) - len(failed), len(cases)))
    return 1 if failed else 0


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs",
                    default="elevenlabs,gemini,openai",
                    help="comma list of config dir suffixes to score "
                         "('gemini' = the bare <fid>/ dir; e.g. "
                         "elevenlabs,gemini,openai,elevenlabs__gpt,"
                         "openai__gpt)")
    ap.add_argument("--files", default=None,
                    help="comma list of file_ids (default: every base dir "
                         "under --pred-root with a transcript.json)")
    ap.add_argument("--pred-root", default="output")
    ap.add_argument("--out", default="output/judge/precision_recall")
    ap.add_argument("--span-cache", default="output/judge/pseudo_gt",
                    help="judge span cache dir (SHARED with judge_pseudo_gt "
                         "so verified spans are reused, not re-judged)")
    ap.add_argument("--judge-model", default=judge_client.DEFAULT_JUDGE)
    ap.add_argument("--verify-model", default=judge_client.DEFAULT_VERIFIER)
    ap.add_argument("--profile", default=None,
                    choices=sorted(policy.ANNOTATE_ONLY),
                    help="override the render profile (counterfactual "
                         "scoring only; default: read per file from "
                         "rendered_intervals.json)")
    ap.add_argument("--no-pverify", action="store_true",
                    help="skip the supplementary policy-verified precision")
    ap.add_argument("--listeners", default=",".join(LISTENERS),
                    help="comma list of the reference listener engines "
                         "(transcripts read from <pred-root>/<fid><suffix>; "
                         "default: %(default)s)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    LISTENERS[:] = [e.strip() for e in args.listeners.split(",") if e.strip()]
    sys.exit(run(args))


if __name__ == "__main__":
    main()
