#!/usr/bin/env python3
"""Phase B — cross-engine pseudo-ground-truth and leak scoring via the
independent LLM judge.

Each engine's RAW transcript.json (gemini at <pred-root>/<fid>/, elevenlabs at
<pred-root>/<fid>__elevenlabs/, openai at <pred-root>/<fid>__openai/) is run
through the SAME Taxonomy v0.2 detection policy our stage-C detector uses, but
by the independent judge model (OpenRouter, strict structured outputs). Judge
spans are word-indexed; each is mapped to time via that engine's own word
timestamps, then the per-engine span sets are UNIONED by temporal clustering
(IoU >= 0.3, or one span covering >= 70% of the other) into pseudo-GT items.

Each engine is then scored against the pseudo-GT:
  PSEUDO LEAK   a pseudo-GT item whose action under the profile
                (policy.action_for, default medical_consult) is "beep" is
                LEAKED for engine X unless >= 95% of it is covered by X's
                rendered_intervals.json (channel semantics as in
                tools/score_run.py).
  AGREEMENT     fraction of pseudo-GT items overlapping (same cluster rule)
                an interval in the engine's intervals.json — for gemini this
                measures our stage-C detector against the independent judge.
  SURFACED-BY   how many engines' transcripts surfaced each item; items seen
                by only ONE engine are the transcription-dependent ones.

SCOPE (also printed in every report): pseudo-GT is a UNION of what the
engines heard — a shared benchmark for comparing engines fairly. PII missed
by ALL engines sits outside it, so human spot-checks complement these
numbers.

Time-anchorable judge spans additionally pass an adversarial VERIFICATION
PASS per engine-file before clustering (default ON, --no-verify to skip):
judge_client.verify_findings sends ONE batched call per engine-file to the
verifier model (default openai/gpt-5.6-luna, --verify-model to override)
asking, per span, for the specific INCLUDE rule it satisfies — otherwise the
span is REJECTED as over-reach (rule 11 calibration exclusions,
amount-less/product-name/generic-process/no-person-link cases). Fail-open per
span: a malformed or missing verdict keeps it. Rejection counts are reported
per engine.

Judge spans are cached at <out>/<fid>__<engine>.spans.json (--force to redo;
cache entries carry a "schema" field — entries without it predate the
verification pass and are stale for --verify runs).

Usage:
  python3 tools/judge_pseudo_gt.py [--engines gemini,elevenlabs,openai]
      [--files <fid>,...] [--judge-model openai/gpt-5.6-sol]
      [--no-verify] [--verify-model openai/gpt-5.6-luna]
      [--out output/judge/pseudo_gt] [--pred-root output]
      [--profile medical_consult] [--force]
  python3 tools/judge_pseudo_gt.py --self-test
"""
import argparse
import datetime
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

JUDGE_CHUNK = 4500           # words per judge call (llm_detect-style chunking)
CLUSTER_IOU = 0.30           # two spans cluster at this temporal IoU ...
CLUSTER_COVER = 0.70         # ... or when one covers this much of the other
COVERAGE_MIN = score_run.COVERAGE_MIN   # rendered coverage below this => leak
CONF_RANK = {"high": 2, "medium": 1, "low": 0}
ENGINE_SUFFIX = {"gemini": "", "elevenlabs": "__elevenlabs",
                 "openai": "__openai",
                 # reference-listener candidates (2026-09-03): transcripts
                 # live at <pred-root>/<fid>__<engine>/ like any engine
                 "el_scribe_v1": "__el_scribe_v1",
                 "gem_25_pro": "__gem_25_pro",
                 "gem_38_flash": "__gem_38_flash",
                 "gem_35_transcribe": "__gem_35_transcribe"}
# Span-cache schema. Bumped to 2 when the adversarial verification pass was
# added; entries without the "schema" field predate it and are stale for
# --verify runs (they carry no verification data).
CACHE_SCHEMA = 2
VERIFY_CONTEXT_PAD = 8       # words of context each side for the verifier

CAVEATS = [
    "Pseudo-GT is a UNION of what the engines' transcripts surfaced. PII that "
    "EVERY engine failed to transcribe (or the judge missed on all of them) is "
    "INVISIBLE here — it appears in no denominator.",
    "All rates are measured against this reference: a 0% pseudo leak rate "
    "means nothing the reference contains leaked; human spot-checks "
    "complement it for values outside the reference.",
    "The judge applies the same written policy as stage C but is itself an "
    "LLM; its spans are unvalidated. Human ground truth remains the arbiter "
    "and will later be used to score the judge itself.",
    "Items surfaced by only one engine are transcription-dependent: the other "
    "engines never wrote the value down, so their detectors never had a "
    "chance — leaks on such items measure the whole engine pipeline, not the "
    "detector alone.",
]

# Strict-mode mirror of policy.DETECTION_SCHEMA for OpenRouter structured
# outputs: lowercase JSON-Schema types, additionalProperties false, ALL
# properties required.
JUDGE_DETECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_word": {"type": "integer"},
                    "end_word": {"type": "integer"},
                    "categories": {"type": "array",
                                   "items": {"type": "string"}},
                    "spoken_text": {"type": "string"},
                    "speaker": {"type": "string"},
                    "public_figure": {"type": "boolean"},
                    "confidence": {"type": "string",
                                   "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["start_word", "end_word", "categories",
                             "spoken_text", "speaker", "public_figure",
                             "confidence", "reason"],
            },
        }
    },
    "required": ["spans"],
}


def engine_dir(pred_root, fid, engine):
    return Path(pred_root) / (fid + ENGINE_SUFFIX.get(engine, "__" + engine))


# ------------------------------------------------------------- judge pass

def judge_detect(words, model):
    """Run the judge over the numbered word stream (chunked like
    detect_pii.llm_detect) and return validated word-index spans."""
    spans = []
    for base in range(0, len(words), JUDGE_CHUNK):
        sub = words[base: base + JUDGE_CHUNK]
        prompt = (policy.DETECTION_POLICY + "\n\nTRANSCRIPT WORDS:\n" +
                  detect_pii.words_as_prompt(sub, JUDGE_CHUNK))
        got = judge_client.judge_call(prompt, JUDGE_DETECTION_SCHEMA,
                                      model=model)
        lo, hi = sub[0]["i"], sub[-1]["i"]
        for s in got.get("spans", []):
            try:
                a, b = int(s["start_word"]), int(s["end_word"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (lo <= a <= b <= hi):
                continue
            cats = [c for c in s.get("categories", [])
                    if c in policy.CATEGORIES]
            if not cats:
                continue
            spans.append({"start_word": a, "end_word": b, "categories": cats,
                          "spoken_text": s.get("spoken_text", ""),
                          "speaker": s.get("speaker", ""),
                          "public_figure": bool(s.get("public_figure", False)),
                          "confidence": s.get("confidence", "medium"),
                          "reason": s.get("reason", "")})
    return spans


def _cache_fresh(cached, verify):
    """--verify runs need a schema-2 cache entry written WITH verification;
    anything older (incl. pre-schema entries) is stale for them. --no-verify
    runs accept any cache entry."""
    if not verify:
        return True
    return (cached.get("schema") == CACHE_SCHEMA
            and bool((cached.get("verification") or {}).get("enabled")))


def _no_verification():
    return {"enabled": False, "model": None, "checked": 0, "rejected": 0,
            "rejected_spans": []}


def _cache_view(cached, verify):
    """(spans, verification) as this run should see them. A --no-verify run
    reading a verified cache entry must see the PRE-verification spans
    (verification only removes; without --verify nothing may be removed)."""
    ver = cached.get("verification") or _no_verification()
    if verify or not ver.get("enabled"):
        return cached.get("spans", []), ver
    return (list(cached.get("spans_pre_verify", cached.get("spans", []))),
            _no_verification())


def _span_context(words, s, pad=VERIFY_CONTEXT_PAD):
    """One line of surrounding transcript words for the verifier."""
    try:
        a, b = int(s["start_word"]), int(s["end_word"])
    except (KeyError, TypeError, ValueError):
        return ""
    a = max(0, min(a, len(words) - 1))
    b = max(a, min(b, len(words) - 1))
    return " ".join(w.get("w", "") for w in
                    words[max(0, a - pad): min(len(words), b + pad + 1)])


def judge_spans_cached(fid, engine, words, out_dir, model, force,
                       verify=False,
                       verify_model=judge_client.DEFAULT_VERIFIER):
    """Judge one engine-file, caching at <out>/<fid>__<engine>.spans.json.
    Returns (spans, verification).

    With verify=True the time-anchorable spans additionally pass the
    adversarial verification pass (judge_client.verify_findings — ONE batched
    call per engine-file) before clustering; rejected spans are kept in the
    cache under verification.rejected_spans. Spans that fail anchoring are
    not sent to the verifier (they are dropped at anchoring regardless)."""
    cache = Path(out_dir) / ("%s__%s.spans.json" % (fid, engine))
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if _cache_fresh(cached, verify):
            return _cache_view(cached, verify)
    spans = judge_detect(words, model)
    spans_pre_verify = list(spans)
    verification = {"enabled": bool(verify),
                    "model": verify_model if verify else None,
                    "checked": 0, "rejected": 0, "rejected_spans": []}
    if verify and spans:
        idxs = [k for k, s in enumerate(spans)
                if anchor_span(words, s, engine) is not None]
        to_check = [spans[k] for k in idxs]
        if to_check:
            keep = judge_client.verify_findings(
                to_check, policy.DETECTION_POLICY, model=verify_model,
                context_texts=[_span_context(words, s) for s in to_check])
            rejected_at = set(idxs[j] for j, k in enumerate(keep) if not k)
            verification["checked"] = len(to_check)
            verification["rejected"] = len(rejected_at)
            verification["rejected_spans"] = [spans[k]
                                              for k in sorted(rejected_at)]
            spans = [s for k, s in enumerate(spans) if k not in rejected_at]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "schema": CACHE_SCHEMA,
        "file_id": fid, "engine": engine, "model": model,
        "judged_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_words": len(words), "spans": spans,
        "spans_pre_verify": spans_pre_verify,
        "verification": verification},
        indent=2, ensure_ascii=False))
    return spans, verification


# --------------------------------------------------- time anchoring + trim

def anchor_span(words, s, engine):
    """Map one word-index judge span to time via the engine's own word
    timestamps. Invalid indices -> None. Applies the span-width trim guard
    from detect_pii.build_intervals: a range far wider than the quoted
    spoken_text is trimmed to the best-matching window (confidence -> low)."""
    try:
        a, b = int(s["start_word"]), int(s["end_word"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= a <= b < len(words)):
        return None
    trimmed = False
    expected = max(1, len((s.get("spoken_text") or "").split()))
    span_n = b - a + 1
    if span_n > expected * 3 + 5:
        def key(t):
            return t.strip().lower().strip(".,!?;:।")
        first = (key((s.get("spoken_text") or "").split()[0])
                 if s.get("spoken_text") else "")
        best = a
        for cand in range(a, b - expected + 2):
            if first and key(words[cand]["w"]) == first:
                best = cand
                break
        a, b = best, min(best + expected - 1, b)
        trimmed = True
    ws = words[a: b + 1]
    channels = {w.get("channel") for w in ws}
    conf = s.get("confidence", "medium")
    if conf not in CONF_RANK:
        conf = "medium"
    return {"start_ms": ws[0]["start_ms"],
            "end_ms": max(w["end_ms"] for w in ws),
            "categories": list(s["categories"]),
            "spoken_text": s.get("spoken_text", ""),
            "engine": engine,
            "public_figure": bool(s.get("public_figure", False)),
            "confidence": "low" if trimmed else conf,
            "channel": channels.pop() if len(channels) == 1 else None,
            "trimmed": trimmed}


# ------------------------------------------------------ cross-engine union

def overlap_pair(a, b):
    """Cluster rule: temporal IoU >= CLUSTER_IOU, or one span covers
    >= CLUSTER_COVER of the other."""
    inter = min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"])
    if inter <= 0:
        return False
    la = a["end_ms"] - a["start_ms"]
    lb = b["end_ms"] - b["start_ms"]
    union = la + lb - inter
    iou = inter / float(union) if union else 1.0
    cover = inter / float(min(la, lb)) if min(la, lb) else 1.0
    return iou >= CLUSTER_IOU or cover >= CLUSTER_COVER


def _value_tokens(s):
    """Normalized token set of a span's quoted value, for cross-engine
    same-value matching. Uses the scorer's normalizer (incl. number-phrase
    canonicalization) so "Rs 30 635" and "thirty thousand six hundred and
    thirty five rupees" share the token "30635"."""
    return set(score_run.norm_span_tokens(str(s.get("spoken_text") or "")))


# fold_indic moved to pipeline/textnorm.py (2026-09-02) so the pipeline's
# edge verifier can share it; re-exported here for compatibility.
from textnorm import fold_indic  # noqa: E402


def same_value_nearby(a, b, slop_ms=2500):
    """Engines timestamp the same audio differently (offsets of 1-2 s are
    common), which breaks pure temporal-IoU clustering and double-counts one
    spoken value as two pseudo-GT items. Cluster additionally when two spans
    from DIFFERENT engines are within slop_ms and share the same category and
    half their value tokens."""
    if a["engine"] == b["engine"]:
        return False
    if not (set(a["categories"]) & set(b["categories"])):
        return False
    gap = max(a["start_ms"], b["start_ms"]) - min(a["end_ms"], b["end_ms"])
    if gap > slop_ms:
        return False
    ta, tb = _value_tokens(a), _value_tokens(b)
    if ta and tb:
        inter = len(ta & tb)
        if inter >= max(1, min(len(ta), len(tb)) // 2):
            return True
    # engines mis-hear the same name differently ("Keerthana" vs "Seerana")
    # or write it in different scripts ("ಸುಧಾ" vs "Sudha"); fold Indic scripts
    # to a Latin skeleton, then fall back to character similarity for short
    # single-category values
    import difflib
    va = " ".join(sorted(ta)) or str(a.get("spoken_text") or "").lower()
    vb = " ".join(sorted(tb)) or str(b.get("spoken_text") or "").lower()
    va, vb = fold_indic(va), fold_indic(vb)
    if va and vb and max(len(va), len(vb)) <= 40:
        return difflib.SequenceMatcher(None, va, vb).ratio() >= 0.55
    return False


NAME_SLOT_GAP_MS = 600       # cross-engine name-respelling merge window


def _name_slot_pairs(spans):
    """Cross-engine FULL_NAME spans occupying the SAME time slot are one
    occurrence heard with different spellings (e.g. 'Vidya' vs 'Ritika' at
    13.0-13.8 s: barely-overlapping spans, no value match — two items without
    this guard). Chain FULL_NAME spans whose gaps are <= NAME_SLOT_GAP_MS
    into slots; a slot merges ONLY when it holds several engines and no
    engine contributes more than one span — an engine hearing TWO names in
    the window ('Ramesh' then 'Vidya') marks genuinely distinct names, and
    the slot is left alone. Returns the set of index pairs to force-merge."""
    names = sorted((i for i, s in enumerate(spans)
                    if "FULL_NAME" in s["categories"]),
                   key=lambda i: (spans[i]["start_ms"], spans[i]["end_ms"]))
    chains, last_end = [], None
    for i in names:
        if chains and spans[i]["start_ms"] - last_end <= NAME_SLOT_GAP_MS:
            chains[-1].append(i)
            last_end = max(last_end, spans[i]["end_ms"])
        else:
            chains.append([i])
            last_end = spans[i]["end_ms"]
    pairs = set()
    for ch in chains:
        per_engine = {}
        for i in ch:
            per_engine.setdefault(spans[i]["engine"], []).append(i)
        if len(per_engine) > 1 and all(len(v) == 1
                                       for v in per_engine.values()):
            ids = sorted(v[0] for v in per_engine.values())
            pairs.update((a, b) for x, a in enumerate(ids)
                         for b in ids[x + 1:])
    return pairs


def cluster_spans(spans):
    """Union-find clustering of time-anchored spans; one pseudo-GT item per
    cluster: interval hull, union of categories, surfacing engines, majority
    public_figure (ties -> False, the safety-conservative choice: the item
    stays in the leak denominator), best confidence. Spans cluster on temporal
    overlap, on same-value-nearby (cross-engine timestamp slop), or on the
    FULL_NAME same-slot rule (cross-engine respellings of one occurrence)."""
    n = len(spans)
    parent = list(range(n))
    slot_pairs = _name_slot_pairs(spans)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if (overlap_pair(spans[i], spans[j])
                    or same_value_nearby(spans[i], spans[j])
                    or (i, j) in slot_pairs):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(spans[i])
    items = []
    for g in groups.values():
        pf_true = sum(1 for s in g if s["public_figure"])
        best = max(g, key=lambda s: (CONF_RANK[s["confidence"]],
                                     len(s.get("spoken_text") or "")))
        channels = {s.get("channel") for s in g}
        items.append({
            "start_ms": min(s["start_ms"] for s in g),
            "end_ms": max(s["end_ms"] for s in g),
            "categories": sorted({c for s in g for c in s["categories"]}),
            "engines": sorted({s["engine"] for s in g}),
            "public_figure": pf_true > len(g) - pf_true,
            "confidence": max(g, key=lambda s: CONF_RANK[s["confidence"]]
                              )["confidence"],
            "spoken_text": best.get("spoken_text", ""),
            "channel": channels.pop() if len(channels) == 1 else None,
            # the constituent spans: value-match clustering can chain REPEATED
            # mentions of one value into an item whose hull spans minutes of
            # non-PII conversation — coverage must be judged on what was
            # actually spoken, not the hull
            "source_spans": [{"start_ms": s["start_ms"],
                              "end_ms": s["end_ms"],
                              "engine": s["engine"],
                              "channel": s.get("channel")} for s in g],
        })
    items.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    return items


# ------------------------------------------------------------ engine scoring

def item_coverage(it, rendered, slop_ms=2500):
    """Duration-weighted beep coverage of an item's constituent spans (falls
    back to the item hull for items without source_spans). Two artifacts of
    cross-listener clustering are corrected here:
    1. Repeated mentions of one value cluster into an item whose HULL can span
       minutes of non-PII speech — every occurrence must be covered, but the
       gaps between occurrences must not count against coverage.
    2. Listeners timestamp the same occurrence 1-2 s apart, so the OTHER
       listener's span of an occurrence sits shifted against this engine's
       render timeline. A span therefore counts as covered when a
       different-listener span of the same occurrence (overlap or gap <=
       slop_ms within the item) is covered — the engine beeped that
       occurrence on its own clock; the residue is skew, not a leak. An
       occurrence NO listener's covered span accounts for still leaks."""
    spans = it.get("source_spans") or [it]
    covs = [score_run.coverage_of(
        {"start_ms": s["start_ms"], "end_ms": s["end_ms"],
         "channel": s.get("channel", it.get("channel"))}, rendered)
        for s in spans]
    eff = list(covs)
    for i, s in enumerate(spans):
        for j, t in enumerate(spans):
            if i == j or s.get("engine") == t.get("engine"):
                continue
            gap = (max(s["start_ms"], t["start_ms"])
                   - min(s["end_ms"], t["end_ms"]))
            if gap <= slop_ms:
                eff[i] = max(eff[i], covs[j])
    total = sum(max(0, s["end_ms"] - s["start_ms"]) for s in spans)
    if total <= 0:
        return 1.0
    return sum(max(0, s["end_ms"] - s["start_ms"]) * e
               for s, e in zip(spans, eff)) / total


def score_engine_for_items(items, rendered, intervals, profile):
    """Score one engine's rendered beeps + detector intervals against the
    pseudo-GT items. rendered = rendered_merged list ([] when the file's
    rendered_intervals.json is missing: everything must-beep leaks);
    intervals = intervals.json 'intervals' list or None (agreement skipped).
    Returns (counts, leaked_indices, agreed_indices)."""
    counts = {"items": len(items), "must_beep": 0, "annotate_only": 0,
              "leaked": 0, "covered": 0, "agree": 0, "agree_eval": 0}
    leaked_idx, agreed_idx = [], []
    for k, it in enumerate(items):
        action = policy.action_for(it["categories"], it["public_figure"],
                                   profile)
        if action == "beep":
            counts["must_beep"] += 1
            cov = item_coverage(it, rendered)
            if cov >= COVERAGE_MIN:
                counts["covered"] += 1
            else:
                counts["leaked"] += 1
                leaked_idx.append(k)
        else:
            counts["annotate_only"] += 1
        if intervals is not None:
            counts["agree_eval"] += 1
            if any(overlap_pair(it, p) for p in intervals):
                counts["agree"] += 1
                agreed_idx.append(k)
    return counts, leaked_idx, agreed_idx


# ------------------------------------------------------------- per-file run

def _load_optional(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def process_file(fid, engines, pred_root, out_dir, model, profile, force,
                 verify=False, verify_model=judge_client.DEFAULT_VERIFIER):
    """Judge + union + score one file. Returns the per-file result dict."""
    notes = []
    all_spans = []
    verifier_rejected = {}
    for eng in engines:
        tpath = engine_dir(pred_root, fid, eng) / "transcript.json"
        transcript = _load_optional(tpath)
        if transcript is None:
            notes.append("%s: transcript.json missing (absent from union)"
                         % eng)
            continue
        words = detect_pii.flatten_words(transcript)
        if not words:
            notes.append("%s: empty transcript (absent from union)" % eng)
            continue
        raw, verification = judge_spans_cached(fid, eng, words, out_dir,
                                               model, force, verify,
                                               verify_model)
        verifier_rejected[eng] = verification.get("rejected", 0)
        if verification.get("rejected"):
            notes.append("%s: %d span(s) rejected by verifier (over-reach)"
                         % (eng, verification["rejected"]))
        anchored = [x for x in (anchor_span(words, s, eng) for s in raw) if x]
        dropped = len(raw) - len(anchored)
        if dropped:
            notes.append("%s: %d judge span(s) dropped (invalid indices)"
                         % (eng, dropped))
        all_spans.extend(anchored)

    items = cluster_spans(all_spans)
    for k, it in enumerate(items):
        it["item_id"] = "%s_P%03d" % (fid, k + 1)
        it["action"] = policy.action_for(it["categories"],
                                         it["public_figure"], profile)
        it["leaked_by"], it["covered_by"], it["agreed_by"] = [], [], []

    per_engine = {}
    for eng in engines:
        edir = engine_dir(pred_root, fid, eng)
        rend_doc = _load_optional(edir / "rendered_intervals.json")
        if rend_doc is None:
            notes.append("%s: rendered_intervals.json missing "
                         "(all must-beep count as leaked)" % eng)
        int_doc = _load_optional(edir / "intervals.json")
        if int_doc is None:
            notes.append("%s: intervals.json missing (agreement skipped)"
                         % eng)
        counts, leaked_idx, agreed_idx = score_engine_for_items(
            items, (rend_doc or {}).get("rendered_merged", []),
            int_doc.get("intervals", []) if int_doc is not None else None,
            profile)
        for k in leaked_idx:
            items[k]["leaked_by"].append(eng)
        for k, it in enumerate(items):
            if it["action"] == "beep" and k not in leaked_idx:
                it["covered_by"].append(eng)
        for k in agreed_idx:
            items[k]["agreed_by"].append(eng)
        per_engine[eng] = counts

    return {"file_id": fid, "items": items, "engines": per_engine,
            "verifier_rejected": verifier_rejected, "notes": notes}


# --------------------------------------------------------------- aggregation

def aggregate(files, engines):
    per_engine = {e: {"must_beep": 0, "leaked": 0, "covered": 0,
                      "agree": 0, "agree_eval": 0, "verifier_rejected": 0}
                  for e in engines}
    surfaced = {}          # n_engines -> {"items": x, "must_beep": y, "leaked_any": z}
    unique = {e: {"items": 0, "must_beep": 0} for e in engines}
    totals = {"items": 0, "must_beep": 0, "annotate_only": 0}
    for f in files:
        for e in engines:
            per_engine[e]["verifier_rejected"] += f.get(
                "verifier_rejected", {}).get(e, 0)
            c = f["engines"].get(e)
            if not c:
                continue
            for k in ("must_beep", "leaked", "covered", "agree", "agree_eval"):
                per_engine[e][k] += c[k]
        for it in f["items"]:
            totals["items"] += 1
            beep = it["action"] == "beep"
            totals["must_beep"] += 1 if beep else 0
            totals["annotate_only"] += 0 if beep else 1
            n = len(it["engines"])
            b = surfaced.setdefault(n, {"items": 0, "must_beep": 0,
                                        "leaked_any": 0})
            b["items"] += 1
            b["must_beep"] += 1 if beep else 0
            b["leaked_any"] += 1 if it["leaked_by"] else 0
            if n == 1 and it["engines"][0] in unique:
                unique[it["engines"][0]]["items"] += 1
                unique[it["engines"][0]]["must_beep"] += 1 if beep else 0
    for e, c in per_engine.items():
        c["pseudo_leak_rate"] = _ratio(c["leaked"], c["must_beep"])
        c["coverage"] = _ratio(c["covered"], c["must_beep"])
        c["agreement"] = _ratio(c["agree"], c["agree_eval"])
    return {"per_engine": per_engine, "surfaced_by": surfaced,
            "unique_by_engine": unique, "totals": totals}


def _ratio(num, den):
    return (num / float(den)) if den else None


# ----------------------------------------------------------------- reporting

def _pct(num, den):
    if not den:
        return "n/a"
    return "%d/%d = %.1f%%" % (num, den, 100.0 * num / den)


def build_report(cfg, files, agg, engines):
    verify_line = ("verifier `%s` (adversarial over-reach filter ON)"
                   % cfg.get("verify_model")
                   if cfg.get("verify") else "verification OFF")
    L = ["# Pseudo-ground-truth report — judge `%s`, profile `%s`"
         % (cfg["judge_model"], cfg["profile"]),
         "",
         "Generated %s by `tools/judge_pseudo_gt.py` | engines: %s | "
         "%d file(s) | predictions: `%s` | %s"
         % (cfg["generated_at"], ", ".join(engines), len(files),
            cfg["pred_root"], verify_line),
         "",
         "## CAVEAT — read before using these numbers", ""]
    for c in CAVEATS:
        L.append("> **%s**" % c)
        L.append(">")
    L[-1] = ""
    t = agg["totals"]
    L += ["## Aggregate per engine", "",
          "Pseudo-GT: %d item(s) total — %d must-beep under `%s`, %d "
          "annotate-only (excluded from the leak denominator)."
          % (t["items"], t["must_beep"], cfg["profile"], t["annotate_only"]),
          "",
          "| engine | must-beep | leaked | pseudo leak rate | covered >=%d%% "
          "| agreement (detector vs judge) | spans rejected by verifier |"
          % int(COVERAGE_MIN * 100),
          "|---|---|---|---|---|---|---|"]
    for e in engines:
        c = agg["per_engine"][e]
        L.append("| %s | %d | %d | %s | %s | %s | %d |" % (
            e, c["must_beep"], c["leaked"], _pct(c["leaked"], c["must_beep"]),
            _pct(c["covered"], c["must_beep"]),
            _pct(c["agree"], c["agree_eval"]),
            c.get("verifier_rejected", 0)))
    L += ["",
          "## Surfaced-by breakdown", "",
          "Items surfaced by a single engine are the transcription-dependent "
          "ones — the other engines never heard the value.", "",
          "| surfaced by N engine(s) | items | must-beep | leaked by >=1 engine |",
          "|---|---|---|---|"]
    for n in sorted(agg["surfaced_by"]):
        b = agg["surfaced_by"][n]
        L.append("| %d | %d | %d | %d |" % (n, b["items"], b["must_beep"],
                                            b["leaked_any"]))
    L += ["", "| engine | items surfaced ONLY by it | of which must-beep |",
          "|---|---|---|"]
    for e in engines:
        u = agg["unique_by_engine"][e]
        L.append("| %s | %d | %d |" % (e, u["items"], u["must_beep"]))
    L += ["", "## Per file", "",
          "| file_id | items | must-beep | ann-only | " +
          " | ".join("%s leaked" % e for e in engines) + " | notes |",
          "|---" * (4 + len(engines) + 1) + "|"]
    for f in files:
        beep = sum(1 for it in f["items"] if it["action"] == "beep")
        cells = [f["file_id"], str(len(f["items"])), str(beep),
                 str(len(f["items"]) - beep)]
        for e in engines:
            c = f["engines"].get(e)
            cells.append(_pct(c["leaked"], c["must_beep"]) if c else "n/a")
        cells.append("; ".join(f["notes"]) or "")
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Leaked pseudo-GT items (leaked by at least one engine)", ""]
    rows = [(f["file_id"], it) for f in files for it in f["items"]
            if it["leaked_by"]]
    if rows:
        L += ["| file_id | item | interval ms | categories | surfaced by | "
              "leaked by | spoken_text |",
              "|---|---|---|---|---|---|---|"]
        for fid, it in rows:
            L.append("| %s | %s | %d-%d | %s | %s | %s | %s |" % (
                fid, it["item_id"], it["start_ms"], it["end_ms"],
                ",".join(it["categories"]), ",".join(it["engines"]),
                ",".join(it["leaked_by"]),
                (it["spoken_text"] or "")[:60].replace("|", "/")))
    else:
        L.append("None.")
    L.append("")
    return "\n".join(L) + "\n"


def write_reports(out_dir, cfg, files, agg, engines):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "pseudo_gt_report.md"
    md.write_text(build_report(cfg, files, agg, engines))
    js = out / "pseudo_gt_report.json"
    js.write_text(json.dumps({
        "config": cfg, "caveats": CAVEATS,
        "thresholds": {"cluster_iou": CLUSTER_IOU,
                       "cluster_cover": CLUSTER_COVER,
                       "coverage_min": COVERAGE_MIN,
                       "judge_chunk_words": JUDGE_CHUNK},
        "aggregate": agg,
        "files": files}, indent=2, ensure_ascii=False))
    return md, js


# ----------------------------------------------------------------------- run

def discover_files(pred_root, engines):
    """file_ids = dirs under pred_root holding a transcript.json whose name is
    not a config-suffixed dir. Config dirs always carry a double-underscore
    suffix (__elevenlabs, __openai, __elevenlabs__gpt, ...); raw file ids
    never contain a double underscore — filtering on "__" keeps newly added
    detector suffixes out without listing them here (bug found 2026-09-02:
    __gpt dirs were discovered as seven spurious extra files)."""
    fids = []
    for d in sorted(Path(pred_root).iterdir()):
        if not d.is_dir() or not (d / "transcript.json").exists():
            continue
        if "__" in d.name:
            continue
        fids.append(d.name)
    return fids


def run(args):
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    fids = ([f.strip() for f in args.files.split(",") if f.strip()]
            if args.files else discover_files(args.pred_root, engines))
    if not fids:
        print("no files to judge under %s" % args.pred_root)
        return 0
    # profile: explicit --profile wins (counterfactual); else read the profile
    # the outputs were actually rendered under from a rendered_intervals.json
    resolved_profile = args.profile
    if not resolved_profile:
        for fid in fids:
            ri = Path(args.pred_root) / fid / "rendered_intervals.json"
            if ri.exists():
                resolved_profile = json.loads(ri.read_text()).get("profile")
                if resolved_profile:
                    break
    if not resolved_profile:
        import common
        resolved_profile = common.CONFIG.get("profile", "strict")
    print("profile: %s (from rendered outputs)" % resolved_profile)
    files = []
    for fid in fids:
        print("judging %s ..." % fid)
        res = process_file(fid, engines, args.pred_root, args.out,
                           args.judge_model, resolved_profile, args.force,
                           verify=args.verify, verify_model=args.verify_model)
        print("  %d pseudo-GT item(s); leaks: %s%s" % (
            len(res["items"]),
            ", ".join("%s=%d" % (e, res["engines"][e]["leaked"])
                      for e in engines if e in res["engines"]),
            ("; verifier rejected: %s" % ", ".join(
                "%s=%d" % (e, n)
                for e, n in sorted(res["verifier_rejected"].items()))
             if any(res.get("verifier_rejected", {}).values()) else "")))
        files.append(res)
    agg = aggregate(files, engines)
    cfg = {"generated_at": datetime.date.today().isoformat(),
           "judge_model": args.judge_model, "profile": resolved_profile,
           "engines": engines, "pred_root": str(args.pred_root),
           "out": str(args.out), "verify": bool(args.verify),
           "verify_model": args.verify_model if args.verify else None}
    md, js = write_reports(args.out, cfg, files, agg, engines)
    for e in engines:
        c = agg["per_engine"][e]
        print("%-12s pseudo leak rate %-16s coverage %-16s agreement %-12s "
              "verifier-rejected %d"
              % (e, _pct(c["leaked"], c["must_beep"]),
                 _pct(c["covered"], c["must_beep"]),
                 _pct(c["agree"], c["agree_eval"]),
                 c.get("verifier_rejected", 0)))
    print("report -> %s\njson   -> %s" % (md, js))
    return 0


# ------------------------------------------------------------------ self-test

def _w(word, s, e, channel=None):
    return {"w": word, "start_ms": s, "end_ms": e, "conf": "high",
            "channel": channel}


def _tspan(s, e, engine, cats=("PHONE",), conf="medium", pf=False,
           text="x", channel=None):
    return {"start_ms": s, "end_ms": e, "categories": list(cats),
            "spoken_text": text, "engine": engine, "public_figure": pf,
            "confidence": conf, "channel": channel, "trimmed": False}


def _mk_transcript(word_triples, channel=None):
    """word_triples: [(word, start_ms, end_ms), ...] -> transcript.json dict."""
    words = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in word_triples]
    return {"duration_ms": max(e for _, _, e in word_triples) + 1000,
            "split_channels": False, "primary_language": "en",
            "languages": [{"lang": "en", "word_share": 1.0}],
            "utterances": [{"speaker": "S1", "lang": "en",
                            "channel": channel, "words": words}]}


DIGITS = ["nine", "eight", "seven", "six", "five", "four", "three", "two",
          "one", "zero"]


def _fake_judge_factory(counter):
    """Content-driven fake judge: parses the numbered word lines back out of
    the prompt and emits spans for known trigger words. Dispatches on the
    schema: VERIFY_SCHEMA calls get verdicts (rejecting the amount-less
    "paid the bill" plant), detection calls get spans."""
    import re

    def fake(prompt, schema, model="fake", max_tokens=0):
        if schema is judge_client.VERIFY_SCHEMA:
            counter["verify"] = counter.get("verify", 0) + 1
            counter.setdefault("verify_prompts", []).append(prompt)
            verdicts = []
            for m in re.finditer(r'(?m)^(\d+)\. text: "([^"]*)"', prompt):
                keep = "paid the bill" not in m.group(2)
                verdicts.append({"index": int(m.group(1)), "keep": keep,
                                 "rule": ("include rule satisfied" if keep
                                          else "REJECT: rule 11 amount-less "
                                               "payment claim")})
            return {"verdicts": verdicts}
        counter["calls"] += 1
        idx_word = {}
        for line in prompt.splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0].isdigit():
                idx_word[int(parts[0])] = parts[2].lower().strip(".,")
        spans = []
        for i, w in sorted(idx_word.items()):
            if w == "nine" and all(idx_word.get(i + k) == DIGITS[k]
                                   for k in range(10)):
                spans.append({"start_word": i, "end_word": i + 9,
                              "categories": ["PHONE"],
                              "spoken_text": " ".join(DIGITS),
                              "speaker": "S1", "public_figure": False,
                              "confidence": "high", "reason": "phone"})
            elif w == "ramesh" and idx_word.get(i + 1) == "kumar":
                spans.append({"start_word": i, "end_word": i + 1,
                              "categories": ["FULL_NAME"],
                              "spoken_text": "Ramesh Kumar", "speaker": "S1",
                              "public_figure": False, "confidence": "high",
                              "reason": "name"})
            elif w == "diabetes":
                spans.append({"start_word": i, "end_word": i,
                              "categories": ["HEALTH_MEDICAL_INFO"],
                              "spoken_text": "diabetes", "speaker": "S1",
                              "public_figure": False, "confidence": "medium",
                              "reason": "condition"})
            elif (w == "paid" and idx_word.get(i + 1) == "the"
                    and idx_word.get(i + 2) == "bill"):
                # planted over-reach: amount-less FINANCIAL_INFO claim
                spans.append({"start_word": i, "end_word": i + 2,
                              "categories": ["FINANCIAL_INFO"],
                              "spoken_text": "paid the bill",
                              "speaker": "S1", "public_figure": False,
                              "confidence": "medium",
                              "reason": "payment claim, no amount"})
            elif w == "thirty" and idx_word.get(i + 1) == "thousand":
                # valued FINANCIAL_INFO -> the verifier must keep it
                spans.append({"start_word": i, "end_word": i + 1,
                              "categories": ["FINANCIAL_INFO"],
                              "spoken_text": "thirty thousand",
                              "speaker": "S1", "public_figure": False,
                              "confidence": "high",
                              "reason": "salary amount"})
            elif w == "vidya":
                # deliberately absurd width -> must trigger the trim guard
                spans.append({"start_word": max(0, i - 20),
                              "end_word": min(max(idx_word), i + 5),
                              "categories": ["FULL_NAME"],
                              "spoken_text": "Vidya", "speaker": "S2",
                              "public_figure": False, "confidence": "high",
                              "reason": "name, wide span"})
        return {"spans": spans}
    return fake


def self_test():
    import tempfile
    global JUDGE_CHUNK
    cases = []

    def case(name, ok, detail=""):
        cases.append((name, ok))
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  [%s]" % detail) if (detail and not ok) else ""))

    # 1. clustering: same value from 3 engines -> ONE item; unique item kept
    spans = [_tspan(10000, 12000, "gemini"),
             _tspan(10100, 12100, "elevenlabs"),
             _tspan(9900, 11900, "openai"),
             _tspan(30000, 31000, "elevenlabs", cats=("FULL_NAME",))]
    items = cluster_spans(spans)
    case("clustering: 3-engine value -> 1 item, unique item kept",
         len(items) == 2
         and items[0]["engines"] == ["elevenlabs", "gemini", "openai"]
         and items[0]["start_ms"] == 9900 and items[0]["end_ms"] == 12100
         and items[1]["engines"] == ["elevenlabs"], str(items))

    # 2. cluster fields: category union, majority public_figure (tie -> False),
    #    best confidence
    items = cluster_spans([
        _tspan(1000, 2000, "gemini", cats=("PHONE",), conf="low", pf=True),
        _tspan(1000, 2000, "elevenlabs", cats=("GOVERNMENT_ID",),
               conf="high", pf=True),
        _tspan(1050, 1950, "openai", cats=("PHONE",), conf="medium", pf=False)])
    tie = cluster_spans([_tspan(1, 1000, "gemini", pf=True),
                         _tspan(1, 1000, "openai", pf=False)])
    case("cluster union/majority/best-confidence",
         items[0]["categories"] == ["GOVERNMENT_ID", "PHONE"]
         and items[0]["public_figure"] is True
         and items[0]["confidence"] == "high"
         and tie[0]["public_figure"] is False, str(items) + str(tie))

    # 3a. low overlap AND different values -> stay separate
    a = _tspan(1000, 2000, "gemini")
    b = _tspan(1900, 4000, "elevenlabs")
    a["spoken_text"], b["spoken_text"] = "nine eight seven", "priya anand"
    items = cluster_spans([a, b])
    case("low-overlap different-value spans stay separate",
         len(items) == 2, str(items))

    # 3b. same value from two engines with a cross-engine timestamp offset
    #     (no temporal overlap, < 2.5 s apart) -> ONE item, not a duplicate
    c = _tspan(10420, 10980, "elevenlabs")
    d = _tspan(11000, 12000, "gemini")
    c["spoken_text"] = d["spoken_text"] = "noor jahan"
    items = cluster_spans([c, d])
    case("same-value nearby cross-engine spans merge",
         len(items) == 1 and set(items[0]["engines"]) ==
         {"elevenlabs", "gemini"}, str(items))

    # 3c. cross-script same-name merge: Indic fold pushes difflib similarity
    #     over the threshold ("ಸುಧಾ" ~ "Sudha", "ఆయుష్" ~ "Ayush")
    import difflib
    r_kn = difflib.SequenceMatcher(None, fold_indic("ಸುಧಾ"), "sudha").ratio()
    r_te = difflib.SequenceMatcher(None, fold_indic("ఆయుష్"), "ayush").ratio()
    e1 = _tspan(10400, 10900, "elevenlabs", cats=("FULL_NAME",))
    e2 = _tspan(11000, 11800, "gemini", cats=("FULL_NAME",))
    e1["spoken_text"], e2["spoken_text"] = "ಸುಧಾ", "Sudha"
    items = cluster_spans([e1, e2])
    case("indic fold: Kannada/Latin same-name spans merge",
         r_kn >= 0.55 and r_te >= 0.55 and len(items) == 1
         and set(items[0]["engines"]) == {"elevenlabs", "gemini"},
         "r_kn=%.2f r_te=%.2f fold_kn=%r fold_te=%r items=%s"
         % (r_kn, r_te, fold_indic("ಸುಧಾ"), fold_indic("ఆయుష్"), items))

    # 3d. name-slot guard: cross-engine respelling of ONE name occurrence
    #     (dissimilar values, barely-touching spans — the Vidya/Ritika case)
    #     merges into one item
    f1 = _tspan(13000, 13400, "gemini", cats=("FULL_NAME",), text="Vidya")
    f2 = _tspan(13388, 13763, "el_scribe_v1", cats=("FULL_NAME",),
                text="Ritika")
    items = cluster_spans([f1, f2])
    case("name-slot guard: cross-engine respelling merges",
         len(items) == 1 and set(items[0]["engines"]) ==
         {"el_scribe_v1", "gemini"}, str(items))

    # 3e. name-slot ambiguity discriminator: an engine hearing TWO names in
    #     the window marks genuinely distinct names — no forced merge
    g1 = _tspan(13000, 13300, "gemini", cats=("FULL_NAME",), text="Ramesh")
    g2 = _tspan(13350, 13700, "gemini", cats=("FULL_NAME",), text="Vidya")
    g3 = _tspan(13720, 14100, "el_scribe_v1", cats=("FULL_NAME",),
                text="Ritika")
    items = cluster_spans([g1, g2, g3])
    case("name-slot guard: two-name engine disables the merge",
         len(items) == 3, str(items))

    # 3f. name-slot guard leaves non-name categories alone: adjacent
    #     dissimilar PHONE spans stay separate
    h1 = _tspan(13000, 13400, "gemini", text="nine eight")
    h2 = _tspan(13450, 13800, "el_scribe_v1", text="four two")
    items = cluster_spans([h1, h2])
    case("name-slot guard: non-name categories unaffected",
         len(items) == 2, str(items))

    # 4. leak coverage math incl. channel semantics (channel None covers all;
    #    wrong channel leaks; 90% coverage still leaks)
    def item(ch=None, cats=("PHONE",)):
        return {"start_ms": 1000, "end_ms": 2000, "categories": list(cats),
                "engines": ["gemini"], "public_figure": False,
                "confidence": "high", "spoken_text": "x", "channel": ch}
    c_none, _, _ = score_engine_for_items(
        [item(ch=0)], [{"start_ms": 900, "end_ms": 2100, "channel": None}],
        None, "medical_consult")
    c_wrong, _, _ = score_engine_for_items(
        [item(ch=0)], [{"start_ms": 900, "end_ms": 2100, "channel": 1}],
        None, "medical_consult")
    c_part, li, _ = score_engine_for_items(
        [item()], [{"start_ms": 1000, "end_ms": 1900, "channel": None}],
        None, "medical_consult")
    case("leak coverage: channel None covers, wrong channel leaks, 90% leaks",
         c_none["leaked"] == 0 and c_none["covered"] == 1
         and c_wrong["leaked"] == 1 and c_part["leaked"] == 1 and li == [0],
         "none=%s wrong=%s part=%s" % (c_none, c_wrong, c_part))

    # 5. action_for medical_consult: HEALTH-only item excluded from the leak
    #    denominator; HEALTH+PHONE stays must-beep; strict profile beeps both
    two = [item(cats=("HEALTH_MEDICAL_INFO",)),
           item(cats=("HEALTH_MEDICAL_INFO", "PHONE"))]
    c_med, _, _ = score_engine_for_items(two, [], None, "medical_consult")
    c_str, _, _ = score_engine_for_items(two, [], None, "strict")
    case("medical_consult excludes HEALTH-only from leak denominator",
         c_med["must_beep"] == 1 and c_med["annotate_only"] == 1
         and c_med["leaked"] == 1 and c_str["must_beep"] == 2
         and c_str["leaked"] == 2, "med=%s strict=%s" % (c_med, c_str))

    # 6. agreement rule + intervals None skips the denominator
    it6 = [item(), dict(item(), start_ms=8000, end_ms=9000)]
    c6, _, ag = score_engine_for_items(
        it6, [], [{"start_ms": 950, "end_ms": 2050}], "medical_consult")
    c6n, _, _ = score_engine_for_items(it6, [], None, "medical_consult")
    case("agreement overlap rule; missing intervals.json skipped",
         c6["agree"] == 1 and c6["agree_eval"] == 2 and ag == [0]
         and c6n["agree_eval"] == 0, str(c6))

    # 7. span trim guard: wide judge range vs one-word spoken_text
    words = detect_pii.flatten_words(_mk_transcript(
        [("w%d" % k, k * 500, k * 500 + 400) for k in range(25)]
        + [("Vidya", 12500, 12900)]
        + [("t%d" % k, 13000 + k * 500, 13400 + k * 500) for k in range(5)]))
    got = anchor_span(words, {"start_word": 0, "end_word": 30,
                              "categories": ["FULL_NAME"],
                              "spoken_text": "Vidya", "confidence": "high"},
                      "gemini")
    case("span trim guard anchors to the quoted word, confidence -> low",
         got is not None and got["start_ms"] == 12500
         and got["end_ms"] == 12900 and got["trimmed"]
         and got["confidence"] == "low", str(got))

    # 8. invalid judge indices are dropped
    case("invalid indices dropped",
         anchor_span(words, {"start_word": 5, "end_word": 99,
                             "categories": ["PHONE"], "spoken_text": "x",
                             "confidence": "high"}, "g") is None
         and anchor_span(words, {"start_word": -1, "end_word": 3,
                                 "categories": ["PHONE"], "spoken_text": "x",
                                 "confidence": "high"}, "g") is None)

    # 9. chunking: spans found in a later chunk keep global indices
    counter = {"calls": 0}
    old_call, old_chunk = judge_client.judge_call, JUDGE_CHUNK
    judge_client.judge_call = _fake_judge_factory(counter)
    JUDGE_CHUNK = 5
    try:
        w9 = detect_pii.flatten_words(_mk_transcript(
            [("f%d" % k, k * 500, k * 500 + 400) for k in range(6)]
            + [("Ramesh", 3000, 3400), ("Kumar", 3500, 3900)]
            + [("f%d" % k, 4000 + k * 500, 4400 + k * 500) for k in range(4)]))
        s9 = judge_detect(w9, "fake")
        case("chunked judge pass keeps global word indices",
             counter["calls"] == 3 and len(s9) == 1
             and s9[0]["start_word"] == 6 and s9[0]["end_word"] == 7, str(s9))
    finally:
        JUDGE_CHUNK = old_chunk
        judge_client.judge_call = old_call

    # 10. end-to-end with a monkeypatched judge across 3 synthetic engines
    counter = {"calls": 0}
    judge_client.judge_call = _fake_judge_factory(counter)
    try:
        tmp = Path(tempfile.mkdtemp(prefix="pseudo_gt_selftest_"))
        pred, out = tmp / "output", tmp / "judge_out"
        fid = "synthfile"

        def digits_at(t0):
            return [(DIGITS[k], t0 + k * 500, t0 + k * 500 + 400)
                    for k in range(10)]

        fillers = [("hello%d" % k, k * 700, k * 700 + 300) for k in range(8)]
        gem = fillers + [("Vidya", 13000, 13400)] + digits_at(20000) + \
            [("diabetes", 45000, 45400)]
        elv = fillers + digits_at(20100) + \
            [("Ramesh", 30000, 30400), ("Kumar", 30500, 30900),
             ("diabetes", 45050, 45450)]
        opn = fillers + digits_at(19900) + [("diabetes", 44950, 45350)]
        docs = {
            "gemini": (gem,
                       [{"start_ms": 12800, "end_ms": 13600, "channel": None},
                        {"start_ms": 19800, "end_ms": 25100, "channel": None}],
                       [{"start_ms": 13000, "end_ms": 13400},
                        {"start_ms": 20000, "end_ms": 24900},
                        {"start_ms": 45000, "end_ms": 45400}]),
            "elevenlabs": (elv,
                           [{"start_ms": 19800, "end_ms": 25100,
                             "channel": None},
                            {"start_ms": 29800, "end_ms": 31100,
                             "channel": None}],
                           [{"start_ms": 20100, "end_ms": 25000}]),
            "openai": (opn,
                       [{"start_ms": 19900, "end_ms": 22000, "channel": None}],
                       []),
        }
        for eng, (wtrip, rend, ints) in docs.items():
            d = engine_dir(pred, fid, eng)
            d.mkdir(parents=True)
            (d / "transcript.json").write_text(
                json.dumps(_mk_transcript(wtrip)))
            (d / "rendered_intervals.json").write_text(
                json.dumps({"rendered_merged": rend}))
            (d / "intervals.json").write_text(
                json.dumps({"file_id": fid, "intervals": ints}))
        engines = ["gemini", "elevenlabs", "openai"]
        res = process_file(fid, engines, pred, out, "fake", "medical_consult",
                           False)
        by_id = {tuple(it["engines"]): it for it in res["items"]}
        g, e, o = (res["engines"][x] for x in engines)
        case("end-to-end: 4 items (phone x3 engines, vidya gem-only, "
             "ramesh elv-only, diabetes ann-only)",
             len(res["items"]) == 4
             and ("elevenlabs", "gemini", "openai") in by_id
             and ("gemini",) in by_id and ("elevenlabs",) in by_id
             and sum(1 for it in res["items"]
                     if it["action"] == "beep") == 3, str(res["items"]))
        case("end-to-end leaks: gemini leaks ramesh; elevenlabs leaks vidya; "
             "openai leaks all 3 must-beep",
             g["leaked"] == 1 and g["covered"] == 2
             and e["leaked"] == 1 and e["covered"] == 2
             and o["leaked"] == 3 and o["covered"] == 0
             and by_id[("elevenlabs",)]["leaked_by"] == ["gemini", "openai"],
             "g=%s e=%s o=%s" % (g, e, o))
        case("end-to-end agreement: gemini 3/4, elevenlabs 1/4, openai 0/4",
             (g["agree"], g["agree_eval"]) == (3, 4)
             and (e["agree"], e["agree_eval"]) == (1, 4)
             and (o["agree"], o["agree_eval"]) == (0, 4),
             "g=%s e=%s o=%s" % (g, e, o))
        # trim guard fired inside the pipeline for the wide vidya span
        vid = by_id[("gemini",)]
        case("end-to-end trim: vidya anchored to its own word",
             vid["start_ms"] == 13000 and vid["end_ms"] == 13400
             and vid["confidence"] == "low", str(vid))
        # caching: re-run makes no further judge calls; --force redoes them
        n0 = counter["calls"]
        case("first run: one judge call per engine", n0 == 3, str(n0))
        process_file(fid, engines, pred, out, "fake", "medical_consult", False)
        n1 = counter["calls"]
        process_file(fid, engines, pred, out, "fake", "medical_consult", True)
        n2 = counter["calls"]
        case("caching: cached re-run adds 0 calls, --force adds 3",
             n1 == n0 and n2 == n0 + 3, "n0=%d n1=%d n2=%d" % (n0, n1, n2))
        # aggregate + reports
        agg = aggregate([res], engines)
        case("aggregate: surfaced-by 1->2 items, 3->2 items; uniques counted",
             agg["surfaced_by"][1]["items"] == 2
             and agg["surfaced_by"][3]["items"] == 2
             and agg["unique_by_engine"]["gemini"]["items"] == 1
             and agg["unique_by_engine"]["elevenlabs"]["items"] == 1
             and agg["per_engine"]["openai"]["pseudo_leak_rate"] == 1.0
             and agg["per_engine"]["gemini"]["agreement"] == 0.75,
             str(agg))
        cfg = {"generated_at": "test", "judge_model": "fake",
               "profile": "medical_consult", "engines": engines,
               "pred_root": str(pred), "out": str(out)}
        md, js = write_reports(out, cfg, [res], agg, engines)
        txt = md.read_text()
        case("reports written with caveat block",
             md.exists() and js.exists()
             and "CAVEAT" in txt and "UNION" in txt
             and "measured against this reference" in txt
             and json.loads(js.read_text())["aggregate"]["totals"]["items"] == 4)
    finally:
        judge_client.judge_call = old_call

    # 11. adversarial verification pass: planted amount-less FINANCIAL_INFO
    #     span rejected before clustering, valued one kept; ONE batched
    #     verify call per engine-file; cache schema handling
    counter = {"calls": 0, "verify": 0}
    judge_client.judge_call = _fake_judge_factory(counter)
    try:
        tmp = Path(tempfile.mkdtemp(prefix="pseudo_gt_verify_selftest_"))
        pred, out = tmp / "output", tmp / "judge_out"
        fid = "verifile"

        def digits_at(t0):
            return [(DIGITS[k], t0 + k * 500, t0 + k * 500 + 400)
                    for k in range(10)]

        wtrip = ([("hello%d" % k, k * 700, k * 700 + 300) for k in range(6)]
                 + digits_at(10000)
                 + [("paid", 30000, 30300), ("the", 30400, 30600),
                    ("bill", 30700, 31000)]
                 + [("thirty", 40000, 40300), ("thousand", 40400, 40800)])
        engines2 = ["gemini", "elevenlabs"]
        for eng in engines2:
            d = engine_dir(pred, fid, eng)
            d.mkdir(parents=True)
            (d / "transcript.json").write_text(
                json.dumps(_mk_transcript(wtrip)))
        res = process_file(fid, engines2, pred, out, "fake",
                           "medical_consult", False, verify=True,
                           verify_model="fake-verifier")
        texts = sorted(it["spoken_text"] for it in res["items"])
        case("verify: amount-less FINANCIAL_INFO span rejected before "
             "clustering, valued one kept",
             len(res["items"]) == 2
             and "paid the bill" not in texts
             and "thirty thousand" in texts
             and res["verifier_rejected"] == {"gemini": 1, "elevenlabs": 1}
             and any("rejected by verifier" in n for n in res["notes"]),
             str((texts, res["verifier_rejected"], res["notes"])))
        case("batching: one detect + one verify call per engine-file",
             counter["calls"] == 2 and counter["verify"] == 2, str(counter))
        vp = counter["verify_prompts"][-1]
        case("verify prompt: numbered findings + categories + word context",
             '2. text: "paid the bill"' in vp
             and "categories: FINANCIAL_INFO" in vp
             and "must be REJECTED" in vp
             and "hello0" in vp, vp[:400])
        cache = out / ("%s__gemini.spans.json" % fid)
        cdoc = json.loads(cache.read_text())
        case("span cache: schema field + pre/post-verify span lists",
             cdoc.get("schema") == CACHE_SCHEMA
             and len(cdoc["spans_pre_verify"]) == 3
             and len(cdoc["spans"]) == 2
             and cdoc["verification"]["rejected"] == 1
             and cdoc["verification"]["model"] == "fake-verifier",
             str(cdoc.get("verification")))
        # cached --verify re-run adds no calls
        process_file(fid, engines2, pred, out, "fake", "medical_consult",
                     False, verify=True, verify_model="fake-verifier")
        case("verified span cache reused by --verify run",
             counter["calls"] == 2 and counter["verify"] == 2, str(counter))
        # --no-verify view of a verified cache = pre-verify spans, no calls
        words11 = detect_pii.flatten_words(_mk_transcript(wtrip))
        spans_nv, ver_nv = judge_spans_cached(fid, "gemini", words11, out,
                                              "fake", False, verify=False)
        case("--no-verify view of verified span cache = pre-verify spans",
             len(spans_nv) == 3 and not ver_nv["enabled"]
             and counter["calls"] == 2 and counter["verify"] == 2,
             str((len(spans_nv), ver_nv)))
        # cache staleness on schema bump: strip schema/verification from the
        # gemini entry (old-style cache) — stale for --verify, fresh for
        # --no-verify
        cdoc.pop("schema", None)
        cdoc.pop("verification", None)
        cdoc.pop("spans_pre_verify", None)
        cache.write_text(json.dumps(cdoc))
        spans_old, _ = judge_spans_cached(fid, "gemini", words11, out,
                                          "fake", False, verify=False)
        case("pre-schema span cache still fresh for --no-verify runs",
             counter["calls"] == 2 and len(spans_old) == 2)
        res2 = process_file(fid, engines2, pred, out, "fake",
                            "medical_consult", False, verify=True,
                            verify_model="fake-verifier")
        case("cache staleness on schema bump: --verify re-judges old entry "
             "only",
             counter["calls"] == 3 and counter["verify"] == 3
             and res2["verifier_rejected"] == {"gemini": 1, "elevenlabs": 1},
             str(counter))
        # aggregation + report carry the per-engine rejection count
        agg2 = aggregate([res2], engines2)
        case("aggregate: verifier_rejected per engine",
             agg2["per_engine"]["gemini"]["verifier_rejected"] == 1
             and agg2["per_engine"]["elevenlabs"]["verifier_rejected"] == 1,
             str(agg2["per_engine"]))
        cfg2 = {"generated_at": "test", "judge_model": "fake",
                "profile": "medical_consult", "engines": engines2,
                "pred_root": str(pred), "out": str(out),
                "verify": True, "verify_model": "fake-verifier"}
        md2 = build_report(cfg2, [res2], agg2, engines2)
        case("report: rejection count per engine + verifier named",
             "rejected by verifier" in md2
             and "verifier `fake-verifier`" in md2
             and "| gemini |" in md2, md2[:300])
    finally:
        judge_client.judge_call = old_call

    failed = [n for n, ok in cases if not ok]
    print("\nself-test: %d/%d passed" % (len(cases) - len(failed), len(cases)))
    return 1 if failed else 0


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engines", default="gemini,elevenlabs,openai",
                    help="comma list of engines to union + score")
    ap.add_argument("--files", default=None,
                    help="comma list of file_ids (default: every base dir "
                         "under --pred-root with a transcript.json)")
    ap.add_argument("--judge-model", default=judge_client.DEFAULT_JUDGE,
                    help="OpenRouter judge model id")
    ap.add_argument("--verify", dest="verify", action="store_true",
                    default=True,
                    help="adversarial verification pass over time-anchored "
                         "judge spans (default ON)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the verification pass")
    ap.add_argument("--verify-model", default=judge_client.DEFAULT_VERIFIER,
                    help="OpenRouter verifier model (default %s)"
                         % judge_client.DEFAULT_VERIFIER)
    ap.add_argument("--out", default="output/judge/pseudo_gt",
                    help="cache + report directory")
    ap.add_argument("--pred-root", default="output",
                    help="directory holding <file_id><engine-suffix>/ dirs")
    ap.add_argument("--profile", default=None,
                    choices=sorted(policy.ANNOTATE_ONLY),
                    help="Action Matrix profile defining the must-beep set "
                         "(default: read per file from the gemini output's "
                         "rendered_intervals.json; pass explicitly for "
                         "counterfactual scoring)")
    ap.add_argument("--force", action="store_true",
                    help="re-judge even when a span cache exists")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic cases with a mocked judge and exit")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    sys.exit(run(args))


if __name__ == "__main__":
    main()
