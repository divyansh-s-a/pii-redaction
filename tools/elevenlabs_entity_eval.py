#!/usr/bin/env python3
"""Vendor-native detection comparison — ElevenLabs scribe_v2's built-in
entity detection vs our Gemini policy detector, on the SAME ElevenLabs
transcripts.

Per file the tool
  1. POSTs the ElevenLabs speech-to-text endpoint exactly like
     pipeline/adapters/elevenlabs_transcribe.py does, plus
     entity_detection=["pii"], and caches the full response at
     <out>/<fid>.entities.json (--score-only never fetches, --force
     refetches). Usage is logged via adapters.log_usage (engine
     "elevenlabs_entity").
  2. Maps each vendor entity's CHAR offsets (into the response's top-level
     "text") to a time interval: word char positions are reconstructed by
     walking words[] in order (str.find at-or-after the previous end —
     whitespace between words varies), then the entity's covered words give
     {start_ms, end_ms}. Entities whose chars hit no mapped word are counted
     as unmapped and reported.
  3. Maps the vendor's NER-style entity_type into our Taxonomy v0.2
     categories (pipeline/policy.py). Pure NER noise (cardinal, ordinal,
     time, ...) is dropped but counted per type; unknown types are counted
     and listed so the vendor vocabulary stays visible.
  4. Scores the mapped entities with the pseudo-GT overlap rule
     (tools/judge_pseudo_gt.py: temporal IoU >= 0.3 OR one interval covering
     >= 70% of the other) against BOTH references:
       a. the judge pseudo-GT union (output/judge/pseudo_gt/
          pseudo_gt_report.json) — coverage of union items (recall-like) and
          share of mapped entities matching a union item (precision-like);
       b. our detector on the same transcript
          (output/<fid>__elevenlabs/intervals.json) — found by both / only
          ours / only theirs, with a per-category breakdown.
  5. Writes <out>/el_entity_report.md + .json with per-file and aggregate
     numbers, the dropped-noise histogram, the unknown-type list and a
     caveats block.

Usage:
  python3 tools/elevenlabs_entity_eval.py [--files <fid>,...]
      [--out output/el_entity] [--force] [--score-only]
  python3 tools/elevenlabs_entity_eval.py --self-test   # no network
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import adapters
import common
import policy

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL_ID = "scribe_v2"
ENTITY_DETECTION = '["pii"]'
USAGE_ENGINE = "elevenlabs_entity"
SKIP_TYPES = {"spacing", "audio_event"}       # as in the transcribe adapter
MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
        "mp4": "video/mp4"}
EL_SUFFIX = "__elevenlabs"
PSEUDO_GT = ROOT / "output" / "judge" / "pseudo_gt" / "pseudo_gt_report.json"

# Overlap rule copied from tools/judge_pseudo_gt.py (CLUSTER_IOU /
# CLUSTER_COVER): temporal IoU >= 0.3, or one interval covers >= 70% of the
# other.
IOU_MIN = 0.30
COVER_MIN = 0.70

# vendor entity_type -> our Taxonomy v0.2 category (policy.CATEGORIES).
ENTITY_MAP = {
    "name": "FULL_NAME", "name_given": "FULL_NAME",
    "name_family": "FULL_NAME", "person": "FULL_NAME",
    "phone_number": "PHONE", "phone": "PHONE",
    "email": "EMAIL", "email_address": "EMAIL",
    # over-broad: any location becomes ADDRESS (flagged in report notes)
    "location": "ADDRESS", "address": "ADDRESS", "gpe": "ADDRESS",
    "facility": "ADDRESS",
    "organization": "ORGANIZATION_PERSON_LINKED",
    "org": "ORGANIZATION_PERSON_LINKED",
    "company": "ORGANIZATION_PERSON_LINKED",
    "money": "FINANCIAL_INFO", "salary": "FINANCIAL_INFO",
    "date_of_birth": "DATE_OF_BIRTH", "dob": "DATE_OF_BIRTH",
    "age": "AGE",
    "ssn": "GOVERNMENT_ID", "national_id": "GOVERNMENT_ID",
    "id_number": "GOVERNMENT_ID", "government_id": "GOVERNMENT_ID",
    "credit_card": "PAYMENT_CARD_NUMBER",
    "card_number": "PAYMENT_CARD_NUMBER",
    "bank_account": "BANK_ACCOUNT_NUMBER", "iban": "BANK_ACCOUNT_NUMBER",
    "ip": "IP_ADDRESS", "ip_address": "IP_ADDRESS",
    "url": "USERNAME_HANDLE", "username": "USERNAME_HANDLE",
    "handle": "USERNAME_HANDLE",
    "passport": "PASSPORT_NUMBER",
    "drivers_license": "DRIVERS_LICENSE", "driver_license": "DRIVERS_LICENSE",
    "medical": "HEALTH_MEDICAL_INFO", "health": "HEALTH_MEDICAL_INFO",
    "condition": "HEALTH_MEDICAL_INFO", "drug": "HEALTH_MEDICAL_INFO",
    "nationality": "NATIONALITY", "norp": "NATIONALITY",
}
# generic NER noise the vendor emits that our policy never redacts: dropped,
# but counted per type so the report shows what was thrown away.
NOISE_TYPES = {"cardinal", "ordinal", "quantity", "number", "percent",
               "time", "duration", "event", "product", "work_of_art",
               "language"}
OVERBROAD_TYPES = {"location", "address", "gpe", "facility"}

CAVEATS = [
    "The vendor taxonomy is a FIXED NER-style vocabulary — it cannot express "
    "our policy: no fiction rule, no person-link requirement for "
    "organizations, no Action Matrix profiles, no masked-value exclusion. "
    "Every location becomes ADDRESS and every money mention becomes "
    "FINANCIAL_INFO, so over-flagging here is expected by construction.",
    "Vendor entities are CHAR-offset based; timing is derived by OUR word "
    "mapping (str.find walk of words[] over the response text), not by the "
    "vendor. A mapping miss drops the entity (counted as unmapped).",
    "The vendor's detection runs only on its OWN transcript: PII scribe_v2 "
    "never wrote down is invisible to it, and both references here (judge "
    "pseudo-GT union, our detector's intervals) come from earlier "
    "transcription runs whose word timestamps can differ slightly from the "
    "entity fetch's — small timing slop can cost borderline matches.",
    "Pseudo-GT is itself an LLM-judge union (see tools/judge_pseudo_gt.py): "
    "coverage/precision against it are agreement numbers, not ground truth.",
]


# ------------------------------------------------------------------- fetch

def fetch_entities(media, audio_seconds):
    """POST the STT endpoint with entity detection on; return the parsed
    full response. Mirrors the transcribe adapter's retry + usage logging."""
    mime = MIME.get(media.suffix.lower().lstrip("."), "audio/mpeg")
    body, ctype = adapters.multipart_body(
        {"model_id": MODEL_ID, "diarize": "true",
         "timestamps_granularity": "word", "tag_audio_events": "false",
         "entity_detection": ENTITY_DETECTION},
        "file", media.name, media.read_bytes(), mime)
    headers = {"xi-api-key": adapters.env_key("ELEVENLABS_API_KEY"),
               "Content-Type": ctype}
    for attempt in range(4):
        status, _, raw = common._request(ENDPOINT, data=body, headers=headers,
                                         timeout=1200)
        adapters.log_usage(USAGE_ENGINE, ENDPOINT, media, audio_seconds,
                           status)
        if status == 200:
            try:
                return json.loads(raw)
            except ValueError:
                raise RuntimeError(
                    "elevenlabs returned non-JSON body: %r" % raw[:400])
        if status in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print("  HTTP %d, retrying in %ds ..." % (status, wait))
            time.sleep(wait)
            continue
        raise RuntimeError("elevenlabs entity fetch HTTP %d: %r"
                           % (status, raw[:400]))
    raise RuntimeError("elevenlabs entity fetch: retries exhausted")


# -------------------------------------------------------- char -> time map

def build_char_map(text, words):
    """Reconstruct each word's char position in `text` by walking words[] in
    order and finding each word's text at-or-after the previous end position
    (whitespace between words varies). Returns (char_words, misses) where
    char_words = [{cs, ce, start_ms, end_ms, text}] and misses counts words
    whose text could not be found (they are skipped; the walk continues)."""
    pos, out, misses = 0, [], 0
    for w in words:
        if not isinstance(w, dict) or w.get("type") in SKIP_TYPES:
            continue
        t = str(w.get("text", ""))
        if not t.strip():
            continue
        i = text.find(t, pos)
        if i < 0:
            t2 = t.strip()
            i = text.find(t2, pos) if t2 else -1
            if i < 0:
                misses += 1
                continue
            t = t2
        try:
            s_ms = int(round(float(w["start"]) * 1000))
            e_ms = int(round(float(w["end"]) * 1000))
        except (KeyError, TypeError, ValueError):
            misses += 1
            continue
        out.append({"cs": i, "ce": i + len(t), "start_ms": s_ms,
                    "end_ms": e_ms, "text": t})
        pos = i + len(t)
    return out, misses


def entity_interval(ent, char_words):
    """[start_char, end_char) -> {start_ms, end_ms} from the covered words,
    or None when no mapped word overlaps the char range. Zero-width word
    timings are widened to 1 ms so the overlap rule can still fire."""
    try:
        cs, ce = int(ent["start_char"]), int(ent["end_char"])
    except (KeyError, TypeError, ValueError):
        return None
    cov = [w for w in char_words if w["cs"] < ce and w["ce"] > cs]
    if not cov:
        return None
    s = min(w["start_ms"] for w in cov)
    e = max(w["end_ms"] for w in cov)
    return {"start_ms": s, "end_ms": max(e, s + 1)}


# ------------------------------------------------------- category mapping

def classify(entity_type):
    """vendor entity_type -> (category|None, kind) with kind in
    {"mapped", "noise", "unknown"}."""
    et = str(entity_type or "").strip().lower()
    if et in ENTITY_MAP:
        return ENTITY_MAP[et], "mapped"
    if et in NOISE_TYPES:
        return None, "noise"
    return None, "unknown"


# ---------------------------------------------------------- overlap scoring

def overlap_pair(a, b):
    """Copied cluster rule from tools/judge_pseudo_gt.py: temporal
    IoU >= IOU_MIN, or one interval covers >= COVER_MIN of the other."""
    inter = min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"])
    if inter <= 0:
        return False
    la = a["end_ms"] - a["start_ms"]
    lb = b["end_ms"] - b["start_ms"]
    union = la + lb - inter
    iou = inter / float(union) if union else 1.0
    cover = inter / float(min(la, lb)) if min(la, lb) else 1.0
    return iou >= IOU_MIN or cover >= COVER_MIN


def score_vs_union(mapped, union_items):
    """Coverage of pseudo-GT union items (recall-like) + share of mapped EL
    entities matching a union item (precision-like). union_items may be None
    (pseudo-GT missing for the file)."""
    if union_items is None:
        return {"available": False, "union_items": 0, "union_matched": 0,
                "must_beep": 0, "must_beep_matched": 0,
                "el_mapped": len(mapped), "el_matched": 0}
    res = {"available": True, "union_items": len(union_items),
           "union_matched": 0, "must_beep": 0, "must_beep_matched": 0,
           "el_mapped": len(mapped), "el_matched": 0}
    for it in union_items:
        hit = any(overlap_pair(it, m) for m in mapped)
        res["union_matched"] += 1 if hit else 0
        if it.get("action") == "beep":
            res["must_beep"] += 1
            res["must_beep_matched"] += 1 if hit else 0
    for m in mapped:
        m["matched_union"] = any(overlap_pair(m, it) for it in union_items)
        res["el_matched"] += 1 if m["matched_union"] else 0
    return res


def score_vs_ours(mapped, ours):
    """Head-to-head vs our detector's intervals.json on the same transcript:
    both / only-ours / only-theirs + per-category breakdown. ours may be None
    (intervals.json missing)."""
    cats = {}

    def cat(c):
        return cats.setdefault(c, {"ours": 0, "ours_matched": 0,
                                   "el": 0, "el_matched": 0})

    if ours is None:
        for m in mapped:
            c = cat(m["category"])
            c["el"] += 1
        return {"available": False, "ours_total": 0, "both": 0,
                "only_ours": 0, "only_theirs": 0, "el_total": len(mapped),
                "categories": cats}
    both = 0
    for o in ours:
        hit = any(overlap_pair(o, m) for m in mapped)
        both += 1 if hit else 0
        for c in o.get("categories", []):
            cat(c)["ours"] += 1
            cat(c)["ours_matched"] += 1 if hit else 0
    only_theirs = 0
    for m in mapped:
        m["matched_ours"] = any(overlap_pair(m, o) for o in ours)
        only_theirs += 0 if m["matched_ours"] else 1
        c = cat(m["category"])
        c["el"] += 1
        c["el_matched"] += 1 if m["matched_ours"] else 0
    return {"available": True, "ours_total": len(ours), "both": both,
            "only_ours": len(ours) - both, "only_theirs": only_theirs,
            "el_total": len(mapped), "categories": cats}


# --------------------------------------------------------------- per file

def eval_file(fid, pred_root, out_dir, union_items, score_only=False,
              force=False):
    """Fetch (or read cached) entities for one file, map and score them.
    Returns the per-file result dict."""
    notes = []
    edir = Path(pred_root) / (fid + EL_SUFFIX)
    cache = Path(out_dir) / (fid + ".entities.json")
    resp = None
    if cache.exists() and not force:
        resp = json.loads(cache.read_text())
    elif score_only:
        notes.append("no cached entities (--score-only never fetches)")
    else:
        prep_p = edir / "prepare.json"
        if not prep_p.exists():
            notes.append("prepare.json missing — cannot locate media, "
                         "fetch skipped")
        else:
            prep = json.loads(prep_p.read_text())
            media = Path(prep["source"])
            if not media.exists():
                notes.append("media %s missing — fetch skipped" % media)
            else:
                secs = float(prep.get("duration_ms", 0)) / 1000.0
                print("  fetching entities (%s) ..." % media.name)
                resp = fetch_entities(media, secs)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(resp, indent=1,
                                            ensure_ascii=False))
    res = {"file_id": fid, "fetched": resp is not None,
           "entities_total": 0, "mapped": 0, "dropped_noise": 0,
           "dropped_unknown": 0, "unmapped": 0, "word_map_misses": 0,
           "overbroad_address": 0, "noise_hist": {}, "unknown_hist": {},
           "mapped_entities": [], "unmapped_entities": [],
           "vs_union": None, "vs_ours": None, "notes": notes}
    if resp is None:
        res["vs_union"] = score_vs_union([], union_items)
        res["vs_ours"] = score_vs_ours([], None)
        return res
    text = str(resp.get("text") or "")
    char_words, misses = build_char_map(text, resp.get("words") or [])
    res["word_map_misses"] = misses
    if misses:
        notes.append("%d word(s) failed char mapping" % misses)
    ents = resp.get("entities")
    if ents is None:
        notes.append("response has no entities field")
        ents = []
    res["entities_total"] = len(ents)
    mapped = []
    for ent in ents:
        etype = str(ent.get("entity_type") or "").strip().lower()
        category, kind = classify(etype)
        if kind == "noise":
            res["dropped_noise"] += 1
            res["noise_hist"][etype] = res["noise_hist"].get(etype, 0) + 1
            continue
        if kind == "unknown":
            res["dropped_unknown"] += 1
            res["unknown_hist"][etype] = res["unknown_hist"].get(etype, 0) + 1
            continue
        iv = entity_interval(ent, char_words)
        if iv is None:
            res["unmapped"] += 1
            res["unmapped_entities"].append(
                {"entity_type": etype, "text": str(ent.get("text", ""))[:80],
                 "start_char": ent.get("start_char"),
                 "end_char": ent.get("end_char")})
            continue
        if etype in OVERBROAD_TYPES:
            res["overbroad_address"] += 1
        mapped.append({"start_ms": iv["start_ms"], "end_ms": iv["end_ms"],
                       "category": category, "entity_type": etype,
                       "text": str(ent.get("text", ""))[:80],
                       "start_char": ent.get("start_char"),
                       "end_char": ent.get("end_char")})
    res["mapped"] = len(mapped)
    if res["overbroad_address"]:
        notes.append("%d location-family entit(ies) mapped to ADDRESS "
                     "(over-broad)" % res["overbroad_address"])
    if union_items is None:
        notes.append("file absent from pseudo-GT report (union scoring "
                     "skipped)")
    res["vs_union"] = score_vs_union(mapped, union_items)
    ints_p = edir / "intervals.json"
    ours = None
    if ints_p.exists():
        ours = json.loads(ints_p.read_text()).get("intervals", [])
    else:
        notes.append("intervals.json missing (head-to-head skipped)")
    res["vs_ours"] = score_vs_ours(mapped, ours)
    res["mapped_entities"] = mapped
    return res


# --------------------------------------------------------------- aggregate

def aggregate(files):
    agg = {"files": len(files), "entities_total": 0, "mapped": 0,
           "dropped_noise": 0, "dropped_unknown": 0, "unmapped": 0,
           "word_map_misses": 0, "overbroad_address": 0,
           "noise_hist": {}, "unknown_hist": {},
           "vs_union": {"union_items": 0, "union_matched": 0,
                        "must_beep": 0, "must_beep_matched": 0,
                        "el_mapped": 0, "el_matched": 0},
           "vs_ours": {"ours_total": 0, "both": 0, "only_ours": 0,
                       "only_theirs": 0, "el_total": 0},
           "categories": {}}
    for f in files:
        for k in ("entities_total", "mapped", "dropped_noise",
                  "dropped_unknown", "unmapped", "word_map_misses",
                  "overbroad_address"):
            agg[k] += f[k]
        for h in ("noise_hist", "unknown_hist"):
            for t, n in f[h].items():
                agg[h][t] = agg[h].get(t, 0) + n
        vu = f["vs_union"]
        if vu["available"]:
            for k in ("union_items", "union_matched", "must_beep",
                      "must_beep_matched", "el_mapped", "el_matched"):
                agg["vs_union"][k] += vu[k]
        vo = f["vs_ours"]
        if vo["available"]:
            for k in ("ours_total", "both", "only_ours", "only_theirs",
                      "el_total"):
                agg["vs_ours"][k] += vo[k]
            for c, d in vo["categories"].items():
                a = agg["categories"].setdefault(
                    c, {"ours": 0, "ours_matched": 0, "el": 0,
                        "el_matched": 0})
                for k in d:
                    a[k] += d[k]
    u, o = agg["vs_union"], agg["vs_ours"]
    agg["union_coverage"] = _ratio(u["union_matched"], u["union_items"])
    agg["union_must_beep_coverage"] = _ratio(u["must_beep_matched"],
                                             u["must_beep"])
    agg["union_precision"] = _ratio(u["el_matched"], u["el_mapped"])
    agg["ours_agreement"] = _ratio(o["both"], o["ours_total"])
    return agg


def _ratio(num, den):
    return (num / float(den)) if den else None


def _pct(num, den):
    if not den:
        return "n/a"
    return "%d/%d = %.1f%%" % (num, den, 100.0 * num / den)


# ---------------------------------------------------------------- reporting

def build_report(cfg, files, agg):
    L = ["# ElevenLabs native entity detection vs our policy detector",
         "",
         "Generated %s by `tools/elevenlabs_entity_eval.py` | model `%s`, "
         "`entity_detection=%s` | %d file(s) | predictions: `%s` | "
         "pseudo-GT: `%s`"
         % (cfg["generated_at"], MODEL_ID, ENTITY_DETECTION, len(files),
            cfg["pred_root"], cfg["pseudo_gt"]),
         "",
         "## Caveats — read before using these numbers", ""]
    for c in CAVEATS:
        L.append("> **%s**" % c)
        L.append(">")
    L[-1] = ""
    L += ["## Aggregate", "",
          "| entities returned | mapped to taxonomy | dropped (NER noise) | "
          "dropped (unknown type) | unmapped (char->time miss) | "
          "over-broad ADDRESS |",
          "|---|---|---|---|---|---|",
          "| %d | %d | %d | %d | %d | %d |"
          % (agg["entities_total"], agg["mapped"], agg["dropped_noise"],
             agg["dropped_unknown"], agg["unmapped"],
             agg["overbroad_address"]),
          "",
          "### vs judge pseudo-GT union", "",
          "| union items matched (coverage, recall-like) | must-beep items "
          "matched | mapped EL entities matching a union item "
          "(precision-like) |",
          "|---|---|---|",
          "| %s | %s | %s |"
          % (_pct(agg["vs_union"]["union_matched"],
                  agg["vs_union"]["union_items"]),
             _pct(agg["vs_union"]["must_beep_matched"],
                  agg["vs_union"]["must_beep"]),
             _pct(agg["vs_union"]["el_matched"],
                  agg["vs_union"]["el_mapped"])),
          "",
          "### head-to-head vs our detector (same ElevenLabs transcript)", "",
          "| our intervals | found by both | only ours | only theirs "
          "(mapped EL entities we lack) |",
          "|---|---|---|---|",
          "| %d | %s | %d | %d |"
          % (agg["vs_ours"]["ours_total"],
             _pct(agg["vs_ours"]["both"], agg["vs_ours"]["ours_total"]),
             agg["vs_ours"]["only_ours"], agg["vs_ours"]["only_theirs"]),
          ""]
    if agg["categories"]:
        L += ["### Category breakdown (head-to-head)", "",
              "| category | ours | ours matched by EL | EL mapped | "
              "EL matched by ours |",
              "|---|---|---|---|---|"]
        for c in sorted(agg["categories"]):
            d = agg["categories"][c]
            L.append("| %s | %d | %s | %d | %s |"
                     % (c, d["ours"], _pct(d["ours_matched"], d["ours"]),
                        d["el"], _pct(d["el_matched"], d["el"])))
        L.append("")
    if agg["noise_hist"]:
        L += ["### Dropped NER-noise types", "",
              "| entity_type | dropped |", "|---|---|"]
        for t in sorted(agg["noise_hist"]):
            L.append("| %s | %d |" % (t, agg["noise_hist"][t]))
        L.append("")
    if agg["unknown_hist"]:
        L += ["### Unknown vendor types (not in our mapping)", "",
              "| entity_type | count |", "|---|---|"]
        for t in sorted(agg["unknown_hist"]):
            L.append("| %s | %d |" % (t, agg["unknown_hist"][t]))
        L.append("")
    L += ["## Per file", "",
          "| file_id | entities | mapped | noise | unknown | unmapped | "
          "union coverage | union precision | both | only ours | "
          "only theirs | notes |",
          "|---" * 12 + "|"]
    for f in files:
        vu, vo = f["vs_union"], f["vs_ours"]
        L.append("| %s | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s "
                 "| %s |" % (
                     f["file_id"], f["entities_total"], f["mapped"],
                     f["dropped_noise"], f["dropped_unknown"], f["unmapped"],
                     (_pct(vu["union_matched"], vu["union_items"])
                      if vu["available"] else "n/a"),
                     (_pct(vu["el_matched"], vu["el_mapped"])
                      if vu["available"] else "n/a"),
                     str(vo["both"]) if vo["available"] else "n/a",
                     str(vo["only_ours"]) if vo["available"] else "n/a",
                     str(vo["only_theirs"]) if vo["available"] else "n/a",
                     "; ".join(f["notes"]) or ""))
    L += ["", "## Mapped EL entities we lack (only theirs)", ""]
    rows = [(f["file_id"], m) for f in files for m in f["mapped_entities"]
            if f["vs_ours"]["available"] and not m.get("matched_ours", False)]
    if rows:
        L += ["| file_id | interval ms | vendor type | our category | "
              "matched union item | text |",
              "|---|---|---|---|---|---|"]
        for fid, m in rows:
            L.append("| %s | %d-%d | %s | %s | %s | %s |"
                     % (fid, m["start_ms"], m["end_ms"], m["entity_type"],
                        m["category"],
                        "yes" if m.get("matched_union") else "no",
                        m["text"][:60].replace("|", "/")))
    else:
        L.append("None.")
    L.append("")
    return "\n".join(L) + "\n"


def write_reports(out_dir, cfg, files, agg):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "el_entity_report.md"
    md.write_text(build_report(cfg, files, agg))
    js = out / "el_entity_report.json"
    js.write_text(json.dumps({
        "config": cfg, "caveats": CAVEATS,
        "thresholds": {"iou_min": IOU_MIN, "cover_min": COVER_MIN},
        "mapping": {"entity_map": ENTITY_MAP,
                    "noise_types": sorted(NOISE_TYPES),
                    "overbroad_types": sorted(OVERBROAD_TYPES)},
        "aggregate": agg, "files": files}, indent=2, ensure_ascii=False))
    return md, js


# ---------------------------------------------------------------------- run

def discover_fids(pred_root):
    """file_ids of every <fid>__elevenlabs dir under pred_root."""
    return sorted(d.name[: -len(EL_SUFFIX)]
                  for d in Path(pred_root).glob("*" + EL_SUFFIX)
                  if d.is_dir())


def load_pseudo_gt(path):
    """file_id -> items[] from the pseudo-GT report, or {} when missing."""
    p = Path(path)
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return {f["file_id"]: f.get("items", []) for f in doc.get("files", [])}


def run(args):
    fids = ([f.strip() for f in args.files.split(",") if f.strip()]
            if args.files else discover_fids(args.pred_root))
    if not fids:
        print("no %s dirs under %s" % (EL_SUFFIX, args.pred_root))
        return 0
    union_by_fid = load_pseudo_gt(args.pseudo_gt)
    if not union_by_fid:
        print("WARNING: pseudo-GT report missing at %s (union scoring "
              "skipped)" % args.pseudo_gt)
    files = []
    for fid in fids:
        print("evaluating %s ..." % fid)
        res = eval_file(fid, args.pred_root, args.out,
                        union_by_fid.get(fid), score_only=args.score_only,
                        force=args.force)
        vu, vo = res["vs_union"], res["vs_ours"]
        print("  entities %d | mapped %d noise %d unknown %d unmapped %d | "
              "union cov %s | both %s only-ours %s only-theirs %s" % (
                  res["entities_total"], res["mapped"], res["dropped_noise"],
                  res["dropped_unknown"], res["unmapped"],
                  (_pct(vu["union_matched"], vu["union_items"])
                   if vu["available"] else "n/a"),
                  vo["both"] if vo["available"] else "n/a",
                  vo["only_ours"] if vo["available"] else "n/a",
                  vo["only_theirs"] if vo["available"] else "n/a"))
        files.append(res)
    agg = aggregate(files)
    cfg = {"generated_at": datetime.date.today().isoformat(),
           "model_id": MODEL_ID, "entity_detection": ENTITY_DETECTION,
           "pred_root": str(args.pred_root), "out": str(args.out),
           "pseudo_gt": str(args.pseudo_gt),
           "score_only": bool(args.score_only), "files": fids}
    md, js = write_reports(args.out, cfg, files, agg)
    print("aggregate: union coverage %s | union precision %s | "
          "head-to-head both %s" % (
              _pct(agg["vs_union"]["union_matched"],
                   agg["vs_union"]["union_items"]),
              _pct(agg["vs_union"]["el_matched"],
                   agg["vs_union"]["el_mapped"]),
              _pct(agg["vs_ours"]["both"], agg["vs_ours"]["ours_total"])))
    print("report -> %s\njson   -> %s" % (md, js))
    return 0


# ------------------------------------------------------------------ self-test

def self_test():
    """Offline checks against output/el_entity_probe.json — no network."""
    import tempfile
    cases = []

    def case(name, ok, detail=""):
        cases.append((name, ok))
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  [%s]" % detail) if (detail and not ok)
                            else ""))

    probe_p = ROOT / "output" / "el_entity_probe.json"
    if not probe_p.exists():
        case("probe fixture present", False, str(probe_p))
        print("\nself-test: 0/1 passed")
        return 1
    probe = json.loads(probe_p.read_text())
    text, words = probe["text"], probe["words"]

    # 1. char map: every word found verbatim at its recorded position
    cw, misses = build_char_map(text, words)
    ok_slices = all(text[w["cs"]:w["ce"]] == w["text"] for w in cw)
    ok_order = all(cw[i]["cs"] < cw[i + 1]["cs"] for i in range(len(cw) - 1))
    case("char map: probe words found, positions verbatim and ordered",
         misses == 0 and len(cw) > 400 and ok_slices and ok_order,
         "misses=%d n=%d" % (misses, len(cw)))

    # 2. entity intervals monotonic in char order, valid widths
    ivs = [entity_interval(e, cw) for e in probe["entities"]]
    case("entity intervals: all mapped, monotonic, positive width",
         all(ivs)
         and all(iv["end_ms"] > iv["start_ms"] for iv in ivs)
         and all(a["start_ms"] <= b["start_ms"]
                 for a, b in zip(ivs, ivs[1:])), str(ivs))

    # 3. Vidya (name, chars 113-118) maps to its words near 11-15 s
    vid = probe["entities"][0]
    viv = entity_interval(vid, cw)
    case("Vidya maps to 11-15 s",
         vid["entity_type"] == "name" and text[113:118] == "Vidya"
         and viv is not None and 11000 <= viv["start_ms"] <= 15000
         and viv["end_ms"] > viv["start_ms"], str(viv))

    # 4. category mapping: kept / noise-with-count / unknown-with-count;
    #    every mapped category exists in the taxonomy
    case("category mapping: name kept, ordinal noise, unknown counted",
         classify("name") == ("FULL_NAME", "mapped")
         and classify("ORDINAL") == (None, "noise")
         and classify("frobnicate") == (None, "unknown")
         and all(v in policy.CATEGORIES for v in ENTITY_MAP.values()))

    # 5. overlap rule math (copied from judge_pseudo_gt)
    iv_ = lambda s, e: {"start_ms": s, "end_ms": e}
    case("overlap rule: IoU/cover thresholds and no-touch",
         overlap_pair(iv_(0, 1000), iv_(500, 1500))          # IoU 1/3
         and not overlap_pair(iv_(0, 1000), iv_(900, 3000))  # IoU .03 cov .1
         and overlap_pair(iv_(0, 1000), iv_(100, 800))       # covered 100%
         and not overlap_pair(iv_(0, 1000), iv_(1000, 2000))  # touching
         and overlap_pair(iv_(13210, 13211), iv_(13000, 14000)))  # 1ms inside

    # 6. synthetic scoring math
    mapped = [{"start_ms": 0, "end_ms": 900, "category": "PHONE",
               "entity_type": "phone", "text": "x"},
              {"start_ms": 9000, "end_ms": 9500, "category": "EMAIL",
               "entity_type": "email", "text": "y"}]
    ours = [{"start_ms": 0, "end_ms": 1000, "categories": ["PHONE"]},
            {"start_ms": 5000, "end_ms": 6000, "categories": ["FULL_NAME"]}]
    union = [{"start_ms": 0, "end_ms": 1000, "action": "beep"},
             {"start_ms": 5000, "end_ms": 6000, "action": "annotate_only"}]
    vu = score_vs_union([dict(m) for m in mapped], union)
    vo = score_vs_ours([dict(m) for m in mapped], ours)
    case("vs union: coverage 1/2, must-beep 1/1, precision 1/2",
         (vu["union_matched"], vu["union_items"]) == (1, 2)
         and (vu["must_beep_matched"], vu["must_beep"]) == (1, 1)
         and (vu["el_matched"], vu["el_mapped"]) == (1, 2), str(vu))
    case("vs ours: both 1, only-ours 1, only-theirs 1, category split",
         (vo["both"], vo["only_ours"], vo["only_theirs"]) == (1, 1, 1)
         and vo["categories"]["PHONE"] == {"ours": 1, "ours_matched": 1,
                                           "el": 1, "el_matched": 1}
         and vo["categories"]["FULL_NAME"]["ours_matched"] == 0
         and vo["categories"]["EMAIL"]["el_matched"] == 0, str(vo))

    # 7. end-to-end eval_file offline: probe as cache fixture, synthetic
    #    intervals.json + union items; --score-only style (no fetch path)
    tmp = Path(tempfile.mkdtemp(prefix="el_entity_selftest_"))
    pred, out = tmp / "output", tmp / "el_entity"
    fid = "probefile"
    edir = pred / (fid + EL_SUFFIX)
    edir.mkdir(parents=True)
    out.mkdir(parents=True)
    (out / (fid + ".entities.json")).write_text(json.dumps(probe))
    (edir / "intervals.json").write_text(json.dumps({
        "file_id": fid, "intervals": [
            {"start_ms": 13100, "end_ms": 13400,
             "categories": ["FULL_NAME"]}]}))
    union7 = [{"start_ms": 13000, "end_ms": 13600,
               "categories": ["FULL_NAME"], "action": "beep"}]
    res = eval_file(fid, pred, out, union7, score_only=True)
    vu, vo = res["vs_union"], res["vs_ours"]
    case("end-to-end: probe counts (8 entities: 5 mapped, 3 noise, "
         "0 unknown/unmapped)",
         res["entities_total"] == 8 and res["mapped"] == 5
         and res["dropped_noise"] == 3 and res["dropped_unknown"] == 0
         and res["unmapped"] == 0
         and res["noise_hist"] == {"cardinal": 1, "ordinal": 2}
         and res["overbroad_address"] == 1,
         json.dumps({k: res[k] for k in ("entities_total", "mapped",
                                         "dropped_noise", "noise_hist",
                                         "overbroad_address")}))
    case("end-to-end: union cov 1/1, precision 2/5; head-to-head both 1, "
         "only-ours 0, only-theirs 3",
         (vu["union_matched"], vu["union_items"]) == (1, 1)
         and (vu["el_matched"], vu["el_mapped"]) == (2, 5)
         and (vo["both"], vo["only_ours"], vo["only_theirs"]) == (1, 0, 3)
         and vo["categories"]["FULL_NAME"] == {"ours": 1, "ours_matched": 1,
                                               "el": 2, "el_matched": 2},
         json.dumps({"vu": vu, "vo": {k: vo[k] for k in ("both", "only_ours",
                                                         "only_theirs")}}))
    # score-only with no cache: file reported, nothing fetched
    res_nc = eval_file("nocache", pred, out, None, score_only=True)
    case("score-only without cache: skipped with note, no fetch",
         not res_nc["fetched"] and res_nc["entities_total"] == 0
         and any("never fetches" in n for n in res_nc["notes"]),
         str(res_nc["notes"]))

    # 8. report renders with the key blocks
    agg = aggregate([res, res_nc])
    cfg = {"generated_at": "test", "model_id": MODEL_ID,
           "entity_detection": ENTITY_DETECTION, "pred_root": str(pred),
           "out": str(out), "pseudo_gt": "test", "score_only": True,
           "files": [fid, "nocache"]}
    md, js = write_reports(out, cfg, [res, res_nc], agg)
    txt = md.read_text()
    jdoc = json.loads(js.read_text())
    case("report renders: caveats, histograms, tables, json aggregate",
         md.exists() and js.exists()
         and "Caveats" in txt and "over-broad" in txt.lower()
         and "cannot express" in txt
         and "| cardinal | 1 |" in txt and "| ordinal | 2 |" in txt
         and "head-to-head" in txt and "| probefile |" in txt
         and "| nocache |" in txt
         and jdoc["aggregate"]["entities_total"] == 8
         and jdoc["aggregate"]["vs_ours"]["only_theirs"] == 3
         and jdoc["thresholds"] == {"iou_min": IOU_MIN,
                                    "cover_min": COVER_MIN},
         txt[:400])

    failed = [n for n, ok in cases if not ok]
    print("\nself-test: %d/%d passed" % (len(cases) - len(failed),
                                         len(cases)))
    return 1 if failed else 0


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", default=None,
                    help="comma list of file_ids (default: every "
                         "<fid>__elevenlabs dir under --pred-root)")
    ap.add_argument("--out", default="output/el_entity",
                    help="cache + report directory")
    ap.add_argument("--pred-root", default="output",
                    help="directory holding <fid>__elevenlabs/ dirs")
    ap.add_argument("--pseudo-gt", default=str(PSEUDO_GT),
                    help="pseudo-GT report json (judge union reference)")
    ap.add_argument("--force", action="store_true",
                    help="refetch even when a cached entities json exists")
    ap.add_argument("--score-only", action="store_true",
                    help="never call the API; score cached responses only")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline checks on the probe fixture and exit")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    sys.exit(run(args))


if __name__ == "__main__":
    main()
