#!/usr/bin/env python3
"""Mapped-entity leak audit — is any real name still readable or audible?

Runs after the PII judge audit whenever a mapping is active (tools/redact.py
--mapping). Three checks, all against the DELIVERABLES:

1. readable, deterministic: entity_map's matcher (regex, aliases,
   Devanagari->Latin keys, sound-alike) over the redacted transcript, with
   invented names and markers masked out.
2. readable, judge: a different model family (judge_client) reads the
   numbered redacted transcript with the real entity list and the invented
   names as an allowlist (prompts/judge_mapped_entity_v1.txt). Quotes are
   validated against the cited line; a judge-only finding counts when a
   second independent read confirms it (the audit judge is unstable on
   borderline cases, as the PII audit learned).
3. audible (--listen): the delivered AUDIO is transcribed again by
   ElevenLabs scribe_v2, biased toward the real names (keyterms), and the
   matcher scans that transcript. This is the check that hears what the
   transcript cannot show: a name a beep or voiced slot only partly covered.
   It also measures how often the invented name itself is heard (a
   replacement no listener recognises is a failed replacement, not a leak).

Writes <dir>/mapped_audit/mapped_audit.json (the listen transcript holds any
surviving audio content, so it stays in the work dir, never a deliverable).

  python3 tools/mapped_audit.py --dir output/<fid>__elevenlabs --fid <fid>
      [--mapping m.yaml] [--listen] [--no-judge] [--judge-model M]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import common
import entity_map
import judge_client

PROMPT = (ROOT / "prompts" / "judge_mapped_entity_v1.txt").read_text()
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"findings": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"line": {"type": "integer"},
                       "quoted_text": {"type": "string"},
                       "entity_id": {"type": "string"},
                       "reason": {"type": "string"}},
        "required": ["line", "quoted_text", "entity_id", "reason"]}}},
    "required": ["findings"],
}
HEARD_MIN = 0.75        # key similarity for "the invented name was heard"
# where a mapped mention was REMOVED, a remnant is likely, so a looser
# sound-alike counts (a near spelling heard where a brand was replaced)
FUZZY_AT_REMOVED = 0.70


def _norm(s):
    return " ".join(str(s or "").split())


def judge_read(em, text, model):
    lines = text.splitlines()
    numbered = "\n".join(f"{i + 1}: {l}" for i, l in enumerate(lines))
    prompt = (PROMPT + "\n\n" + em.prompt_block()
              + "\n\nINVENTED NAMES (deliberate, never findings): "
              + "; ".join(em.invented_terms())
              + "\n\n=== REDACTED TRANSCRIPT (numbered lines) ===\n"
              + numbered)
    raw = judge_client.judge_call(prompt, SCHEMA, model=model,
                                  log_engine="judge_mapped")
    invented = {t.lower() for t in em.invented_terms()}
    norm = [_norm(l) for l in lines]
    kept = []
    for f in raw.get("findings", []):
        try:
            ln = int(f.get("line", 0))
        except (TypeError, ValueError):
            continue
        q = _norm(f.get("quoted_text"))
        win = " ".join(norm[max(0, ln - 2): min(len(norm), ln + 1)])
        if not q or not (1 <= ln <= len(norm)) or q not in win:
            continue                        # hallucination guard
        if q.lower() in invented or "REDACTED>" in q:
            continue
        kept.append({"line": ln, "quoted_text": q,
                     "entity": f.get("entity_id"),
                     "reason": str(f.get("reason", ""))[:200]})
    return kept


def _same(f, g):
    return (f["quoted_text"].lower() == g["quoted_text"].lower()
            and abs(f["line"] - g["line"]) <= 3)


def listen(src_dir, fid, em, out_dir):
    """Re-transcribe the delivered audio; scan it; measure intelligibility."""
    from adapters import elevenlabs_transcribe
    import audio_utils
    import detect_pii
    deliverable = src_dir / common.beeped_name(fid)
    work = out_dir / "listen_16k.wav"
    tr = out_dir / "listen_transcript.json"
    audio_utils.decode_to_wav(deliverable, work, rate=16000, mono=True)
    common.set_usage_context(f"listen_check:{fid}")
    try:
        elevenlabs_transcribe.transcribe_file(None, work, work, tr)
    finally:
        common.set_usage_context("")
    work.unlink(missing_ok=True)
    words = detect_pii.flatten_words(common.load_json(tr))
    inv_rx = em._invented_rx
    removed_ivs = [iv for iv in common.load_json(
        src_dir / "intervals.json")["intervals"]
        if iv.get("mapping_id") or "MAPPED_ENTITY" in iv["categories"]]
    removed = [(iv["start_ms"], iv["end_ms"]) for iv in removed_ivs]
    hits, suspected = [], []
    near_removed, run, prev = [], 0, None
    for w in words:
        if any(w["start_ms"] <= e + 500 and s0 - 500 <= w["end_ms"]
               for s0, e in removed):
            if prev is None or w["i"] != prev + 1:
                run += 1               # contiguous runs only: never join
            near_removed.append(dict(w, ui=("near", run)))  # distant words
            prev = w["i"]
    loose = em.find_spans(near_removed, fuzzy_min=FUZZY_AT_REMOVED)
    strict = em.find_spans(words)
    seen = set()
    spans = []
    for sp in strict + [dict(x, _loose=True) for x in loose]:
        if (sp["start_word"], sp["end_word"]) in seen:
            continue
        seen.add((sp["start_word"], sp["end_word"]))
        spans.append(sp)
    for sp in spans:
        ws = words[sp["start_word"]: sp["end_word"] + 1]
        text = " ".join(w["w"] for w in ws)
        if inv_rx is not None and inv_rx.fullmatch(text.strip(" .,!?")):
            continue
        h = {"start_ms": ws[0]["start_ms"], "end_ms": ws[-1]["end_ms"],
             "heard": text, "entity": sp["mapping_id"], "match": sp["match"]}
        # an exact form heard anywhere is a leak; a sound-alike only where a
        # mapped mention was removed (a partly covered name) — elsewhere a
        # common word that sounds close is reported, not gated
        # ... and only when it sounds like the entity removed THERE: with a
        # large mapping, an ordinary word near any replacement sounds close
        # to some name ("finally", "coming") and is reported, not gated
        at_removed = any(iv["start_ms"] - 500 <= h["end_ms"]
                         and h["start_ms"] <= iv["end_ms"] + 500
                         and (iv.get("mapping_id") is None
                              or iv.get("mapping_id") == sp["mapping_id"])
                         for iv in removed_ivs)
        (hits if sp["match"] != "fuzzy" or at_removed
         else suspected).append(h)
    ri = common.load_json(src_dir / "rendered_intervals.json")
    voiced = ri.get("rendered_voiced") or []
    heard = 0
    import difflib
    for v in voiced:
        near = [w["w"] for w in words
                if w["end_ms"] >= v["start_ms"] - 300
                and w["start_ms"] <= v["end_ms"] + 300]
        target = entity_map.key(v["text"])
        best = 0.0
        for i in range(len(near)):
            for j in range(i, min(len(near), i + 3)):
                k = entity_map.key("".join(near[i:j + 1]))
                best = max(best, difflib.SequenceMatcher(None, k,
                                                         target).ratio())
        heard += best >= HEARD_MIN
    return {"hits": hits, "suspected": suspected,
            "voiced_slots": len(voiced),
            "invented_name_heard": heard,
            "invented_name_heard_rate": (round(heard / len(voiced), 3)
                                         if voiced else None),
            "transcript": str(tr)}


def run(src_dir, fid, do_listen=False, do_judge=True,
        model=judge_client.DEFAULT_JUDGE):
    src_dir = Path(src_dir)
    em = entity_map.active()
    if em is None:
        raise RuntimeError("no mapping active (--mapping / PII_MAPPING_FILE)")
    out_dir = src_dir / "mapped_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = (src_dir / f"{fid}_redacted.txt").read_text()
    det = em.scan_text(text)
    judge = {"enabled": do_judge, "first_read": [], "confirmed": []}
    if do_judge:
        common.set_usage_context(f"audit_mapped:{fid}")
        try:
            first = judge_read(em, text, model)
            judge["first_read"] = first
            if first:
                second = judge_read(em, text, model)
                judge["second_read"] = second
                judge["confirmed"] = [f for f in first
                                      if any(_same(f, g) for g in second)]
        finally:
            common.set_usage_context("")
    lst = listen(src_dir, fid, em, out_dir) if do_listen else None
    n = len(det) + len(judge["confirmed"]) + (len(lst["hits"]) if lst else 0)
    rep = {"file_id": fid, "mapping": em.summary(),
           "judge_model": model if do_judge else None,
           "transcript_hits": det, "judge": judge, "listen": lst,
           "leak_candidates": n}
    (out_dir / "mapped_audit.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"  [mapped audit] transcript hits {len(det)}, judge confirmed "
          f"{len(judge['confirmed'])}"
          + (f", listen hits {len(lst['hits'])}, invented name heard "
             f"{lst['invented_name_heard']}/{lst['voiced_slots']}"
             if lst else "")
          + f" -> {n} leak candidate(s)")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fid", required=True)
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-model", default=judge_client.DEFAULT_JUDGE)
    a = ap.parse_args()
    if a.mapping:
        os.environ[entity_map.ENV] = str(Path(a.mapping).resolve())
    rep = run(a.dir, a.fid, a.listen, not a.no_judge, a.judge_model)
    sys.exit(1 if rep["leak_candidates"] else 0)


if __name__ == "__main__":
    main()
