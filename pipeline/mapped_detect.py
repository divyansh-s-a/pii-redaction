#!/usr/bin/env python3
"""Stage C add-on — mapped-entity detection (active only with a mapping).

Two detectors over the same numbered word stream as the PII pass:
1. LLM pass (prompts/mapped_entity_detection_v1.txt + the mapping's entity
   list): every mention of a listed entity, in any language or script, plus
   "hints" that reveal one without naming it (slogans, mascots, ...).
2. Deterministic net (entity_map): regex, aliases, Devanagari->Latin keys,
   sound-alike fuzzy match.

Merge rule — voice only what is confirmed:
  net regex/alias match                     -> voice (invented name)
  LLM mention that also resembles the entity -> voice
  LLM mention agreeing with any net hit      -> voice
  net fuzzy match alone, LLM mention that does not resemble the entity,
  or any hint                                -> beep (suspected; never
                                                put a name in a mouth)
Person entries (categories FULL_NAME) are spoken as their invented name in replace mode when confirmed.
"""
import difflib
import json
from pathlib import Path

import common
import entity_map

PROMPT_FILE = "mapped_entity_detection_v1.txt"
PROMPT = (Path(__file__).resolve().parent.parent / "prompts"
          / PROMPT_FILE).read_text()
RESEMBLE_MIN = 0.7     # LLM-only mention: key similarity to an entity form

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "spans": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_word": {"type": "INTEGER"},
                    "end_word": {"type": "INTEGER"},
                    "entity_id": {"type": "STRING"},
                    "kind": {"type": "STRING", "enum": ["mention", "hint"]},
                    "spoken_text": {"type": "STRING"},
                    "confidence": {"type": "STRING",
                                   "enum": ["high", "medium", "low"]},
                    "reason": {"type": "STRING"},
                },
                "required": ["start_word", "end_word", "entity_id", "kind",
                             "spoken_text", "confidence"],
            },
        }
    },
    "required": ["spans"],
}
GPT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"spans": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "start_word": {"type": "integer"},
            "end_word": {"type": "integer"},
            "entity_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["mention", "hint"]},
            "spoken_text": {"type": "string"},
            "confidence": {"type": "string",
                           "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"}},
        "required": ["start_word", "end_word", "entity_id", "kind",
                     "spoken_text", "confidence", "reason"]}}},
    "required": ["spans"],
}


def resembles(e, text):
    """Best key similarity between `text` and any form of entry `e` (a
    containment counts as a match: an inflected word holds the name)."""
    k = entity_map.key(text)
    if not k:
        return 0.0
    best = 0.0
    for fk in e["keys"]:
        if len(fk) >= 3 and (fk in k or k in fk) and \
                min(len(k), len(fk)) / max(len(k), len(fk)) >= 0.5:
            return 1.0
        best = max(best, difflib.SequenceMatcher(None, k, fk).ratio())
    return best


def llm_spans(model, words, em, backend="gemini"):
    import detect_pii
    out = []
    CHUNK = 5000
    for base in range(0, len(words), CHUNK):
        sub = words[base: base + CHUNK]
        prompt = (PROMPT + "\n\n" + em.prompt_block()
                  + "\n\nTRANSCRIPT WORDS:\n"
                  + detect_pii.words_as_prompt(sub, CHUNK))
        if backend == "gpt":
            import judge_client
            got = judge_client.judge_call(
                prompt, GPT_SCHEMA, model=model,
                log_engine="detector_gpt").get("spans", [])
        else:
            got = json.loads(common.generate(
                model, [{"text": prompt}], SCHEMA)).get("spans", [])
        lo, hi = sub[0]["i"], sub[-1]["i"]
        for s in got:
            try:
                a, b = int(s["start_word"]), int(s["end_word"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (lo <= a <= b <= hi) or b - a > 12:
                continue
            e = em.entry_for_prompt_id(s.get("entity_id"))
            spoken = " ".join(w["w"] for w in words[a: b + 1])
            kind = s.get("kind", "mention")
            if kind == "mention" and e is not None:
                how = "llm" if resembles(e, spoken) >= RESEMBLE_MIN \
                    else "llm_unconfirmed"
                sp = em.span(a, b, e, spoken, words[a]["speaker"],
                             "mapping_llm", how, s.get("reason", ""))
            else:
                sp = em.span(a, b, None, spoken, words[a]["speaker"],
                             "mapping_llm", "hint",
                             "brand hint: " + str(s.get("reason", "")))
                if e is not None:
                    sp["hint_reveals"] = e["id"]
            out.append(sp)
    return out


def merge(llm, net):
    """Union LLM and net spans into one list without overlaps; agreement
    upgrades the match to 'llm+net' (confirmed)."""
    allsp = sorted(llm + net, key=lambda s: (s["start_word"], -s["end_word"]))
    merged = []
    for s in allsp:
        if merged and s["start_word"] <= merged[-1]["end_word"]:
            m = merged[-1]
            m["end_word"] = max(m["end_word"], s["end_word"])
            srcs = {m["source"], s["source"]}
            # prefer a confirmed entry; a person entry always wins (beep)
            cands = [x for x in (m, s) if x.get("mapping_id")]
            persons = [x for x in cands if x["categories"] == ["FULL_NAME"]]
            confirmed = [x for x in cands if x["render"] == "voice"]
            pick = (persons or confirmed or cands or [m])[0]
            agree = (len(srcs) == 2 and m.get("mapping_id")
                     and m.get("mapping_id") == s.get("mapping_id"))
            for k in ("mapping_id", "mapping_category", "categories",
                      "render", "match", "confidence"):
                m[k] = pick[k]
            if agree and not persons and m["mapping_id"]:
                m["match"], m["render"], m["confidence"] = \
                    "llm+net", "voice", "high"
            if len(srcs) == 2:
                m["source"] = "mapping_llm+net"
            m["spoken_text"] = ""          # rebuilt from the words below
        else:
            merged.append(dict(s))
    return merged


def detect(model, words, em, backend="gemini"):
    net = em.find_spans(words)
    llm = llm_spans(model, words, em, backend)
    spans = merge(llm, net)
    for s in spans:
        if not s["spoken_text"]:
            s["spoken_text"] = " ".join(
                w["w"] for w in words[s["start_word"]: s["end_word"] + 1])
    print(f"    mapped entities: {len(spans)} span(s) "
          f"({sum(s['render'] == 'voice' for s in spans)} confirmed, "
          f"{len(net)} net / {len(llm)} llm hits)")
    return spans


def _self_test():
    import tempfile
    import yaml
    mapping = {"brands": [{"pattern": r"Zorvex", "replace": "Kelmora"}],
               "people": [{"pattern": r"\bRavindra\b", "replace": "x"}]}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.yaml"
        p.write_text(yaml.safe_dump(mapping))
        em = entity_map.load(p)
        e = em.by_id["brands:0"]
        assert resembles(e, "zorvexwala") == 1.0
        assert resembles(e, "television") < RESEMBLE_MIN
        llm = [em.span(3, 3, e, "zorvecks", "s", "mapping_llm", "llm"),
               em.span(9, 9, None, "slogan", "s", "mapping_llm", "hint")]
        net = [em.span(3, 4, e, "zorvecks tv", "s", "mapping_net", "fuzzy"),
               em.span(7, 7, em.by_id["people:0"], "Ravindra", "s",
                       "mapping_net", "regex")]
        m = merge(llm, net)
        assert [(s["start_word"], s["end_word"], s["render"]) for s in m] == \
            [(3, 4, "voice"), (7, 7, "voice"), (9, 9, "beep")], m
        assert m[0]["match"] == "llm+net" and m[1]["categories"] == ["FULL_NAME"]
        # a lone fuzzy hit stays a beep
        m2 = merge([], [em.span(3, 3, e, "zorvecks", "s", "mapping_net",
                                "fuzzy")])
        assert m2[0]["render"] == "beep"
    print("mapped_detect self-test: OK")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _self_test()
