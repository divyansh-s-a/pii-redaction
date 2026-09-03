#!/usr/bin/env python3
"""Stage C — detect PII in a transcript.json and emit intervals.json.

Two detectors, merged:
1. LLM policy pass (Taxonomy v0.2 prompt) over the numbered word stream.
   Pluggable backend: "gemini" (default, common.generate) or "gpt"
   (OpenRouter chat completions via the judge_client transport, default model
   openai/gpt-5.6-luna) — SAME policy prompt, SAME word-index schema, so
   detector models compare fairly.
2. Deterministic safety net: long spoken digit/letter sequences (English,
   Hindi/Bengali/Telugu number words, Devanagari/Bengali/Telugu/Latin digits,
   double/triple constructions) that the LLM missed become recall-first
   intervals flagged needs_review.

intervals.json is the single source of truth for BOTH audio beeps and
transcript redaction.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import policy

DIGIT_WORDS = {
    # English
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hindi (Latin)
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "tin": "3", "char": "4",
    "chaar": "4", "panch": "5", "paanch": "5", "chhe": "6", "che": "6",
    "chha": "6", "saat": "7", "sat": "7", "aath": "8", "ath": "8", "nau": "9",
    # Hindi (Devanagari)
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5",
    "पाँच": "5", "छह": "6", "छे": "6", "सात": "7", "आठ": "8", "नौ": "9",
    # Bengali
    "শূন্য": "0", "এক": "1", "দুই": "2", "তিন": "3", "চার": "4", "পাঁচ": "5",
    "ছয়": "6", "সাত": "7", "আট": "8", "নয়": "9",
    # Telugu (Latin)
    "sunna": "0", "okati": "1", "rendu": "2", "moodu": "3", "nalugu": "4",
    "aidu": "5", "ayidu": "5", "aaru": "6", "edu": "7", "yedu": "7",
    "enimidi": "8", "tommidi": "9",
    # Telugu (native script)
    "సున్న": "0", "సున్నా": "0", "ఒకటి": "1", "రెండు": "2", "మూడు": "3",
    "నాలుగు": "4", "ఐదు": "5", "ఆరు": "6", "ఏడు": "7", "ఎనిమిది": "8",
    "తొమ్మిది": "9",
}
MULTIPLIERS = {"double": 2, "triple": 3, "quadruple": 4}
FILLERS = {"umm", "um", "uh", "haan", "haa", "ha", "hmm", "hm", "acha",
           "accha", "जी", "हाँ", "हं", "आज্ঞা", "আজ্ঞা", "and", "aur", "और"}
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
BEN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
TEL_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")


def flatten_words(transcript):
    """Global word list with utterance/back-references."""
    words = []
    for ui, u in enumerate(transcript["utterances"]):
        for wi, w in enumerate(u["words"]):
            words.append({"i": len(words), "ui": ui, "wi": wi,
                          "speaker": u["speaker"],
                          "channel": u.get("channel"), **w})
    return words


_CANON_TRANS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"',
                                  "\u201d": '"', "\u2013": "-", "\u2014": "-",
                                  "\u00a0": " "})
_STRIP_PUNCT = ".,!?;:\"'()[]{}\u0964\u0965"


def canon_token(token):
    """Detection-input canonicalization (Fix 1, 2026-09-01): engines format the
    same speech differently (attached punctuation, fancy quotes, danda), and
    detection quality was measurably sensitive to that dialect — the detector
    flagged values on gemini's formatting that it skipped on elevenlabs'.
    Presenting every engine's words in one house style removes the variable.
    Word indices and timestamps are untouched — this cleans only the text the
    detector reads; an empty result falls back to the original token."""
    t = str(token).translate(_CANON_TRANS).strip()
    cleaned = t.strip(_STRIP_PUNCT)
    return cleaned if cleaned else t


def words_as_prompt(words, max_per_call=6000):
    lines = []
    for w in words[:max_per_call]:
        lines.append(f"{w['i']}\t{w['speaker']}\t{canon_token(w['w'])}")
    return "word_index\tspeaker\tword\n" + "\n".join(lines)


# Strict-mode mirror of policy.DETECTION_SCHEMA for OpenRouter structured
# outputs (lowercase types, additionalProperties false, all required) — used
# by the "gpt" detector backend.
GPT_DETECTOR_DEFAULT = "openai/gpt-5.6-luna"
GPT_DETECTION_SCHEMA = {
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


def llm_detect(model, words, backend="gemini"):
    spans = []
    CHUNK = 5000
    for base in range(0, len(words), CHUNK):
        sub = words[base: base + CHUNK]
        prompt = (policy.DETECTION_POLICY +
                  "\n\nTRANSCRIPT WORDS:\n" + words_as_prompt(sub, CHUNK))
        if backend == "gpt":
            import judge_client
            got = judge_client.judge_call(
                prompt, GPT_DETECTION_SCHEMA, model=model,
                log_engine="detector_gpt").get("spans", [])
        else:
            raw = common.generate(model, [{"text": prompt}],
                                  policy.DETECTION_SCHEMA)
            got = json.loads(raw).get("spans", [])
        for s in got:
            a, b = int(s["start_word"]), int(s["end_word"])
            lo, hi = sub[0]["i"], sub[-1]["i"]
            if not (lo <= a <= b <= hi):
                continue
            cats = [c for c in s.get("categories", []) if c in policy.CATEGORIES]
            if not cats:
                continue
            spans.append({"start_word": a, "end_word": b, "categories": cats,
                          "spoken_text": s.get("spoken_text", ""),
                          "speaker": s.get("speaker", ""),
                          "public_figure": bool(s.get("public_figure", False)),
                          "confidence": s.get("confidence", "medium"),
                          "source": "llm", "reason": s.get("reason", "")})
    return spans


def norm_digit(token: str):
    t = (token.strip().lower().strip(".,;:!?-")
         .translate(DEV_DIGITS).translate(BEN_DIGITS).translate(TEL_DIGITS))
    if t.isdigit():
        return t
    return DIGIT_WORDS.get(t)


def safety_net(words):
    """Find runs of >=7 spoken digits (allowing fillers/multipliers inside)."""
    spans = []
    i = 0
    while i < len(words):
        j = i
        digits = ""
        used = []
        pending_mult = 1
        while j < len(words):
            tok = words[j]["w"].strip().lower().strip(".,;:!?-")
            d = norm_digit(words[j]["w"])
            if tok in MULTIPLIERS:
                pending_mult = MULTIPLIERS[tok]
                used.append(j)
                j += 1
                continue
            if d is not None and len(d) <= 4:
                digits += d * pending_mult if len(d) == 1 else d
                pending_mult = 1
                used.append(j)
                j += 1
                continue
            if used and tok in FILLERS and len(used) >= 1:
                nxt = norm_digit(words[j + 1]["w"]) if j + 1 < len(words) else None
                if nxt is not None:
                    used.append(j)
                    j += 1
                    continue
            break
        if len(digits) >= 7:
            n = len(digits)
            if n in (10, 11) or digits.startswith("91") and n == 12:
                cats = ["PHONE"]
            elif n == 12:
                cats = ["GOVERNMENT_ID"]
            elif 15 <= n <= 16:
                cats = ["PAYMENT_CARD_NUMBER"]
            else:
                cats = ["OTHER"]
            spans.append({
                "start_word": words[used[0]]["i"], "end_word": words[used[-1]]["i"],
                "categories": cats,
                "spoken_text": " ".join(words[k]["w"] for k in range(used[0], used[-1] + 1)),
                "speaker": words[used[0]]["speaker"],
                "public_figure": False, "confidence": "low",
                "source": "safety_net",
                "reason": f"{n}-digit spoken sequence"})
            i = j
        else:
            i += 1
    return spans


def overlaps(a, b):
    return a["start_word"] <= b["end_word"] and b["start_word"] <= a["end_word"]


def build_intervals(file_id, transcript, spans, gap_merge_ms):
    words = flatten_words(transcript)
    out = []
    for k, s in enumerate(sorted(spans, key=lambda x: x["start_word"])):
        notes = s["reason"]
        # Guard: the LLM sometimes returns absurdly wide word ranges (observed:
        # a one-word value spanning to end-of-file). If the range is far wider
        # than the quoted spoken_text, trim to the best-matching window and
        # force review.
        expected = max(1, len((s.get("spoken_text") or "").split()))
        span_n = s["end_word"] - s["start_word"] + 1
        if span_n > expected * 3 + 5:
            def key(t):
                return t.strip().lower().strip(".,!?;:\u0964")
            first = key((s.get("spoken_text") or "").split()[0]) if s.get("spoken_text") else ""
            best = s["start_word"]
            for cand in range(s["start_word"], s["end_word"] - expected + 2):
                if first and key(words[cand]["w"]) == first:
                    best = cand
                    break
            s = dict(s, start_word=best,
                     end_word=min(best + expected - 1, s["end_word"]),
                     confidence="low")
            notes = (f"SPAN TRIMMED from {span_n} to {expected} words "
                     f"(model range far exceeded quoted value). " + notes)
        ws = words[s["start_word"]: s["end_word"] + 1]
        start, end = ws[0]["start_ms"], max(w["end_ms"] for w in ws)
        if s["source"] == "safety_net":
            notes = "SAFETY NET: " + notes
        out.append({
            "annotation_id": f"{file_id}_{k+1:03d}",
            "start_ms": start, "end_ms": end,
            "speaker": s.get("speaker") or ws[0]["speaker"],
            "spoken_text": s["spoken_text"] or " ".join(w["w"] for w in ws),
            "categories": s["categories"],
            "person_link": "explicit" if s["confidence"] == "high" else "contextual",
            "confidence": s["confidence"],
            "public_figure": s["public_figure"],
            "needs_review": s["confidence"] == "low" or s["source"] == "safety_net"
                            or "OTHER" in s["categories"] or s["public_figure"],
            "notes": notes,
            "start_word": s["start_word"], "end_word": s["end_word"],
            "source": s["source"],
            "channel": ws[0].get("channel"),
        })
    return out


def detect_file(model, file_id, transcript_path: Path, out_path: Path,
                gap_merge_ms: int, detector="gemini", detector_model=None):
    """detector: 'gemini' (default; `model` is the Gemini model) or 'gpt'
    (OpenRouter; detector_model defaults to GPT_DETECTOR_DEFAULT)."""
    transcript = common.load_json(transcript_path)
    words = flatten_words(transcript)
    if not words:
        common.save_json(out_path, {"file_id": file_id, "intervals": []})
        return out_path
    if detector == "gpt":
        llm_spans = llm_detect(detector_model or GPT_DETECTOR_DEFAULT, words,
                               backend="gpt")
    else:
        llm_spans = llm_detect(model, words)
    net_spans = [n for n in safety_net(words)
                 if not any(overlaps(n, l) for l in llm_spans)]
    intervals = build_intervals(file_id, transcript, llm_spans + net_spans,
                                gap_merge_ms)
    common.save_json(out_path, {"file_id": file_id,
                                "taxonomy_version": "0.2",
                                "detector": detector,
                                "detector_model": (
                                    (detector_model or GPT_DETECTOR_DEFAULT)
                                    if detector == "gpt" else model),
                                "intervals": intervals})
    n_net = sum(1 for i in intervals if i["source"] == "safety_net")
    print(f"  detected {len(intervals)} intervals "
          f"({len(intervals) - n_net} llm, {n_net} safety-net) -> {out_path}")
    return out_path


if __name__ == "__main__":
    # tiny inline self-test: norm_digit understands Telugu (words + native
    # digit chars) and the existing Hindi/Bengali/English paths are unchanged.
    assert norm_digit("nalugu") == "4"
    assert norm_digit("tommidi") == "9"
    assert norm_digit("తొమ్మిది") == "9"
    assert norm_digit("౪") == "4"          # native Telugu 4
    assert norm_digit("౧౨") == "12"
    assert norm_digit("nau") == "9"        # Hindi (Latin) unchanged
    assert norm_digit("९") == "9"          # Devanagari digit unchanged
    assert norm_digit("নয়") == "9"         # Bengali unchanged
    assert norm_digit("seven") == "7"      # English unchanged
    assert norm_digit("hello") is None
    print("detect_pii self-test OK")
