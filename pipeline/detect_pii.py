#!/usr/bin/env python3
"""Stage C — detect PII in a transcript.json and emit intervals.json.

Two detectors, merged:
1. LLM policy pass (Taxonomy v0.2 prompt) over the numbered word stream.
   Pluggable backend: "gemini" (default, common.generate) or "gpt"
   (OpenRouter chat completions via the judge_client transport, default model
   openai/gpt-5.6-luna) — SAME policy prompt, SAME word-index schema, so
   detector models compare fairly.
2. Deterministic safety net: long spoken digit/letter sequences that the
   LLM missed become recall-first intervals flagged needs_review. Digit words
   cover English + 9 Indian languages (Hindi, Bengali, Telugu, Tamil,
   Malayalam, Kannada, Odia, Marathi, Gujarati, Punjabi — native + romanized);
   numerals of ALL scripts count (Unicode isdigit + a combined normalization
   table); double/triple constructions understood.

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
    # Tamil (native + Latin; several romanizations shared with Malayalam)
    "பூஜ்ஜியம்": "0", "ஒன்று": "1", "இரண்டு": "2", "மூன்று": "3",
    "நான்கு": "4", "ஐந்து": "5", "ஆறு": "6", "ஏழு": "7", "எட்டு": "8",
    "ஒன்பது": "9",
    "poojiyam": "0", "pujjiyam": "0", "onru": "1", "ondru": "1", "onnu": "1",
    "moonu": "3", "moondru": "3", "naalu": "4", "nangu": "4", "ainthu": "5",
    "anju": "5", "ezhu": "7", "elu": "7", "ettu": "8", "onbathu": "9",
    "onpathu": "9", "ombathu": "9",
    # Malayalam (native; Latin forms above are shared)
    "പൂജ്യം": "0", "ഒന്ന്": "1", "രണ്ട്": "2", "മൂന്ന്": "3", "നാല്": "4",
    "അഞ്ച്": "5", "ആറ്": "6", "ഏഴ്": "7", "എട്ട്": "8", "ഒൻപത്": "9",
    "ഒമ്പത്": "9", "randu": "2", "nalu": "4",
    # Kannada (native + Latin)
    "ಸೊನ್ನೆ": "0", "ಒಂದು": "1", "ಎರಡು": "2", "ಮೂರು": "3", "ನಾಲ್ಕು": "4",
    "ಐದು": "5", "ಆರು": "6", "ಏಳು": "7", "ಎಂಟು": "8", "ಒಂಬತ್ತು": "9",
    "sonne": "0", "ondu": "1", "eradu": "2", "mooru": "3", "nalku": "4",
    "naalku": "4", "eelu": "7", "entu": "8", "ombattu": "9",
    # Odia (native + Latin; "shunya"/"sat" shared with Hindi)
    "ଶୂନ": "0", "ଏକ": "1", "ଦୁଇ": "2", "ତିନି": "3", "ଚାରି": "4", "ପାଞ୍ଚ": "5",
    "ଛଅ": "6", "ସାତ": "7", "ଆଠ": "8", "ନଅ": "9",
    "eka": "1", "dui": "2", "tini": "3", "chari": "4", "pancha": "5",
    "chhaa": "6", "sata": "7", "atha": "8", "nao": "9",
    # Marathi (Devanagari forms not shared with Hindi + Latin)
    "दोन": "2", "पाच": "5", "सहा": "6", "नऊ": "9",
    "don": "2", "pach": "5", "saha": "6",
    # Gujarati (native + Latin). Romanized "be" (2) is deliberately OMITTED:
    # it is a ubiquitous English word and would corrupt digit runs in
    # code-switched speech; the native form and the LLM detector cover it.
    "શૂન્ય": "0", "એક": "1", "બે": "2", "ત્રણ": "3", "ચાર": "4", "પાંચ": "5",
    "છ": "6", "સાત": "7", "આઠ": "8", "નવ": "9",
    "tran": "3", "nav": "9",
    # Punjabi (Gurmukhi + Latin; do/char/che shared with Hindi)
    "ਸਿਫ਼ਰ": "0", "ਇੱਕ": "1", "ਦੋ": "2", "ਤਿੰਨ": "3", "ਚਾਰ": "4", "ਪੰਜ": "5",
    "ਛੇ": "6", "ਸੱਤ": "7", "ਅੱਠ": "8", "ਨੌਂ": "9",
    "sifar": "0", "ikk": "1", "tinn": "3", "panj": "5", "satt": "7",
    "atth": "8", "naun": "9",
    # Malay / Indonesian (Latin; the two share most numerals; "sifar" (0,
    # Malay) is already mapped by Punjabi above). "dua" (2) also romanizes
    # the Hindi/Urdu word for blessing/prayer — harmless behind the
    # >=7-digit run rule, which no isolated word can trigger.
    "kosong": "0", "nol": "0", "satu": "1", "dua": "2", "tiga": "3",
    "empat": "4", "lima": "5", "enam": "6", "tujuh": "7", "lapan": "8",
    "delapan": "8", "sembilan": "9",
    # Malay/Indonesian se- contractions are complete VALUES (seribu = satu
    # ribu = 1000), unlike the bare magnitude words ribu/ratus which are
    # unit words (exempt as edge spill per owner policy 2026-09-11)
    "sepuluh": "10", "sebelas": "11", "seratus": "100", "seribu": "1000",
    # Swahili. "sita" (6) collides with the romanized Indian name Sita and
    # "saba" (7) with given names — likewise safe behind the run rule.
    "sifuri": "0", "moja": "1", "mbili": "2", "tatu": "3", "nne": "4",
    "tano": "5", "sita": "6", "saba": "7", "nane": "8", "tisa": "9",
}
MULTIPLIERS = {"double": 2, "triple": 3, "quadruple": 4,
               # Hinglish transcripts often write these in Devanagari
               "डबल": 2, "ट्रिपल": 3}
FILLERS = {"umm", "um", "uh", "haan", "haa", "ha", "hmm", "hm", "acha",
           "accha", "जी", "हाँ", "हं", "आज্ঞা", "আজ্ঞা", "and", "aur", "और",
           "ok", "okay", "ji", "haanji", "ho", "हो",
           "aama", "ஆமா", "howdu", "ಹೌದು", "athe", "അതെ", "avunu", "అవును",
           "ହଁ", "હા", "ਹਾਂ"}
# one combined numeral table: Devanagari, Bengali, Telugu, Tamil, Odia,
# Gujarati, Kannada, Malayalam, Gurmukhi -> ASCII (norm_digit's isdigit()
# check already accepts any Unicode numeral; translating normalizes them so
# length/prefix classification like startswith("91") works uniformly)
INDIC_DIGITS = str.maketrans(
    "०१२३४५६७८९" "০১২৩৪৫৬৭৮৯" "౦౧౨౩౪౫౬౭౮౯" "௦௧௨௩௪௫௬௭௮௯"
    "୦୧୨୩୪୫୬୭୮୯" "૦૧૨૩૪૫૬૭૮૯" "೦೧೨೩೪೫೬೭೮೯" "൦൧൨൩൪൫൬൭൮൯"
    "੦੧੨੩੪੫੬੭੮੯",
    "0123456789" * 9)
# aliases kept for compatibility (tools/score_run.py chains them; chaining
# the same table twice is a no-op)
DEV_DIGITS = BEN_DIGITS = TEL_DIGITS = INDIC_DIGITS


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
        import langpack
        prompt = (policy.DETECTION_POLICY + langpack.detection_supplement() +
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
    t = token.strip().lower().strip(".,;:!?-").translate(INDIC_DIGITS)
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
    # 2026-09-03 language extension: Tamil/Kannada/Malayalam/Odia/Marathi/
    # Gujarati/Punjabi digit words + all-script numeral normalization
    assert norm_digit("ஒன்பது") == "9"      # Tamil word
    assert norm_digit("moondru") == "3"     # Tamil (Latin)
    assert norm_digit("ombattu") == "9"     # Kannada (Latin)
    assert norm_digit("രണ്ട്") == "2"        # Malayalam word
    assert norm_digit("pancha") == "5"      # Odia (Latin)
    assert norm_digit("दोन") == "2"          # Marathi word
    assert norm_digit("tran") == "3"        # Gujarati (Latin)
    assert norm_digit("ਪੰਜ") == "5"          # Punjabi word
    assert norm_digit("௧") == "1"           # Tamil numeral
    assert norm_digit("୫୬") == "56"         # Odia numerals
    assert norm_digit("૯") == "9"           # Gujarati numeral
    assert norm_digit("೪") == "4"           # Kannada numeral
    assert norm_digit("൮") == "8"           # Malayalam numeral
    assert norm_digit("੭") == "7"           # Gurmukhi numeral
    assert norm_digit("be") is None         # deliberately excluded (English)
    # 2026-09-10 extension: Malay/Indonesian + Swahili (target datasets)
    assert norm_digit("satu") == "1"        # Malay/Indonesian
    assert norm_digit("dua") == "2"
    assert norm_digit("lapan") == "8"       # Malay 8
    assert norm_digit("delapan") == "8"     # Indonesian 8
    assert norm_digit("sembilan") == "9"
    assert norm_digit("kosong") == "0"
    assert norm_digit("sifuri") == "0"      # Swahili
    assert norm_digit("mbili") == "2"
    assert norm_digit("sita") == "6"
    assert norm_digit("tisa") == "9"
    # collision words cannot fire alone: the net needs a >=7-digit run
    lone = [{"w": w, "i": k, "start_ms": k * 500, "end_ms": k * 500 + 300,
             "speaker": "S1"} for k, w in enumerate(
        ["meri", "dua", "hai", "aapke", "saath", "Sita", "ji", "namaste"])]
    assert safety_net(lone) == []
    # a real Malay number readout does fire
    run = [{"w": w, "i": k, "start_ms": k * 500, "end_ms": k * 500 + 300,
            "speaker": "S1"} for k, w in enumerate(
        ["kosong", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh"])]
    hits = safety_net(run)
    assert len(hits) == 1 and hits[0]["reason"] == "8-digit spoken sequence"
    print("detect_pii self-test OK (incl. Malay/Indonesian + Swahili)")
