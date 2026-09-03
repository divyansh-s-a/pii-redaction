#!/usr/bin/env python3
"""Score one engine's predictions against human ground truth (Evaluation Spec v0.2).

Per GT clip <file_id>.gt.json (Annotation Schema v0.2), predictions are read from
  <pred-root>/<file_id><engine-suffix>/intervals.json            stage-C detector output
  <pred-root>/<file_id><engine-suffix>/rendered_intervals.json   actually-beeped, padded+merged
                                                                 intervals {start_ms,end_ms,channel}

Metrics (imports pipeline/policy.py — the exact Action Matrix the renderer uses):
  LEAK RATE (headline safety)  a must-beep GT annotation (action_for(categories, public_figure,
    profile) == "beep") is LEAKED unless >= 95% of its [start_ms, end_ms] is covered by the
    union of rendered beeped intervals on a compatible channel. Channel rule: a rendered
    interval with channel None covers every channel; channel k covers a GT annotation on
    channel k, or one with no channel recorded. public_figure intervals and intervals whose
    categories are all annotate-only under the profile are EXCLUDED from the leak denominator
    and reported separately. Also reported: leaks per audio-hour (GT duration_ms).
  DETECTION RECALL / PRECISION  GT annotations matched one-to-one to predicted intervals,
    greedily by temporal IoU; a pair is eligible if IoU >= 0.3 OR the prediction covers
    >= 70% of the GT interval. Padding/rendering never enters this metric.
  CATEGORY ACCURACY  among matched pairs, fraction whose category sets intersect.
  OVER-REDACTION  beeped seconds (rendered timeline union, channel-agnostic) not overlapping
    any GT annotation padded by 250 ms, as seconds per audio-hour; plus false beep events
    (rendered segments touching no padded GT at all).
  TIMING  mean |start delta| and |end delta| in ms over matched pairs.
  Breakdowns: per category, per clip language (languages_observed), per GT confidence.

Engine-quality metrics (need transcript.json + prepare.json in the prediction dir):
  SPAN TRANSCRIPTION ACCURACY  per GT annotation: reference = GT spoken_text, hypothesis =
    engine words whose midpoint falls in [start_ms-150, end_ms+150] joined in time order.
    Both sides normalized: lowercase; punctuation/currency symbols and currency words
    (rs/rupees/...) stripped; Devanagari/Bengali/Telugu digit chars -> ASCII; single-digit
    number words -> digits (detect_pii.DIGIT_WORDS); number phrases (English plus
    Hindi/Telugu/Bengali number words, native script and romanized) collapsed to one digit
    token ("thirty thousand six hundred and thirty five" -> "30635", "పదిహేను వేలు" ->
    "15000"); double/triple/quadruple digit constructions expanded ("double nine" -> "99");
    adjacent pure-digit tokens merged ("30 635" -> "30635"). Word-level edit distance ->
    span WER. Reported:
    mean span WER, % exact after normalization, % windows fully containing the GT text
    (robust to neighbour-word insertions the padded window pulls in), "span not
    transcribed" tally (no engine words in the window => WER 1.0), breakdowns by GT clip
    language and category.
  LANGUAGE ID ACCURACY  language codes on both sides normalized through one alias table
    (eng/english/en-IN -> en, hin/hindi -> hi, ...; mixed/unknown kept). GT = clip
    languages_observed; engine = transcript languages[] with word_share >= 0.05 plus
    primary_language. Reported: primary-language match rate (engine primary in GT set),
    mean language-set Jaccard, per-file listing.
  SPEAKER COUNT / CHANNEL CONSISTENCY  GT num_speakers vs distinct engine speakers.
    Reported: exact-match rate, mean absolute error, per-file listing with prepare.json
    channel_analysis context ({channels, split}) and a consistency note when split is true
    but engine speakers != channels. A missing engine prepare.json falls back to the base
    gemini dir's prepare.json (same media), noted.

Outputs a markdown report (per-file table + aggregate + breakdowns + one row per leak) and a
machine-readable score.json next to it. Scoring runs ALWAYS exit 0 — scores are information,
not gates. --self-test runs synthetic in-memory cases (no files, no APIs) and exits 1 on FAIL.

Usage:
  python3 tools/score_run.py --gt "annotations/dev/*.gt.json" --pred-root output \\
      --profile medical_consult [--report output/score_report.md] [--engine-suffix ""]
  python3 tools/score_run.py --self-test
"""
import argparse
import datetime
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import detect_pii
import policy

COVERAGE_MIN = 0.95        # GT must-beep interval covered below this => leak
IOU_MIN = 0.30             # detection-match eligibility
GT_COVER_MIN = 0.70        # ... OR prediction covers this much of the GT interval
OVER_PAD_MS = 250          # GT tolerance pad when measuring over-redaction
MS_PER_HOUR = 3600000.0
SPAN_WIN_PAD_MS = 150      # GT interval pad when collecting engine words for span WER
LANG_SHARE_MIN = 0.05      # engine languages[] entries below this word_share are ignored

COUNT_KEYS = [
    "duration_ms", "gt_total", "must_beep_total", "annotate_only_total",
    "public_figure_total", "leaked", "annotate_only_covered", "public_figure_covered",
    "matched_must_beep", "matched_annotate_only", "pred_total", "pred_matched",
    "matched_pairs", "cat_ok", "over_ms", "false_beep_events",
    "sum_abs_start_ms", "sum_abs_end_ms",
]


# ---------------------------------------------------------------- interval math

def _overlap_ms(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def _merge(segs):
    """Disjoint sorted union of (start, end) pairs."""
    out = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _channel_covers(rendered_ch, gt_ch):
    """Rendered channel None mutes all channels; a channel-specific beep covers a GT
    annotation on the same channel, or one with no channel recorded."""
    return rendered_ch is None or gt_ch is None or rendered_ch == gt_ch


def coverage_of(ann, rendered):
    """Fraction of the GT annotation covered by channel-compatible rendered beeps."""
    span = ann["end_ms"] - ann["start_ms"]
    if span <= 0:
        return 1.0
    segs = [(max(ann["start_ms"], r["start_ms"]), min(ann["end_ms"], r["end_ms"]))
            for r in rendered
            if _channel_covers(r.get("channel"), ann.get("channel"))
            and _overlap_ms(ann["start_ms"], ann["end_ms"], r["start_ms"], r["end_ms"]) > 0]
    return sum(e - s for s, e in _merge(segs)) / float(span)


def match_intervals(gt_anns, preds):
    """Greedy one-to-one matching by temporal IoU (desc). A (gt, pred) pair is eligible
    when IoU >= IOU_MIN or the prediction covers >= GT_COVER_MIN of the GT interval."""
    cands = []
    for gi, g in enumerate(gt_anns):
        g_len = g["end_ms"] - g["start_ms"]
        for pi, p in enumerate(preds):
            inter = _overlap_ms(g["start_ms"], g["end_ms"], p["start_ms"], p["end_ms"])
            if inter <= 0:
                continue
            union = g_len + (p["end_ms"] - p["start_ms"]) - inter
            iou = inter / float(union) if union else 0.0
            cover = inter / float(g_len) if g_len else 0.0
            if iou >= IOU_MIN or cover >= GT_COVER_MIN:
                cands.append((iou, cover, gi, pi))
    cands.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    used_g, used_p, pairs = set(), set(), []
    for _iou, _cov, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return pairs


# ---------------------------------------------------------------- engine quality:
# ------------------------------------------- span text normalization + span WER

# punctuation/currency stripped before tokenizing; '-' becomes a space so
# hyphen-joined number words ("thirty-five") split back into number words.
_STRIP_CHARS = ".,!?;:'\"()" + "‘’“”" + "₹$€£¥¢৳"
_SPAN_TRANS = {ord(c): None for c in _STRIP_CHARS}
_SPAN_TRANS[ord("-")] = " "
# Devanagari nukta folding: the combining nukta is dropped and precomposed
# nukta consonants fold to their base consonant, so "हज़ार" and "हजार" become
# the same token (number-word dictionary keys are written nukta-free).
_SPAN_TRANS[0x093C] = None
for _cp, _base in ((0x0958, "क"), (0x0959, "ख"), (0x095A, "ग"), (0x095B, "ज"),
                   (0x095C, "ड"), (0x095D, "ढ"), (0x095E, "फ"), (0x095F, "य")):
    _SPAN_TRANS[_cp] = _base

# spoken currency markers are stripped like currency symbols so the amount
# itself is what gets compared ("rs 30 635" vs "... rupees" -> same value)
CURRENCY_WORDS = {"rs", "rupees", "rupee", "rupaye", "rupaiya", "inr",
                  "paise", "paisa", "taka", "dollars", "dollar"}

_UNITS_EN = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS_EN = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19}
_TENS_EN = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES_EN = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000,
              "lakhs": 100000, "crore": 10000000, "crores": 10000000}

# Indic number-word vocabularies (native script + common Latin romanizations)
# for Hindi, Telugu and Bengali. Same grammar classes as English. Single-word
# composites above 20 (Hindi "ikkis" = 21, ...) sit in the teens table: they
# are standalone values that combine rightward only with scale words.
_UNITS_IN = {
    # Hindi (Latin + Devanagari)
    "shunya": 0, "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
    "panch": 5, "paanch": 5, "chhe": 6, "che": 6, "chha": 6, "saat": 7,
    "sat": 7, "aath": 8, "ath": 8, "nau": 9,
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "छह": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9,
    # Telugu (Latin + native)
    "sunna": 0, "okati": 1, "rendu": 2, "moodu": 3, "nalugu": 4, "aidu": 5,
    "ayidu": 5, "aaru": 6, "edu": 7, "yedu": 7, "enimidi": 8, "tommidi": 9,
    "సున్న": 0, "సున్నా": 0, "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4,
    "ఐదు": 5, "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9,
    # Bengali (Latin + native)
    "shunno": 0, "dui": 2, "chhoy": 6, "choy": 6, "noy": 9, "noi": 9,
    "শূন্য": 0, "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6,
    "সাত": 7, "আট": 8, "নয়": 9,
}
_TEENS_IN = {
    # Hindi 10-19 + common single-word composites
    "das": 10, "dus": 10, "gyarah": 11, "barah": 12, "terah": 13,
    "chaudah": 14, "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18,
    "unnis": 19, "ikkis": 21, "ikkees": 21, "bais": 22, "pachis": 25,
    "pachchis": 25,
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "इक्कीस": 21,
    "बाईस": 22, "पच्चीस": 25,
    # Telugu 10-19
    "padi": 10, "padakondu": 11, "pannendu": 12, "padamoodu": 13,
    "padnalugu": 14, "padihenu": 15, "padahaaru": 16, "padihedu": 17,
    "paddenimidi": 18, "pantommidi": 19,
    "పది": 10, "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13, "పద్నాలుగు": 14,
    "పదిహేను": 15, "పదహారు": 16, "పదిహేడు": 17, "పద్దెనిమిది": 18,
    "పంతొమ్మిది": 19,
    # Bengali 10-19
    "dosh": 10, "egaro": 11, "baro": 12, "tero": 13, "choddo": 14,
    "ponero": 15, "sholo": 16, "shotero": 17, "atharo": 18, "unish": 19,
    "দশ": 10, "এগারো": 11, "বারো": 12, "তেরো": 13, "চোদ্দ": 14, "পনেরো": 15,
    "ষোলো": 16, "সতেরো": 17, "আঠারো": 18, "উনিশ": 19,
}
_TENS_IN = {
    # Hindi (60 "saath" is romanized like 7 "saat", so 60 is native-only)
    "bees": 20, "bis": 20, "tees": 30, "chalis": 40, "chaalis": 40,
    "pachas": 50, "sattar": 70, "assi": 80, "nabbe": 90,
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70,
    "अस्सी": 80, "नब्बे": 90,
    # Telugu
    "iravai": 20, "muppai": 30, "nalabhai": 40, "yabhai": 50, "aravai": 60,
    "debbai": 70, "enabhai": 80, "tombhai": 90,
    "ఇరవై": 20, "ముప్పై": 30, "నలభై": 40, "యాభై": 50, "అరవై": 60,
    "డెబ్బై": 70, "ఎనభై": 80, "తొంభై": 90, "తొంబై": 90,
    # Bengali (60 "shat" is romanized like 7, so 60 is native-only)
    "bish": 20, "kuri": 20, "trish": 30, "chollish": 40, "ponchash": 50,
    "sottor": 70, "nobboi": 90,
    "বিশ": 20, "ত্রিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50, "ষাট": 60,
    "সত্তর": 70, "আশি": 80, "নব্বই": 90,
}
_SCALES_IN = {
    # Hindi ("हजार" nukta-free: _SPAN_TRANS folds "हज़ार" onto it)
    "sau": 100, "hazaar": 1000, "hazar": 1000, "hajar": 1000, "hajaar": 1000,
    "सौ": 100, "हजार": 1000, "लाख": 100000, "करोड": 10000000,
    "karod": 10000000,
    # Telugu
    "vanda": 100, "vandala": 100, "vandalu": 100, "veyyi": 1000, "veyi": 1000,
    "vela": 1000, "velu": 1000, "laksha": 100000, "lakshala": 100000,
    "lakshalu": 100000, "koti": 10000000, "kotla": 10000000,
    "వంద": 100, "వందల": 100, "వందలు": 100, "వెయ్యి": 1000, "వేయి": 1000,
    "వేల": 1000, "వేలు": 1000, "లక్ష": 100000, "లక్షల": 100000,
    "లక్షలు": 100000, "కోటి": 10000000, "కోట్ల": 10000000,
    # Bengali
    "sho": 100, "shoto": 100, "lokkho": 100000,
    "শো": 100, "শত": 100, "হাজার": 1000, "লক্ষ": 100000, "কোটি": 10000000,
}

_UNITS = dict(_UNITS_EN, **_UNITS_IN)
_TEENS = dict(_TEENS_EN, **_TEENS_IN)
_TENS = dict(_TENS_EN, **_TENS_IN)
_SCALES = dict(_SCALES_EN, **_SCALES_IN)


def _parse_number_run(toks, i):
    """Longest number PHRASE starting at toks[i] -> (value, end_index).
    Vocabulary: English plus Hindi/Telugu/Bengali number words (native script
    and common romanizations); every English key parses exactly as before.

    The grammar keeps digit recitations apart from arithmetic phrases: a
    unit/teen directly after a unit/teen (phone-number style "nine eight")
    ends the phrase, so each digit is emitted alone and the adjacent-digit
    merge concatenates them instead of summing. end_index == i means toks[i]
    does not start a number phrase. "and" joins only after a scale word and
    only when a number word follows ("six hundred and thirty five")."""
    total, cur = 0, 0
    j, last, started = i, "", False
    while j < len(toks):
        t = toks[j]
        if t in _UNITS:
            if last in ("unit", "teen"):
                break
            cur += _UNITS[t]
            last = "unit"
        elif t in _TEENS:
            if last in ("unit", "teen", "tens"):
                break
            cur += _TEENS[t]
            last = "teen"
        elif t in _TENS:
            if last in ("unit", "teen", "tens"):
                break
            cur += _TENS[t]
            last = "tens"
        elif t in _SCALES:
            scale = _SCALES[t]
            if scale == 100:
                cur = max(cur, 1) * 100
            else:
                total += max(cur, 1) * scale
                cur = 0
            last = "scale"
        elif t == "and":
            nxt = toks[j + 1] if j + 1 < len(toks) else ""
            if last != "scale" or not (nxt in _UNITS or nxt in _TEENS
                                       or nxt in _TENS):
                break
            last = "and"
            j += 1
            continue
        else:
            break
        started = True
        j += 1
    if not started:
        return 0, i
    return total + cur, j


def _mult_digit(tok):
    """tok as one spoken digit ("nine", "సున్న", "7") -> "0".."9", else None."""
    if len(tok) == 1 and tok.isdigit():
        return tok
    if tok in _UNITS:
        return str(_UNITS[tok])
    d = detect_pii.DIGIT_WORDS.get(tok)
    return d if d is not None and len(d) == 1 else None


def _collapse_number_phrases(toks):
    out, i = [], 0
    while i < len(toks):
        # "double"/"triple"/"quadruple" + digit -> repeated digit STRING
        # ("double nine" -> "99", never arithmetic), then the adjacent-digit
        # merge concatenates it with neighbours ("double nine eight" -> "998").
        mult = detect_pii.MULTIPLIERS.get(toks[i])
        if mult and i + 1 < len(toks):
            d = _mult_digit(toks[i + 1])
            if d is not None:
                out.append(d * mult)
                i += 2
                continue
        val, j = _parse_number_run(toks, i)
        if j > i:
            out.append(str(val))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return out


def norm_span_tokens(text, strip_currency_words=True):
    """Normalize spoken text into comparable tokens: lowercase; strip
    punctuation + currency symbols/words; Devanagari/Bengali/Telugu digits ->
    ASCII; collapse number phrases (English + Hindi/Telugu/Bengali) and
    double/triple digit constructions; map single-digit number words via
    detect_pii.DIGIT_WORDS; merge adjacent pure-digit tokens."""
    t = str(text or "").lower()
    t = (t.translate(detect_pii.DEV_DIGITS).translate(detect_pii.BEN_DIGITS)
         .translate(detect_pii.TEL_DIGITS))
    t = t.translate(_SPAN_TRANS)
    toks = t.split()
    if strip_currency_words:
        toks = [x for x in toks if x not in CURRENCY_WORDS]
    toks = _collapse_number_phrases(toks)
    toks = [detect_pii.DIGIT_WORDS.get(x, x) for x in toks]
    out = []
    for x in toks:
        if out and x.isdigit() and out[-1].isdigit():
            out[-1] += x
        else:
            out.append(x)
    return out


def word_edit_distance(ref, hyp):
    """Word-level Levenshtein distance between two token lists."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        row = [i]
        for j, h in enumerate(hyp, 1):
            row.append(min(prev[j] + 1, row[j - 1] + 1,
                           prev[j - 1] + (0 if r == h else 1)))
        prev = row
    return prev[-1]


def _sublist_in(hay, needle):
    """True when `needle` occurs as a contiguous sublist of `hay`."""
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def span_accuracy_for_file(gt, transcript):
    """One record per GT annotation: WER of the engine's words in the padded
    GT window vs the GT spoken_text (both sides normalized). No engine word
    in the window => WER 1.0 and not_transcribed. Because the +/-150 ms window
    can pull in neighbouring words (counted as insertions by WER), `contains`
    additionally records whether the full normalized GT text appears
    contiguously in the window — robust to that contamination."""
    words = []
    for u in (transcript or {}).get("utterances", []):
        for w in u.get("words", []):
            words.append((w["start_ms"], w["end_ms"], str(w.get("w", ""))))
    words.sort()
    recs = []
    for a in gt.get("annotations", []):
        ref = norm_span_tokens(a.get("spoken_text", ""))
        lo = a["start_ms"] - SPAN_WIN_PAD_MS
        hi = a["end_ms"] + SPAN_WIN_PAD_MS
        hyp_raw = [w for s, e, w in words if lo <= (s + e) / 2.0 <= hi]
        hyp = norm_span_tokens(" ".join(hyp_raw))
        not_transcribed = not hyp_raw
        wer = 1.0 if not_transcribed else (
            word_edit_distance(ref, hyp) / float(max(1, len(ref))))
        recs.append({"annotation_id": a.get("annotation_id", ""),
                     "categories": a.get("categories", []),
                     "wer": round(wer, 4),
                     "exact": bool(ref) and ref == hyp,
                     "contains": bool(ref) and _sublist_in(hyp, ref),
                     "not_transcribed": not_transcribed,
                     "ref_tokens": len(ref), "hyp_tokens": len(hyp)})
    return recs


# ------------------------------------------------------- language ID accuracy

_LANG_FAMILIES = {
    "en": ["en", "eng", "english", "en-in", "en-us", "en-gb"],
    "hi": ["hi", "hin", "hindi"],
    "bn": ["bn", "ben", "bengali", "bangla"],
    "or": ["or", "ori", "odia", "oriya"],
    "ta": ["ta", "tam", "tamil"],
    "te": ["te", "tel", "telugu"],
    "mr": ["mr", "mar", "marathi"],
    "gu": ["gu", "guj", "gujarati"],
    "kn": ["kn", "kan", "kannada"],
    "ml": ["ml", "mal", "malayalam"],
    "pa": ["pa", "pan", "punjabi", "panjabi"],
    "ur": ["ur", "urd", "urdu"],
    "mixed": ["mixed"],
    "unknown": ["unknown"],
}
LANG_NORM = {v: k for k, vs in _LANG_FAMILIES.items() for v in vs}


def norm_lang(code):
    """One family code per language: en/eng/english/en-IN -> en, etc.
    Region-tagged codes fall back to their prefix; unmapped codes pass
    through lowercased; mixed/unknown stay as-is."""
    c = str(code or "").strip().lower().replace("_", "-")
    if not c:
        return "unknown"
    if c in LANG_NORM:
        return LANG_NORM[c]
    return LANG_NORM.get(c.split("-")[0], c)


def lang_id_for_file(gt, transcript):
    """GT languages_observed vs engine primary_language + languages[]
    (word_share >= LANG_SHARE_MIN), both normalized."""
    gt_set = sorted({norm_lang(l) for l in gt.get("languages_observed", [])})
    rec = {"gt": gt_set, "primary": None, "engine": [],
           "primary_match": None, "jaccard": None}
    if transcript is None:
        return rec
    primary = norm_lang(transcript.get("primary_language", "unknown"))
    eng = {norm_lang(l.get("lang")) for l in transcript.get("languages", [])
           if l.get("word_share", 0) >= LANG_SHARE_MIN}
    eng.add(primary)
    rec["primary"] = primary
    rec["engine"] = sorted(eng)
    if gt_set:
        rec["primary_match"] = primary in gt_set
    union = set(gt_set) | eng
    if union:
        rec["jaccard"] = round(len(set(gt_set) & eng) / float(len(union)), 4)
    return rec


# --------------------------------------- speaker count / channel consistency

def speaker_stats_for_file(gt, transcript, channel_analysis, prep_note=""):
    """GT num_speakers vs distinct engine speaker labels, with channel context
    from prepare.json channel_analysis ({channels, split})."""
    gt_n = gt.get("num_speakers")
    eng_n = None
    if transcript is not None:
        eng_n = len({u.get("speaker", "unknown")
                     for u in transcript.get("utterances", [])})
    ca = channel_analysis or {}
    channels = ca.get("channels")
    split = bool(ca.get("split", False))
    inconsistent = bool(split and eng_n is not None and channels
                        and eng_n != channels)
    note = prep_note
    if inconsistent:
        extra = ("split-channel media (%d channels) but engine reports %d "
                 "distinct speaker(s)" % (channels, eng_n))
        note = (note + "; " + extra) if note else extra
    rec = {"gt": gt_n, "engine": eng_n, "channels": channels, "split": split,
           "split_inconsistent": inconsistent, "note": note,
           "exact": None, "abs_err": None}
    if gt_n is not None and eng_n is not None:
        rec["exact"] = (gt_n == eng_n)
        rec["abs_err"] = abs(gt_n - eng_n)
    return rec


# -------------------------------------------- engine-quality aggregation

def _span_bucket():
    return {"spans": 0, "wer_sum": 0.0, "exact": 0, "contains": 0,
            "not_transcribed": 0}


def aggregate_engine_quality(files):
    """Aggregate span/lang/speaker records; each file dict carries
    span_records, languages, lang_id, speakers."""
    def bump(b, r):
        b["spans"] += 1
        b["wer_sum"] += r["wer"]
        b["exact"] += 1 if r["exact"] else 0
        b["contains"] += 1 if r.get("contains") else 0
        b["not_transcribed"] += 1 if r["not_transcribed"] else 0

    span, by_cat, by_lang = _span_bucket(), {}, {}
    lang = {"files": 0, "primary_match": 0, "primary_eval": 0,
            "jaccard_sum": 0.0, "jaccard_n": 0}
    spk = {"files": 0, "exact": 0, "abs_err_sum": 0, "split_inconsistent": 0}
    for f in files:
        for r in f.get("span_records") or []:
            bump(span, r)
            for cat in r["categories"]:
                bump(by_cat.setdefault(cat, _span_bucket()), r)
            for lg in f.get("languages", []):
                bump(by_lang.setdefault(lg, _span_bucket()), r)
        li = f.get("lang_id") or {}
        if li.get("primary") is not None:
            lang["files"] += 1
            if li.get("primary_match") is not None:
                lang["primary_eval"] += 1
                lang["primary_match"] += 1 if li["primary_match"] else 0
            if li.get("jaccard") is not None:
                lang["jaccard_sum"] += li["jaccard"]
                lang["jaccard_n"] += 1
        sp = f.get("speakers") or {}
        if sp.get("abs_err") is not None:
            spk["files"] += 1
            spk["exact"] += 1 if sp["exact"] else 0
            spk["abs_err_sum"] += sp["abs_err"]
        spk["split_inconsistent"] += 1 if sp.get("split_inconsistent") else 0
    span["mean_wer"] = (span["wer_sum"] / span["spans"]) if span["spans"] else None
    lang["primary_match_rate"] = _ratio(lang["primary_match"], lang["primary_eval"])
    lang["mean_jaccard"] = ((lang["jaccard_sum"] / lang["jaccard_n"])
                            if lang["jaccard_n"] else None)
    spk["exact_match_rate"] = _ratio(spk["exact"], spk["files"])
    spk["mae"] = _ratio(spk["abs_err_sum"], spk["files"])
    return {"span": span, "span_by_category": by_cat, "span_by_language": by_lang,
            "language_id": lang, "speakers": spk}


# ---------------------------------------------------------------- per-file scoring

def score_file(gt, preds, rendered, profile):
    """Score one clip. gt = parsed .gt.json dict; preds = intervals.json 'intervals' list;
    rendered = rendered_intervals.json 'rendered_merged' list. Returns raw counters plus
    per-annotation records (for breakdowns) and one row per leak."""
    anns = gt.get("annotations", [])
    c = dict.fromkeys(COUNT_KEYS, 0)
    c["duration_ms"] = gt.get("duration_ms", 0)
    c["gt_total"] = len(anns)
    c["pred_total"] = len(preds)

    pairs = match_intervals(anns, preds)
    matched_g = {gi for gi, _ in pairs}
    c["matched_pairs"] = len(pairs)
    c["pred_matched"] = len({pi for _, pi in pairs})
    for gi, pi in pairs:
        if set(anns[gi]["categories"]) & set(preds[pi].get("categories", [])):
            c["cat_ok"] += 1
        c["sum_abs_start_ms"] += abs(anns[gi]["start_ms"] - preds[pi]["start_ms"])
        c["sum_abs_end_ms"] += abs(anns[gi]["end_ms"] - preds[pi]["end_ms"])

    records, leaks = [], []
    for gi, a in enumerate(anns):
        pf = bool(a.get("public_figure", False))
        action = policy.action_for(a["categories"], pf, profile)
        cov = coverage_of(a, rendered)
        covered = cov >= COVERAGE_MIN
        matched = gi in matched_g
        leaked = False
        if action == "beep":
            c["must_beep_total"] += 1
            c["matched_must_beep"] += 1 if matched else 0
            if not covered:
                leaked = True
                c["leaked"] += 1
                leaks.append({"annotation_id": a.get("annotation_id", ""),
                              "categories": a["categories"],
                              "start_ms": a["start_ms"], "end_ms": a["end_ms"],
                              "coverage": round(cov, 3),
                              "spoken_text": a.get("spoken_text", "")})
        else:
            c["annotate_only_total"] += 1
            c["matched_annotate_only"] += 1 if matched else 0
            c["annotate_only_covered"] += 1 if covered else 0
            if pf:
                c["public_figure_total"] += 1
                c["public_figure_covered"] += 1 if covered else 0
        records.append({"categories": a["categories"],
                        "confidence": a.get("confidence", "medium"),
                        "action": action, "public_figure": pf,
                        "leaked": leaked, "matched": matched})

    # over-redaction: rendered timeline union (channel-agnostic) minus padded GT union
    gt_pad = _merge([(a["start_ms"] - OVER_PAD_MS, a["end_ms"] + OVER_PAD_MS) for a in anns])
    for s, e in _merge([(r["start_ms"], r["end_ms"]) for r in rendered]):
        inside = sum(_overlap_ms(s, e, gs, ge) for gs, ge in gt_pad)
        c["over_ms"] += (e - s) - inside
        if inside == 0:
            c["false_beep_events"] += 1
    return {"counts": c, "records": records, "leaks": leaks}


def _ratio(num, den):
    return (num / float(den)) if den else None


def derive(c):
    """Metrics derived from raw counters (works for a file or an aggregate)."""
    hours = c["duration_ms"] / MS_PER_HOUR
    return {
        "leak_rate": _ratio(c["leaked"], c["must_beep_total"]),
        "leaks_per_hour": (c["leaked"] / hours) if hours else None,
        "recall_must_beep": _ratio(c["matched_must_beep"], c["must_beep_total"]),
        "recall_annotate_only": _ratio(c["matched_annotate_only"], c["annotate_only_total"]),
        "recall_all": _ratio(c["matched_must_beep"] + c["matched_annotate_only"], c["gt_total"]),
        "precision": _ratio(c["pred_matched"], c["pred_total"]),
        "category_accuracy": _ratio(c["cat_ok"], c["matched_pairs"]),
        "over_redaction_s_per_hour": ((c["over_ms"] / 1000.0) / hours) if hours else None,
        "false_beeps_per_hour": (c["false_beep_events"] / hours) if hours else None,
        "mean_abs_start_ms": _ratio(c["sum_abs_start_ms"], c["matched_pairs"]),
        "mean_abs_end_ms": _ratio(c["sum_abs_end_ms"], c["matched_pairs"]),
    }


# ---------------------------------------------------------------- aggregation + breakdowns

def aggregate_counts(files):
    agg = dict.fromkeys(COUNT_KEYS, 0)
    for f in files:
        for k in COUNT_KEYS:
            agg[k] += f["counts"][k]
    return agg


def breakdowns_for(files):
    """category / language / confidence buckets, each with the same small counter set."""
    def blank():
        return {"gt": 0, "must_beep": 0, "leaked": 0, "matched": 0}

    by_cat, by_conf, by_lang = {}, {}, {}
    for f in files:
        for rec in f["records"]:
            buckets = [by_cat.setdefault(cat, blank()) for cat in rec["categories"]]
            buckets.append(by_conf.setdefault(rec["confidence"], blank()))
            buckets += [by_lang.setdefault(lang, blank()) for lang in f["languages"]]
            for b in buckets:
                b["gt"] += 1
                b["matched"] += 1 if rec["matched"] else 0
                if rec["action"] == "beep":
                    b["must_beep"] += 1
                    b["leaked"] += 1 if rec["leaked"] else 0
    return {"category": by_cat, "language": by_lang, "confidence": by_conf}


# ---------------------------------------------------------------- report rendering

def _pct(num, den):
    if not den:
        return "n/a"
    return "%d/%d = %.1f%%" % (num, den, 100.0 * num / den)


def _f(x, nd=1):
    return "n/a" if x is None else ("%%.%df" % nd) % x


def _breakdown_table(title, buckets):
    lines = ["## Breakdown — %s" % title, "",
             "| %s | GT | must-beep | leaked | leak rate | matched | recall |" % title,
             "|---|---|---|---|---|---|---|"]
    for key in sorted(buckets):
        b = buckets[key]
        lines.append("| %s | %d | %d | %d | %s | %d | %s |" % (
            key, b["gt"], b["must_beep"], b["leaked"],
            _pct(b["leaked"], b["must_beep"]), b["matched"], _pct(b["matched"], b["gt"])))
    lines.append("")
    return lines


_YN = {True: "yes", False: "NO", None: "n/a"}


def _engine_quality_sections(files, eq):
    sp = eq["span"]
    L = ["## Engine quality — span transcription accuracy", "",
         "Reference = GT `spoken_text`; hypothesis = engine words whose midpoint falls in"
         " the GT interval +/-%d ms. Both sides normalized: case, punctuation, currency"
         " symbols/words, Devanagari/Bengali/Telugu digits, digit words, number phrases"
         " (English + Hindi/Telugu/Bengali), double/triple constructions,"
         " adjacent-digit merge." % SPAN_WIN_PAD_MS, "",
         "| Metric | Value |", "|---|---|",
         "| Spans scored | %d |" % sp["spans"],
         "| Mean span WER | %s |" % _f(sp["mean_wer"], 3),
         "| Exact after normalization | %s |" % _pct(sp["exact"], sp["spans"]),
         "| GT text fully present in window (insertion-robust) | %s |"
         % _pct(sp["contains"], sp["spans"]),
         "| Span not transcribed (no engine words in window) | %s |"
         % _pct(sp["not_transcribed"], sp["spans"]),
         ""]
    for title, buckets in (("category", eq["span_by_category"]),
                           ("language", eq["span_by_language"])):
        L += ["### Span WER by %s" % title, "",
              "| %s | spans | mean WER | exact | not transcribed |" % title,
              "|---|---|---|---|---|"]
        for key in sorted(buckets):
            b = buckets[key]
            mean = (b["wer_sum"] / b["spans"]) if b["spans"] else None
            L.append("| %s | %d | %s | %s | %s |" % (
                key, b["spans"], _f(mean, 3), _pct(b["exact"], b["spans"]),
                _pct(b["not_transcribed"], b["spans"])))
        L.append("")
    lid = eq["language_id"]
    L += ["## Engine quality — language identification", "",
          "| Metric | Value |", "|---|---|",
          "| Primary-language match rate (engine primary in GT set) | %s |"
          % _pct(lid["primary_match"], lid["primary_eval"]),
          "| Mean language-set Jaccard | %s |" % _f(lid["mean_jaccard"], 3),
          "",
          "| file_id | GT languages | engine primary | engine languages (share>=%d%%) |"
          " primary match | Jaccard |" % int(LANG_SHARE_MIN * 100),
          "|---|---|---|---|---|---|"]
    for f in files:
        li = f["lang_id"]
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            f["file_id"], ",".join(li["gt"]) or "-", li["primary"] or "n/a",
            ",".join(li["engine"]) or "-", _YN[li["primary_match"]],
            _f(li["jaccard"], 2)))
    L.append("")
    spk = eq["speakers"]
    L += ["## Engine quality — speaker count / channel consistency", "",
          "| Metric | Value |", "|---|---|",
          "| Exact speaker-count match | %s |" % _pct(spk["exact"], spk["files"]),
          "| Speaker-count mean absolute error | %s |" % _f(spk["mae"], 2),
          "| Split-channel inconsistencies (split media, speakers != channels) | %d |"
          % spk["split_inconsistent"],
          "",
          "| file_id | GT speakers | engine speakers | channels | split | exact | note |",
          "|---|---|---|---|---|---|---|"]
    for f in files:
        s = f["speakers"]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            f["file_id"],
            "n/a" if s["gt"] is None else s["gt"],
            "n/a" if s["engine"] is None else s["engine"],
            "n/a" if s["channels"] is None else s["channels"],
            "yes" if s["split"] else "no", _YN[s["exact"]], s["note"]))
    L.append("")
    return L


def build_report(cfg, files, agg_counts, agg, breakdowns, eq):
    L = ["# Score report — profile `%s`, engine suffix `%s`" % (cfg["profile"],
                                                                cfg["engine_suffix"]),
         "",
         "Generated %s by `tools/score_run.py` | GT: `%s` (%d file(s)) | predictions: `%s`"
         % (cfg["generated_at"], cfg["gt_glob"], len(files), cfg["pred_root"]),
         "",
         "## Aggregate", "",
         "| Metric | Value |", "|---|---|",
         "| **Leak rate** (must-beep GT < %d%% covered) | %s |"
         % (int(COVERAGE_MIN * 100), _pct(agg_counts["leaked"], agg_counts["must_beep_total"])),
         "| Leaks per audio-hour | %s |" % _f(agg["leaks_per_hour"], 2),
         "| Detection recall (must-beep) | %s |"
         % _pct(agg_counts["matched_must_beep"], agg_counts["must_beep_total"]),
         "| Detection recall (annotate-only) | %s |"
         % _pct(agg_counts["matched_annotate_only"], agg_counts["annotate_only_total"]),
         "| Detection precision | %s |"
         % _pct(agg_counts["pred_matched"], agg_counts["pred_total"]),
         "| Category accuracy (matched pairs) | %s |"
         % _pct(agg_counts["cat_ok"], agg_counts["matched_pairs"]),
         "| Over-redaction | %s s/h |" % _f(agg["over_redaction_s_per_hour"], 1),
         "| False beep events per hour | %s |" % _f(agg["false_beeps_per_hour"], 2),
         "| Timing mean abs start / end delta | %s / %s ms |"
         % (_f(agg["mean_abs_start_ms"], 0), _f(agg["mean_abs_end_ms"], 0)),
         "",
         "## Exclusions (reported, not in the leak denominator)", "",
         "- annotate-only GT intervals under `%s`: %d (of which beeped >=95%% anyway: %d)"
         % (cfg["profile"], agg_counts["annotate_only_total"],
            agg_counts["annotate_only_covered"]),
         "- public-figure GT intervals (annotate-only in every profile): %d "
         "(beeped >=95%% anyway: %d — policy violation if > 0)"
         % (agg_counts["public_figure_total"], agg_counts["public_figure_covered"]),
         "",
         "## Per file", "",
         "| file_id | min | GT | must-beep | leaked | leak rate | recall(mb) | precision |"
         " cat-acc | over s/h | dS ms | dE ms | notes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        c, m = f["counts"], f["metrics"]
        L.append("| %s | %.1f | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            f["file_id"], c["duration_ms"] / 60000.0, c["gt_total"], c["must_beep_total"],
            c["leaked"], _pct(c["leaked"], c["must_beep_total"]),
            _pct(c["matched_must_beep"], c["must_beep_total"]),
            _pct(c["pred_matched"], c["pred_total"]), _pct(c["cat_ok"], c["matched_pairs"]),
            _f(m["over_redaction_s_per_hour"], 1), _f(m["mean_abs_start_ms"], 0),
            _f(m["mean_abs_end_ms"], 0), "; ".join(f["notes"]) or ""))
    L.append("")

    all_leaks = [(f["file_id"], lk) for f in files for lk in f["leaks"]]
    L += ["## Leaks (every must-beep GT interval covered < %d%%)" % int(COVERAGE_MIN * 100), ""]
    if all_leaks:
        L += ["| file_id | annotation_id | categories | interval ms | coverage | spoken_text |",
              "|---|---|---|---|---|---|"]
        for fid, lk in all_leaks:
            L.append("| %s | %s | %s | %d-%d | %.0f%% | %s |" % (
                fid, lk["annotation_id"], ",".join(lk["categories"]),
                lk["start_ms"], lk["end_ms"], 100 * lk["coverage"],
                lk["spoken_text"][:60]))
    else:
        L.append("None.")
    L.append("")
    L += _breakdown_table("category", breakdowns["category"])
    L += _breakdown_table("language", breakdowns["language"])
    L += _breakdown_table("confidence", breakdowns["confidence"])
    L += _engine_quality_sections(files, eq)
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- scoring run

def _load_optional(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run_scoring(args):
    gt_paths = sorted(glob.glob(args.gt))
    if not gt_paths and not Path(args.gt).is_absolute():
        gt_paths = sorted(glob.glob(str(ROOT / args.gt)))   # fallback: glob from project root
    if not gt_paths:
        print("no GT files match %r — nothing to score" % args.gt)
        return 0

    files = []
    for gp in gt_paths:
        gt = json.loads(Path(gp).read_text())
        fid = gt.get("file_id") or Path(gp).name.replace(".gt.json", "")
        pred_dir = Path(args.pred_root) / (fid + args.engine_suffix)
        notes = []
        pred_doc = _load_optional(pred_dir / "intervals.json")
        if pred_doc is None:
            notes.append("intervals.json missing")
        rend_doc = _load_optional(pred_dir / "rendered_intervals.json")
        if rend_doc is None:
            notes.append("rendered_intervals.json missing (all must-beep count as leaked)")
        tr_doc = _load_optional(pred_dir / "transcript.json")
        if tr_doc is None:
            notes.append("transcript.json missing (engine-quality metrics skipped)")
        prep_doc = _load_optional(pred_dir / "prepare.json")
        prep_note = ""
        if prep_doc is None and args.engine_suffix:
            prep_doc = _load_optional(Path(args.pred_root) / fid / "prepare.json")
            if prep_doc is not None:
                prep_note = "prepare.json from base gemini dir"
        if prep_doc is None:
            notes.append("prepare.json missing (no channel context)")
        res = score_file(gt, (pred_doc or {}).get("intervals", []),
                         (rend_doc or {}).get("rendered_merged", []), args.profile)
        res["file_id"] = fid
        res["languages"] = gt.get("languages_observed", [])
        res["notes"] = notes
        res["metrics"] = derive(res["counts"])
        res["span_records"] = span_accuracy_for_file(gt, tr_doc) if tr_doc else []
        res["lang_id"] = lang_id_for_file(gt, tr_doc)
        res["speakers"] = speaker_stats_for_file(
            gt, tr_doc, (prep_doc or {}).get("channel_analysis"), prep_note)
        files.append(res)

    agg_counts = aggregate_counts(files)
    agg = derive(agg_counts)
    bd = breakdowns_for(files)
    eq = aggregate_engine_quality(files)
    cfg = {"generated_at": datetime.date.today().isoformat(), "profile": args.profile,
           "engine_suffix": args.engine_suffix, "gt_glob": args.gt,
           "pred_root": args.pred_root}

    report_path = Path(args.report) if args.report else Path(args.pred_root) / "score_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(cfg, files, agg_counts, agg, bd, eq))
    score_path = report_path.parent / "score.json"
    score_path.write_text(json.dumps({
        "spec_version": "0.2", "config": cfg,
        "thresholds": {"coverage_min": COVERAGE_MIN, "iou_min": IOU_MIN,
                       "gt_cover_min": GT_COVER_MIN, "over_redaction_pad_ms": OVER_PAD_MS,
                       "span_win_pad_ms": SPAN_WIN_PAD_MS,
                       "lang_share_min": LANG_SHARE_MIN},
        "aggregate": {"counts": agg_counts, "metrics": agg},
        "files": [{"file_id": f["file_id"], "notes": f["notes"], "counts": f["counts"],
                   "metrics": f["metrics"], "leaks": f["leaks"]} for f in files],
        "breakdowns": bd,
        "engine_quality": {
            "aggregate": eq,
            "files": [{"file_id": f["file_id"], "span_records": f["span_records"],
                       "lang_id": f["lang_id"], "speakers": f["speakers"]}
                      for f in files]},
    }, indent=2, ensure_ascii=False))

    print("scored %d file(s) | profile=%s | leak rate %s | leaks/h %s | over-redaction %s s/h"
          % (len(files), args.profile, _pct(agg_counts["leaked"], agg_counts["must_beep_total"]),
             _f(agg["leaks_per_hour"], 2), _f(agg["over_redaction_s_per_hour"], 1)))
    print("report -> %s\nscores -> %s" % (report_path, score_path))
    return 0


# ---------------------------------------------------------------- self-test

def _ann(aid, s, e, cats, conf="high", pf=False, channel=None, text="synthetic"):
    return {"annotation_id": aid, "start_ms": s, "end_ms": e, "categories": cats,
            "confidence": conf, "public_figure": pf, "channel": channel,
            "speaker": "S1", "spoken_text": text}


def _gt(anns, duration_ms=60000):
    return {"file_id": "synthetic", "duration_ms": duration_ms,
            "languages_observed": ["en-IN"], "annotations": anns}


def _pred(s, e, cats):
    return {"start_ms": s, "end_ms": e, "categories": cats}


def _r(s, e, channel=None):
    return {"start_ms": s, "end_ms": e, "channel": channel}


def _tword(w, s, e):
    return {"w": w, "start_ms": s, "end_ms": e, "conf": "high"}


def _transcript(words, primary="en-in", languages=None, speakers=("S1",)):
    """Synthetic transcript.json: one utterance per speaker; `words` go to the
    first speaker."""
    utts = [{"speaker": speakers[0], "lang": primary, "words": list(words)}]
    for spk in speakers[1:]:
        utts.append({"speaker": spk, "lang": primary,
                     "words": [_tword("hello", 50000, 50500)]})
    return {"duration_ms": 60000, "primary_language": primary,
            "languages": languages or [{"lang": primary, "word_share": 1.0}],
            "utterances": utts}


def self_test():
    cases = []

    def case(name, ok, detail=""):
        cases.append((name, ok, detail))
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  [%s]" % detail) if (detail and not ok) else ""))

    # 1. perfect match: covered, matched, right category, zero deltas
    r = score_file(_gt([_ann("a1", 1000, 2000, ["PHONE"])]),
                   [_pred(1000, 2000, ["PHONE"])], [_r(750, 2250)], "strict")
    c = r["counts"]
    case("perfect match", c["leaked"] == 0 and c["matched_must_beep"] == 1
         and c["pred_matched"] == 1 and c["cat_ok"] == 1
         and c["sum_abs_start_ms"] == 0 and c["sum_abs_end_ms"] == 0, str(c))

    # 2. leaked: must-beep GT with no rendered coverage at all
    r = score_file(_gt([_ann("a1", 10000, 11000, ["PHONE"])]), [], [], "strict")
    case("leaked (uncovered)", r["counts"]["leaked"] == 1
         and derive(r["counts"])["leak_rate"] == 1.0, str(r["counts"]))

    # 3. partial coverage 90% is still a leak (threshold is 95%)
    r = score_file(_gt([_ann("a1", 1000, 2000, ["PHONE"])]),
                   [_pred(1000, 1900, ["PHONE"])], [_r(1000, 1900)], "strict")
    case("partial coverage 90% leaks", r["counts"]["leaked"] == 1
         and r["leaks"][0]["coverage"] == 0.9, str(r["leaks"]))

    # 4. category mismatch: matched pair (IoU 1.0) but category accuracy 0
    r = score_file(_gt([_ann("a1", 1000, 2000, ["PHONE"])]),
                   [_pred(1000, 2000, ["EMAIL"])], [_r(750, 2250)], "strict")
    c = r["counts"]
    case("category mismatch", c["matched_pairs"] == 1 and c["cat_ok"] == 0
         and c["leaked"] == 0, str(c))

    # 5. over-beep: 3 s of beep far from any GT counts as over-redaction + false beep event
    r = score_file(_gt([_ann("a1", 1000, 2000, ["PHONE"])]),
                   [_pred(1000, 2000, ["PHONE"])],
                   [_r(1000, 2000), _r(30000, 33000)], "strict")
    c = r["counts"]
    m = derive(c)
    case("over-beep 3s", c["over_ms"] == 3000 and c["false_beep_events"] == 1
         and abs(m["over_redaction_s_per_hour"] - 180.0) < 1e-9, str(c))

    # 6. annotate-only exclusion: HEALTH_MEDICAL_INFO under medical_consult never in the
    #    leak denominator, but still counts toward annotate-only detection recall
    gt6 = _gt([_ann("a1", 5000, 6000, ["HEALTH_MEDICAL_INFO"])])
    r = score_file(gt6, [_pred(5000, 6000, ["HEALTH_MEDICAL_INFO"])], [], "medical_consult")
    c = r["counts"]
    r_strict = score_file(gt6, [], [], "strict")
    case("annotate-only exclusion (medical_consult)",
         c["must_beep_total"] == 0 and c["leaked"] == 0 and c["annotate_only_total"] == 1
         and c["matched_annotate_only"] == 1 and derive(c)["leak_rate"] is None
         and r_strict["counts"]["leaked"] == 1, str(c))

    # 7. public_figure exclusion: annotate-only in EVERY profile, never a leak
    r = score_file(_gt([_ann("a1", 5000, 6000, ["FULL_NAME"], pf=True)]), [], [], "strict")
    c = r["counts"]
    case("public_figure exclusion", c["must_beep_total"] == 0 and c["leaked"] == 0
         and c["public_figure_total"] == 1 and c["annotate_only_total"] == 1, str(c))

    # 8. channel matching: beep on the wrong channel leaks; channel None covers everything
    gt8 = _gt([_ann("a1", 1000, 2000, ["PHONE"], channel=0)])
    r_wrong = score_file(gt8, [], [_r(750, 2250, channel=1)], "strict")
    r_none = score_file(gt8, [], [_r(750, 2250, channel=None)], "strict")
    case("channel matching", r_wrong["counts"]["leaked"] == 1
         and r_none["counts"]["leaked"] == 0,
         "wrong=%s none=%s" % (r_wrong["counts"]["leaked"], r_none["counts"]["leaked"]))

    # 9. span WER — number-phrase normalization: GT "thirty thousand six hundred
    #    and thirty five rupees" vs engine "Rs 30 635" -> identical after
    #    normalization (WER 0); the rs/rupees residual only exists when currency
    #    words are NOT stripped, and then differs by exactly those two tokens.
    gt9 = _gt([_ann("a1", 1000, 3000, ["FINANCIAL_INFO"],
                    text="thirty thousand six hundred and thirty five rupees")])
    tr9 = _transcript([_tword("Rs", 1000, 1200), _tword("30", 1300, 1800),
                       _tword("635", 1900, 2600)])
    rec9 = span_accuracy_for_file(gt9, tr9)[0]
    ref_resid = norm_span_tokens("thirty thousand six hundred and thirty five rupees",
                                 strip_currency_words=False)
    hyp_resid = norm_span_tokens("rs 30 635", strip_currency_words=False)
    contains9 = _sublist_in(norm_span_tokens("was rs 30 635 and"),
                            norm_span_tokens(gt9["annotations"][0]["spoken_text"]))
    case("span WER: number phrase (rs/rupees residual stripped as currency)",
         rec9["wer"] == 0.0 and rec9["exact"] and rec9["contains"]
         and not rec9["not_transcribed"]
         and ref_resid == ["30635", "rupees"] and hyp_resid == ["rs", "30635"]
         and contains9,
         "rec=%s ref_resid=%s hyp_resid=%s" % (rec9, ref_resid, hyp_resid))

    # 10. span WER — Devanagari digit chars and digit words normalize to ASCII
    gt10 = _gt([_ann("a1", 1000, 2000, ["PHONE"], text="९८७६५४३२१०")])
    tr10 = _transcript([_tword("9876543210", 1200, 1900)])
    rec10 = span_accuracy_for_file(gt10, tr10)[0]
    case("span WER: Devanagari digits", rec10["wer"] == 0.0 and rec10["exact"]
         and norm_span_tokens("नौ आठ सात") == ["987"],
         "rec=%s words=%s" % (rec10, norm_span_tokens("नौ आठ सात")))

    # 11. span WER — empty window: no engine word near the GT interval
    gt11 = _gt([_ann("a1", 10000, 11000, ["PHONE"], text="nine eight seven")])
    tr11 = _transcript([_tword("hello", 1000, 1500)])
    rec11 = span_accuracy_for_file(gt11, tr11)[0]
    eq11 = aggregate_engine_quality([{"span_records": [rec11], "languages": ["en-IN"]}])
    case("span WER: empty window -> WER 1.0 + not-transcribed tally",
         rec11["wer"] == 1.0 and rec11["not_transcribed"]
         and eq11["span"]["not_transcribed"] == 1, str(rec11))

    # 12. language normalization: eng / en-IN / english / EN-US -> one family
    li = lang_id_for_file(
        {"languages_observed": ["en-IN", "hi"]},
        {"primary_language": "eng",
         "languages": [{"lang": "english", "word_share": 0.9},
                       {"lang": "hindi", "word_share": 0.08},
                       {"lang": "odia", "word_share": 0.01}],
         "utterances": []})
    case("language normalization + ID metrics",
         all(norm_lang(x) == "en" for x in ("eng", "en-IN", "english", "EN-US"))
         and li["primary_match"] is True and li["engine"] == ["en", "hi"]
         and li["jaccard"] == 1.0, str(li))

    # 13. speaker MAE: files with |2-1|=1 and |3-3|=0 -> exact 1/2, MAE 0.5
    sp_a = speaker_stats_for_file({"num_speakers": 2},
                                  _transcript([], speakers=("S1",)),
                                  {"channels": 1, "split": False})
    sp_b = speaker_stats_for_file({"num_speakers": 3},
                                  _transcript([], speakers=("S1", "S2", "S3")),
                                  {"channels": 1, "split": False})
    eq13 = aggregate_engine_quality([{"speakers": sp_a}, {"speakers": sp_b}])
    case("speaker count exact/MAE",
         sp_a["abs_err"] == 1 and sp_b["exact"] is True
         and eq13["speakers"]["exact_match_rate"] == 0.5
         and eq13["speakers"]["mae"] == 0.5, str(eq13["speakers"]))

    # 14. split-channel consistency note: split media, engine speakers != channels
    sp_bad = speaker_stats_for_file({"num_speakers": 2},
                                    _transcript([], speakers=("S1",)),
                                    {"channels": 2, "split": True})
    sp_ok = speaker_stats_for_file({"num_speakers": 2},
                                   _transcript([], speakers=("S1", "S2")),
                                   {"channels": 2, "split": True})
    case("split-channel inconsistency note",
         sp_bad["split_inconsistent"] and "split-channel" in sp_bad["note"]
         and not sp_ok["split_inconsistent"] and sp_ok["note"] == "",
         "bad=%s ok=%s" % (sp_bad, sp_ok))

    # 15. Indic number canonicalization: word-form and digit-form values merge
    got_te1 = norm_span_tokens("పదిహేను వేలు")
    got_te2 = norm_span_tokens(
        "రెండు లక్షల నలభై తొమ్మిది వేల నాలుగు వందల తొంభై ఐదు")
    got_hi = norm_span_tokens("इक्कीस हज़ार")           # with nukta on ज़
    got_bn = norm_span_tokens("পঞ্চাশ হাজার")
    case("Indic number phrases (Telugu/Hindi/Bengali)",
         got_te1 == ["15000"] and got_te2 == ["249495"]
         and got_hi == ["21000"] and got_bn == ["50000"]
         and norm_span_tokens("రెండు వందల ఏడు") == ["207"],
         "te1=%s te2=%s hi=%s bn=%s" % (got_te1, got_te2, got_hi, got_bn))

    # 16. double/triple multipliers + Telugu digit chars
    case("double/triple multipliers + Telugu digit chars",
         norm_span_tokens("double nine eight") == ["998"]
         and norm_span_tokens("triple five zero") == ["5550"]
         and norm_span_tokens("nine double seven okati") == ["9771"]
         and norm_span_tokens("౯౮౭") == ["987"]
         and norm_span_tokens("౨౪ hello ౯") == ["24", "hello", "9"],
         "%s %s %s" % (norm_span_tokens("double nine eight"),
                       norm_span_tokens("nine double seven okati"),
                       norm_span_tokens("౨౪ hello ౯")))

    # 17. English regression: pre-Indic behavior byte-identical
    case("English number phrases unchanged (regression)",
         norm_span_tokens("thirty thousand six hundred and thirty five")
         == ["30635"]
         and norm_span_tokens("nine eight seven six five four three")
         == ["9876543"]
         and norm_span_tokens("two lakh forty nine thousand") == ["249000"]
         and norm_span_tokens("call me and forty") == ["call", "me", "and",
                                                       "40"],
         "%s %s" % (norm_span_tokens("thirty thousand six hundred and thirty"
                                     " five"),
                    norm_span_tokens("call me and forty")))

    failed = [n for n, ok, _ in cases if not ok]
    print("\nself-test: %d/%d passed" % (len(cases) - len(failed), len(cases)))
    return 1 if failed else 0


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", help='glob of GT files, e.g. "annotations/dev/*.gt.json"')
    ap.add_argument("--pred-root", default="output",
                    help="directory holding <file_id><engine-suffix>/ prediction dirs")
    ap.add_argument("--profile", default="strict", choices=sorted(policy.ANNOTATE_ONLY),
                    help="Action Matrix profile defining the must-beep set")
    ap.add_argument("--report", default=None,
                    help="markdown report path (default <pred-root>/score_report.md; "
                         "score.json is written next to it)")
    ap.add_argument("--engine-suffix", default="",
                    help='prediction dir suffix per engine, e.g. "_elevenlabs"')
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic in-memory scoring cases and exit")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.gt:
        ap.error("--gt is required (or use --self-test)")
    sys.exit(run_scoring(args))


if __name__ == "__main__":
    main()
