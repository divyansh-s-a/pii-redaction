#!/usr/bin/env python3
"""Benchmark engine transcription quality against human reference transcripts.

Reference pairs live in media/incoming/beeped_with_transcript/: a beeped mp3
(<base>_beep.mp3) plus a human transcript (<base>.txt or <base>.mp3.txt) whose
lines look like

  [00:00:03.39 - 00:00:04.11] पर्सन १ - I am <NAME REDACTED>.

"<... REDACTED>" marks audio that is a 1 kHz beep; "[Disease]"-style bracket
tags are annotations, not spoken words. Per pair and engine this tool:

  1. prepare (cached): decode the beeped mp3 to <out>/<stem>/work_16k.wav
     (16 kHz mono) and detect 1 kHz beep intervals -> <out>/<stem>/beeps.json.
  2. transcribe (cached): <out>/<stem>__<engine>/transcript.json via the
     engine's transcribe_file (skipped if the file exists; --score-only never
     transcribes).
  3. score: per-utterance WER (non-redacted utterances; engine words with
     midpoints inside a beep interval +/-300 ms are dropped first), global
     WER + CER (redacted-utterance windows +/-300 ms also excluded from the
     hypothesis), script-mix diagnostic (Devanagari / Bengali-Odia / Latin
     char fractions of ref vs hyp), language-ID match (filename token
     HIN/ODI/BEN -> hi/or/bn vs engine primary_language through score_run's
     family mapper) and speaker counts. Token normalization (incl. English
     number-phrase canonicalization, "thirty five" == "35") is reused from
     tools/score_run.py.

Outputs <out>/transcription_bench.md (one table per file + an aggregate table)
and <out>/transcription_bench.json. Scoring runs always exit 0.

Usage:
  python3 tools/transcription_bench.py [--engines gemini,elevenlabs,openai]
      [--pairs-dir media/incoming/beeped_with_transcript]
      [--out-dir output/bench] [--score-only]
  python3 tools/transcription_bench.py --self-test     # synthetic cases +
      real .txt parsing + real beep detection (no API calls); exit 1 on FAIL
"""
import argparse
import datetime
import importlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import audio_utils
import common


def _load_score_run():
    """Import tools/score_run.py by path (tools/ is not a package)."""
    p = Path(__file__).resolve().parent / "score_run.py"
    spec = importlib.util.spec_from_file_location("score_run", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score_run = _load_score_run()

BEEP_PAD_MS = 300            # word-drop / window pad around beeps and utterances
BEEP_FREQ_HZ = 1000
EXPECTED_LANG = {"HIN": "hi", "ODI": "or", "BEN": "bn"}
ENGINE_MODULES = {"gemini": "gemini_transcribe",
                  "elevenlabs": "adapters.elevenlabs_transcribe",
                  "openai": "adapters.openai_transcribe",
                  # scribe_v1: benched 2026-09-03 (third-listener evaluation).
                  # OpenRouter audio candidates screen via
                  # adapters/openrouter_audio.py. gem_*: Gemini-version
                  # candidates for the reference-listener replacement
                  # (decision 2026-09-03: decouple reference ears from the
                  # production transcription models).
                  "el_scribe_v1": "adapters.el_scribe_v1",
                  "gem_25_pro": "adapters.gem_25_pro",
                  "gem_38_flash": "adapters.gem_38_flash",
                  "gem_35_transcribe": "adapters.gem_35_transcribe"}

# ------------------------------------------------------------ reference parsing

TS_LINE = re.compile(
    r"^\s*\[(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,3})\s*-\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,3})\]\s*(.*\S)\s*$")
TAG_RE = re.compile(r"\[[^\][]*\]")          # inline "[Disease Name]" tags
# extra punctuation score_run's normalizer does not strip: danda, double
# danda, em/en dash, ellipsis
_PRE_TRANS = {ord(c): " " for c in "।॥—–…"}


def _ts_ms(h, m, s, frac):
    return ((int(h) * 3600 + int(m) * 60 + int(s)) * 1000
            + int(str(frac).ljust(3, "0")[:3]))


def parse_reference(text):
    """Reference .txt -> (utterances, bad_line_count). Utterance:
    {start_ms, end_ms, speaker, text (tags stripped), redacted}."""
    utts, bad = [], 0
    for line in str(text).splitlines():
        if not line.strip():
            continue
        mo = TS_LINE.match(line)
        if not mo:
            bad += 1
            continue
        start = _ts_ms(*mo.group(1, 2, 3, 4))
        end = _ts_ms(*mo.group(5, 6, 7, 8))
        rest = mo.group(9)
        if " - " in rest:
            speaker, utt_text = rest.split(" - ", 1)
        else:                    # tolerate a missing speaker separator
            speaker, utt_text = "unknown", rest
        utts.append({"start_ms": start, "end_ms": end,
                     "speaker": speaker.strip(),
                     "text": TAG_RE.sub(" ", utt_text).strip(),
                     "redacted": "REDACTED>" in utt_text})
    return utts, bad


def norm_tokens(text):
    """score_run's span-token normalizer (numbers canonicalized, currency and
    punctuation stripped) with Indic sentence punctuation pre-stripped."""
    return score_run.norm_span_tokens(str(text or "").translate(_PRE_TRANS))


# ------------------------------------------------------------ edit distance

def edit_distance(ref, hyp):
    """Unit-cost sequence edit distance — Myers bit-parallel with Python big
    ints, O(len_a * len_b / wordsize)-ish; exact, handles the long global
    ref/hyp sequences a plain O(n*m) DP cannot. Symmetric, any hashables."""
    a, b = list(ref), list(hyp)
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):          # shorter sequence -> pattern bitmask
        a, b = b, a
    m = len(a)
    full = (1 << m) - 1
    peq = {}
    bit = 1
    for c in a:
        peq[c] = peq.get(c, 0) | bit
        bit <<= 1
    pv, mv, score = full, 0, m
    high = 1 << (m - 1)
    for c in b:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (full & ~(xh | pv))
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = mh | (full & ~(xv | ph))
        mv = ph & xv
    return score


# ------------------------------------------------------------ scoring

def flatten_words(transcript):
    words = []
    for u in (transcript or {}).get("utterances", []):
        for w in u.get("words", []):
            words.append((int(w["start_ms"]), int(w["end_ms"]),
                          str(w.get("w", ""))))
    words.sort()
    return words


def _in_any(mid, windows):
    return any(lo <= mid <= hi for lo, hi in windows)


def script_mix(text):
    """Char fractions by script over non-space chars: Devanagari
    (U+0900-097F), Bengali+Odia (U+0980-09FF, U+0B00-0B7F), Latin (A-Za-z)."""
    d = b = l = tot = 0
    for ch in text:
        if ch.isspace():
            continue
        tot += 1
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            d += 1
        elif 0x0980 <= o <= 0x09FF or 0x0B00 <= o <= 0x0B7F:
            b += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            l += 1
    if not tot:
        return {"devanagari": 0.0, "bengali_odia": 0.0, "latin": 0.0,
                "chars": 0}
    return {"devanagari": round(d / tot, 3), "bengali_odia": round(b / tot, 3),
            "latin": round(l / tot, 3), "chars": tot}


def score_pair(ref_utts, transcript, beep_intervals, expected_lang):
    """All metrics for one (reference, engine transcript) pair."""
    pad = BEEP_PAD_MS
    beep_win = [(s - pad, e + pad) for s, e in beep_intervals]

    words = flatten_words(transcript)
    kept, dropped_beep = [], 0
    for s, e, w in words:
        if _in_any((s + e) / 2.0, beep_win):
            dropped_beep += 1
        else:
            kept.append((s, e, w))

    scored = sorted((u for u in ref_utts if not u["redacted"]),
                    key=lambda u: (u["start_ms"], u["end_ms"]))
    red_win = [(u["start_ms"] - pad, u["end_ms"] + pad)
               for u in ref_utts if u["redacted"]]

    # per-utterance WER
    wers = []
    for u in scored:
        ref_toks = norm_tokens(u["text"])
        lo, hi = u["start_ms"] - pad, u["end_ms"] + pad
        hyp_toks = norm_tokens(" ".join(
            w for s, e, w in kept if lo <= (s + e) / 2.0 <= hi))
        wers.append(edit_distance(ref_toks, hyp_toks)
                    / float(max(1, len(ref_toks))))
    n_wer0 = sum(1 for x in wers if x <= 1e-9)

    # global WER / CER: hypothesis additionally excludes redacted windows
    ref_join = " ".join(u["text"] for u in scored)
    ref_g = norm_tokens(ref_join)
    hyp_words = [w for s, e, w in kept
                 if not _in_any((s + e) / 2.0, red_win)]
    excluded_red = len(kept) - len(hyp_words)
    hyp_join = " ".join(hyp_words)
    hyp_g = norm_tokens(hyp_join)
    global_wer = edit_distance(ref_g, hyp_g) / float(max(1, len(ref_g)))
    ref_str, hyp_str = " ".join(ref_g), " ".join(hyp_g)
    global_cer = edit_distance(ref_str, hyp_str) / float(max(1, len(ref_str)))

    primary = str((transcript or {}).get("primary_language", "unknown"))
    got = score_run.norm_lang(primary)
    ref_speakers = sorted({u["speaker"] for u in ref_utts})
    eng_speakers = sorted({str(u.get("speaker", "unknown"))
                           for u in (transcript or {}).get("utterances", [])})

    return {
        "utts_scored": len(scored),
        "utts_skipped_redacted": len(ref_utts) - len(scored),
        "mean_utt_wer": round(sum(wers) / len(wers), 4) if wers else None,
        "pct_utt_wer0": round(n_wer0 / float(len(wers)), 4) if wers else None,
        "global_wer": round(global_wer, 4),
        "global_cer": round(global_cer, 4),
        "ref_tokens": len(ref_g), "hyp_tokens": len(hyp_g),
        "script_ref": script_mix(ref_join),
        "script_hyp": script_mix(hyp_join),
        "lang_expected": expected_lang,
        "lang_engine_primary": primary,
        "lang_engine_norm": got,
        "lang_match": bool(got == expected_lang),
        "ref_speaker_count": len(ref_speakers),
        "eng_speaker_count": len(eng_speakers),
        "ref_speakers": ref_speakers,
        "eng_speakers": eng_speakers,
        "engine_words_total": len(words),
        "words_dropped_beep": dropped_beep,
        "words_excluded_redacted": excluded_red,
        "hyp_words": len(hyp_words),
    }


# ------------------------------------------------------------ pipeline steps

def discover_pairs(pairs_dir):
    """[(stem, mp3_path, txt_path)] for every mp3 with a matching reference."""
    pairs = []
    for mp3 in sorted(pairs_dir.glob("*.mp3")):
        stem = mp3.stem
        base = stem[:-5] if stem.endswith("_beep") else stem
        cands = [pairs_dir / (base + ".txt"), pairs_dir / (base + ".mp3.txt"),
                 pairs_dir / (stem + ".txt")]
        txt = next((c for c in cands if c.exists()), None)
        if txt is None:
            near = sorted(p for p in pairs_dir.glob("*.txt")
                          if p.name.startswith(base))
            txt = near[0] if near else None
        if txt is None:
            print("  WARNING: no reference .txt for %s — skipped" % mp3.name)
            continue
        pairs.append((stem, mp3, txt))
    return pairs


def expected_lang_of(stem):
    mo = re.search(r"_(HIN|ODI|BEN)_", stem)
    return EXPECTED_LANG.get(mo.group(1)) if mo else "unknown"


def prepare_pair(stem, mp3, out_dir):
    """Cached decode + beep detection. Returns (work_wav, [(s_ms, e_ms)])."""
    d = out_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    wav = d / "work_16k.wav"
    if not wav.exists():
        print("  decoding %s -> %s ..." % (mp3.name, wav))
        audio_utils.decode_to_wav(mp3, wav, rate=16000, mono=True)
    bj = d / "beeps.json"
    if bj.exists():
        ivs = json.loads(bj.read_text())["intervals"]
    else:
        print("  detecting %d Hz beeps in %s ..." % (BEEP_FREQ_HZ, wav.name))
        ivs = audio_utils.find_pure_tones(wav, freq=BEEP_FREQ_HZ)
        common.save_json(bj, {"freq_hz": BEEP_FREQ_HZ,
                              "intervals": [[int(s), int(e)] for s, e in ivs]})
    return wav, [(int(s), int(e)) for s, e in ivs]


_GEMINI_MODEL = []


def transcribe_pair(engine, mp3, work_wav, out_json):
    mod = importlib.import_module(ENGINE_MODULES[engine])
    model = None
    if engine == "gemini":
        if not _GEMINI_MODEL:
            _GEMINI_MODEL.append(common.resolve_model())
        model = _GEMINI_MODEL[0]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    mod.transcribe_file(model, mp3, work_wav, out_json, channel_wavs=None)


# ------------------------------------------------------------ aggregation

_MEAN_KEYS = ["mean_utt_wer", "pct_utt_wer0", "global_wer", "global_cer"]


def aggregate(files, engines):
    agg = {}
    for eng in engines:
        recs = [f["engines"][eng] for f in files
                if isinstance(f["engines"].get(eng), dict)
                and "global_wer" in f["engines"][eng]]
        a = {"files_scored": len(recs)}
        for k in _MEAN_KEYS:
            vals = [r[k] for r in recs if r.get(k) is not None]
            a[k] = round(sum(vals) / len(vals), 4) if vals else None
        a["lang_match"] = sum(1 for r in recs if r["lang_match"])
        a["speaker_exact"] = sum(
            1 for r in recs if r["eng_speaker_count"] == r["ref_speaker_count"])
        a["words_dropped_beep"] = sum(r["words_dropped_beep"] for r in recs)
        agg[eng] = a
    return agg


# ------------------------------------------------------------ report

def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    return ("%." + str(nd) + "f") % x


def _mix(m):
    return "%.2f/%.2f/%.2f" % (m["devanagari"], m["bengali_odia"], m["latin"])


def build_report(files, agg, engines, cfg):
    L = ["# Transcription benchmark — engines vs human reference transcripts",
         "",
         "Generated %s by `tools/transcription_bench.py` | pairs: `%s` | "
         "engines: %s" % (cfg["generated_at"], cfg["pairs_dir"],
                          ", ".join(engines)),
         "",
         "Engine words inside a detected 1 kHz beep interval +/-%d ms are "
         "dropped before scoring; reference utterances containing a "
         "`<... REDACTED>` marker are skipped (and their windows +/-%d ms "
         "excluded from the global hypothesis). Tokens are normalized with "
         "`score_run.norm_span_tokens` (numbers canonicalized). Script mix "
         "columns are Devanagari/Bengali-Odia/Latin char fractions."
         % (BEEP_PAD_MS, BEEP_PAD_MS), ""]
    hdr = ("| engine | mean utt WER | %utt WER=0 | global WER | global CER "
           "| script ref d/b/l | script hyp d/b/l | lang exp->got | match "
           "| spk ref/eng | utts scored/redacted | words kept/beep/redact |")
    sep = "|" + "---|" * 12
    for f in files:
        L += ["## %s" % f["stem"], "",
              "reference `%s` — %d utterances (%d redacted), %d speakers, "
              "%d beeps detected, expected language `%s`%s"
              % (f["ref_txt"], f["ref_utterances"], f["ref_redacted"],
                 len(f["ref_speakers"]), f["beep_count"], f["expected_lang"],
                 (", %d unparsed lines" % f["bad_lines"])
                 if f["bad_lines"] else ""),
              "", hdr, sep]
        for eng in engines:
            r = f["engines"].get(eng)
            if not isinstance(r, dict) or "global_wer" not in r:
                note = (r or {}).get("note", "not scored")
                L.append("| %s | %s |" % (eng, note) + " |" * 10)
                continue
            L.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s->%s | %s | %d/%d "
                "| %d/%d | %d/%d/%d |" % (
                    eng, _fmt(r["mean_utt_wer"]), _fmt(r["pct_utt_wer0"]),
                    _fmt(r["global_wer"]), _fmt(r["global_cer"]),
                    _mix(r["script_ref"]), _mix(r["script_hyp"]),
                    r["lang_expected"], r["lang_engine_norm"],
                    "yes" if r["lang_match"] else "NO",
                    r["ref_speaker_count"], r["eng_speaker_count"],
                    r["utts_scored"], r["utts_skipped_redacted"],
                    r["hyp_words"], r["words_dropped_beep"],
                    r["words_excluded_redacted"]))
        L.append("")
    L += ["## Aggregate (mean over scored files)", "",
          "| engine | files | mean utt WER | %utt WER=0 | global WER "
          "| global CER | lang match | speaker exact | words beep-dropped |",
          "|" + "---|" * 9]
    for eng in engines:
        a = agg[eng]
        n = a["files_scored"]
        L.append("| %s | %d | %s | %s | %s | %s | %d/%d | %d/%d | %d |" % (
            eng, n, _fmt(a["mean_utt_wer"]), _fmt(a["pct_utt_wer0"]),
            _fmt(a["global_wer"]), _fmt(a["global_cer"]),
            a["lang_match"], n, a["speaker_exact"], n,
            a["words_dropped_beep"]))
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------ bench run

def run_bench(args):
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_absolute():
        pairs_dir = ROOT / pairs_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINE_MODULES]
    if unknown:
        raise SystemExit("unknown engine(s) %s — choose from %s"
                         % (unknown, sorted(ENGINE_MODULES)))

    pairs = discover_pairs(pairs_dir)
    if not pairs:
        print("no (mp3, txt) pairs found in %s — nothing to do" % pairs_dir)
        return 0

    files = []
    for stem, mp3, txt in pairs:
        print("pair %s" % stem)
        ref_utts, bad = parse_reference(txt.read_text())
        wav, beeps = prepare_pair(stem, mp3, out_dir)
        rec = {"stem": stem, "audio": mp3.name, "ref_txt": txt.name,
               "expected_lang": expected_lang_of(stem),
               "beep_count": len(beeps), "bad_lines": bad,
               "ref_utterances": len(ref_utts),
               "ref_redacted": sum(1 for u in ref_utts if u["redacted"]),
               "ref_speakers": sorted({u["speaker"] for u in ref_utts}),
               "engines": {}}
        for eng in engines:
            tj = out_dir / ("%s__%s" % (stem, eng)) / "transcript.json"
            if not tj.exists():
                if args.score_only:
                    rec["engines"][eng] = {
                        "note": "transcript.json missing (score-only)"}
                    print("  %s: transcript missing, skipped (score-only)"
                          % eng)
                    continue
                print("  %s: transcribing ..." % eng)
                try:
                    transcribe_pair(eng, mp3, wav, tj)
                except (Exception, SystemExit) as e:
                    rec["engines"][eng] = {
                        "note": "transcription failed: %s" % e}
                    print("  %s: transcription failed: %s" % (eng, e))
                    continue
            transcript = json.loads(tj.read_text())
            print("  %s: scoring ..." % eng)
            rec["engines"][eng] = score_pair(
                ref_utts, transcript, beeps, rec["expected_lang"])
        files.append(rec)

    agg = aggregate(files, engines)
    cfg = {"generated_at": datetime.date.today().isoformat(),
           "pairs_dir": str(pairs_dir), "out_dir": str(out_dir),
           "engines": engines, "beep_pad_ms": BEEP_PAD_MS,
           "beep_freq_hz": BEEP_FREQ_HZ}
    md = out_dir / "transcription_bench.md"
    md.write_text(build_report(files, agg, engines, cfg))
    common.save_json(out_dir / "transcription_bench.json",
                     {"config": cfg, "files": files, "aggregate": agg})
    print("report -> %s\nscores -> %s" % (md, out_dir /
                                          "transcription_bench.json"))
    return 0


# ------------------------------------------------------------ self-test

def _tw(w, s, e):
    return {"w": w, "start_ms": s, "end_ms": e, "conf": "high"}


_SYN_REF = "\n".join([
    "[00:00:01.00 - 00:00:02.00] डॉक्टर - Hello there",
    "[00:00:03.00 - 00:00:04.00] पर्सन १ - I am <NAME REDACTED>.",
    "[00:00:05.00 - 00:00:06.50] डॉक्टर - fever [Disease] since two days",
    "[00:00:07.00 - 00:00:08.00] पर्सन १ - thirty five tablets",
])


def _syn_transcript(primary="hin"):
    return {"duration_ms": 10000, "primary_language": primary,
            "languages": [{"lang": primary, "word_share": 1.0}],
            "utterances": [
                {"speaker": "S1", "lang": primary,
                 "words": [_tw("Hello", 1000, 1400), _tw("there", 1500, 1900)]},
                {"speaker": "S2", "lang": primary,
                 "words": [_tw("secretname", 3200, 3600)]},
                {"speaker": "S1", "lang": primary,
                 "words": [_tw("fever", 5000, 5300), _tw("since", 5400, 5600),
                           _tw("two", 5700, 5800), _tw("days", 5900, 6100),
                           _tw("35", 7100, 7400), _tw("tablets", 7500, 7900),
                           _tw("beepword", 9000, 9200)]},
            ]}


def self_test(args):
    cases = []

    def case(name, ok, detail=""):
        cases.append((name, ok))
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  [%s]" % detail) if (detail and not ok) else ""))

    utts, bad = parse_reference(_SYN_REF)
    tr = _syn_transcript()
    beeps = [(8900, 9300)]                      # covers "beepword" @ 9000-9200
    r = score_pair(utts, tr, beeps, "hi")

    case("reference parsing (times/speakers/lines)",
         bad == 0 and len(utts) == 4 and utts[0]["start_ms"] == 1000
         and utts[0]["end_ms"] == 2000 and utts[0]["speaker"] == "डॉक्टर"
         and utts[1]["speaker"] == "पर्सन १",
         "bad=%d utts=%s" % (bad, utts))

    case("perfect match -> WER 0 (utterance + global + CER)",
         r["mean_utt_wer"] == 0.0 and r["pct_utt_wer0"] == 1.0
         and r["global_wer"] == 0.0 and r["global_cer"] == 0.0, str(r))

    case("redacted utterance skipped + its window excluded from hypothesis",
         utts[1]["redacted"] and r["utts_scored"] == 3
         and r["utts_skipped_redacted"] == 1
         and r["words_excluded_redacted"] == 1, str(r))

    case("annotation tag stripped from reference text",
         # "two" -> "2" is the number canonicalization, also applied here
         norm_tokens(utts[2]["text"]) == ["fever", "since", "2", "days"]
         and "disease" not in [t.lower() for t in norm_tokens(utts[2]["text"])],
         str(norm_tokens(utts[2]["text"])))

    case("beep-masked engine word dropped",
         r["words_dropped_beep"] == 1 and r["global_wer"] == 0.0, str(r))

    case("number canonicalization: 'thirty five' == '35'",
         norm_tokens("thirty five tablets") == ["35", "tablets"]
         and edit_distance(norm_tokens("thirty five"),
                           norm_tokens("35")) == 0,
         str(norm_tokens("thirty five tablets")))

    mix = script_mix("नमस्ते hello")            # 6 Devanagari + 5 Latin chars
    mix_b = script_mix("নমস্কার")
    case("script-mix fractions",
         mix["devanagari"] == round(6 / 11.0, 3)
         and mix["latin"] == round(5 / 11.0, 3)
         and mix["bengali_odia"] == 0.0 and mix_b["bengali_odia"] == 1.0,
         "mix=%s mix_b=%s" % (mix, mix_b))

    r_mis = score_pair(utts, _syn_transcript(primary="english"),
                       beeps, "hi")
    case("language mapping: HIN->hi match and mismatch",
         EXPECTED_LANG == {"HIN": "hi", "ODI": "or", "BEN": "bn"}
         and expected_lang_of("202510232017_HIN_001_beep") == "hi"
         and r["lang_engine_norm"] == "hi" and r["lang_match"] is True
         and r_mis["lang_engine_norm"] == "en" and r_mis["lang_match"] is False,
         "match=%s mis=%s" % (r["lang_match"], r_mis["lang_match"]))

    case("speaker counts (ref vs engine)",
         r["ref_speaker_count"] == 2 and r["eng_speaker_count"] == 2
         and r["ref_speakers"] == ["डॉक्टर", "पर्सन १"], str(r["ref_speakers"]))

    rng = random.Random(42)
    ok_ed = True
    for _ in range(60):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        if edit_distance(a, b) != score_run.word_edit_distance(a, b):
            ok_ed = False
            break
    case("edit distance (Myers) == score_run DP on 60 random pairs", ok_ed)

    # ---- real-data validation: parsing + beep detection, no API calls
    print("\nreal-data validation (no API):")
    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.is_absolute():
        pairs_dir = ROOT / pairs_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    pairs = discover_pairs(pairs_dir) if pairs_dir.exists() else []
    if not pairs:
        print("  (pairs dir %s empty/missing — real validation skipped)"
              % pairs_dir)
    for stem, mp3, txt in pairs:
        utts_r, bad_r = parse_reference(txt.read_text())
        red = sum(1 for u in utts_r if u["redacted"])
        spk = sorted({u["speaker"] for u in utts_r})
        print("  %s: %d utterances (%d redacted, %d unparsed lines), "
              "speakers %s" % (txt.name, len(utts_r), red, bad_r, spk))
        case("real parse %s" % txt.name,
             len(utts_r) > 10 and bad_r == 0 and 1 <= len(spk) <= 6,
             "utts=%d bad=%d spk=%s" % (len(utts_r), bad_r, spk))
        _, beeps_r = prepare_pair(stem, mp3, out_dir)
        print("    %s: %d beeps at %d Hz: %s"
              % (mp3.name, len(beeps_r), BEEP_FREQ_HZ,
                 [(s, e) for s, e in beeps_r[:15]]))
        case("real beeps %s" % mp3.name, 1 <= len(beeps_r) <= 50,
             "beeps=%d" % len(beeps_r))

    failed = [n for n, ok in cases if not ok]
    print("\nself-test: %d/%d passed" % (len(cases) - len(failed), len(cases)))
    return 1 if failed else 0


# ------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engines", default="gemini,elevenlabs,openai",
                    help="comma list from: %s" % ",".join(sorted(ENGINE_MODULES)))
    ap.add_argument("--pairs-dir",
                    default="media/incoming/beeped_with_transcript",
                    help="directory of (beeped mp3, reference .txt) pairs")
    ap.add_argument("--out-dir", default="output/bench",
                    help="work + report directory")
    ap.add_argument("--score-only", action="store_true",
                    help="never call transcription APIs; score existing "
                         "transcript.json files only")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic scoring cases + real .txt parsing + real "
                         "beep detection (no APIs); exit 1 on FAIL")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test(args))
    sys.exit(run_bench(args))


if __name__ == "__main__":
    main()
