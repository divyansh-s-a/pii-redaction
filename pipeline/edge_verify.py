#!/usr/bin/env python3
"""Empirical beep-edge verification (defect class 4 defense).

Engines can timestamp a value's onset late (observed: ElevenLabs placing a
name ~300 ms after it actually starts), so a beep built from those times clips
the first syllable — inaudible to any transcript-side check because the
transcript already contains the full word. The only trustworthy referee is the
AUDIO ITSELF: this module cuts the seconds around every rendered beep edge
from the BEEPED file (covered speech is already tone, so only genuinely
audible leftovers survive), transcribes the snippets, and fuzzy-matches the
heard tokens against the beeped values. A match extends that beep edge and the
audio is re-rendered; repeat up to MAX_ROUNDS.
"""
import difflib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_utils
import common
import gemini_transcribe
from textnorm import fold_indic

SNIP_MS = 2000          # audible context inspected on each side of an edge
EXTEND_MS = 400         # how far a flagged edge grows per round
EDGE_ADJ_MS = 500       # a fragment only counts within this of the boundary
MAX_ROUNDS = 2

# Snippet ASR: a 2-second boundary clip needs neither the flagship model nor
# deep reasoning — observed ~63k thinking tokens per snippet on 3.1 Pro with
# the fragment hint (minutes of latency, real cost). Flash with a small
# thinking budget transcribes fragments fine; the planted-defect tests are
# the empirical check that this stays sensitive enough.
SNIPPET_MODEL = "gemini-3.5-flash"
SNIPPET_THINKING_BUDGET = 256

# The snippet cut ends EXACTLY at the beep boundary (no reach into the tone):
# a 100 ms slice of loud 1 kHz tone right after a clipped 150 ms fragment was
# observed to make the ASR drop the fragment entirely (planted-defect test,
# 2026-09-02). The prompt below additionally tells the ASR that edge
# fragments are expected and must be written as heard.
SNIPPET_HINT = (
    "NOTE: this audio is a SHORT EXCERPT cut from a longer recording and may "
    "begin or end ABRUPTLY IN THE MIDDLE OF A WORD. Transcribe exactly what "
    "is audible, INCLUDING any partial word fragment at the cut points — "
    "write the fragment as heard (e.g. 'exam' if 'example' is cut off). "
    "Never omit a trailing or leading fragment, never complete it into a "
    "full word you did not hear.")

# Function/filler words never flag an edge: they fuzzy-match halves of many
# values ("tell"~"Shailesh" at ratio 0.5) and legitimately neighbor every beep,
# so matching them makes the loop extend forever without protecting anything.
STOPWORDS = {
    "this", "that", "these", "those", "yeah", "yes", "okay", "okey", "tell",
    "sir", "madam", "maam", "can", "could", "will", "would", "shall", "from",
    "the", "and", "you", "your", "our", "was", "were", "are", "is", "am",
    "not", "but", "for", "with", "have", "has", "had", "just", "like", "what",
    "when", "then", "they", "them", "there", "here", "out", "get", "got",
    "right", "please", "thanks", "thank", "hello", "hmm", "huh", "haan", "ji",
    "acha", "accha", "theek", "hai", "nahi", "aur", "toh", "kya", "mein",
    "paid", "call", "calling", "speaking", "name",
}


def _norm(tok):
    return fold_indic(str(tok).strip().lower().strip(".,!?;:\"'()")).lower()


def _unexpected(heard, expected_norms):
    """True when a content word heard at the beep boundary matches NO word the
    transcript expects to be audible there. This is the PRIMARY leak signal:
    two ASRs render the same clipped Indian name differently (observed:
    listeners wrote 'Ritika', the snippet ASR heard the exposed audio as
    'Vidya'), so similarity-to-the-value cannot be relied on — but a leaked
    fragment never matches the expected NEIGHBOUR words either.
    Match threshold 0.55: the snippet ASR renders neighbours across scripts
    ('Okay' came back as Devanagari ओके -> folded 'oke', ratio vs 'okay' is
    0.571), while real leaked fragments score far lower (~0.3) — the 0.55-0.6
    band belongs to script/romanization variance, not to leaks."""
    h = _norm(heard)
    if len(h) < 3 or h in STOPWORDS or any(ch.isdigit() for ch in h):
        return False        # digits are _fragment_match's job; short/function
    for e in expected_norms:
        if h == e or difflib.SequenceMatcher(None, h, e).ratio() >= 0.55:
            return False
    return True


def _value_tokens(intervals, s, e, tol_ms=500):
    """Tokens of every detected value whose interval touches beep [s,e],
    with tol_ms of slack: a skewed interval sitting just outside a beep edge
    is exactly the geometry this module exists to catch, so its value must
    still contribute tokens."""
    toks = set()
    for iv in intervals:
        if iv["start_ms"] <= e + tol_ms and iv["end_ms"] >= s - tol_ms:
            for t in str(iv.get("spoken_text", "")).split():
                n = _norm(t)
                if len(n) >= 2:
                    toks.add(n)
    return toks


def _fragment_match(heard, value_toks):
    """A heard token betrays the value if it IS a value token, fuzzy-matches
    one (clipped names transcribe as similar-sounding fragments), or is a
    digit run (numbers at edges are always suspicious). Function words and
    short tokens never count — they match everything and converge never."""
    h = _norm(heard)
    if len(h) < 2:
        return False
    if any(ch.isdigit() for ch in h):
        return True
    if len(h) < 3 or h in STOPWORDS:
        return False
    for v in value_toks:
        if h == v:
            return True
        if difflib.SequenceMatcher(None, h, v).ratio() >= 0.66:
            return True
    return False


def verify_and_fix(out_dir: Path, fid: str, original: Path, model: str,
                   freq: int = 1000, gain_db: float = -10.0):
    """Check every beep edge; extend + re-render until clean or MAX_ROUNDS.
    Returns a report dict for the manifest."""
    intervals = common.load_json(out_dir / "intervals.json")["intervals"]
    report = {"rounds": 0, "extensions": [], "clean": True}
    tmp = out_dir / "_edge_snip.wav"
    common.set_usage_context(f"edge_verify:{fid}")

    # transcript words power the expected-audible check; without a transcript
    # the verifier falls back to value/digit matching only
    twords = []
    tpath = out_dir / "transcript.json"
    if tpath.exists():
        for u in common.load_json(tpath).get("utterances", []):
            for w in u["words"]:
                twords.append((str(w["w"]), int(w["start_ms"]),
                               int(w["end_ms"])))

    def _in_interval(ws, we):
        mid = (ws + we) / 2.0
        return any(iv["start_ms"] <= mid <= iv["end_ms"] for iv in intervals)

    def expected_norms(a, b, beeps):
        """Normalized non-PII transcript words audible (>= 60 ms outside every
        rendered beep) inside snippet [a, b]."""
        out = set()
        for tok, ws, we in twords:
            if _in_interval(ws, we):
                continue
            s0, e0 = max(ws, a), min(we, b)
            if e0 - s0 <= 0:
                continue
            covered = sum(max(0, min(e0, y) - max(s0, x)) for x, y in beeps)
            if (e0 - s0) - covered >= 60:
                n = _norm(tok)
                if n:
                    out.add(n)
        return out

    for rnd in range(MAX_ROUNDS + 1):
        chk = out_dir / "_edge_chk.wav"
        audio_utils.decode_to_wav(out_dir / f"{fid}_beeped.m4a", chk,
                                  rate=16000, mono=True)
        beeps = audio_utils.find_pure_tones(chk, freq)
        dur = audio_utils.duration_ms(chk)
        flagged = []      # (beep_index, side)
        report["unchecked"] = []      # edges the snippet ASR failed on
        for bi, (s, e) in enumerate(beeps):
            vtoks = _value_tokens(intervals, s, e)
            if not vtoks:
                continue
            for side, a, b in (("start", max(0, s - SNIP_MS), s),
                               ("end", e, min(dur, e + SNIP_MS))):
                if b - a < 300:
                    continue
                exp = expected_norms(a, b, beeps) if twords else None
                # snippet-relative tone spans (OTHER beeps can sit inside the
                # snippet): anything the ASR "hears" on pure tone is a
                # hallucination, not a leak — observed 'zero' transcribed
                # from the 1 kHz tone itself
                tone_rel = [(max(x, a) - a, min(y, b) - a)
                            for x, y in beeps if x < b and y > a]

                def _on_tone(ws, we):
                    cov = sum(max(0, min(we, y) - max(ws, x))
                              for x, y in tone_rel)
                    return cov / max(1, we - ws) >= 0.5

                def listen_once():
                    """Transcribe the snippet once -> (hits, error).
                    hits None = transcription failed after retries."""
                    gemini_transcribe.cut_wav_chunk(chk, tmp, a / 1000,
                                                    (b - a) / 1000)
                    res, err = None, None
                    for attempt in range(3):
                        try:
                            res = gemini_transcribe.transcribe_bytes(
                                SNIPPET_MODEL, tmp, "audio/wav",
                                extra_text=SNIPPET_HINT,
                                thinking_budget=SNIPPET_THINKING_BUDGET)
                            break
                        except Exception as ex:
                            err = ex
                            print(f"    edge-check snippet error, attempt "
                                  f"{attempt + 1}/3 ({str(ex)[:60]})")
                            time.sleep(3 * (attempt + 1))
                    if res is None:
                        return None, err
                    toks = [(str(w.get("w", "")), int(w.get("start_ms", 0)),
                             int(w.get("end_ms", 0)))
                            for u in res.get("utterances", [])
                            for w in u.get("words", [])]
                    # only tokens whose OWN timestamps sit at the beep
                    # boundary matter: the snippet is cut to END exactly at a
                    # beep start (resp. BEGIN at a beep end), so the boundary
                    # is the snippet edge itself — and tokens sitting on tone
                    # are discarded as hallucinations
                    if side == "start":
                        boundary = b - a       # snippet end == beep start
                        near = [(w, ws, we) for w, ws, we in toks
                                if we >= boundary - EDGE_ADJ_MS]
                    else:
                        near = [(w, ws, we) for w, ws, we in toks
                                if ws <= EDGE_ADJ_MS]
                    found = []
                    for w, ws, we in near:
                        if _on_tone(ws, we):
                            continue
                        if _fragment_match(w, vtoks):
                            found.append(w)
                        elif exp is not None and _unexpected(w, exp):
                            found.append(w + " (unexpected at edge)")
                    return found, None

                hits, err = listen_once()
                if hits is None:
                    # a failed snippet is a HOLE in a safety check, not a
                    # skip: record the edge as unverified
                    report["unchecked"].append(
                        {"beep": [s, e], "side": side,
                         "error": str(err)[:100]})
                    print(f"    !! edge {s}-{e}ms ({side}) UNCHECKED after "
                          "retries — recorded in report")
                    continue
                if hits and not any(any(ch.isdigit() for ch in h)
                                    for h in hits):
                    # ASR hallucinations are unstable across listens; real
                    # leaked fragments persist. Confirm non-digit flags with
                    # a SECOND independent listen — digits always flag hard.
                    hits2, _ = listen_once()
                    if hits2 is not None and not hits2:
                        print(f"    edge {s}-{e}ms ({side}): flag "
                              f"{hits} NOT confirmed on re-listen — dropped")
                        hits = []
                if hits:
                    flagged.append((bi, side, hits))
        chk.unlink(missing_ok=True)
        if not flagged:
            report["clean"] = True
            break
        if rnd == MAX_ROUNDS:
            report["clean"] = False
            report["unresolved"] = [
                {"beep": beeps[bi], "side": side, "heard": hits}
                for bi, side, hits in flagged]
            break
        # extend the flagged edges and re-render from the ORIGINAL audio
        report["rounds"] = rnd + 1
        adjusted = []
        for bi, (s, e) in enumerate(beeps):
            ds = EXTEND_MS if any(b == bi and sd == "start"
                                  for b, sd, _ in flagged) else 0
            de = EXTEND_MS if any(b == bi and sd == "end"
                                  for b, sd, _ in flagged) else 0
            adjusted.append((max(0, s - ds), e + de, None))
            if ds or de:
                report["extensions"].append(
                    {"beep": [s, e], "grew": [ds, de],
                     "heard": [h for b, sd, hs in flagged if b == bi
                               for h in hs]})
                print(f"    edge-verify: extending beep {s}-{e}ms "
                      f"(-{ds}/+{de}) — heard fragment near edge")
        # extensions can make adjacent beeps touch; the physical audio then
        # carries ONE continuous tone, so the written plan must merge them
        # too or QC's orphan check sees a tone that matches no planned beep
        # (observed 2026-09-03: two +400ms end-extensions bridged the name
        # and org beeps and QC quarantined a clean file)
        adjusted.sort(key=lambda x: (x[0], x[1]))
        merged_adj = []
        for s2, e2, c2 in adjusted:
            if (merged_adj and c2 == merged_adj[-1][2]
                    and s2 <= merged_adj[-1][1]):
                merged_adj[-1][1] = max(merged_adj[-1][1], e2)
            else:
                merged_adj.append([s2, e2, c2])
        adjusted = [(s2, e2, c2) for s2, e2, c2 in merged_adj]
        native = out_dir / "_ev_native.wav"
        beeped = out_dir / "_ev_beeped.wav"
        audio_utils.decode_to_wav(original, native)
        audio_utils.mix_beeps(native, beeped, adjusted, freq, gain_db)
        audio_utils.encode_wav(beeped, out_dir / f"{fid}_beeped.m4a")
        native.unlink(missing_ok=True)
        beeped.unlink(missing_ok=True)
        # keep rendered_intervals.json truthful
        ri = common.load_json(out_dir / "rendered_intervals.json")
        ri["rendered_merged"] = [{"start_ms": s, "end_ms": e, "channel": c}
                                 for s, e, c in adjusted]
        ri["edge_verified"] = True
        common.save_json(out_dir / "rendered_intervals.json", ri)
    tmp.unlink(missing_ok=True)
    common.set_usage_context("")
    return report
