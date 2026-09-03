#!/usr/bin/env python3
"""Stage D — render the beeped audio AND the redacted transcript from the SAME
intervals.json (single source of truth; audio and transcript cannot disagree).

Audio path: afconvert (macOS) or ffmpeg (other platforms) + pure-Python mixing.
Beeps REPLACE speech with a 1 kHz tone, matching the incumbent's output style.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_utils
import common
import policy

WORD_GRAZE_MS = 20      # privacy floor against a transcribed neighbour word
# Person-name onsets are the empirically worst-stamped edge (defect class 4:
# engines place a name's start late — ~300 ms observed there, and confirmed
# by ear-calibration on the shared sample 2026-09-02: a beep 150 ms before
# the stamped onset still leaked "Vi/Vid", while 300 ms before it was clean
# to the listener). Sub-150 ms onset exposures are BELOW the edge verifier's
# demonstrated floor, so the render must carry this margin itself. Name
# intervals therefore keep a hard 300 ms start-side floor, even when it eats
# the preceding word(s) — losing "This is" is nothing next to leaking a
# name's first syllable.
ONSET_GUARD_MS = 300
NAME_CATS = {"FULL_NAME"}


def fmt_ts(ms: int) -> str:
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s = rem / 1000
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _flat_words(transcript):
    out = []
    for u in transcript.get("utterances", []):
        for w in u["words"]:
            out.append({"w": w["w"], "start_ms": w["start_ms"],
                        "end_ms": w["end_ms"], "channel": u.get("channel")})
    return out


def render_audio(original: Path, work_wav: Path, beep_intervals, out_dir: Path,
                 file_id: str, pad_ms: int, min_pad_ms: int, freq: int,
                 gain_db: float, words=None, channel_wavs=None):
    """Word-aware beep-edge placement (2026-09-02, user-reported defect 6).

    A blind pad+merge swallowed the neighbours of a value ("This is <name>
    FROM <org>" rendered as one continuous beep; "right"/"sir" clipped next
    to amounts). Each beep edge is now placed at the QUIETEST moment inside a
    corridor bounded by the value's own timestamp and the nearest
    non-redacted word: the pad still reaches up to pad_ms into silence, but
    never crosses a neighbour word that the transcript keeps. min_pad_ms is
    the privacy floor — when a neighbour word sits closer than that, the beep
    keeps min_pad_ms of margin and grazes the neighbour instead of exposing
    the value's onset/tail. Two values separated by an audible word ("from")
    therefore render as two beeps with the word intact between them."""
    dur = audio_utils.duration_ms(work_wav)
    words = words or []

    def word_covered(w):
        mid = (w["start_ms"] + w["end_ms"]) / 2.0
        for iv in beep_intervals:
            if (iv.get("channel") is not None and w.get("channel") is not None
                    and iv.get("channel") != w.get("channel")):
                continue
            if iv["start_ms"] <= mid <= iv["end_ms"]:
                return True
        return False

    free_words = [w for w in words if not word_covered(w)]
    pcm_cache = {}

    def mono(path):
        if path not in pcm_cache:
            pcm_cache[path] = audio_utils.load_mono(path)
        return pcm_cache[path]

    placed = []
    for iv in beep_intervals:
        chn = iv.get("channel")
        src = (channel_wavs[chn] if channel_wavs and chn is not None
               and chn < len(channel_wavs) else work_wav)
        pcm, fs = mono(src)
        s0, e0 = iv["start_ms"], iv["end_ms"]
        near = [w for w in free_words
                if chn is None or w.get("channel") is None
                or w.get("channel") == chn]
        prev_end = max([w["end_ms"] for w in near if w["end_ms"] <= s0]
                       or [0])
        next_start = min([w["start_ms"] for w in near if w["start_ms"] >= e0]
                         or [dur])
        # privacy floor: min_pad_ms into silence, but only WORD_GRAZE_MS when
        # a transcribed neighbour word sits inside that floor — the engine
        # segmented both words there, and the beep-edge listen check is the
        # backstop; this is what keeps "from" audible between name and org
        floor_s = (WORD_GRAZE_MS if prev_end > s0 - min_pad_ms
                   else min_pad_ms)
        if set(iv.get("categories", [])) & NAME_CATS:
            floor_s = max(floor_s, ONSET_GUARD_MS)
        hi = max(0, s0 - floor_s)
        lo = max(0, s0 - pad_ms, min(prev_end, hi))
        s = (audio_utils.quietest_point(pcm, fs, lo, hi) if lo < hi else hi)
        floor_e = (WORD_GRAZE_MS if next_start < e0 + min_pad_ms
                   else min_pad_ms)
        lo2 = min(dur, e0 + floor_e)
        hi2 = min(dur, e0 + pad_ms, max(next_start, lo2))
        e = (audio_utils.quietest_point(pcm, fs, lo2, hi2) if lo2 < hi2
             else lo2)
        placed.append((s, e, chn))
    merged = audio_utils.pad_and_merge(placed, 0, dur)

    # decode the ORIGINAL at native rate/channels, mix, encode m4a
    native_wav = out_dir / f"{file_id}_native.wav"
    beeped_wav = out_dir / f"{file_id}_beeped.wav"
    out_m4a = out_dir / f"{file_id}_beeped.m4a"
    audio_utils.decode_to_wav(original, native_wav)
    audio_utils.mix_beeps(native_wav, beeped_wav, merged, freq, gain_db)
    # NOTE: channel indices in `merged` refer to source channels; mix_beeps
    # beeps only that channel, so the other speaker's audio is preserved.
    audio_utils.encode_wav(beeped_wav, out_m4a)
    native_wav.unlink(missing_ok=True)
    beeped_wav.unlink(missing_ok=True)
    return out_m4a, merged


def redact_transcript(transcript, intervals, out_path: Path):
    """Replace covered words with <CATEGORY REDACTED>, incumbent format."""
    covered = {}   # (ui, wi) -> interval
    for iv in intervals:
        if iv["action"] != "beep":
            continue
        for gi in range(iv["start_word"], iv["end_word"] + 1):
            covered[gi] = iv
    lines = []
    gi = 0
    for u in transcript["utterances"]:
        toks = []
        last_iv = None
        for w in u["words"]:
            iv = covered.get(gi)
            if iv is not None:
                if iv is not last_iv:
                    cat = iv["categories"][0]
                    label = {"FULL_NAME": "NAME",
                             "ORGANIZATION_PERSON_LINKED": "ORGANIZATION",
                             "HEALTH_MEDICAL_INFO": "HEALTH"}.get(cat, cat)
                    toks.append(f"<{label} REDACTED>")
                    last_iv = iv
            else:
                toks.append(w["w"])
                last_iv = None
            gi += 1
        start, end = u["words"][0]["start_ms"], u["words"][-1]["end_ms"]
        lines.append(f"[{fmt_ts(start)} - {fmt_ts(end)}] {u['speaker']} - "
                     + " ".join(toks))
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def redact_file(original: Path, work_wav: Path, intervals_path: Path,
                transcript_path: Path, out_dir: Path, file_id: str,
                profile: str, channel_wavs=None):
    cfg = common.CONFIG
    data = common.load_json(intervals_path)
    transcript = common.load_json(transcript_path)
    intervals = data["intervals"]
    for iv in intervals:
        iv["action"] = policy.action_for(iv["categories"],
                                         iv.get("public_figure", False), profile)
    to_beep = [iv for iv in intervals if iv["action"] == "beep"]
    out_m4a, merged = render_audio(
        original, work_wav, to_beep, out_dir, file_id,
        cfg["pad_ms"], cfg.get("min_pad_ms", 50), cfg["beep_freq_hz"],
        cfg["beep_gain_db"], words=_flat_words(transcript),
        channel_wavs=channel_wavs)
    txt = redact_transcript(transcript, intervals,
                            out_dir / f"{file_id}_redacted.txt")
    common.save_json(out_dir / "rendered_intervals.json", {
        "profile": profile, "pad_ms": cfg["pad_ms"],
        "min_pad_ms": cfg.get("min_pad_ms", 50),
        "beeped_intervals": len(to_beep),
        "annotate_only": len(intervals) - len(to_beep),
        "rendered_merged": [{"start_ms": s, "end_ms": e, "channel": c}
                            for s, e, c in merged],
    })
    print(f"  beeped audio -> {out_m4a}")
    print(f"  redacted transcript -> {txt}")
    print(f"  {len(to_beep)} beeped, {len(intervals) - len(to_beep)} annotate-only "
          f"(profile={profile})")
    return out_m4a, txt
