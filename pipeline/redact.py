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
MIN_BEEP_MS = 200       # narrowest rendered beep: degenerate (bunched) word
                        # timing can shrink a value interval to a few ms; the
                        # tone must stay audible and detectable by the QC
                        # tone scan (over-redaction is the safe direction)
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
                 gain_db: float, words=None, channel_wavs=None,
                 voice_intervals=None):
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
    therefore render as two beeps with the word intact between them.

    voice_intervals (pseudonym mode): mapped mentions whose audio is replaced
    by an invented name. They are placed with the SAME corridor logic, then
    rendered by pseudonym_render; one that overlaps a beep becomes part of
    the beep (the safe direction). Without them the beep path is unchanged.
    Returns (out_audio, merged_beeps, rendered_voiced, voice_fallback_ids)."""
    dur = audio_utils.duration_ms(work_wav)
    words = words or []
    removed = list(beep_intervals) + list(voice_intervals or [])

    def word_covered(w):
        mid = (w["start_ms"] + w["end_ms"]) / 2.0
        for iv in removed:
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

    def place(iv):
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
        if e - s < MIN_BEEP_MS:
            deficit = MIN_BEEP_MS - (e - s)
            s = max(0.0, s - deficit / 2)
            e = min(dur, s + MIN_BEEP_MS)
        return s, e, chn, prev_end, next_start

    placed = [place(iv)[:3] for iv in beep_intervals]
    merged = audio_utils.pad_and_merge(placed, 0, dur)

    voiced = []
    for iv in voice_intervals or []:
        s, e, chn, prev_end, next_start = place(iv)
        widen = PSEUDONYM_WIDEN_MS
        voiced.append({"annotation_id": iv["annotation_id"],
                       "start_ms": int(s), "end_ms": int(e), "channel": chn,
                       "lo_ms": int(max(prev_end, s - widen, 0)),
                       "hi_ms": int(min(next_start, e + widen, dur)),
                       "text": iv["replace"],
                       "mapping_id": iv.get("mapping_id"),
                       "voice_bucket": iv.get("voice_bucket", "low")})

    out_audio = out_dir / common.beeped_name(file_id)
    merged, voiced, fallback_ids = render_mix(
        original, out_audio, merged, voiced, freq, gain_db, out_dir,
        file_id, dur)
    return out_audio, merged, voiced, fallback_ids


PSEUDONYM_WIDEN_MS = 600   # a voiced slot may grow this far into silence
LAST_FALLBACK_REASONS = {}  # annotation_id -> why the last render beeped it


def resolve_overlaps(merged, voiced, dur):
    """A voiced slot that touches a beep is beeped instead (merged into
    it): a partly-beeped name with its remainder voiced — or left as the
    original audio — is never rendered. Returns (merged, voiced, ids)"""
    keep, extra, ids = [], [], []
    for v in voiced:
        hit = any(not (v["end_ms"] <= s or e <= v["start_ms"])
                  and (c is None or v.get("channel") is None
                       or c == v.get("channel"))
                  for s, e, c in merged)
        if hit:
            extra.append((v["start_ms"], v["end_ms"], v.get("channel")))
            ids.append(v["annotation_id"])
        else:
            keep.append(v)
    if extra:
        merged = audio_utils.pad_and_merge(list(merged) + extra, 0, dur)
    return merged, keep, ids


def render_mix(original: Path, out_audio: Path, merged, voiced, freq: int,
               gain_db: float, tmp_dir: Path, tmp_stem: str, dur=None):
    """THE renderer: decode the ORIGINAL at native rate/channels, lay the
    voiced slots (pseudonym mode), mix the beeps, encode the deliverable.
    Every re-render (digit-edge extension, audit remediation, edge verify)
    goes through here so voiced slots survive it — a re-render that mixed
    beeps alone would put the ORIGINAL audio back under every invented name.
    With no voiced slots this is exactly decode -> mix_beeps -> encode.
    Returns (merged, rendered_voiced, voice_fallback_ids)."""
    native_wav = tmp_dir / f"{tmp_stem}_native.wav"
    beeped_wav = tmp_dir / f"{tmp_stem}_beeped.wav"
    audio_utils.decode_to_wav(original, native_wav)
    fallback_ids = []
    LAST_FALLBACK_REASONS.clear()
    src = native_wav
    if voiced:
        import pseudonym_render
        if dur is None:
            dur = audio_utils.duration_ms(native_wav)
        merged, voiced, fallback_ids = resolve_overlaps(merged, voiced, dur)
        LAST_FALLBACK_REASONS.update({i: "touches a beep"
                                      for i in fallback_ids})
        voiced_wav = tmp_dir / f"{tmp_stem}_voiced.wav"
        rendered, fallbacks = pseudonym_render.apply(
            native_wav, voiced_wav, voiced,
            common.OUTPUT_ROOT / "_tts_cache")
        voiced = [v for v in rendered if not v.get("fallback")]
        fallback_ids += [v["annotation_id"] for v in rendered
                         if v.get("fallback")]
        LAST_FALLBACK_REASONS.update({
            v["annotation_id"]: f"{v['fallback']} (clip {v.get('clip_ms')} "
            f"ms, slot {v['end_ms'] - v['start_ms']} ms)"
            for v in rendered if v.get("fallback")})
        if fallbacks:
            merged = audio_utils.pad_and_merge(list(merged) + fallbacks, 0,
                                               dur)
        src = voiced_wav
    audio_utils.mix_beeps(src, beeped_wav, merged, freq, gain_db)
    # NOTE: channel indices in `merged` refer to source channels; mix_beeps
    # beeps only that channel, so the other speaker's audio is preserved.
    audio_utils.encode_audio(beeped_wav, out_audio)
    native_wav.unlink(missing_ok=True)
    beeped_wav.unlink(missing_ok=True)
    if src != native_wav:
        src.unlink(missing_ok=True)
    return merged, voiced, fallback_ids


def redact_transcript(transcript, intervals, out_path: Path):
    """Replace covered words with <CATEGORY REDACTED>, incumbent format.
    Pseudonym mode: a voiced interval (action 'voice') shows its invented
    name instead; a beep that overlaps it wins, as it does in the audio."""
    covered = {}   # (ui, wi) -> interval
    for want in ("voice", "beep"):
        for iv in intervals:
            if iv["action"] != want or iv.get("start_word") is None:
                continue                   # None: an audio-only beep
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
                    if iv["action"] == "voice":
                        toks.append(iv["replace"])
                    else:
                        if cat == "MAPPED_ENTITY":
                            cat = iv.get("mapping_category") or "BRAND"
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


def replace_mode() -> str:
    """'beep' (default) or 'pseudonym' — tools/redact.py --replace sets the
    PII_REPLACE_MODE env var so audit subprocesses agree."""
    import os
    m = (os.environ.get("PII_REPLACE_MODE")
         or common.CONFIG.get("replace_mode") or "beep").strip().lower()
    if m not in ("beep", "pseudonym"):
        raise ValueError(f"unknown replace mode {m!r} (beep|pseudonym)")
    return m


def _new_modes_active():
    """True when anything beyond today's default render is in play; only
    then does rendered_intervals.json gain the new fields."""
    import entity_map
    import profiles
    return (replace_mode() != "beep" or entity_map.active() is not None
            or profiles.active() is not None)


def redact_file(original: Path, work_wav: Path, intervals_path: Path,
                transcript_path: Path, out_dir: Path, file_id: str,
                profile: str, channel_wavs=None):
    cfg = common.CONFIG
    import profiles
    profiles.ensure(profile)
    data = common.load_json(intervals_path)
    transcript = common.load_json(transcript_path)
    intervals = data["intervals"]
    mode = replace_mode()
    em = None
    if mode == "pseudonym":
        import entity_map
        em = entity_map.active()
        if em is None:
            raise RuntimeError("--replace pseudonym needs a mapping "
                               "(--mapping / PII_MAPPING_FILE)")
    for iv in intervals:
        iv["action"] = policy.action_for(iv["categories"],
                                         iv.get("public_figure", False), profile)
        # pseudonym mode voices a CONFIRMED mapped mention with nothing
        # personal stacked on it; everything else keeps its policy action
        if (em is not None and iv["action"] == "beep"
                and iv["categories"] in (["MAPPED_ENTITY"], ["FULL_NAME"])
                and iv.get("render") == "voice"
                and iv.get("mapping_id") in em.by_id
                and em.by_id[iv["mapping_id"]]["render"] == "voice"):
            iv["action"] = "voice"
            iv["replace"] = em.by_id[iv["mapping_id"]]["replace"]
    to_beep = [iv for iv in intervals if iv["action"] == "beep"]
    to_voice = [iv for iv in intervals if iv["action"] == "voice"]
    words = _flat_words(transcript)
    voices = None
    if to_voice:
        import pseudonym_render
        voices = pseudonym_render.speaker_buckets(
            work_wav, [dict(w, speaker=u["speaker"])
                       for u in transcript["utterances"] for w in u["words"]])
        for iv in to_voice:
            iv["voice_bucket"] = (voices.get(iv.get("speaker"))
                                  or {}).get("bucket", "low")
    out_audio, merged, voiced, fallback_ids = render_audio(
        original, work_wav, to_beep, out_dir, file_id,
        cfg["pad_ms"], cfg.get("min_pad_ms", 50), cfg["beep_freq_hz"],
        cfg["beep_gain_db"], words=words,
        channel_wavs=channel_wavs, voice_intervals=to_voice)
    for iv in intervals:
        if iv["annotation_id"] in fallback_ids:
            iv["action"] = "beep"         # the audio beeped it; so does text
    txt = redact_transcript(transcript, intervals,
                            out_dir / f"{file_id}_redacted.txt")
    n_voice = sum(iv["action"] == "voice" for iv in intervals)
    n_beep = sum(iv["action"] == "beep" for iv in intervals)
    rendered = {
        "profile": profile, "pad_ms": cfg["pad_ms"],
        "min_pad_ms": cfg.get("min_pad_ms", 50),
        "beeped_intervals": len(to_beep),
        "annotate_only": len(intervals) - len(to_beep),
        "rendered_merged": [{"start_ms": s, "end_ms": e, "channel": c}
                            for s, e, c in merged],
    }
    if _new_modes_active():
        import entity_map
        emx = entity_map.active()
        pdef = profiles.active()
        rendered.update({
            "beeped_intervals": n_beep,
            "annotate_only": len(intervals) - n_beep - n_voice,
            "replace_mode": mode,
            "voiced_intervals": n_voice,
            "rendered_voiced": voiced,
            "voice_fallbacks": fallback_ids,
            "voice_fallback_reasons": dict(LAST_FALLBACK_REASONS),
            "speaker_voices": voices,
            "voiced_ids": [iv["annotation_id"] for iv in intervals
                           if iv["action"] == "voice"],
            "annotate_only_ids": [iv["annotation_id"] for iv in intervals
                                  if iv["action"] == "annotate_only"],
            "mapping_sha256": emx.sha256 if emx else None,
            "profile_file_sha256": pdef["sha256"] if pdef else None,
            "tts_engine": (__import__("pseudonym_render").cfg()["engine"]
                           if mode == "pseudonym" else None),
        })
    common.save_json(out_dir / "rendered_intervals.json", rendered)
    print(f"  beeped audio -> {out_audio}")
    print(f"  redacted transcript -> {txt}")
    if to_voice:
        print(f"  {n_beep} beeped, {n_voice} voiced (invented name), "
              f"{len(intervals) - n_beep - n_voice} annotate-only "
              f"(profile={profile}; {len(fallback_ids)} voice fallback(s) "
              "beeped)")
    else:
        print(f"  {len(to_beep)} beeped, {len(intervals) - len(to_beep)} "
              f"annotate-only (profile={profile})")
    return out_audio, txt


def rerender(src_dir: Path, file_id: str, media: Path, merged, tmp_stem: str):
    """Re-render after the beep plan changed (digit-edge extension, audit
    remediation, edge verify), keeping the voiced slots of the last render.
    Updates rendered_intervals.json; when a voiced slot had to become a beep,
    the transcript is rewritten so it shows the marker the audio now has.
    Returns the final merged beep list."""
    cfg = common.CONFIG
    ri = common.load_json(src_dir / "rendered_intervals.json")
    voiced = ri.get("rendered_voiced") or []
    merged, voiced, fb = render_mix(
        media, src_dir / common.beeped_name(file_id), merged, voiced,
        cfg["beep_freq_hz"], cfg["beep_gain_db"], src_dir, tmp_stem)
    ri["rendered_merged"] = [{"start_ms": s, "end_ms": e, "channel": c}
                             for s, e, c in merged]
    if "rendered_voiced" in ri:
        ri["rendered_voiced"] = voiced
        if fb:
            ri["voice_fallbacks"] = sorted(set(ri.get("voice_fallbacks", []))
                                           | set(fb))
            ri["voiced_ids"] = [i for i in ri.get("voiced_ids", [])
                                if i not in fb]
            ri["voiced_intervals"] = len(ri["voiced_ids"])
            _rewrite_transcript(src_dir, file_id, ri)
    common.save_json(src_dir / "rendered_intervals.json", ri)
    return merged


def _rewrite_transcript(src_dir: Path, file_id: str, ri):
    import entity_map
    em = entity_map.active()
    transcript = common.load_json(src_dir / "transcript.json")
    data = common.load_json(src_dir / "intervals.json")
    voiced = set(ri.get("voiced_ids", []))
    kept = set(ri.get("annotate_only_ids", []))
    for iv in data["intervals"]:
        if iv["annotation_id"] in voiced and em is not None:
            iv["action"] = "voice"
            iv["replace"] = em.by_id[iv["mapping_id"]]["replace"]
        elif iv["annotation_id"] in kept:
            iv["action"] = "annotate_only"
        else:
            iv["action"] = "beep"
    redact_transcript(transcript, data["intervals"],
                      src_dir / f"{file_id}_redacted.txt")
