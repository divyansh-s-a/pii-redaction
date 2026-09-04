#!/usr/bin/env python3
"""Stage B — transcribe one audio file with Gemini: utterances, speakers,
word-level start/end timestamps, language tags, overlap marking.

Long files are transcribed in overlapping chunks (config: chunk_len_s /
chunk_step_s applied above chunk_threshold_s) so timestamp drift cannot
accumulate; chunks are stitched by dropping overlap-zone duplicates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_utils
import common

TRANSCRIBE_PROMPT = (Path(__file__).resolve().parent.parent / "prompts"
                     / "transcription_prompt_v1.txt").read_text()

# Reference-listener mode ONLY: a safety-blocked 30 s floor window is skipped
# (the other reference ear covers the hole) instead of failing the file.
# NEVER enable for production transcription — an untranscribed window is
# audio the detector never sees.
SKIP_BLOCKED_WINDOWS = False

TRANSCRIBE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "utterances": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "speaker": {"type": "STRING"},
                    "lang": {"type": "STRING"},
                    "overlaps_with": {"type": "STRING"},
                    "words": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "w": {"type": "STRING"},
                                "start_ms": {"type": "INTEGER"},
                                "end_ms": {"type": "INTEGER"},
                                "conf": {"type": "STRING",
                                         "enum": ["high", "medium", "low"]},
                            },
                            "required": ["w", "start_ms", "end_ms"],
                        },
                    },
                },
                "required": ["speaker", "words"],
            },
        }
    },
    "required": ["utterances"],
}


def language_summary(utterances):
    """Aggregate per-utterance lang tags into primary + full list, weighted by
    word count. 'mixed' from the model stays its own entry."""
    counts = {}
    for u in utterances:
        lang = (u.get("lang") or "unknown").strip().lower() or "unknown"
        counts[lang] = counts.get(lang, 0) + len(u["words"])
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return {
        "primary_language": ranked[0][0] if ranked else "unknown",
        "languages": [{"lang": k, "word_share": round(v / total, 3)}
                      for k, v in ranked],
    }


def cut_wav_chunk(wav: Path, out: Path, start_s: float, len_s: float) -> Path:
    pcm, fs, ch = audio_utils.read_wav(wav)
    a = int(start_s * fs) * ch
    b = min(len(pcm), int((start_s + len_s) * fs) * ch)
    audio_utils.write_wav(out, pcm[a:b], fs, ch)
    return out


def transcribe_bytes(model: str, media: Path, mime: str, context_text: str = "",
                     extra_text: str = "", thinking_budget=None):
    info = common.upload_file(media, mime, media.name)
    parts = [{"file_data": {"file_uri": info["uri"], "mime_type": mime}}]
    prompt = TRANSCRIBE_PROMPT
    if context_text:
        prompt += ("\n\nThis audio CONTINUES a longer recording. The previous "
                   "part ended with:\n" + context_text +
                   "\nKeep the same speaker labels for the same voices.")
    if extra_text:
        prompt += "\n\n" + extra_text
    parts.append({"text": prompt})
    raw = common.generate(model, parts, TRANSCRIBE_SCHEMA,
                          thinking_budget=thinking_budget)
    import json
    return json.loads(raw)


def sanity(utts, dur_ms, offset_ms=0):
    """Clamp/repair word times; drop empty words; add chunk offset."""
    clean = []
    for u in utts:
        words = []
        prev = 0
        for w in u.get("words", []):
            if not str(w.get("w", "")).strip():
                continue
            s = max(0, int(w.get("start_ms", 0))) + offset_ms
            e = max(s + 1, int(w.get("end_ms", s + 1))) + offset_ms
            s, e = min(s, dur_ms), min(e, dur_ms)
            if e <= s:
                continue
            if s < prev - 2000:   # wildly non-monotonic -> distrust, clamp
                s = prev
                e = max(e, s + 1)
            if e - s > 5000:      # no spoken word lasts >5s: chunk-boundary artifact
                e = s + 5000
            prev = s
            words.append({"w": str(w["w"]), "start_ms": s, "end_ms": e,
                          "conf": w.get("conf", "high")})
        if words:
            # Degenerate-timing repair: some engines bunch word times inside
            # an utterance (many ~1 ms "words" sharing one instant), which
            # breaks beep boundaries built from them. If a third of the words
            # are near-zero-length, redistribute the utterance's total span
            # across its words proportionally to word length.
            tiny = sum(1 for w in words if w["end_ms"] - w["start_ms"] < 20)
            if len(words) >= 3 and tiny >= max(1, len(words) // 3):
                span_s = min(w["start_ms"] for w in words)
                span_e = max(w["end_ms"] for w in words)
                total_chars = sum(max(1, len(w["w"])) for w in words) or 1
                t = span_s
                dur = max(span_e - span_s, len(words) * 40)
                for w in words:
                    share = dur * max(1, len(w["w"])) / total_chars
                    w["start_ms"] = int(t)
                    w["end_ms"] = int(min(span_s + dur, t + share))
                    t += share
            clean.append({"speaker": u.get("speaker", "unknown"),
                          "lang": u.get("lang", ""),
                          "overlaps_with": u.get("overlaps_with", ""),
                          "words": words})
    return clean


def _tail_context(utts):
    if not utts:
        return ""
    tail = utts[-1]
    return tail["speaker"] + ": " + " ".join(w["w"] for w in tail["words"][-25:])


def _truncated(err) -> bool:
    return "truncated at max tokens" in str(err)


def transcribe_window(model, work_wav, out_dir, start_s, length_s, dur_ms,
                      context):
    """Transcribe one window of work_wav (offset applied). Dense speech can
    overflow the model's output-token cap; on truncation the window bisects
    and each half is transcribed separately (down to 30 s windows — a real
    corpus file overflowed even at 60 s, 2026-09-02)."""
    chunk = out_dir / "chunk_tmp.wav"
    cut_wav_chunk(work_wav, chunk, start_s, length_s)
    try:
        try:
            res = transcribe_bytes(model, chunk, "audio/wav", context)
        except RuntimeError as e:
            if "no candidates" in str(e):
                # input-side safety block (e.g. PROHIBITED_CONTENT) — bisect
                # to isolate the offending segment; at the floor, skipping is
                # allowed ONLY in reference-listener mode. Production
                # transcription must fail loudly: an untranscribed window is
                # audio the detector never sees.
                if length_s >= 60:
                    half = length_s / 2
                    print(f"  window @{start_s:.0f}s blocked ({e}) -> "
                          f"bisecting into {half:.0f}s halves")
                    left = transcribe_window(model, work_wav, out_dir,
                                             start_s, half, dur_ms, context)
                    right = transcribe_window(model, work_wav, out_dir,
                                              start_s + half,
                                              length_s - half, dur_ms,
                                              _tail_context(left) or context)
                    return left + right
                if SKIP_BLOCKED_WINDOWS:
                    print(f"  WARNING: window @{start_s:.0f}s "
                          f"({length_s:.0f}s) blocked ({e}) -> SKIPPED "
                          "(reference-listener mode; stretch is "
                          "single-listener)")
                    return []
                raise
            if not _truncated(e):
                raise
            # Truncation is usually THINKING eating the output budget, not
            # the word JSON itself (observed 2026-09-02: dense telephone
            # audio truncated even 30 s windows). Retry once with a capped
            # thinking budget — transcription needs little reasoning and the
            # cap leaves ~57k tokens for words — before paying for a
            # bisection cascade.
            print(f"  window @{start_s:.0f}s ({length_s:.0f}s) truncated -> "
                  "retrying with capped thinking")
            res = None
            try:
                # with thinking capped the model was observed to romanize
                # Indic speech (2026-09-02: Devanagari reference scored 0.94
                # WER purely from script) — restate script fidelity explicitly
                res = transcribe_bytes(
                    model, chunk, "audio/wav", context,
                    extra_text=("CRITICAL: write every word in its ORIGINAL "
                                "SPOKEN SCRIPT (Devanagari for Hindi, Odia "
                                "script for Odia, Bengali script for Bengali, "
                                "Telugu script for Telugu). NEVER romanize or "
                                "transliterate."),
                    thinking_budget=4096)
            except Exception:
                res = None
            if res is None:
                if length_s >= 60:
                    half = length_s / 2
                    print(f"  still truncated -> bisecting into "
                          f"{half:.0f}s halves")
                    left = transcribe_window(model, work_wav, out_dir,
                                             start_s, half, dur_ms, context)
                    right = transcribe_window(model, work_wav, out_dir,
                                              start_s + half,
                                              length_s - half, dur_ms,
                                              _tail_context(left) or context)
                    return left + right
                raise
    finally:
        chunk.unlink(missing_ok=True)
    return sanity(res["utterances"], dur_ms, offset_ms=int(start_s * 1000))


def transcribe_wav(model: str, media: Path, mime: str, work_wav: Path,
                   out_dir: Path):
    """Transcribe one audio stream (chunked when long) -> sanitized utterances."""
    dur_ms = audio_utils.duration_ms(work_wav)
    cfg = common.CONFIG
    if dur_ms / 1000 <= cfg["chunk_threshold_s"]:
        try:
            return sanity(transcribe_bytes(model, media, mime)["utterances"],
                          dur_ms), dur_ms
        except RuntimeError as e:
            if not _truncated(e):
                raise
            print("  single-call transcript truncated -> falling back to chunks")
    step, length = cfg["chunk_step_s"], cfg["chunk_len_s"]
    utts = []
    start = 0.0
    idx = 0
    context = ""
    while start * 1000 < dur_ms:
        print(f"  chunk {idx} @ {start:.0f}s ...")
        res = transcribe_window(model, work_wav, out_dir, start, length,
                                dur_ms, context)
        keep_until = (start + step + (length - step) / 2) * 1000
        drop_before = (start + (length - step) / 2) * 1000 if idx else 0
        kept = []
        for u in res:
            words = [w for w in u["words"]
                     if drop_before <= (w["start_ms"] + w["end_ms"]) / 2
                     < (keep_until if start * 1000 + length * 1000 < dur_ms
                        else dur_ms + 1)]
            if words:
                kept.append({**u, "words": words})
        utts.extend(kept)
        if kept:
            context = _tail_context(kept)
        start += step
        idx += 1
    return utts, dur_ms


def transcribe_file(model: str, original: Path, work_wav: Path, out_json: Path,
                    channel_wavs=None):
    """channel_wavs: list of per-channel mono WAVs when the source is a
    split-channel recording (one speaker per channel). Each channel is
    transcribed separately — better diarization, and intervals inherit the
    channel so the renderer can beep only the offending channel."""
    cfg = common.CONFIG
    if channel_wavs:
        utts = []
        dur_ms = 0
        for chn, cw in enumerate(channel_wavs):
            print(f"  transcribing channel {chn} ...")
            ch_utts, d = transcribe_wav(model, cw, "audio/wav", cw,
                                        out_json.parent)
            dur_ms = max(dur_ms, d)
            for u in ch_utts:
                u["speaker"] = f"C{chn}_{u['speaker']}"
                u["channel"] = chn
            utts.extend(ch_utts)
        utts.sort(key=lambda u: u["words"][0]["start_ms"])
        common.save_json(out_json, {"duration_ms": dur_ms,
                                    "split_channels": True,
                                    **language_summary(utts),
                                    "utterances": utts})
        n_words = sum(len(u["words"]) for u in utts)
        print(f"  transcript ({len(channel_wavs)} channels): {len(utts)} "
              f"utterances, {n_words} words -> {out_json}")
        return out_json
    mime = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
            "mp4": "video/mp4"}.get(original.suffix.lower().lstrip("."),
                                    "audio/mpeg")
    utts, dur_ms = transcribe_wav(model, original, mime, work_wav,
                                  out_json.parent)
    for u in utts:
        u["channel"] = None
    utts.sort(key=lambda u: u["words"][0]["start_ms"])
    common.save_json(out_json, {"duration_ms": dur_ms,
                                "split_channels": False,
                                **language_summary(utts),
                                "utterances": utts})
    n_words = sum(len(u["words"]) for u in utts)
    print(f"  transcript: {len(utts)} utterances, {n_words} words -> {out_json}")
    return out_json
