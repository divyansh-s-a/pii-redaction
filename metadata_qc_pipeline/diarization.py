#!/usr/bin/env python3
"""Speaker / turn / language metadata from transcript.json.

All derived from word timestamps and speaker labels — no audio needed, so it
is effectively free. Note: true speaker-overlap %% is NOT derivable here — the
ASR emits a serialized, non-overlapping word stream, so double-talk cannot be
recovered from these timestamps. It requires acoustic or stereo-channel
analysis and lives outside this transcript-only stage.
"""


def _merge(intervals, gap_ms=0):
    """Merge (start, end) intervals that touch/overlap within gap_ms."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1] + gap_ms:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _union_speech(groups):
    """Total speech time (union of all speech intervals) across merged groups."""
    events = []
    for ivs in groups:
        for s, e in ivs:
            if e > s:
                events.append((s, 1))
                events.append((e, -1))
    events.sort()
    active, last, union = 0, None, 0
    for t, delta in events:
        if last is not None and active > 0:
            union += t - last
        active += delta
        last = t
    return union


def diarization_stats(transcript: dict) -> dict:
    utts = transcript.get("utterances", [])
    split = bool(transcript.get("split_channels"))
    dur = int(transcript.get("duration_ms") or 0)

    per_speaker = {}          # speaker -> list[(s,e)]
    per_channel = {}          # channel -> list[(s,e)]
    total_words = 0
    speaker_words = {}
    for u in utts:
        spk = u.get("speaker", "unknown")
        chn = u.get("channel")
        for w in u.get("words", []):
            s, e = int(w["start_ms"]), int(w["end_ms"])
            per_speaker.setdefault(spk, []).append((s, e))
            per_channel.setdefault(chn, []).append((s, e))
            speaker_words[spk] = speaker_words.get(spk, 0) + 1
            total_words += 1

    speaker_segments = {k: _merge(v) for k, v in per_speaker.items()}
    speaker_talk_ms = {k: sum(e - s for s, e in segs)
                       for k, segs in speaker_segments.items()}
    total_talk = sum(speaker_talk_ms.values()) or 1

    # turns: contiguous runs of one speaker across the utterance sequence
    turns = 0
    longest_monologue = 0
    prev = None
    run_start = run_end = None
    turn_speakers = [u.get("speaker", "unknown") for u in utts if u.get("words")]
    for u in utts:
        if not u.get("words"):
            continue
        spk = u.get("speaker", "unknown")
        s = int(u["words"][0]["start_ms"])
        e = int(u["words"][-1]["end_ms"])
        if spk != prev:
            if run_start is not None:
                longest_monologue = max(longest_monologue, run_end - run_start)
            turns += 1
            run_start, run_end = s, e
            prev = spk
        else:
            run_end = max(run_end, e)
    if run_start is not None:
        longest_monologue = max(longest_monologue, run_end - run_start)

    # total speech time: group by channel for split recordings, else by speaker
    if split and len([c for c in per_channel if c is not None]) >= 2:
        groups = [_merge(v) for k, v in per_channel.items() if k is not None]
    else:
        groups = list(speaker_segments.values())
    union = _union_speech(groups)

    distribution = sorted(
        ({"speaker": k,
          "talk_ms": speaker_talk_ms[k],
          "talk_pct": round(100.0 * speaker_talk_ms[k] / total_talk, 1),
          "words": speaker_words.get(k, 0)}
         for k in speaker_segments),
        key=lambda d: -d["talk_ms"])

    speech_ms = union or total_talk
    wpm = round(total_words / (speech_ms / 60000.0), 1) if speech_ms else 0.0

    langs = transcript.get("languages", [])
    return {
        "speaker_count": len(speaker_segments),
        "channel_count": transcript.get("channel_count")
        or len([c for c in per_channel if c is not None]) or 1,
        "split_channels": split,
        "speaker_distribution": distribution,
        "turn_count": turns,
        "avg_turn_ms": int(total_talk / turns) if turns else 0,
        "longest_monologue_ms": longest_monologue,
        "total_speech_ms": speech_ms,
        "speech_coverage_pct": round(100.0 * speech_ms / dur, 1) if dur else None,
        "words_per_minute": wpm,
        "total_words": total_words,
        "language": {
            "primary_language": transcript.get("primary_language"),
            "languages": langs,
            "num_languages": len(langs),
            "code_switching": len(langs) > 1,
        },
    }
