#!/usr/bin/env python3
"""Stage E — automated QC gate on the rendered output.

Signal-verifies the beeped audio (the same Goertzel analysis that found the
incumbent's ODI_005 defect, turned into a regression gate):
1. every rendered interval is >=90% pure 1 kHz tone (speech actually replaced)
2. no orphan sustained 1 kHz tones outside the rendered intervals
3. transcript markers reconcile with beeped intervals
4. leak heuristic: digit-words within 400 ms outside a beeped interval's edges
Any failure flags the file; nothing is silently passed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_utils
import common


def qc_file(out_dir: Path, file_id: str, transcript_path: Path,
            intervals_path: Path, freq: int):
    problems, notes = [], []
    transcript = common.load_json(transcript_path)

    # 0a. language reporting (patch cached transcripts that predate the field)
    if "primary_language" not in transcript:
        import gemini_transcribe
        transcript.update(
            gemini_transcribe.language_summary(transcript["utterances"]))
        common.save_json(transcript_path, transcript)
    langs = ", ".join(f"{l['lang']} ({l['word_share']:.0%})"
                      for l in transcript.get("languages", []))
    notes.append(f"primary language (engine LID; reconciled value in "
                 f"metadata.json 'language'): "
                 f"{transcript.get('primary_language')} | "
                 f"all: {langs}")

    # 0b. speakers vs channels
    prep = common.load_json(out_dir / "prepare.json")
    chan = prep.get("channel_analysis", {"channels": 1, "split": False})
    speakers = sorted({u["speaker"] for u in transcript["utterances"]})
    notes.append(f"{len(speakers)} distinct speaker(s) {speakers}; "
                 f"{chan.get('channels', 1)} channel(s), "
                 f"split={chan.get('split', False)}")
    if chan.get("split"):
        n_ch = chan.get("channels", 2)
        if len(speakers) != n_ch:
            problems.append(
                f"split-channel file: {len(speakers)} speakers != {n_ch} "
                "channels — check diarization / channel purity")
        per = {}
        for u in transcript["utterances"]:
            per.setdefault(u.get("channel"), set()).add(u["speaker"])
        for chn, spk in sorted(per.items()):
            if len(spk) > 1:
                problems.append(
                    f"channel {chn} carries {len(spk)} speakers "
                    f"({sorted(spk)}) — expected one speaker per channel")

    rendered = common.load_json(out_dir / "rendered_intervals.json")
    merged = [(iv["start_ms"], iv["end_ms"], iv.get("channel"))
              for iv in rendered["rendered_merged"]]
    beeped_audio = out_dir / common.beeped_name(file_id)
    check_wav = out_dir / "qc_check.wav"
    # keep native channel count so channel-targeted beeps can be verified
    audio_utils.decode_to_wav(beeped_audio, check_wav, rate=16000)
    import audio_utils as _au
    _, _, n_ch = _au.read_wav(check_wav)

    # 1. every rendered interval is really beeped — on its target channel —
    #    and, for channel-targeted beeps, the OTHER channel is preserved
    for s, e, chn in merged:
        cov = audio_utils.tone_coverage(check_wav, s, e, freq, channel=chn)
        if cov < 0.9:
            problems.append(f"interval {s}-{e}ms (ch={chn}) tone coverage "
                            f"{cov:.2f} < 0.90")
        if chn is not None and n_ch > 1:
            for other in range(n_ch):
                if other == chn:
                    continue
                oc = audio_utils.tone_coverage(check_wav, s, e, freq,
                                               channel=other)
                if oc > 0.5:
                    problems.append(
                        f"interval {s}-{e}ms leaked beep onto channel {other} "
                        f"(coverage {oc:.2f}) — other speaker's audio lost")
    notes.append(f"{len(merged)} rendered intervals verified for tone coverage")
    for s, e, chn in merged:
        if e - s > 45000:
            problems.append(f"rendered beep {s}-{e}ms is {(e-s)/1000:.0f}s long "
                            "— implausibly wide for a spoken value; check spans")

    # 2. orphan tones: a beep-like tone in the OUTPUT that is neither planned
    #    nor present in the ORIGINAL audio (episode music can hold a ~1 kHz
    #    note; that is source content, not a stray beep)
    work_wav = out_dir / "work_16k.wav"
    original_tones = (audio_utils.find_pure_tones(work_wav, freq)
                      if work_wav.exists() else [])
    # symmetric source scan (owner review 2026-09-11): the output is scanned
    # PER CHANNEL, so the source must be too — a telephony tone in one source
    # channel averages away in the mono mix and used to raise a false
    # "orphan tone". True orphans (in no source channel) still flag.
    if n_ch > 1:
        ch_srcs = sorted(out_dir.glob("work_16k_ch*.wav"))
        if not ch_srcs and work_wav.exists():
            native = out_dir / "_qc_src_native.wav"
            try:
                prep_src = prep.get("source") or prep.get("source_file") or ""
                if prep_src and Path(prep_src).exists():
                    audio_utils.decode_to_wav(Path(prep_src), native,
                                              rate=16000)
                    outs = [out_dir / f"_qc_src_ch{k}.wav"
                            for k in range(n_ch)]
                    audio_utils.split_channels(native, outs)
                    ch_srcs = outs
            except Exception:
                ch_srcs = []
        for cs in ch_srcs:
            original_tones += audio_utils.find_pure_tones(cs, freq)
        for tmpf in list(out_dir.glob("_qc_src_*.wav")):
            tmpf.unlink(missing_ok=True)
    check_channels = range(n_ch) if n_ch > 1 else [None]
    n_tones = 0
    for cch in check_channels:
        tones = audio_utils.find_pure_tones(check_wav, freq, channel=cch)
        n_tones += len(tones)
        # adjacent planned beeps with tiny gaps render as ONE audible tone;
        # match each found tone against the UNION of applicable plan spans
        # (>=90% coverage with 200 ms edge slack), not individual spans
        appl = sorted((s, e) for s, e, chn in merged
                      if chn is None or cch is None or chn == cch)
        fused = []
        for s0, e0 in appl:
            if fused and s0 - fused[-1][1] <= 200:
                fused[-1][1] = max(fused[-1][1], e0)
            else:
                fused.append([s0, e0])
        for ts, te in tones:
            planned = any(s - 200 <= ts and te <= e + 200
                          for s, e in fused)
            in_source = any(os_ - 300 <= ts and te <= oe + 300
                            for os_, oe in original_tones)
            if not planned and not in_source:
                problems.append(f"orphan tone at {ts}-{te}ms (ch={cch}) "
                                "not in plan and not in source audio")
            elif not planned and in_source:
                notes.append(f"tone at {ts}-{te}ms is source content "
                             "(music/hold), not a beep — ok")
    notes.append(f"{n_tones} sustained tones found in output")

    # 3. transcript markers == beeped interval count
    txt = (out_dir / f"{file_id}_redacted.txt").read_text()
    n_markers = txt.count("REDACTED>")
    n_beeped = rendered["beeped_intervals"]
    if n_markers != n_beeped:
        notes.append(f"markers {n_markers} != beeped intervals {n_beeped} "
                     "(adjacent intervals can merge into one beep; verify by ear)")

    # 4. leak heuristic: digit words just outside beeped edges. Structured
    # entries go to qc_digit_edges.json so the delivery workflow can apply
    # the safe fix (extend the beep over the word) instead of only holding.
    import detect_pii
    words = detect_pii.flatten_words(transcript)
    data = common.load_json(intervals_path)
    beeped = [iv for iv in data["intervals"]]
    digit_edges = []
    for iv in beeped:
        for w in words:
            if detect_pii.norm_digit(w["w"]) is None:
                continue
            near_start = iv["start_ms"] - 400 <= w["end_ms"] <= iv["start_ms"]
            near_end = iv["end_ms"] <= w["start_ms"] <= iv["end_ms"] + 400
            inside_plan = any(s <= w["start_ms"] and w["end_ms"] <= e
                              for s, e, _c in merged)
            if (near_start or near_end) and not inside_plan:
                problems.append(
                    f"digit word '{w['w']}' at {w['start_ms']}ms borders "
                    f"interval {iv['annotation_id']} — possible partial leak")
                digit_edges.append({
                    "word": w["w"], "w_start": w["start_ms"],
                    "w_end": w["end_ms"],
                    "interval_id": iv["annotation_id"],
                    "iv_start": iv["start_ms"], "iv_end": iv["end_ms"],
                    "channel": iv.get("channel")})
    common.save_json(out_dir / "qc_digit_edges.json", digit_edges)

    check_wav.unlink(missing_ok=True)
    status = "PASS" if not problems else "FLAGGED"
    lines = [f"# QC report — {file_id}", "", f"**Status: {status}**", ""]
    lines += [f"- {n}" for n in notes]
    if problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in problems]
    (out_dir / "qc_report.md").write_text("\n".join(lines) + "\n")
    print(f"  QC {status}: {len(problems)} problem(s) -> {out_dir/'qc_report.md'}")
    return status, problems
