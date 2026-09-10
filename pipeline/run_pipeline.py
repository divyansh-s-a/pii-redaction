#!/usr/bin/env python3
"""Audio PII redaction pipeline — orchestrator.

Usage:
  python3 pipeline/run_pipeline.py                     # all raw files
  python3 pipeline/run_pipeline.py --file <path>       # one file
  python3 pipeline/run_pipeline.py --limit 1           # first file only
  python3 pipeline/run_pipeline.py --profile strict    # override profile
  python3 pipeline/run_pipeline.py --force transcribe  # redo a cached stage
  python3 pipeline/run_pipeline.py --engine elevenlabs # vendor stage B

Stages per file (each cached in output/<file_id>/, or
output/<file_id>__<engine>/ for a non-gemini --engine):
  A prepare    hash + 16 kHz mono work WAV + duration
  B transcribe Gemini (or vendor adapter) -> transcript.json
               (words + times + speakers)
  C detect     LLM policy pass + safety net -> intervals.json
  D redact     beeped m4a + redacted txt from the SAME intervals
  E qc         signal-verified beeps, orphan tones, leak heuristics
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_utils
import common
import detect_pii
import gemini_transcribe
import qc_report
import redact

ROOT = common.ROOT
RAW_DIRS = [ROOT / "media" / "incoming" / "raw"]
EXTRA_RAW = [ROOT / "media" / "incoming" / "5252dea2cb5c97b14aa7dcf365211a85.mp3",
             ROOT / "media" / "incoming" / "MP EP - 1.mp3"]


def file_id_for(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")
    return stem or hashlib.sha256(path.name.encode()).hexdigest()[:12]


def gather_files(one):
    if one:
        return [Path(one).resolve()]
    files = []
    for d in RAW_DIRS:
        files += sorted(p for p in d.glob("*.mp3"))
    files += [p for p in EXTRA_RAW if p.exists()]
    return files


def run_file(path: Path, model: str, profile: str, force: set,
             engine: str = "gemini", engine_model=None,
             detector: str = "gemini", detector_model=None):
    fid = file_id_for(path)
    suffix = ("" if engine == "gemini" else f"__{engine}") + \
             ("" if detector == "gemini" else f"__{detector}")
    out = ROOT / "output" / (fid + suffix)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {path.name}  (file_id={fid}) ===")

    # A prepare (incl. split-channel detection: a true 2-channel call
    #   recording gets per-channel work WAVs so beeps can target one speaker)
    prep = out / "prepare.json"
    work = out / "work_16k.wav"
    if "prepare" in force or not prep.exists() or not work.exists():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        native16 = out / "work_16k_native.wav"
        audio_utils.decode_to_wav(path, native16, rate=16000)
        chan = audio_utils.channel_analysis(native16)
        ch_wav_names = []
        if chan["split"]:
            outs = [out / f"work_16k_ch{k}.wav" for k in range(chan["channels"])]
            audio_utils.split_channels(native16, outs)
            ch_wav_names = [o.name for o in outs]
        audio_utils.decode_to_wav(path, work, rate=16000, mono=True)
        native16.unlink(missing_ok=True)
        common.save_json(prep, {"file_id": fid, "source": str(path),
                                "sha256": sha,
                                "duration_ms": audio_utils.duration_ms(work),
                                "channel_analysis": chan,
                                "channel_wavs": ch_wav_names})
        d = common.load_json(prep)
        print(f"  prepared: {d['duration_ms']} ms | channels="
              f"{chan['channels']} corr={chan['correlation']} "
              f"split={chan['split']}")
    else:
        print("  prepare: cached")
    prep_data = common.load_json(prep)
    channel_wavs = [out / n for n in prep_data.get("channel_wavs", [])] or None

    # B transcribe (engine-specific; C/D/E below only read transcript.json).
    # A non-gemini detector changes nothing about stage B, so its output dir
    # seeds the transcript from the plain-engine dir when one exists — the
    # same audio is never paid for twice just to swap detectors.
    tr = out / "transcript.json"
    if detector != "gemini" and "transcribe" not in force and not tr.exists():
        base_tr = (ROOT / "output"
                   / (fid if engine == "gemini" else f"{fid}__{engine}")
                   / "transcript.json")
        if base_tr.exists():
            import shutil
            shutil.copy2(base_tr, tr)
            print(f"  transcribe: reused cached {engine} transcript")
    if "transcribe" in force or not tr.exists():
        if engine == "elevenlabs":
            from adapters import elevenlabs_transcribe
            elevenlabs_transcribe.transcribe_file(engine_model, path, work, tr,
                                                  channel_wavs=channel_wavs)
        elif engine == "openai":
            from adapters import openai_transcribe
            openai_transcribe.transcribe_file(engine_model, path, work, tr,
                                              channel_wavs=channel_wavs)
        else:
            common.set_usage_context(f"transcribe:{fid}{suffix}")
            try:
                gemini_transcribe.transcribe_file(engine_model or model, path,
                                                  work, tr,
                                                  channel_wavs=channel_wavs)
            finally:
                common.set_usage_context("")
    else:
        print("  transcribe: cached")

    # C detect — pluggable detector, default Gemini (`model`); --detector gpt
    #   swaps the model behind the SAME policy prompt + word-index schema so
    #   detector models compare fairly on identical transcripts.
    iv = out / "intervals.json"
    if "detect" in force or not iv.exists():
        common.set_usage_context(f"detect:{fid}{suffix}")
        try:
            detect_pii.detect_file(model, fid, tr, iv,
                                   common.CONFIG["value_gap_merge_ms"],
                                   detector=detector,
                                   detector_model=detector_model)
        finally:
            common.set_usage_context("")
    else:
        print("  detect: cached")

    # D redact
    beeped = out / common.beeped_name(fid)
    if "redact" in force or not beeped.exists():
        redact.redact_file(path, work, iv, tr, out, fid, profile,
                           channel_wavs=channel_wavs)
    else:
        print("  redact: cached")

    # E qc
    status, problems = qc_report.qc_file(out, fid, tr, iv,
                                         common.CONFIG["beep_freq_hz"])
    tdata = common.load_json(tr)
    return {"file_id": fid, "qc": status, "problems": len(problems),
            "primary_language": tdata.get("primary_language"),
            "languages": [l["lang"] for l in tdata.get("languages", [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="run a single media file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", default=common.CONFIG["profile"],
                    choices=["strict", "medical_consult"])
    ap.add_argument("--force", action="append", default=[],
                    choices=["prepare", "transcribe", "detect", "redact"],
                    help="redo a cached stage (repeatable)")
    ap.add_argument("--engine", default="gemini",
                    choices=["gemini", "elevenlabs", "openai"],
                    help="transcription engine for stage B; detection (stage "
                         "C) always uses Gemini so engines compare fairly")
    ap.add_argument("--engine-model", default=None,
                    help="override the engine's transcription model id "
                         "(e.g. scribe_v2, whisper-1)")
    ap.add_argument("--detector", default="gemini",
                    choices=["gemini", "gpt"],
                    help="stage-C detector backend (gpt = OpenRouter "
                         "%s, same policy prompt; output dirs gain a "
                         "__gpt suffix)" % detect_pii.GPT_DETECTOR_DEFAULT)
    ap.add_argument("--detector-model", default=None,
                    help="override the gpt detector's model id")
    args = ap.parse_args()

    model = common.resolve_model()
    print(f"model: {model} | profile: {args.profile}")
    if args.engine != "gemini":
        print(f"engine: {args.engine}"
              + (f" ({args.engine_model})" if args.engine_model else "")
              + f" -> output/<file_id>__{args.engine}/")
    files = gather_files(args.file)
    if args.limit:
        files = files[: args.limit]
    results = []
    for p in files:
        try:
            results.append(run_file(p, model, args.profile, set(args.force),
                                    args.engine, args.engine_model,
                                    args.detector, args.detector_model))
        except Exception as e:
            print(f"  ERROR on {p.name}: {e}")
            results.append({"file_id": file_id_for(p), "qc": "ERROR",
                            "problems": -1, "error": str(e)})
    print("\n=== corpus summary ===")
    for r in results:
        print(f"  {r['file_id']}: {r['qc']}"
              + (f" ({r['problems']} problems)" if r["problems"] > 0 else "")
              + (f" | {r.get('primary_language')} {r.get('languages')}"
                 if r.get("primary_language") else "")
              + (f" — {r.get('error','')}" if r["qc"] == "ERROR" else ""))
    sfx = ("" if args.engine == "gemini" else f"__{args.engine}") + \
          ("" if args.detector == "gemini" else f"__{args.detector}")
    summary = f"corpus_summary{sfx}.json"
    common.save_json(ROOT / "output" / summary, results)


if __name__ == "__main__":
    main()
