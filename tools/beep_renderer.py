#!/usr/bin/env python3
"""Shared beep renderer — the ONE renderer used for every vendor's output.

Takes a media file (audio or video) plus a JSON list of intervals, mutes the
original audio inside each (padded, merged) interval and overlays a sine tone,
then writes a new file. Video streams are copied untouched; only audio changes.

Usage:
  python3 tools/beep_renderer.py INPUT_MEDIA INTERVALS_JSON OUTPUT_MEDIA \
      [--pad-ms 250] [--freq 1000] [--gain-db -10]

INTERVALS_JSON is either:
  [{"start_ms": 12400, "end_ms": 15100}, ...]
or a GT/prediction file with an "annotations" or "intervals" key holding that list.

Requires ffmpeg + ffprobe on PATH (macOS: brew install ffmpeg).
The pad is applied symmetrically, clamped to [0, duration]; overlapping or
touching intervals are merged BEFORE rendering, and the merged list is printed
so scoring can use exactly what was rendered.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def load_intervals(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("intervals") or data.get("annotations") or []
    out = []
    for item in data:
        s, e = int(item["start_ms"]), int(item["end_ms"])
        if e <= s:
            raise ValueError(f"bad interval start_ms={s} end_ms={e}")
        out.append((s, e))
    return out


def probe_duration_ms(media: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(media)],
        capture_output=True, text=True, check=True)
    return int(float(r.stdout.strip()) * 1000)


def has_video(media: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(media)],
        capture_output=True, text=True, check=True)
    return "video" in r.stdout


def pad_and_merge(intervals, pad_ms, duration_ms):
    padded = sorted((max(0, s - pad_ms), min(duration_ms, e + pad_ms))
                    for s, e in intervals)
    merged = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def build_filter(merged, freq, gain_db):
    def secs(ms):
        return f"{ms / 1000:.3f}"
    gate = "+".join(f"between(t,{secs(s)},{secs(e)})" for s, e in merged)
    # Mute original inside intervals; play tone only inside intervals; mix.
    return (
        f"[0:a]volume=0:enable='{gate}'[muted];"
        f"[1:a]volume=0:enable='not({gate})',volume={gain_db}dB[tone];"
        f"[muted][tone]amix=inputs=2:duration=first:normalize=0[aout]"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("intervals", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--pad-ms", type=int, default=250,
                    help="safety pad added to each side of every interval (default 250)")
    ap.add_argument("--freq", type=int, default=1000, help="beep frequency Hz")
    ap.add_argument("--gain-db", type=float, default=-10.0, help="beep gain in dB")
    ap.add_argument("--rendered-out", type=Path, default=None,
                    help="where to write the merged rendered intervals JSON "
                         "(default: OUTPUT.rendered_intervals.json)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe not found on PATH. Install with: brew install ffmpeg")
    if args.output.resolve() == args.input.resolve():
        sys.exit("refusing to overwrite the input file; choose a different output path")

    duration_ms = probe_duration_ms(args.input)
    intervals = load_intervals(args.intervals)
    if not intervals:
        print("no intervals -> copying input unchanged")
        shutil.copyfile(args.input, args.output)
        merged = []
    else:
        merged = pad_and_merge(intervals, args.pad_ms, duration_ms)
        fc = build_filter(merged, args.freq, args.gain_db)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(fc)
            fc_path = f.name
        cmd = ["ffmpeg", "-y", "-i", str(args.input),
               "-f", "lavfi", "-t", f"{duration_ms / 1000:.3f}",
               "-i", f"sine=frequency={args.freq}:sample_rate=48000",
               "-filter_complex_script", fc_path,
               "-map", "[aout]"]
        if has_video(args.input):
            cmd += ["-map", "0:v", "-c:v", "copy"]
        cmd += [str(args.output)]
        subprocess.run(cmd, check=True)

    rendered_out = args.rendered_out or args.output.with_suffix(
        args.output.suffix + ".rendered_intervals.json")
    rendered_out.write_text(json.dumps({
        "source": str(args.input), "pad_ms": args.pad_ms,
        "rendered_intervals": [{"start_ms": s, "end_ms": e} for s, e in merged],
    }, indent=2))
    print(f"wrote {args.output}")
    print(f"rendered intervals ({len(merged)} after pad+merge) -> {rendered_out}")


if __name__ == "__main__":
    main()
