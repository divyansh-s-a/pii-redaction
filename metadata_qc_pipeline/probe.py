#!/usr/bin/env python3
"""Native container/codec probe for the delivered audio.

Tries ffprobe (richest, cross-platform), then afinfo (macOS), and finally a
raw WAV header parse, so it returns something useful on any box without a hard
dependency on ffmpeg. All fields are best-effort; missing ones come back None.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path


def _ffprobe(path: Path):
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    d = json.loads(out.stdout)
    astream = next((s for s in d.get("streams", [])
                    if s.get("codec_type") == "audio"), {})
    fmt = d.get("format", {})
    dur = fmt.get("duration") or astream.get("duration")
    return {
        "codec": astream.get("codec_name"),
        "container": fmt.get("format_name"),
        "sample_rate_hz": int(astream["sample_rate"]) if astream.get(
            "sample_rate") else None,
        "channel_count": astream.get("channels"),
        "bit_depth": (int(astream["bits_per_raw_sample"])
                      if astream.get("bits_per_raw_sample")
                      else int(astream["bits_per_sample"])
                      if astream.get("bits_per_sample") else None),
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else (
            int(astream["bit_rate"]) if astream.get("bit_rate") else None),
        "duration_ms": int(float(dur) * 1000) if dur else None,
        "probe": "ffprobe",
    }


def _afinfo(path: Path):
    if not shutil.which("afinfo"):
        return None
    out = subprocess.run(["afinfo", str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    txt = out.stdout
    info = {"codec": None, "container": None, "sample_rate_hz": None,
            "channel_count": None, "bit_depth": None, "bit_rate": None,
            "duration_ms": None, "probe": "afinfo"}
    m = re.search(r"File type ID:\s*(\S+)", txt)
    if m:
        info["container"] = m.group(1)
    m = re.search(r"Data format:\s*(\d+)\s*ch,\s*([\d.]+)\s*Hz", txt)
    if m:
        info["channel_count"] = int(m.group(1))
        info["sample_rate_hz"] = int(float(m.group(2)))
    m = re.search(r"Data format:.*?'([A-Za-z0-9 ]+)'", txt)
    if m:
        info["codec"] = m.group(1).strip()
    m = re.search(r"(\d+)\s*bits?/channel", txt) or re.search(
        r"LEI(\d+)|BEI(\d+)", txt)
    if m:
        info["bit_depth"] = int(next(g for g in m.groups() if g))
    m = re.search(r"estimated duration:\s*([\d.]+)\s*sec", txt)
    if m:
        info["duration_ms"] = int(float(m.group(1)) * 1000)
    m = re.search(r"bit rate:\s*(\d+)\s*bits", txt)
    if m:
        info["bit_rate"] = int(m.group(1))
    return info


def _wave_header(path: Path):
    try:
        import wave
        with wave.open(str(path), "rb") as w:
            fr = w.getframerate()
            return {"codec": "pcm", "container": "wav",
                    "sample_rate_hz": fr,
                    "channel_count": w.getnchannels(),
                    "bit_depth": w.getsampwidth() * 8,
                    "bit_rate": fr * w.getnchannels() * w.getsampwidth() * 8,
                    "duration_ms": int(w.getnframes() / fr * 1000) if fr else None,
                    "probe": "wave"}
    except Exception:
        return None


def probe(path) -> dict:
    p = Path(path)
    for fn in (_ffprobe, _afinfo, _wave_header):
        try:
            info = fn(p)
        except Exception:
            info = None
        if info:
            info["file_size_bytes"] = p.stat().st_size if p.exists() else None
            info["ext"] = p.suffix.lower().lstrip(".")
            return info
    return {"probe": "none", "ext": p.suffix.lower().lstrip("."),
            "file_size_bytes": p.stat().st_size if p.exists() else None}
