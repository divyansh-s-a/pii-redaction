#!/usr/bin/env python3
"""Signal-level metrics for the FINAL delivered audio.

The deliverable (FLAC/m4a) is decoded once to PCM and summarized with a single
vectorized numpy pass: level (peak/rms/crest), clipping, DC offset, and a
frame-energy VAD for silence / speech / noise-floor / SNR. Fast enough to run
on every file (milliseconds for minutes of audio).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import audio_utils  # noqa: E402

_FS = 32768.0


def _db(v: float) -> float:
    return round(20.0 * math.log10(v), 2) if v > 1e-9 else -120.0


def signal_stats(audio_path: Path, cfg: dict) -> dict:
    """Decode the deliverable and compute level/silence/snr metrics."""
    tmp = Path(audio_path).parent / "_meta_probe.wav"
    try:
        audio_utils.decode_to_wav(Path(audio_path), tmp)   # native rate/chans
        pcm, fs, ch = audio_utils.read_wav(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    allx = np.frombuffer(pcm.tobytes(), dtype=np.int16).astype(np.float64)
    if allx.size == 0:
        return {"error": "empty audio"}
    mono = allx.reshape(-1, ch).mean(axis=1) if ch > 1 else allx

    peak = float(np.max(np.abs(allx))) / _FS
    rms = float(np.sqrt(np.mean((mono / _FS) ** 2)))
    peak_db, rms_db = _db(peak), _db(rms)
    clip_thr = (10 ** (cfg["clip_dbfs"] / 20.0)) * _FS
    clip_pct = round(100.0 * float(np.mean(np.abs(allx) >= clip_thr)), 3)
    dc_offset = round(float(np.mean(mono)) / _FS, 5)

    out = {
        "measured_on": Path(audio_path).name,
        "sample_rate_hz": fs,
        "channel_count": ch,
        "duration_ms": int(mono.size / fs * 1000) if fs else None,
        "peak_dbfs": peak_db,
        "rms_dbfs": rms_db,
        "crest_factor_db": round(peak_db - rms_db, 2),
        "clipping_pct": clip_pct,
        "dc_offset": dc_offset,
    }

    fl = max(1, int(fs * cfg["frame_ms"] / 1000))
    nf = mono.size // fl
    if nf >= 1:
        fr = (mono[:nf * fl] / _FS).reshape(nf, fl)
        frms = np.sqrt(np.mean(fr ** 2, axis=1))
        frdb = 20.0 * np.log10(np.maximum(frms, 1e-9))
        silent = frdb < cfg["silence_dbfs"]
        noise_floor = float(np.percentile(frdb, cfg["noise_floor_percentile"]))
        speech = frdb[~silent]
        speech_lvl = float(speech.mean()) if speech.size else noise_floor
        out.update({
            "silence_pct": round(100.0 * float(np.mean(silent)), 2),
            "speech_pct": round(100.0 * float(np.mean(~silent)), 2),
            "noise_floor_dbfs": round(noise_floor, 2),
            "estimated_snr_db": round(speech_lvl - noise_floor, 2),
        })
    return out
