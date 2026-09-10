#!/usr/bin/env python3
"""Dedupe stage 1 — collapse one file into a tiny fingerprint record.

  file_sha      sha256 of the raw bytes          -> exact-duplicate blocking
  pcm_sha       sha256 of normalized 16-bit PCM  -> same audio, other container
  audio_tokens  set of 32-bit sub-fingerprints   -> perceptual near-dup (MinHash)
  duration_ms   decoded length
  size          raw byte size (canonical-rep tiebreak)

The acoustic tokens are Shazam-style constellation landmarks: a spectrogram is
reduced to its local spectral peaks (dominant energy, which survives lossy
re-encoding and gain changes), and each anchor peak is paired with a few
later peaks into a hash of (freq1, freq2, delta-time). Because the time is
stored only as a peak-to-peak DELTA, the token set is offset/trim tolerant,
and near-duplicates share a large fraction of landmarks even after
re-encoding — where a whole-frame bit hash would miss them. When the
Chromaprint `fpcalc` binary is on PATH it is used instead (faster, proven at
library scale); both backends yield an int-token SET, so the MinHash/LSH
matcher downstream is identical regardless of backend.

Transcript tokens (k-word shingle hashes) are produced separately by
transcript_tokens() from an existing transcript.json, so the same MinHash/LSH
path can dedupe on conversation content when transcripts are available.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import audio_utils  # noqa: E402
import common  # noqa: E402
try:
    from textnorm import fold_indic
except Exception:                                   # pragma: no cover
    def fold_indic(s):
        return s

# Scratch WAVs live inside the workspace output tree (not the system temp dir):
# predictable, easy to clean, and avoids sandbox/permission surprises when the
# audio converter writes its output file. Override with DEDUPE_TMP.
_TMP = Path(os.environ.get("DEDUPE_TMP",
                           common.ROOT / "output" / "dedupe" / "tmp"))


def _load_mono(path: Path, rate: int):
    """Decode any input to mono PCM at `rate` and return float32 in [-1, 1]."""
    _TMP.mkdir(parents=True, exist_ok=True)
    tmp = _TMP / f"dedfp_{hashlib.md5(str(path).encode()).hexdigest()}.wav"
    try:
        audio_utils.decode_to_wav(Path(path), tmp, rate=rate, mono=True)
        pcm, fs, _ = audio_utils.read_wav(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    x = np.frombuffer(pcm.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
    return x, fs


def _spectrogram(x: np.ndarray, fs: int, cfg: dict):
    """Log-magnitude spectrogram (T x fp_freq_bins) over the speech band, with
    the frequency axis max-pooled down to fp_freq_bins for stable peak
    coordinates across encodings."""
    frame, hop = int(cfg["fp_frame"]), int(cfg["fp_hop"])
    if x.size < frame + hop:
        return None
    win = np.hanning(frame).astype(np.float32)
    idx = np.arange(0, x.size - frame + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[idx] * win
    mag = np.abs(np.fft.rfft(frames, axis=1))       # T x F
    freqs = np.fft.rfftfreq(frame, 1.0 / fs)
    lo = int(np.searchsorted(freqs, float(cfg["fp_fmin"])))
    hi = int(np.searchsorted(freqs, min(float(cfg["fp_fmax"]), fs / 2 - 1)))
    mag = mag[:, lo:hi]
    if mag.shape[1] < 2:
        return None
    nb = int(cfg["fp_freq_bins"])
    F = mag.shape[1]
    if F > nb:
        edges = np.linspace(0, F, nb + 1).astype(int)
        S = np.empty((mag.shape[0], nb), dtype=np.float32)
        for b in range(nb):
            a, c = edges[b], max(edges[b] + 1, edges[b + 1])
            S[:, b] = mag[:, a:c].max(axis=1)
    else:
        S = mag
    return np.log(S + 1e-9)


def _audio_tokens_numpy(x: np.ndarray, fs: int, cfg: dict):
    """Constellation landmarks -> set of (f1, f2, dt) hash tokens."""
    S = _spectrogram(x, fs, cfg)
    if S is None or S.shape[0] < 3:
        return []
    pt, pf = int(cfg["fp_peak_dt"]), int(cfg["fp_peak_df"])
    local_max = maximum_filter(S, size=(pt, pf), mode="nearest")
    thr = np.percentile(S, float(cfg["fp_peak_percentile"]))
    peaks = np.argwhere((S == local_max) & (S >= thr))
    if len(peaks) < 2:
        return []
    peaks = peaks[np.lexsort((peaks[:, 1], peaks[:, 0]))]   # by time, then freq
    times, freqs = peaks[:, 0], peaks[:, 1]
    fan = int(cfg["fp_fan_out"])
    dmin, dmax = int(cfg["fp_dt_min"]), int(cfg["fp_dt_max"])
    n = len(peaks)
    tokens = set()
    for i in range(n):
        t1, f1 = int(times[i]), int(freqs[i])
        cnt = 0
        j = i + 1
        while j < n and times[j] <= t1 + dmax:
            dt = int(times[j]) - t1
            if dt >= dmin:
                f2 = int(freqs[j])
                tokens.add(((f1 & 0x1FF) << 18) | ((f2 & 0x1FF) << 9)
                           | (dt & 0x1FF))
                cnt += 1
                if cnt >= fan:
                    break
            j += 1
    return sorted(tokens)


def _audio_tokens_fpcalc(path: Path):
    out = subprocess.run(["fpcalc", "-raw", "-json", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"fpcalc failed: {out.stderr[:200]}")
    d = json.loads(out.stdout)
    return sorted({int(v) & 0xFFFFFFFF for v in d.get("fingerprint", [])})


def _use_fpcalc(backend: str) -> bool:
    if backend == "fpcalc":
        return True
    if backend == "numpy":
        return False
    return bool(shutil.which("fpcalc"))             # "auto"


def fingerprint_file(path, cfg: dict, backend: str = "auto",
                     want_audio: bool = True) -> dict:
    """Return the fingerprint record for one file (never raises for audio: a
    decode/fingerprint failure is recorded in the record's 'error' field so a
    bad file never aborts a large batch)."""
    p = Path(path)
    raw = p.read_bytes()
    rec = {"path": str(p), "name": p.name, "size": len(raw),
           "file_sha": hashlib.sha256(raw).hexdigest()}
    if not want_audio:
        return rec
    try:
        x, fs = _load_mono(p, int(cfg["fp_rate"]))
        rec["duration_ms"] = int(x.size / fs * 1000) if fs else 0
        rec["pcm_sha"] = hashlib.sha256(
            np.round(x * 32768.0).astype(np.int16).tobytes()).hexdigest()
        try:
            rec["audio_tokens"] = (_audio_tokens_fpcalc(p)
                                   if _use_fpcalc(backend)
                                   else _audio_tokens_numpy(x, fs, cfg))
        except Exception:
            rec["audio_tokens"] = _audio_tokens_numpy(x, fs, cfg)
    except Exception as e:
        rec["error"] = str(e)[:200]
        rec["audio_tokens"] = []
    return rec


_STRIP = ".,!?;:\"'()[]{}\u0964\u0965"


def _norm_word(w: str) -> str:
    return fold_indic(str(w).strip().lower().strip(_STRIP)).lower()


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"),
                                          digest_size=8).digest(), "big")


def transcript_tokens(transcript_path, cfg: dict):
    """k-word shingle hashes for a transcript.json (empty when absent)."""
    d = json.loads(Path(transcript_path).read_text())
    words = [_norm_word(w.get("w", ""))
             for u in d.get("utterances", []) for w in u.get("words", [])]
    words = [w for w in words if w]
    k = int(cfg["transcript_shingle_k"])
    if len(words) < k:
        return [_hash64(" ".join(words))] if words else []
    return sorted({_hash64(" ".join(words[i:i + k]))
                   for i in range(len(words) - k + 1)})
