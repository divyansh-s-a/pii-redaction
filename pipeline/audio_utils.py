#!/usr/bin/env python3
"""Audio helpers: decode/encode via afconvert (macOS) or ffmpeg (Linux/Windows),
pure-Python WAV beep mixing, energy edge-snapping, Goertzel tone verification.

All functions operate on 16-bit PCM WAV. Beeps REPLACE the original audio
(mute + tone), matching the incumbent process's style.
"""
import array
import audioop
import math
import statistics
import subprocess
import wave
from pathlib import Path


def _converter() -> str:
    """Preferred audio converter on this machine: afconvert on macOS,
    ffmpeg elsewhere. (afconvert first: it is what the pilot was validated
    on; the ffmpeg branch keeps the workflow runnable on Linux/Windows —
    same PCM format, same AAC output.)"""
    import shutil
    if shutil.which("afconvert"):
        return "afconvert"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    raise RuntimeError(
        "no audio converter found: install ffmpeg (Linux/Windows: "
        "`apt-get install ffmpeg` / `brew install ffmpeg`); macOS ships "
        "afconvert with the system")


def decode_to_wav(src: Path, dst: Path, rate=None,
                  mono: bool = False) -> Path:
    if _converter() == "afconvert":
        cmd = ["afconvert", "-f", "WAVE", "-d",
               f"LEI16{'@' + str(rate) if rate else ''}"]
        if mono:
            cmd += ["-c", "1"]
        cmd += [str(src), str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
               "-acodec", "pcm_s16le"]
        if rate:
            cmd += ["-ar", str(rate)]
        if mono:
            cmd += ["-ac", "1"]
        cmd += ["-f", "wav", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def encode_wav(src: Path, dst: Path) -> Path:
    """WAV -> m4a (AAC). Neither converter writes mp3 freely; m4a plays
    everywhere."""
    if _converter() == "afconvert":
        cmd = ["afconvert", "-f", "m4af", "-d", "aac", "-q", "127",
               str(src), str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
               "-c:a", "aac", "-b:a", "160k", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def read_wav(path: Path):
    """Manual RIFF parse: accepts plain PCM (1) and WAVE_FORMAT_EXTENSIBLE
    (65534, as written by afconvert), 16-bit only."""
    import struct
    blob = Path(path).read_bytes()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos, fs, ch, bits, data = 12, None, None, None, None
    while pos + 8 <= len(blob):
        cid, size = blob[pos:pos + 4], struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body = blob[pos + 8: pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, fs = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag not in (1, 65534):
                raise ValueError(f"{path}: unsupported WAV format tag {tag}")
        elif cid == b"data":
            data = body
        pos += 8 + size + (size % 2)
    if data is None or fs is None:
        raise ValueError(f"{path}: missing fmt/data chunk")
    if bits != 16:
        raise ValueError(f"{path}: expected 16-bit PCM, got {bits}-bit")
    pcm = array.array("h")
    pcm.frombytes(data[: len(data) - (len(data) % 2)])
    return pcm, fs, ch


def write_wav(path: Path, pcm: array.array, fs: int, ch: int):
    w = wave.open(str(path), "wb")
    w.setnchannels(ch)
    w.setsampwidth(2)
    w.setframerate(fs)
    w.writeframes(pcm.tobytes())
    w.close()


def duration_ms(path: Path) -> int:
    pcm, fs, ch = read_wav(path)
    return int(len(pcm) / ch / fs * 1000)


def mix_beeps(wav_in: Path, wav_out: Path, intervals_ms, freq: int = 1000,
              gain_db: float = -10.0):
    """Replace each interval with a sine tone. Interval items are (start_ms,
    end_ms) — beep ALL channels — or (start_ms, end_ms, channel) — beep only
    that channel, preserving the other speaker's audio (split-channel mode)."""
    pcm, fs, ch = read_wav(wav_in)
    amp = int(32767 * (10 ** (gain_db / 20)))
    for iv in intervals_ms:
        s_ms, e_ms = iv[0], iv[1]
        target = iv[2] if len(iv) > 2 else None
        chans = range(ch) if target is None or target >= ch else [target]
        s = max(0, int(s_ms * fs / 1000)) * ch
        e = min(len(pcm) // ch, int(e_ms * fs / 1000)) * ch
        for i in range(s, e, ch):
            frame = i // ch
            v = int(amp * math.sin(2 * math.pi * freq * frame / fs))
            for c in chans:
                pcm[i + c] = v
    write_wav(wav_out, pcm, fs, ch)


def channel_analysis(wav_path: Path, max_frames: int = 400000):
    """Are the two channels of a stereo file distinct speakers (a split-channel
    call recording) or a near-identical dual-mono mix?

    Returns {"channels", "correlation", "energy_ratio", "split"} where split is
    True only when both channels carry real energy AND they are decorrelated.
    """
    pcm, fs, ch = read_wav(wav_path)
    if ch < 2:
        return {"channels": ch, "correlation": 1.0, "energy_ratio": 0.0,
                "split": False}
    left, right = pcm[0::ch], pcm[1::ch]
    stride = max(1, len(left) // max_frames)
    sl = sr = slr = sll = srr = 0.0
    for i in range(0, len(left), stride):
        l, r = float(left[i]), float(right[i])
        sl += l
        sr += r
        slr += l * r
        sll += l * l
        srr += r * r
    n = len(range(0, len(left), stride))
    cov = slr / n - (sl / n) * (sr / n)
    vl, vr = sll / n - (sl / n) ** 2, srr / n - (sr / n) ** 2
    corr = cov / math.sqrt(vl * vr) if vl > 0 and vr > 0 else 1.0
    rms_l, rms_r = math.sqrt(max(vl, 0)), math.sqrt(max(vr, 0))
    ratio = min(rms_l, rms_r) / max(rms_l, rms_r) if max(rms_l, rms_r) else 0.0
    return {"channels": ch, "correlation": round(corr, 3),
            "energy_ratio": round(ratio, 3),
            "split": bool(corr < 0.5 and ratio > 0.15)}


def split_channels(wav_in: Path, outs):
    """Write each channel of a multichannel WAV as its own mono WAV."""
    pcm, fs, ch = read_wav(wav_in)
    for k, out in enumerate(outs[:ch]):
        write_wav(out, pcm[k::ch], fs, 1)
    return outs[:ch]


def load_mono(wav_path: Path):
    """(mono_pcm, fs) for edge-placement analysis; downmixes stereo."""
    pcm, fs, ch = read_wav(wav_path)
    if ch != 1:
        mono = audioop.tomono(pcm.tobytes(), 2, 0.5, 0.5)
        pcm = array.array("h")
        pcm.frombytes(mono)
    return pcm, fs


def quietest_point(pcm, fs, lo_ms: int, hi_ms: int, win_ms: int = 20) -> int:
    """The start of the quietest win_ms window inside [lo_ms, hi_ms]."""
    win = max(1, int(win_ms * fs / 1000))
    step = win_ms // 2 or 1
    best_ms, best_rms = lo_ms, None
    for t in range(int(lo_ms), int(hi_ms) + 1, step):
        a = int(t * fs / 1000)
        frag = pcm[a: a + win]
        if len(frag) < win:
            continue
        r = audioop.rms(frag.tobytes(), 2)
        if best_rms is None or r < best_rms:
            best_ms, best_rms = t, r
    return best_ms


def snap_edges(wav_path: Path, start_ms: int, end_ms: int,
               search_ms: int = 150, win_ms: int = 20):
    """Move interval edges to the quietest nearby moment so the beep starts and
    ends in a speech gap instead of mid-phoneme. Never shrinks the interval.
    (Legacy helper: stage D now places edges with word-aware corridors in
    redact.render_audio; kept for standalone use.)"""
    pcm, fs = load_mono(wav_path)
    total = int(len(pcm) / fs * 1000)
    new_start = quietest_point(pcm, fs, max(0, start_ms - search_ms),
                               start_ms, win_ms)
    new_end = quietest_point(pcm, fs, end_ms, min(total, end_ms + search_ms),
                             win_ms)
    return min(new_start, start_ms), max(new_end, end_ms)


def pad_and_merge(intervals, pad_ms: int, duration: int):
    """Pad and merge intervals. Items are (s, e) or (s, e, channel); merging
    happens per channel — a beep on channel 0 never swallows one on channel 1.
    Returns (s, e, channel) triples (channel None = all channels)."""
    groups = {}
    for iv in intervals:
        chn = iv[2] if len(iv) > 2 else None
        groups.setdefault(chn, []).append((iv[0], iv[1]))
    out = []
    for chn, ivs in groups.items():
        padded = sorted((max(0, s - pad_ms), min(duration, e + pad_ms))
                        for s, e in ivs)
        merged = []
        for s, e in padded:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out += [(s, e, chn) for s, e in merged]
    return sorted(out, key=lambda t: (t[0], t[1]))


def _goertzel_frac(seg, fs, f):
    w = 2 * math.pi * f / fs
    coeff = 2 * math.cos(w)
    s1 = s2 = 0.0
    total = 0.0
    for x in seg:
        fx = float(x)
        total += fx * fx
        s0 = fx + coeff * s1 - s2
        s2, s1 = s1, s0
    if total == 0:
        return 0.0
    p = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return 2 * p / (total * len(seg))


def _as_mono(pcm, ch, channel=None):
    """Downmix to mono, or extract one channel when `channel` is given."""
    if ch == 1:
        return pcm
    if channel is not None and channel < ch:
        return pcm[channel::ch]
    mono = audioop.tomono(pcm.tobytes(), 2, 0.5, 0.5)
    out = array.array("h")
    out.frombytes(mono)
    return out


def tone_coverage(wav_path: Path, start_ms: int, end_ms: int, freq: int = 1000,
                  win_ms: int = 80, channel=None) -> float:
    """Fraction of 80 ms windows inside [start,end) that are a pure tone at
    freq. channel=None analyses the downmix; an int analyses that channel."""
    pcm, fs, ch = read_wav(wav_path)
    pcm = _as_mono(pcm, ch, channel)
    win = int(win_ms * fs / 1000)
    a, b = int(start_ms * fs / 1000), int(end_ms * fs / 1000)
    wins = range(a, max(a + 1, b - win + 1), win)
    if not wins:
        return 0.0
    good = total = 0
    for off in wins:
        seg = pcm[off: off + win]
        if len(seg) < win:
            continue
        total += 1
        purity = max(_goertzel_frac(seg, fs, freq * m) for m in (0.99, 1.0, 1.01))
        if purity > 0.7:
            good += 1
    return good / total if total else 0.0


def find_pure_tones(wav_path: Path, freq: int = 1000, win_ms: int = 80,
                    min_windows: int = 3, channel=None):
    """Sustained pure tones at freq anywhere in the file -> [(start_ms, end_ms)]."""
    pcm, fs, ch = read_wav(wav_path)
    pcm = _as_mono(pcm, ch, channel)
    win = int(win_ms * fs / 1000)
    flags = []
    for wi in range(len(pcm) // win):
        seg = pcm[wi * win: (wi + 1) * win]
        if audioop.rms(seg.tobytes(), 2) < 200:
            flags.append(False)
            continue
        purity = max(_goertzel_frac(seg, fs, freq * m) for m in (0.99, 1.0, 1.01))
        flags.append(purity > 0.7)
    tones, run = [], 0
    for i, f in enumerate(flags + [False]):
        if f:
            run += 1
        else:
            if run >= min_windows:
                tones.append(((i - run) * win_ms, i * win_ms))
            run = 0
    return tones
