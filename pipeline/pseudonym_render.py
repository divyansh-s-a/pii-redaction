#!/usr/bin/env python3
"""Pseudonym mode — speak an invented name in place of a mapped entity.

For each voiced slot (a mapped mention whose original audio is removed):
  1. synthesise the invented name with a STOCK voice (ElevenLabs premade
     library or macOS `say`) — a speaker's own voice is never cloned; the
     speaker's median pitch only picks between a low and a high stock voice;
  2. trim the clip's silence and time-fit it to the slot (pitch-preserving
     WSOLA, at most max_compress faster / max_stretch slower), first widening
     the slot into neighbouring SILENCE (never over a kept word) when the clip
     is longer than the slot;
  3. match its loudness to the speech it replaces, band-limit it to the
     source's bandwidth, and lay it over room tone sampled nearby, with short
     fades at both edges;
  4. if the clip cannot fit even at hard_max_compress, that slot is BEEPED and
     reported — privacy never depends on the voice fitting.

Clips are cached per (engine, voice, text); a batch synthesises each invented
name once per voice. numpy is required (scipy optional, for resampling).
"""
import hashlib
import json
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

import audio_utils
import common

DEFAULTS = {
    "engine": "elevenlabs",            # or "say" (macOS, offline, free)
    "elevenlabs_model": "eleven_multilingual_v2",
    "elevenlabs_voices": {"low": None, "high": None},   # premade names
    "say_voices": {"low": "Aman", "high": "Lekha"},
    "pitch_split_hz": 165.0,
    "max_compress": 1.35, "hard_max_compress": 1.5, "max_stretch": 1.25,
    "widen_ms": 600, "fade_ms": 12, "edge_pad_ms": 20,
}
TTS_FS = 24000


def cfg():
    c = dict(DEFAULTS)
    c.update(common.CONFIG.get("pseudonym", {}))
    if os.environ.get("PII_TTS_ENGINE"):
        c["engine"] = os.environ["PII_TTS_ENGINE"]
    return c


# ------------------------------------------------------------------ TTS

def _elevenlabs_voice(bucket, c, cache_dir):
    """A premade (stock) ElevenLabs voice for the bucket: by configured name,
    else the first premade voice whose gender label fits (low=male)."""
    import urllib.request
    import adapters
    vfile = cache_dir / "elevenlabs_voices.json"
    if vfile.exists():
        voices = json.loads(vfile.read_text())
    else:
        key = adapters.env_key("ELEVENLABS_API_KEY")
        req = urllib.request.Request("https://api.elevenlabs.io/v1/voices",
                                     headers={"xi-api-key": key})
        voices = [v for v in json.loads(urllib.request.urlopen(
            req, timeout=60).read())["voices"]
            if v.get("category") == "premade"]
        voices = [{"voice_id": v["voice_id"], "name": v.get("name"),
                   "gender": (v.get("labels") or {}).get("gender")}
                  for v in voices]
        vfile.parent.mkdir(parents=True, exist_ok=True)
        vfile.write_text(json.dumps(voices, indent=1))
    want = (c.get("elevenlabs_voices") or {}).get(bucket)
    gender = "male" if bucket == "low" else "female"
    for v in voices:
        if want and v["name"] == want:
            return v
    for v in voices:
        if v.get("gender") == gender:
            return v
    if not voices:
        raise RuntimeError("no premade ElevenLabs voices available")
    return voices[0]


def _tts_elevenlabs(text, bucket, c, cache_dir):
    import urllib.request
    import adapters
    v = _elevenlabs_voice(bucket, c, cache_dir)
    url = ("https://api.elevenlabs.io/v1/text-to-speech/"
           f"{v['voice_id']}?output_format=pcm_{TTS_FS}")
    body = json.dumps({"text": text,
                       "model_id": c["elevenlabs_model"]}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": adapters.env_key("ELEVENLABS_API_KEY"),
        "Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    pcm = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2")
    _log_tts("elevenlabs", v["name"], text)
    return pcm.astype(np.float32) / 32768.0, v["name"]


def _tts_say(text, bucket, c, cache_dir):
    voice = (c.get("say_voices") or {}).get(bucket) or "Aman"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "say.wav"
        subprocess.run(["say", "-v", voice, "-o", str(out),
                        "--file-format=WAVE", f"--data-format=LEI16@{TTS_FS}",
                        text], check=True, capture_output=True)
        pcm, fs, ch = audio_utils.read_wav(out)
    x = np.frombuffer(pcm.tobytes(), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    _log_tts("say", voice, text)
    return x, voice


def _log_tts(engine, voice, text):
    log = common.OUTPUT_ROOT / "api_usage.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps({"engine": "tts_" + engine, "voice": voice,
                            "characters": len(text),
                            "context": common.USAGE_CONTEXT["tag"] or None})
                + "\n")


class BadClip(RuntimeError):
    """The synthesiser returned no usable speech (seen with macOS `say`:
    an occasional near-silent 23 ms clip for a ten-letter name)."""


def plausible(x, text):
    """A spoken name lasts at least ~30 ms per letter and is not silence."""
    letters = sum(ch.isalnum() for ch in text)
    if not len(x) or float(np.max(np.abs(x))) < 0.05:
        return False
    return len(trim(x, TTS_FS)) / TTS_FS * 1000 >= max(200, 30 * letters)


def tts_clip(text, bucket, cache_dir: Path, c=None):
    """(mono float32 at TTS_FS, voice name), cached on disk. A clip that is
    not plausible speech is never cached: it is synthesised again (twice at
    most) and, if still unusable, BadClip is raised — the caller beeps."""
    c = c or cfg()
    eng = c["engine"]
    h = hashlib.sha1(f"{eng}|{bucket}|{c.get('elevenlabs_model')}|"
                     f"{json.dumps(c.get(eng + '_voices'), sort_keys=True)}|"
                     f"{text}".encode()).hexdigest()[:16]
    f = cache_dir / f"{eng}_{h}.npy"
    meta = cache_dir / f"{eng}_{h}.json"
    if f.exists() and meta.exists():
        x = np.load(f)
        if plausible(x, text):
            return x, json.loads(meta.read_text())["voice"]
        f.unlink()                     # a bad clip cached before the check
        meta.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)
    fn = _tts_elevenlabs if eng == "elevenlabs" else _tts_say
    for _attempt in range(3):
        x, voice = fn(text, bucket, c, cache_dir)
        if plausible(x, text):
            break
    else:
        raise BadClip(f"{eng} returned no usable speech for a "
                      f"{len(text)}-character name")
    np.save(f, x)
    meta.write_text(json.dumps({"voice": voice, "engine": eng,
                                "text_sha1": hashlib.sha1(
                                    text.encode()).hexdigest()}))
    return x, voice


# ---------------------------------------------------------------- signal

def _frames_rms(x, fs, win_ms=20):
    n = max(1, int(fs * win_ms / 1000))
    m = len(x) // n
    if m == 0:
        return np.array([np.sqrt(np.mean(x ** 2)) if len(x) else 0.0])
    return np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1))


def speech_db(x, fs):
    """Loudness of the speech in x: RMS of its loudest half of 20 ms frames."""
    r = _frames_rms(x, fs)
    if not len(r) or r.max() <= 1e-9:
        return -120.0
    top = np.sort(r)[len(r) // 2:]
    return float(20 * np.log10(np.sqrt(np.mean(top ** 2)) + 1e-12))


def trim(x, fs, rel_db=-35.0):
    r = _frames_rms(x, fs, 10)
    if not len(r) or r.max() <= 1e-9:
        return x
    thr = r.max() * 10 ** (rel_db / 20)
    idx = np.where(r >= thr)[0]
    n = int(fs * 0.01)
    a = max(0, idx[0] * n - n)
    b = min(len(x), (idx[-1] + 1) * n + n)
    return x[a:b]


def wsola(x, rate, fs, win_ms=30, tol_ms=8):
    """Pitch-preserving time scale; rate > 1 shortens (len ~ len(x)/rate)."""
    if abs(rate - 1.0) < 1e-3 or len(x) < fs * win_ms / 1000 * 2:
        return x.copy()
    N = int(fs * win_ms / 1000)
    Hs = N // 2
    Ha = Hs * rate
    tol = int(fs * tol_ms / 1000)
    win = np.hanning(N).astype(np.float32)
    pad = np.concatenate([np.zeros(N + tol, np.float32), x,
                          np.zeros(2 * N + tol, np.float32)])
    out_len = int(len(x) / rate)
    nfr = out_len // Hs + 1
    y = np.zeros(nfr * Hs + N, np.float32)
    ws = np.zeros_like(y)
    prev = N + tol
    for k in range(nfr):
        target = int(N + tol + k * Ha)
        if k == 0:
            best = target
        else:
            ref = pad[prev + Hs: prev + Hs + N]
            lo = max(0, target - tol)
            hi = min(len(pad) - N, target + tol)
            if hi <= lo or len(ref) < N:
                best = min(max(target, 0), len(pad) - N)
            else:
                seg = pad[lo: hi + N]
                corr = np.correlate(seg, ref, mode="valid")
                best = lo + int(np.argmax(corr))
        frame = pad[best: best + N]
        if len(frame) < N:
            break
        y[k * Hs: k * Hs + N] += frame * win
        ws[k * Hs: k * Hs + N] += win
        prev = best
    y = y / np.maximum(ws, 1e-3)
    return y[:out_len]


def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    try:
        from scipy.signal import resample_poly
        fr = Fraction(fs_out, fs_in).limit_denominator(1000)
        return resample_poly(x, fr.numerator, fr.denominator).astype(
            np.float32)
    except ImportError:
        t_out = np.arange(int(len(x) * fs_out / fs_in)) * fs_in / fs_out
        return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


def rolloff_hz(x, fs, frac=0.99):
    if len(x) < 256:
        return fs / 2
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    c = np.cumsum(spec)
    if c[-1] <= 0:
        return fs / 2
    k = int(np.searchsorted(c, frac * c[-1]))
    return k * fs / 2 / len(spec)


def lowpass(x, fs, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    taper = np.clip((cutoff * 1.15 - f) / (cutoff * 0.3), 0, 1)
    return np.fft.irfft(X * taper, n=len(x)).astype(np.float32)


def f0_median(x, fs):
    """Median F0 (Hz) of the voiced frames of x, or None."""
    n = int(fs * 0.04)
    hop = n // 2
    if len(x) < n * 4:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    sel = frames[rms > np.percentile(rms, 60)]
    lo, hi = int(fs / 400), int(fs / 70)
    f0s = []
    for fr in sel[:600]:
        fr = fr - fr.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * n)) ** 2)[: n]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.35:
            f0s.append(fs / lag)
    return float(np.median(f0s)) if len(f0s) >= 5 else None


def speaker_buckets(work_wav_16k: Path, words, c=None):
    """{speaker: 'low'|'high'} from each speaker's median pitch (up to ~30 s
    of their words). Unknown pitch -> 'low'."""
    c = c or cfg()
    pcm, fs = audio_utils.load_mono(work_wav_16k)
    x = np.frombuffer(pcm.tobytes(), dtype="<i2").astype(np.float32) / 32768
    per = {}
    for w in words:
        per.setdefault(w.get("speaker"), []).append(w)
    out = {}
    for spk, ws in per.items():
        segs, total = [], 0
        for w in sorted(ws, key=lambda w: -(w["end_ms"] - w["start_ms"])):
            a, b = int(w["start_ms"] * fs / 1000), int(w["end_ms"] * fs / 1000)
            if b > a:
                segs.append(x[a:b])
                total += b - a
            if total > 30 * fs:
                break
        f0 = f0_median(np.concatenate(segs), fs) if segs else None
        out[spk] = {"bucket": ("high" if f0 and f0 >= c["pitch_split_hz"]
                               else "low"), "f0_hz": f0}
    return out


# ------------------------------------------------------------------- mix

def _room_tone(ch_x, fs, a, b, need):
    """`need` samples of room tone: the quietest 300 ms within 3 s of the
    slot (outside it), looped with crossfades."""
    seg = int(0.3 * fs)
    step = int(0.02 * fs)
    best, best_r = None, None
    lo = max(0, a - 3 * fs)
    hi = min(len(ch_x) - seg, b + 3 * fs)
    for t in range(lo, max(lo, hi), step):
        if t + seg > a and t < b:
            continue
        r = float(np.sqrt(np.mean(ch_x[t: t + seg] ** 2)))
        if best_r is None or r < best_r:
            best, best_r = t, r
    if best is None:
        return np.zeros(need, np.float32)
    tone = ch_x[best: best + seg].astype(np.float32)
    xf = int(0.02 * fs)
    out = np.zeros(need + seg, np.float32)
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    t = 0
    while t < need:
        piece = tone.copy()
        piece[:xf] *= ramp
        piece[-xf:] *= ramp[::-1]
        out[t: t + seg] += piece
        t += seg - xf
    return out[:need]


def apply(native_wav: Path, out_wav: Path, slots, cache_dir: Path, c=None):
    """Render voiced slots into native_wav -> out_wav.

    slots: dicts with start_ms, end_ms, channel, text, voice_bucket, and
    optional lo_ms/hi_ms (silence corridor the slot may widen into).
    Returns (rendered_slots, fallbacks): rendered slots carry the final
    edges and fit metrics; fallbacks are (start_ms, end_ms, channel) to beep.
    """
    c = c or cfg()
    pcm, fs, ch = audio_utils.read_wav(native_wav)
    X = np.frombuffer(pcm.tobytes(), dtype="<i2").astype(np.float32)
    X = X.reshape(-1, ch) / 32768.0
    out = X.copy()
    dur = len(X)
    fade = max(1, int(fs * c["fade_ms"] / 1000))
    rendered, fallbacks = [], []
    for sl in slots:
        s = int(sl["start_ms"] * fs / 1000)
        e = min(dur, int(sl["end_ms"] * fs / 1000))
        lo = int(sl.get("lo_ms", sl["start_ms"]) * fs / 1000)
        hi = min(dur, int(sl.get("hi_ms", sl["end_ms"]) * fs / 1000))
        chans = (list(range(ch)) if sl.get("channel") is None
                 or sl["channel"] >= ch else [sl["channel"]])
        try:
            clip, voice = tts_clip(sl["text"], sl.get("voice_bucket", "low"),
                                   cache_dir, c)
        except BadClip as exc:
            fallbacks.append((sl["start_ms"], sl["end_ms"],
                              sl.get("channel")))
            rendered.append(dict(sl, fallback=f"no usable voice clip "
                                 f"({str(exc)[:60]})", clip_ms=0))
            continue
        clip = resample(trim(clip, TTS_FS), TTS_FS, fs)
        pad = int(fs * c["edge_pad_ms"] / 1000)
        avail = (e - s) - 2 * pad
        need = len(clip)
        if need > avail * c["max_compress"]:
            # widen into the silence corridor, symmetrically where possible
            want = int(need / c["max_compress"]) + 2 * pad - (e - s)
            grow_l = min(max(0, s - lo), want // 2 + max(0, want // 2 - (hi - e)))
            grow_r = min(max(0, hi - e), want - grow_l)
            s, e = s - grow_l, e + grow_r
            avail = (e - s) - 2 * pad
        rate = 1.0
        if need > avail:
            rate = need / max(avail, 1)
            if rate > c["hard_max_compress"]:
                fallbacks.append((sl["start_ms"], sl["end_ms"],
                                  sl.get("channel")))
                rendered.append(dict(sl, fallback="does not fit",
                                     clip_ms=int(need * 1000 / fs)))
                continue
        elif need < avail / 1.6:
            rate = max(1 / c["max_stretch"], need / avail)
        fitted = wsola(clip, rate, fs) if rate != 1.0 else clip
        mono_orig = X[s:e, chans].mean(axis=1)
        target_db = speech_db(mono_orig, fs)
        clip_db = speech_db(fitted, fs)
        gain = 10 ** ((target_db - clip_db) / 20) if clip_db > -100 else 0
        fitted = fitted * gain
        roll = rolloff_hz(mono_orig, fs)
        if roll < 7000:
            fitted = lowpass(fitted, fs, max(roll, 2500))
        peak = float(np.max(np.abs(fitted))) if len(fitted) else 0
        if peak > 0.98:
            fitted *= 0.98 / peak
        n = e - s
        off = s + max(0, (n - len(fitted)) // 2)
        ramp_in = np.linspace(0, 1, fade, dtype=np.float32)
        for cc in chans:
            bed = _room_tone(X[:, cc], fs, s, e, n)
            seg = bed.copy()
            k0 = off - s
            seg[k0: k0 + len(fitted)] += fitted[: n - k0]
            # crossfade original -> new at both edges
            orig = X[s:e, cc]
            env = np.ones(n, np.float32)
            env[:fade] = ramp_in[: min(fade, n)]
            env[-fade:] = np.minimum(env[-fade:], ramp_in[::-1][: min(fade, n)])
            out[s:e, cc] = orig * (1 - env) + seg * env
        rendered.append(dict(
            sl, start_ms=int(s * 1000 / fs), end_ms=int(e * 1000 / fs),
            clip_ms=int(len(fitted) * 1000 / fs), fit_rate=round(rate, 3),
            target_db=round(target_db, 1),
            clip_db=round(speech_db(fitted, fs), 1),
            engine=c["engine"], voice=voice, fallback=None))
    pcm_out = np.clip(np.round(out * 32768), -32768, 32767).astype("<i2")
    import array
    arr = array.array("h")
    arr.frombytes(pcm_out.reshape(-1).tobytes())
    audio_utils.write_wav(out_wav, arr, fs, ch)
    return rendered, fallbacks


def _self_test():
    fs = 16000
    t = np.arange(fs) / fs
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = wsola(x, 1.25, fs)
    assert abs(len(y) - len(x) / 1.25) < fs * 0.01, len(y)
    # pitch preserved: dominant frequency stays ~220 Hz
    f = np.fft.rfftfreq(len(y), 1 / fs)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(f - 220) < 8, f
    assert abs(f0_median(x, fs) - 220) < 10
    v = (0.1 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    assert abs(speech_db(v, fs) - 20 * np.log10(0.1 / np.sqrt(2))) < 0.5
    tr = trim(np.concatenate([np.zeros(4000, np.float32), x[:8000],
                              np.zeros(4000, np.float32)]), fs)
    assert 7800 <= len(tr) <= 8800, len(tr)
    assert abs(len(resample(x, 24000, 48000)) - 2 * len(x)) <= 2
    # a near-silent or too-short clip is not plausible speech
    tts = (0.3 * np.sin(2 * np.pi * 180 * np.arange(TTS_FS) / TTS_FS)
           ).astype(np.float32)
    assert plausible(tts[: TTS_FS // 2], "Kelmora")
    assert not plausible(tts[:600] * 0.01, "Insightwell")
    assert not plausible(tts[: TTS_FS // 10], "Insightwell")
    print("pseudonym_render self-test: OK")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _self_test()
