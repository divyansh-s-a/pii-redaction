#!/usr/bin/env python3
"""Audible evaluation watermark for beeped deliverables.

  python3 tools/watermark.py --make-assets            # one-time TTS of the clips
  python3 tools/watermark.py deliveries/<fid> [...]   # watermark deliveries
  python3 tools/watermark.py path/to/audio.flac --work-dir output/<fid>__elevenlabs
  python3 tools/watermark.py --self-test

Scheme (owner decision 2026-09-10):
  1. The FULL statement is PREPENDED as a clean leader — every listener hears
     the terms once, no content is obstructed, the file grows by ~12 s.
  2. A SHORT tag ("Deccan AI evaluation sample — not for production use.") is
     OVERLAID over the program at length-scaled intervals with the program
     ducked beneath it — audible, and destructive to strip (cutting the tag
     cuts content).
Placement is transcript-aware: within each target window the tag start is
scored toward silence gaps (word timestamps from transcript.json), and any
overlap with a rendered 1 kHz beep (rendered_intervals.json, +buffer) is a
hard reject — beeps stay clean for the Goertzel QC. Missing metadata degrades
gracefully to a uniform schedule (still never lands on beeps when they are
known). Levels are matched to the file's measured speech RMS.

The original deliverable is never modified: output is
<name>_watermarked.<ext> plus a watermark.json sidecar (schedule, levels,
source sha256). This is a deterrent / provenance notice, not DRM.
"""
import argparse
import array
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import audio_utils        # noqa: E402
import common             # noqa: E402
from adapters import env_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "prompts" / "watermark"

WM = common.CONFIG.get("watermark", {})
STATEMENT = WM.get("statement_text", (
    "This file is a watermarked sample provided by Deccan AI for evaluation. "
    "It is not a production dataset, does not confer a licence to train, and "
    "must not be copied, shared, or used beyond the stated evaluation "
    "purpose."))
TAG = WM.get("tag_text", "Deccan AI evaluation sample — not for production use.")
TAG_GAIN_DB = WM.get("tag_gain_db", 2.0)      # tag RMS relative to speech RMS
DUCK_DB = WM.get("duck_db", 6.0)              # program attenuation under tag
LEADER_GAP_MS = WM.get("leader_gap_ms", 400)  # silence between leader and program
BEEP_BUFFER_MS = WM.get("beep_buffer_ms", 300)
RAMP_MS = 100                                  # duck fade in/out
SEARCH_S = 10                                  # tag placement search window +/-
VOICE_NAME = WM.get("voice_name", "Rachel")
TTS_MODEL = WM.get("tts_model", "eleven_multilingual_v2")


# ------------------------------------------------------------ tiny DSP utils

def _mono(pcm, ch):
    if ch == 1:
        return pcm
    return array.array("h", (sum(pcm[i + c] for c in range(ch)) // ch
                             for i in range(0, len(pcm) - ch + 1, ch)))


def _rms_db(samples):
    vals = [s * s for s in samples if abs(s) > 60]   # skip near-digital-silence
    if not vals:
        vals = [s * s for s in samples] or [1]
    return 10.0 * math.log10(sum(vals) / len(vals) / (32768.0 ** 2) + 1e-12)


def _scale(pcm, factor):
    out = array.array("h")
    for s in pcm:
        v = int(s * factor)
        out.append(max(-32768, min(32767, v)))
    return out


# ------------------------------------------------------------ asset handling

def _tts(text: str, out_wav: Path):
    """One-time ElevenLabs TTS -> 16 kHz mono PCM wav (stdlib urllib)."""
    import urllib.request
    key = env_key("ELEVENLABS_API_KEY")
    req = urllib.request.Request("https://api.elevenlabs.io/v1/voices",
                                 headers={"xi-api-key": key})
    voices = json.loads(urllib.request.urlopen(req, timeout=60).read())["voices"]
    voice = next((v for v in voices if v.get("name") == VOICE_NAME),
                 next((v for v in voices
                       if v.get("category") == "premade"), voices[0]))
    url = ("https://api.elevenlabs.io/v1/text-to-speech/"
           f"{voice['voice_id']}?output_format=pcm_16000")
    body = json.dumps({"text": text, "model_id": TTS_MODEL}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": key, "Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=300).read()
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        pcm.byteswap()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    audio_utils.write_wav(out_wav, pcm, 16000, 1)
    print(f"  asset ({voice['name']}): {out_wav}  "
          f"({len(pcm) / 16000:.1f}s, {len(text)} chars)")


def make_assets(force=False):
    for name, text in (("statement", STATEMENT), ("tag", TAG)):
        out = ASSET_DIR / f"{name}.wav"
        if out.exists() and not force:
            print(f"  asset cached: {out}")
            continue
        _tts(text, out)


def _load_asset(name: str, fs: int, tmp_dir: Path):
    src = ASSET_DIR / f"{name}.wav"
    if not src.exists():
        raise SystemExit(f"missing watermark asset {src} — run "
                         "`python3 tools/watermark.py --make-assets` once "
                         "(needs ELEVENLABS_API_KEY)")
    if fs == 16000:
        pcm, _, ch = audio_utils.read_wav(src)
        return _mono(pcm, ch)
    tmp = tmp_dir / f"_wm_{name}_{fs}.wav"
    audio_utils.decode_to_wav(src, tmp, rate=fs, mono=True)
    pcm, _, ch = audio_utils.read_wav(tmp)
    tmp.unlink(missing_ok=True)
    return _mono(pcm, ch)


# ------------------------------------------------------------ scheduling

def tag_interval_s(dur_s: float) -> float:
    if dur_s < 60:
        return dur_s / 2.0        # a single mid-file tag
    if dur_s < 300:
        return WM.get("interval_medium_s", 40)
    return WM.get("interval_long_s", 60)


def schedule_tags(dur_ms: int, tag_ms: int, words, beeps):
    """Choose tag start times. words: [(s,e)] spoken words; beeps: [(s,e)]
    rendered beep intervals. Score candidates toward silence; hard-reject any
    overlap with a beep +/- BEEP_BUFFER_MS."""
    interval = tag_interval_s(dur_ms / 1000.0) * 1000.0
    blocked = [(max(0, s - BEEP_BUFFER_MS), e + BEEP_BUFFER_MS)
               for s, e in beeps]

    def hits_beep(a, b):
        return any(a < e and s < b for s, e in blocked)

    def in_word(t):
        return any(s <= t < e for s, e in words)

    def silence_frac(a, b):
        if not words:
            return 0.0
        span = b - a
        covered = 0
        for s, e in words:
            covered += max(0, min(e, b) - max(s, a))
        return max(0.0, 1.0 - covered / max(1, span))

    placements = []
    t = interval
    while t + tag_ms < dur_ms - 5000:
        best, best_score = None, -1.0
        lo = int(max(0, t - SEARCH_S * 1000))
        hi = int(min(dur_ms - tag_ms - 1000, t + SEARCH_S * 1000))
        for cand in range(lo, hi + 1, 250):
            a, b = cand, cand + tag_ms
            if hits_beep(a, b):
                continue
            score = silence_frac(a, b) + (0.5 if not in_word(a) else 0.0)
            if score > best_score:
                best, best_score = cand, score
        if best is not None:
            placements.append({"start_ms": best,
                               "silence_frac": round(best_score, 3)})
        t += interval
    return placements


# ------------------------------------------------------------ mixing

def _mix_overlay(pcm, fs, ch, tag, start_ms, duck_factor):
    """Duck the program under the tag (cosine ramps) and add the tag to every
    channel, with a clip guard over the mixed region."""
    n_tag = len(tag)
    start = int(start_ms * fs / 1000)
    ramp = int(RAMP_MS * fs / 1000)
    for i in range(n_tag + 2 * ramp):
        idx = start - ramp + i
        if idx < 0 or (idx + 1) * ch > len(pcm):
            continue
        if i < ramp:
            g = 1.0 - (1.0 - duck_factor) * (i / ramp)
        elif i < ramp + n_tag:
            g = duck_factor
        else:
            g = duck_factor + (1.0 - duck_factor) * ((i - ramp - n_tag) / ramp)
        add = tag[i - ramp] if 0 <= i - ramp < n_tag else 0
        for c in range(ch):
            j = idx * ch + c
            v = int(pcm[j] * g) + add
            pcm[j] = max(-32768, min(32767, v))


def watermark_file(audio: Path, out_path: Path, words, beeps, tmp_dir: Path,
                   statement=None, tag=None):
    """Returns the watermark.json dict. statement/tag: optional preloaded
    16 kHz mono arrays (self-test injects synthetic ones)."""
    wav = tmp_dir / "_wm_program.wav"
    audio_utils.decode_to_wav(audio, wav)          # native rate + channels
    pcm, fs, ch = audio_utils.read_wav(wav)
    wav.unlink(missing_ok=True)
    dur_ms = int(len(pcm) / ch * 1000 / fs)

    stmt = statement if statement is not None else _load_asset("statement", fs, tmp_dir)
    tg = tag if tag is not None else _load_asset("tag", fs, tmp_dir)
    if statement is not None and fs != 16000:      # self-test convenience
        raise SystemExit("self-test fixtures must be 16 kHz")

    speech = _mono(pcm, ch)
    if words:
        sel = array.array("h")
        for s, e in words[:400]:
            a, b = int(s * fs / 1000), int(e * fs / 1000)
            sel.extend(speech[a:b])
        speech = sel or speech
    prog_db = _rms_db(speech)
    for name, clip in (("stmt", stmt), ("tag", tg)):
        cur = _rms_db(clip)
        factor = 10 ** ((prog_db + TAG_GAIN_DB - cur) / 20.0)
        scaled = _scale(clip, min(factor, 32000 / (max(
            (abs(s) for s in clip), default=1))))
        if name == "stmt":
            stmt = scaled
        else:
            tg = scaled

    placements = schedule_tags(dur_ms, int(len(tg) * 1000 / fs), words, beeps)
    duck = 10 ** (-DUCK_DB / 20.0)
    for p in placements:
        _mix_overlay(pcm, fs, ch, tg, p["start_ms"], duck)

    # leader: statement + gap, replicated to all channels, then the program
    gap = array.array("h", [0] * int(LEADER_GAP_MS * fs / 1000))
    leader_mono = stmt + gap
    leader = array.array("h")
    for s in leader_mono:
        leader.extend([s] * ch)
    mixed = leader + pcm
    leader_ms = int(len(leader_mono) * 1000 / fs)

    if out_path.suffix.lower() == ".wav":
        audio_utils.write_wav(out_path, mixed, fs, ch)
    else:
        out_wav = tmp_dir / "_wm_out.wav"
        audio_utils.write_wav(out_wav, mixed, fs, ch)
        audio_utils.encode_audio(out_wav, out_path)
        out_wav.unlink(missing_ok=True)

    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(audio), "output": str(out_path),
        "source_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "statement_text": STATEMENT, "tag_text": TAG,
        "leader_ms": leader_ms,
        "content_offset_ms": leader_ms,
        "tag_placements": placements,
        "levels": {"program_speech_rms_db": round(prog_db, 2),
                   "tag_gain_db_over_rms": TAG_GAIN_DB,
                   "duck_db": DUCK_DB},
        "note": ("all content timestamps (beeps, transcript) shift by "
                 "content_offset_ms in the watermarked file"),
    }
    return doc


# ------------------------------------------------------------ target lookup

def _resolve(target: Path, work_dir):
    """Return (audio, words, beeps, out_dir). Accepts a delivery dir or an
    audio file; transcript/rendered intervals come from --work-dir or the
    conventional output/<fid>__<engine> location."""
    if target.is_dir():
        audio = next((f for f in sorted(target.iterdir())
                      if "_beeped." in f.name and "_watermarked" not in f.name),
                     None)
        if audio is None:
            raise SystemExit(f"{target}: no *_beeped.* audio inside")
    else:
        audio = target
    fid = audio.name.split("_beeped.")[0]
    wd = Path(work_dir) if work_dir else None
    if wd is None:
        for cand in sorted((ROOT / "output").glob(f"{fid}__*")) + [ROOT / "output" / fid]:
            if (cand / "transcript.json").exists():
                wd = cand
                break
    words, beeps = [], []
    if wd and (wd / "transcript.json").exists():
        t = json.loads((wd / "transcript.json").read_text())
        words = sorted((w["start_ms"], w["end_ms"])
                       for u in t.get("utterances", []) for w in u["words"])
    if wd and (wd / "rendered_intervals.json").exists():
        r = json.loads((wd / "rendered_intervals.json").read_text())
        beeps = [(iv["start_ms"], iv["end_ms"])
                 for iv in r.get("rendered_merged", [])]
    return audio, words, beeps


# ------------------------------------------------------------------ self-test

def self_test():
    import tempfile
    ok = True

    def case(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("%s  %s%s" % ("PASS" if cond else "FAIL", name,
                            f"  [{detail}]" if (detail and not cond) else ""))

    tmp = Path(tempfile.mkdtemp(prefix="wm_selftest_"))
    fs = 16000
    # synthetic program: 100 s speech-shaped noise with silences + one beep
    import random
    random.seed(7)
    n = 100 * fs
    pcm = array.array("h", [0] * n)
    words = []
    t = 2000
    while t < 96000:                       # words of 400 ms, gaps of 300 ms
        a, b = int(t * fs / 1000), int((t + 400) * fs / 1000)
        for i in range(a, b):
            pcm[i] = random.randint(-6000, 6000)
        words.append((t, t + 400))
        t += 700
        if len(words) % 12 == 0:
            t += 2500                       # a periodic long silence gap
    beeps = [(30000, 32000)]
    for i in range(30 * fs, 32 * fs):
        pcm[i] = int(8000 * math.sin(2 * math.pi * 1000 * i / fs))
    src = tmp / "prog_beeped.wav"
    audio_utils.write_wav(src, pcm, fs, 1)

    stmt = array.array("h", [random.randint(-5000, 5000) for _ in range(3 * fs)])
    tag = array.array("h", [random.randint(-5000, 5000) for _ in range(2 * fs)])

    out = tmp / "prog_beeped_watermarked.wav"
    doc = watermark_file(src, out, words, beeps, tmp, statement=stmt, tag=tag)

    case("leader = statement + gap", abs(doc["leader_ms"] - (3000 + LEADER_GAP_MS)) < 20,
         str(doc["leader_ms"]))
    pl = doc["tag_placements"]
    exp = len(range(1, int(100 / tag_interval_s(100))))
    case("length-scaled schedule (100s @ %ds interval)" % tag_interval_s(100),
         1 <= len(pl) <= max(1, exp + 1), str(pl))
    case("no tag overlaps the beep (+buffer)",
         all(not (p["start_ms"] < 32000 + BEEP_BUFFER_MS
                  and 30000 - BEEP_BUFFER_MS < p["start_ms"] + 2000)
             for p in pl), str(pl))
    case("placements favour silence", all(p["silence_frac"] > 0 for p in pl),
         str(pl))

    mixed, mfs, mch = audio_utils.read_wav(out)
    case("output = leader + program length",
         abs(len(mixed) - (len(pcm) + int((3000 + LEADER_GAP_MS) * fs / 1000)))
         < fs // 10, str(len(mixed)))
    off = int(doc["content_offset_ms"] * fs / 1000)
    beep_seg = mixed[off + 30 * fs + fs: off + 30 * fs + fs + 400]
    orig_seg = pcm[30 * fs + fs: 30 * fs + fs + 400]
    case("beep region untouched (shifted only)",
         list(beep_seg) == list(orig_seg))
    case("no clipping", max(abs(s) for s in mixed) < 32768)

    # short file: single mid tag
    short = array.array("h", [random.randint(-4000, 4000) for _ in range(40 * fs)])
    src2 = tmp / "short_beeped.wav"
    audio_utils.write_wav(src2, short, fs, 1)
    doc2 = watermark_file(src2, tmp / "short_beeped_watermarked.wav",
                          [], [], tmp, statement=stmt, tag=tag)
    case("short file gets exactly one tag", len(doc2["tag_placements"]) == 1,
         str(doc2["tag_placements"]))

    print("\nself-test:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("targets", nargs="*",
                    help="delivery dirs or *_beeped.* audio files")
    ap.add_argument("--work-dir", default=None,
                    help="dir holding transcript.json / rendered_intervals.json "
                         "(default: found by fid under output/)")
    ap.add_argument("--outdir", default=None,
                    help="write outputs here (default: next to each source)")
    ap.add_argument("--make-assets", action="store_true")
    ap.add_argument("--force-assets", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if args.make_assets or args.force_assets:
        make_assets(force=args.force_assets)
        if not args.targets:
            return
    if not args.targets:
        ap.error("nothing to do: pass delivery dirs / audio files")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="wm_"))
    for tstr in args.targets:
        target = Path(tstr).resolve()
        audio, words, beeps = _resolve(target, args.work_dir)
        stem, ext = audio.name.rsplit(".", 1)
        out_dir = Path(args.outdir) if args.outdir else audio.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{stem}_watermarked.{ext}"
        print(f"=== {audio.name} -> {out.name}")
        doc = watermark_file(audio, out, words, beeps, tmp)
        (out_dir / f"{stem}_watermark.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"  leader {doc['leader_ms']} ms | {len(doc['tag_placements'])} "
              f"tag(s) at {[p['start_ms'] for p in doc['tag_placements']]} | "
              f"speech RMS {doc['levels']['program_speech_rms_db']} dBFS")


if __name__ == "__main__":
    main()
