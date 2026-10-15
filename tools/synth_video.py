#!/usr/bin/env python3
"""Synthetic advertisement for the video arm (invented brands only).

One 12 s, 1280x720, 25 fps clip with four shots whose ground truth is known
per frame because every element is drawn here:
  1  moving   a wordmark panning across the frame + a drawn cartoon face
  2  moving   a Devanagari wordmark sliding down (OCR cannot read it)
  3  static   packshot: a product box with brand text and a round logo
  4  static   end card: wordmark, tagline, URL, toll-free number
Voice-over (macOS `say`) names the brands; a quiet chord bed runs under it.
Writes synth_ad.mp4 and synth_ad.gt.json (per-frame boxes, static flags,
audio entity times) next to the audio set's mapping.yaml.
"""
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "multimodal" / "video_pipeline"))
W, H, FPS, SECS = 1280, 720, 25, 12
LATIN = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
DEVA = "/System/Library/Fonts/Kohinoor.ttc"
VO = [(0.4, ["Introducing the new", ("Zorvex Vision", "brands:0"), "."]),
      (6.3, [("Zorvex", "brands:0"), ".",
             ("See more, live more", "brand_facts:0"), "."]),
      (9.4, ["Visit", ("Quillfin", "brands:3"), "today."])]


def _font(path, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)


def _text(d, xy, text, font, fill):
    bb = d.textbbox(xy, text, font=font)
    d.text(xy, text, font=font, fill=fill)
    return [bb[0], bb[1], bb[2], bb[3]]


def frame(k):
    """-> (RGB array, [gt elements])"""
    from PIL import Image, ImageDraw
    t = k / FPS
    els = []
    if t < 3:                                   # shot 1, moving
        im = Image.new("RGB", (W, H), (30, 60, 140))
        d = ImageDraw.Draw(im)
        x = int(60 + t * 260)
        els.append({"kind": "brand_text", "entity": "brands:0",
                    "box": _text(d, (x, 120), "ZORVEX", _font(LATIN, 110),
                                 (255, 255, 255))})
        cx, cy = int(1000 - t * 120), 470
        d.ellipse((cx - 90, cy - 110, cx + 90, cy + 110),
                  fill=(225, 185, 150))
        for ex in (-38, 38):
            d.ellipse((cx + ex - 18, cy - 40, cx + ex + 18, cy - 18),
                      fill=(255, 255, 255))
            d.ellipse((cx + ex - 8, cy - 36, cx + ex + 8, cy - 22),
                      fill=(30, 20, 10))
        d.polygon([(cx, cy - 10), (cx - 14, cy + 30), (cx + 14, cy + 30)],
                  fill=(200, 155, 125))
        d.arc((cx - 40, cy + 30, cx + 40, cy + 75), 15, 165,
              fill=(150, 50, 50), width=7)
        d.ellipse((cx - 95, cy - 130, cx + 95, cy - 40), fill=(60, 40, 30))
        els.append({"kind": "face", "entity": None,
                    "box": [cx - 95, cy - 130, cx + 90, cy + 110]})
        shot = 1
    elif t < 6:                                 # shot 2, moving Devanagari
        im = Image.new("RGB", (W, H), (230, 120, 30))
        d = ImageDraw.Draw(im)
        y = int(80 + (t - 3) * 120)
        els.append({"kind": "brand_text", "entity": "brands:0",
                    "box": _text(d, (380, y), "ज़ोरवेक्स", _font(DEVA, 120),
                                 (255, 255, 255))})
        shot = 2
    elif t < 9:                                 # shot 3, static packshot
        im = Image.new("RGB", (W, H), (235, 235, 240))
        d = ImageDraw.Draw(im)
        d.rectangle((380, 160, 900, 600), fill=(20, 20, 25))
        els.append({"kind": "packaging", "entity": "brands:0",
                    "box": [380, 160, 900, 600]})
        d.ellipse((440, 200, 560, 320), fill=(220, 30, 40))
        d.text((475, 222), "Z", font=_font(LATIN, 80), fill=(255, 255, 255))
        els.append({"kind": "logo", "entity": "brands:0",
                    "box": [440, 200, 560, 320]})
        els.append({"kind": "brand_text", "entity": "brands:0",
                    "box": _text(d, (440, 420), "Zorvex Vision",
                                 _font(LATIN, 64), (255, 255, 255))})
        shot = 3
    else:                                       # shot 4, static end card
        im = Image.new("RGB", (W, H), (250, 250, 250))
        d = ImageDraw.Draw(im)
        els.append({"kind": "brand_text", "entity": "brands:3",
                    "box": _text(d, (380, 140), "QUILLFIN", _font(LATIN, 96),
                                 (10, 90, 60))})
        els.append({"kind": "brand_text", "entity": "brand_facts:0",
                    "box": _text(d, (390, 290), "see more, live more",
                                 _font(LATIN, 48), (60, 60, 60))})
        els.append({"kind": "url_handle", "entity": "brands:3",
                    "box": _text(d, (390, 420), "www.quillfin-example.in",
                                 _font(LATIN, 40), (30, 30, 120))})
        els.append({"kind": "phone", "entity": None,
                    "box": _text(d, (390, 500), "1800 123 4567",
                                 _font(LATIN, 40), (30, 30, 30))})
        shot = 4
    return np.asarray(im), els, shot


def _voice_over(out_wav):
    """VO chunks at fixed times over a quiet chord bed; entity times exact."""
    fs = 16000
    n = SECS * fs
    bed = np.zeros(n, np.float32)
    t = np.arange(n) / fs
    for f in (196.0, 246.9, 293.7):
        bed += 0.012 * np.sin(2 * math.pi * f * t)
    ents = []
    with tempfile.TemporaryDirectory() as td:
        for start, chunks in VO:
            pos = int(start * fs)
            for ci, ch in enumerate(chunks):
                text, ent = (ch, None) if isinstance(ch, str) else ch
                if not text.strip(" ."):
                    continue
                p = Path(td) / f"{start}_{ci}.wav"
                subprocess.run(["say", "-v", "Rishi", "-o", str(p),
                                "--file-format=WAVE",
                                f"--data-format=LEI16@{fs}", text],
                               check=True, capture_output=True)
                with wave.open(str(p)) as w:
                    x = np.frombuffer(w.readframes(w.getnframes()),
                                      dtype="<i2").astype(np.float32) / 32768
                nz = np.nonzero(np.abs(x) > 0.01)[0]
                x = x[max(0, nz[0] - 80): nz[-1] + 80] if len(nz) else x
                end = min(n, pos + len(x))
                bed[pos:end] += x[: end - pos]
                if ent:
                    ents.append({"text": text, "kind": "M", "label": ent,
                                 "speaker": "VO",
                                 "start_ms": pos * 1000 // fs,
                                 "end_ms": end * 1000 // fs})
                pos = end + int(0.07 * fs)
    pcm = np.clip(bed * 32768, -32768, 32767).astype("<i2")
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())
    return ents


def make_video(out_dir: Path):
    import media_io
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = {"clip": "synth_ad", "fps": FPS, "width": W, "height": H,
          "shots": [{"shot": 1, "static": False}, {"shot": 2, "static": False},
                    {"shot": 3, "static": True}, {"shot": 4, "static": True}],
          "frames": {}}
    silent = out_dir / "_synth_ad_video.mp4"
    wr = media_io.Writer(silent, FPS, W, H, crf=18)
    for k in range(SECS * FPS):
        rgb, els, shot = frame(k)
        wr.write(rgb[:, :, ::-1].copy())
        gt["frames"][str(k)] = {"shot": shot, "elements": els}
    wr.close()
    wav = out_dir / "_synth_ad_vo.wav"
    gt["audio_entities"] = _voice_over(wav)
    media_io.mux(silent, wav, out_dir / "synth_ad.mp4")
    silent.unlink()
    wav.unlink()
    (out_dir / "synth_ad.gt.json").write_text(json.dumps(gt, ensure_ascii=False))
    print(f"  synth_ad: {SECS}s, {len(gt['frames'])} frames, "
          f"{len(gt['audio_entities'])} spoken entities -> "
          f"{out_dir / 'synth_ad.mp4'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1]).resolve()
    if ROOT in out.parents or out == ROOT:
        sys.exit("refusing to write media inside the repository")
    make_video(out)
