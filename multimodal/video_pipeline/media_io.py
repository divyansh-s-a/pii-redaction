#!/usr/bin/env python3
"""ffmpeg/OpenCV plumbing for the video arm.

ffmpeg is taken from PATH, else from the imageio-ffmpeg wheel (a static
build with libx264 and AAC; no ffprobe needed — OpenCV reads the stream
geometry and ffmpeg's own banner says whether there is audio).
"""
import re
import shutil
import subprocess
from pathlib import Path


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: install ffmpeg or "
                           "`pip install imageio-ffmpeg`")


def probe(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    banner = r.stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", banner)
    dur = (int((int(m.group(1)) * 3600 + int(m.group(2)) * 60
                + float(m.group(3))) * 1000) if m else int(n / fps * 1000))
    return {"fps": fps, "width": w, "height": h, "frames": n,
            "duration_ms": dur, "has_audio": " Audio: " in banner}


def extract_audio(src: Path, dst_wav: Path) -> Path:
    """Audio track -> 16-bit PCM WAV at its native rate and channel count."""
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(src), "-vn",
                    "-acodec", "pcm_s16le", str(dst_wav)], check=True,
                   capture_output=True)
    return dst_wav


def frames(path: Path):
    """Yield (index, BGR frame) for every frame."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield k, fr
        k += 1
    cap.release()


class Writer:
    """BGR frames -> H.264 (yuv420p, CRF 17) through an ffmpeg pipe."""

    def __init__(self, dst: Path, fps: float, w: int, h: int, crf: int = 17):
        self.proc = subprocess.Popen(
            [ffmpeg(), "-y", "-v", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
             "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags",
             "+faststart", str(dst)], stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def mux(video: Path, audio: Path, dst: Path) -> Path:
    """Redacted picture + redacted sound -> deliverable mp4 (AAC 192k)."""
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(video),
                    "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(dst)], check=True,
                   capture_output=True)
    return dst


def save_frame(frame, path: Path, max_side: int = 1280) -> Path:
    """JPEG copy of a frame for a model or OCR call (downscaled)."""
    import cv2
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path
