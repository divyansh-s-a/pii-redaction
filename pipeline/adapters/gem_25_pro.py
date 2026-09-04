"""Listener candidate: gemini-2.5-pro (previous generation, pro tier).
Same prompt/schema/windowing as the production Gemini engine — only the
model differs, so the bench isolates the model-version variable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gemini_transcribe


def transcribe_file(model, original, work_wav, out_json, channel_wavs=None):
    return gemini_transcribe.transcribe_file(
        "gemini-2.5-pro", original, work_wav, out_json,
        channel_wavs=channel_wavs)
