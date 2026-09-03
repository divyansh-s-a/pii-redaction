"""Listener candidate: ElevenLabs Scribe v1 (previous generation, same key).
Same adapter and keyterms as production scribe_v2 — only the model differs,
so the bench isolates the model-version variable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapters import elevenlabs_transcribe


def transcribe_file(model, original, work_wav, out_json, channel_wavs=None):
    return elevenlabs_transcribe.transcribe_file(
        "scribe_v1", original, work_wav, out_json, channel_wavs=channel_wavs)
