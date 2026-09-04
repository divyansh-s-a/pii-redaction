"""Listener candidate: gemini-3.8-flash (newest generation, flash tier).
Same prompt/schema/windowing as the production Gemini engine — only the
model differs, so the bench isolates the model-version variable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gemini_transcribe


def transcribe_file(model, original, work_wav, out_json, channel_wavs=None):
    # reference-listener mode, two adjustments (2026-09-03):
    # - a safety-blocked floor window becomes a single-listener hole instead
    #   of failing the file (3.8-flash raises PROHIBITED_CONTENT on episode
    #   segments that 3.1-pro transcribes fine)
    # - 120 s windows: at the default 240 s, 3.8-flash silently
    #   under-transcribes dense speech (Kannada call minutes 0-3: 14-29
    #   words/min vs ~195 actual; at 120 s it matches 3.1-pro word counts)
    import common
    gemini_transcribe.SKIP_BLOCKED_WINDOWS = True
    old_len = common.CONFIG["chunk_len_s"]
    old_step = common.CONFIG["chunk_step_s"]
    old_thresh = common.CONFIG["chunk_threshold_s"]
    common.CONFIG["chunk_len_s"] = 120
    common.CONFIG["chunk_step_s"] = 105
    common.CONFIG["chunk_threshold_s"] = min(old_thresh, 120)
    try:
        return gemini_transcribe.transcribe_file(
            "gemini-3.8-flash", original, work_wav, out_json,
            channel_wavs=channel_wavs)
    finally:
        gemini_transcribe.SKIP_BLOCKED_WINDOWS = False
        common.CONFIG["chunk_len_s"] = old_len
        common.CONFIG["chunk_step_s"] = old_step
        common.CONFIG["chunk_threshold_s"] = old_thresh
