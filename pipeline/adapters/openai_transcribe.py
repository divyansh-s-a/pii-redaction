#!/usr/bin/env python3
"""Stage B (vendor) — transcribe one audio file with OpenAI (Whisper API).

Same output contract as pipeline/gemini_transcribe.transcribe_file so stages
C/D/E run unchanged. Uses response_format=verbose_json with word-level
timestamp granularity (float seconds -> int ms).

LIMITATION: the OpenAI transcription endpoint does NOT diarize, so every
utterance is attributed to a single speaker "S1"; words are split into
utterances only on silence gaps > 1.5 s. Speaker-dependent comparisons
(person_link, per-speaker QC) are therefore weaker for this engine. On
split-channel recordings each channel is still transcribed separately and
namespaced C0_S1 / C1_S1, which restores two-party separation.

Uploads are capped at 25 MB by OpenAI; an original above ~24 MB is replaced
by the work WAV re-encoded to AAC m4a (same timeline, much smaller).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import adapters
import audio_utils
import common
import gemini_transcribe

# Direct OpenAI only, never the OpenRouter proxy. Every OpenRouter request
# must ask for providers that keep no data (provider.data_collection =
# "deny"), and OpenRouter's multipart /audio/transcriptions endpoint takes no
# provider preferences, so it cannot carry call audio.
ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"
ENGINE = "openai"
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
        "mp4": "video/mp4"}
_FORMAT_ERR_MARKS = (b"response_format", b"verbose_json",
                     b"timestamp_granularities")


def _post(media: Path, mime: str, model: str, audio_seconds: float) -> dict:
    key = adapters.env_key("OPENAI_API_KEY")
    body, ctype = adapters.multipart_body(
        {"model": model or DEFAULT_MODEL, "response_format": "verbose_json",
         "timestamp_granularities[]": "word"},
        "file", media.name, media.read_bytes(), mime)
    headers = {"Authorization": "Bearer " + key, "Content-Type": ctype}
    for attempt in range(4):
        status, _, raw = common._request(ENDPOINT, data=body, headers=headers,
                                         timeout=1200)
        adapters.log_usage(ENGINE, ENDPOINT, media, audio_seconds, status)
        if status == 200:
            try:
                return json.loads(raw)
            except ValueError:
                raise RuntimeError(
                    f"openai returned non-JSON body: {raw[:400]!r}")
        if status in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print(f"  HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        if status == 400 and any(m in raw for m in _FORMAT_ERR_MARKS):
            raise RuntimeError(
                f"model {model!r} rejected verbose_json word timestamps — "
                "this pipeline needs word-level times; use --engine-model "
                f"whisper-1. Raw error: {raw[:400]!r}")
        raise RuntimeError(
            f"openai transcription HTTP {status}: {raw[:400]!r}")
    raise RuntimeError("openai transcription: retries exhausted")


def _upload_media(preferred: Path, work_wav: Path, tmp_dir: Path):
    """Pick the file to upload, honouring the 25 MB cap.

    Returns (path, mime, tmp_path_or_None); a non-None tmp path is a
    re-encoded m4a the caller must delete after use.
    """
    if preferred.stat().st_size <= MAX_UPLOAD_BYTES:
        return (preferred,
                MIME.get(preferred.suffix.lower().lstrip("."), "audio/mpeg"),
                None)
    tmp = tmp_dir / f"openai_upload_{preferred.stem}.m4a"
    print(f"  {preferred.name} exceeds 24 MB — re-encoding to m4a for upload")
    audio_utils.encode_wav(work_wav, tmp)
    return tmp, "audio/mp4", tmp


def to_utterances(resp) -> list:
    """OpenAI verbose_json response -> our utterance dicts (single speaker)."""
    if not isinstance(resp, dict) or not isinstance(resp.get("words"), list):
        raise RuntimeError(
            "unexpected OpenAI response shape (expected verbose_json with "
            "words[] — word timestamps need whisper-1): "
            + json.dumps(resp, ensure_ascii=False)[:400])
    tuples = []
    for w in resp["words"]:
        if not isinstance(w, dict):
            continue
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        try:
            s = int(round(float(w["start"]) * 1000))
            e = int(round(float(w["end"]) * 1000))
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("unexpected OpenAI word shape: "
                               + json.dumps(w, ensure_ascii=False)[:200])
        tuples.append(("spk", text, s, e))  # no diarization: one speaker
    lang = str(resp.get("language") or "unknown").strip() or "unknown"
    utts = adapters.group_words(tuples)
    for u in utts:
        u["lang"] = lang
        u["overlaps_with"] = ""
    return utts


def transcribe_file(model, original: Path, work_wav: Path, out_json: Path,
                    channel_wavs=None):
    """Mirror of gemini_transcribe.transcribe_file for OpenAI Whisper."""
    model_id = model or DEFAULT_MODEL
    dur_ms = audio_utils.duration_ms(work_wav)
    if channel_wavs:
        utts = []
        for chn, cw in enumerate(channel_wavs):
            print(f"  transcribing channel {chn} (openai) ...")
            secs = audio_utils.duration_ms(cw) / 1000.0
            media, mime, tmp = _upload_media(cw, cw, out_json.parent)
            try:
                ch_utts = gemini_transcribe.sanity(
                    to_utterances(_post(media, mime, model_id, secs)), dur_ms)
            finally:
                if tmp is not None:
                    tmp.unlink()
            for u in ch_utts:
                u["speaker"] = f"C{chn}_{u['speaker']}"
                u["channel"] = chn
            utts.extend(ch_utts)
        split = True
    else:
        media, mime, tmp = _upload_media(original, work_wav, out_json.parent)
        try:
            utts = gemini_transcribe.sanity(
                to_utterances(_post(media, mime, model_id, dur_ms / 1000.0)),
                dur_ms)
        finally:
            if tmp is not None:
                tmp.unlink()
        for u in utts:
            u["channel"] = None
        split = False
    utts.sort(key=lambda u: u["words"][0]["start_ms"])
    common.save_json(out_json, {"duration_ms": dur_ms,
                                "split_channels": split,
                                **gemini_transcribe.language_summary(utts),
                                "utterances": utts})
    n_words = sum(len(u["words"]) for u in utts)
    print(f"  transcript (openai): {len(utts)} utterances, {n_words} "
          f"words -> {out_json}")
    return out_json
