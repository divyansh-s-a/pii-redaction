#!/usr/bin/env python3
"""Vendor transcription adapters (ElevenLabs, OpenAI Whisper) for stage B.

Each adapter module exposes transcribe_file(model, original, work_wav,
out_json, channel_wavs=None) with the same semantics and output file shape as
pipeline/gemini_transcribe.transcribe_file, so detection / redaction / QC run
unchanged and engines can be compared like-for-like.

This package module holds the pieces both adapters share: env_key() reads a
vendor API key from the project-root .env (same convention as
common.api_key), multipart_body() builds a multipart/form-data upload with
the stdlib only, group_words() turns a vendor's flat word stream into our
utterance shape, and log_usage() appends one JSON line per API call to
output/api_usage.jsonl.
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common


def env_key(name):
    """Read NAME=value from .env at the project root (then PII_ENV_FILE,
    then the process environment); SystemExit when absent."""
    val = common._read_env_value(name)
    if val:
        return val
    raise SystemExit(
        f"{name} missing: put {name}=<key> in .env at the project root")


def multipart_body(fields, file_field, filename, file_bytes, file_mime):
    """Build a multipart/form-data body (text fields + one file part).

    Returns (body_bytes, content_type_header_value). Stdlib only — manual
    boundary construction, no requests dependency.
    """
    boundary = "----audio-pii-" + uuid.uuid4().hex
    b = boundary.encode()
    parts = []
    for k, v in fields.items():
        values = v if isinstance(v, (list, tuple)) else [v]
        for one in values:   # a list value becomes repeated same-name fields
            parts += [b"--" + b,
                      f'Content-Disposition: form-data; name="{k}"'.encode(),
                      b"", str(one).encode("utf-8")]
    parts += [b"--" + b,
              (f'Content-Disposition: form-data; name="{file_field}"; '
               f'filename="{filename}"').encode(),
              f"Content-Type: {file_mime}".encode(),
              b"", file_bytes,
              b"--" + b + b"--", b""]
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"


def group_words(words, gap_ms=1500):
    """Group a flat, time-ordered word stream into utterances.

    words: iterable of (speaker_key, text, start_ms, end_ms). A new utterance
    starts on a speaker change or a silence gap > gap_ms. Vendor speaker keys
    are mapped to S1, S2, ... in order of first appearance. The caller fills
    in lang / overlaps_with / channel.
    """
    labels = {}
    utts = []
    prev_key, prev_end = None, None
    for key, text, s_ms, e_ms in words:
        if key not in labels:
            labels[key] = f"S{len(labels) + 1}"
        if (not utts or key != prev_key
                or (prev_end is not None and s_ms - prev_end > gap_ms)):
            utts.append({"speaker": labels[key], "words": []})
        utts[-1]["words"].append({"w": text, "start_ms": int(s_ms),
                                  "end_ms": int(e_ms), "conf": "high"})
        prev_key, prev_end = key, e_ms
    return utts


def log_usage(engine, endpoint, file, audio_seconds, http_status):
    """Append one JSON usage line to output/api_usage.jsonl (no timestamps)."""
    p = common.OUTPUT_ROOT / "api_usage.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"engine": engine, "endpoint": endpoint,
                            "file": str(file),
                            "audio_seconds": round(float(audio_seconds), 3),
                            "http_status": http_status}) + "\n")
