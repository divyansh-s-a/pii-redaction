"""Metadata / QC sidecar — runs after PII redaction to describe the delivered
audio.

A standalone pipeline (like dedupe/): given one processed file's working
directory it emits metadata.json with the content domain plus technical,
signal-level, diarization, language and redaction metadata. Almost everything
is read from artifacts the redaction pipeline already produced (prepare.json,
transcript.json, intervals.json) or one numpy pass over the FINAL beeped
deliverable, so the stage is fast. The only network touch is a small
domain-classification LLM call (with an offline keyword fallback).

Modules:
  probe.py        native codec / sample-rate / channels (ffprobe|afinfo|wave)
  audio_stats.py  peak / rms / clipping / silence / snr from the deliverable
  diarization.py  speaker distribution, turns, language from words
  domain.py       free-form content-domain label (LLM + heuristic fallback)
  run_metadata_qc.py  orchestrator CLI -> metadata.json
"""
