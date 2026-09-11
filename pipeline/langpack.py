#!/usr/bin/env python3
"""Per-language resource packs (owner direction 2026-09-11).

The base pipeline must detect well WITHOUT the correction loop, so
language-specific recall lives here: a transcription keyterms vocabulary and
a detection-prompt supplement per language family, selected per run/batch.
Pack content is GENERIC domain vocabulary and synthetic examples — never
harvested verbatim from corpus PII (same principle as keyterms_india_v1).

Selection: tools/redact.py --langpack <id> (sets common.CONFIG['langpack']),
a 'langpack' key in pipeline/config.json, or the AUDIOPII_LANGPACK env var.
Family aliases collapse mutually-intelligible pairs (id -> ms).
"""
import os
from pathlib import Path

import common

PACK_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "langpacks"
ALIASES = {"ms": "ms", "id": "ms", "malay": "ms", "indonesian": "ms",
           "bahasa": "ms"}
KEYTERM_CAP = 100      # ElevenLabs keyterms limit


def active():
    """Active pack id or None."""
    raw = (os.environ.get("AUDIOPII_LANGPACK")
           or str(common.CONFIG.get("langpack") or "")).strip().lower()
    pack = ALIASES.get(raw, raw)
    return pack if pack and (PACK_ROOT / pack).is_dir() else None


def keyterms():
    """Pack keyterms (replace the base vocabulary while active), or None."""
    p = active()
    if not p:
        return None
    f = PACK_ROOT / p / "keyterms.txt"
    if not f.exists():
        return None
    terms = [t.strip() for t in f.read_text().splitlines()
             if t.strip() and not t.startswith("#")]
    return terms[:KEYTERM_CAP] or None


def detection_supplement():
    """Text appended to the detection policy prompt, or ''."""
    p = active()
    if not p:
        return ""
    f = PACK_ROOT / p / "detection_supplement.txt"
    if not f.exists():
        return ""
    return "\n\n" + f.read_text().strip() + "\n"
