#!/usr/bin/env python3
"""Stage B (vendor) — transcribe one audio file with ElevenLabs speech-to-text.

Same output contract as pipeline/gemini_transcribe.transcribe_file (writes
transcript.json with duration_ms / split_channels / language fields /
utterances) so stages C/D/E run unchanged. ElevenLabs returns a flat diarized
word stream with float-second timestamps; we group consecutive words of one
speaker into utterances (speaker change or a silence gap > 1.5 s starts a new
one) and map vendor speaker ids to S1, S2, ... in order of first appearance.
Split-channel recordings mirror the Gemini path: each channel WAV is
transcribed separately, speakers are namespaced C0_/C1_ and the utterance
carries its channel so beeps can target one side of the call.
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

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_MODEL = "scribe_v2"
SKIP_TYPES = {"spacing", "audio_event"}
MIME = {"mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav",
        "mp4": "video/mp4"}


def _keyterms():
    """Recognition-biasing vocabulary (Fix 2, 2026-09-01): scribe drops Indian
    proper nouns it cannot resolve (observed: a thrice-spoken name absent from
    the transcript). The list is GENERIC domain vocabulary — common Indian
    given names + major insurers/TPAs/hospitals — maintained in
    prompts/keyterms_india_v1.txt, deliberately not harvested from corpus
    transcripts (documented in System_Design)."""
    import langpack
    pack_terms = langpack.keyterms()
    if pack_terms:
        terms = pack_terms     # active language pack replaces the base list
    else:
        f = common.ROOT / "prompts" / "keyterms_india_v1.txt"
        if not f.exists():
            return _with_mapped(None)
        terms = [t.strip() for t in f.read_text().splitlines() if t.strip()]
    return _with_mapped(terms[:100] or None)


def _with_mapped(terms):
    """With a mapping active, its real spoken forms go FIRST (the 100-term
    cap then trims the generic tail): a listener that spells a brand the
    same way every time is one the entity matcher can find."""
    import entity_map
    em = entity_map.active()
    if em is None:
        return terms
    merged = list(dict.fromkeys(em.keyterms() + (terms or [])))
    return merged[:100] or None


def _post(media: Path, mime: str, model_id: str, audio_seconds: float) -> dict:
    fields = {"model_id": model_id, "diarize": "true",
              "timestamps_granularity": "word", "tag_audio_events": "false"}
    kt = _keyterms()
    if kt:
        fields["keyterms"] = [t for t in kt if len(t) < 50]
    body, ctype = adapters.multipart_body(
        fields, "file", media.name, media.read_bytes(), mime)
    headers = {"xi-api-key": adapters.env_key("ELEVENLABS_API_KEY"),
               "Content-Type": ctype}
    # a long recording is answered only after the whole file is processed
    # (a 2 h focus group can stay silent past 20 min): scale the read wait
    wait_s = max(1200, int(audio_seconds) + 600)
    timeouts = 0
    for attempt in range(4):
        status, _, raw = common._request(ENDPOINT, data=body, headers=headers,
                                         timeout=wait_s)
        adapters.log_usage("elevenlabs", ENDPOINT, media, audio_seconds, status)
        if status == 0 and b"timed out" in raw and timeouts == 0:
            timeouts += 1              # one retry on a network-level timeout
            print(f"  elevenlabs: {raw[:80]!r}, retrying once ...")
            continue
        if status == 200:
            try:
                return json.loads(raw)
            except ValueError:
                raise RuntimeError(
                    f"elevenlabs returned non-JSON body: {raw[:400]!r}")
        if status in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print(f"  HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        raise RuntimeError(
            f"elevenlabs speech-to-text HTTP {status}: {raw[:400]!r}")
    raise RuntimeError("elevenlabs speech-to-text: retries exhausted")


def to_utterances(resp) -> list:
    """ElevenLabs response -> our utterance dicts (lang/overlaps_with set)."""
    if not isinstance(resp, dict) or not isinstance(resp.get("words"), list):
        raise RuntimeError(
            "unexpected ElevenLabs response shape (expected words[]): "
            + json.dumps(resp, ensure_ascii=False)[:400])
    tuples = []
    for w in resp["words"]:
        if not isinstance(w, dict) or w.get("type") in SKIP_TYPES:
            continue
        text = str(w.get("text", "")).strip()
        if not text:
            continue
        try:
            s = int(round(float(w["start"]) * 1000))
            e = int(round(float(w["end"]) * 1000))
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("unexpected ElevenLabs word shape: "
                               + json.dumps(w, ensure_ascii=False)[:200])
        tuples.append((w.get("speaker_id", "speaker_0"), text, s, e))
    lang = str(resp.get("language_code") or "unknown").strip() or "unknown"
    utts = adapters.group_words(tuples)
    for u in utts:
        u["lang"] = lang
        u["overlaps_with"] = ""
    return utts


def transcribe_file(model, original: Path, work_wav: Path, out_json: Path,
                    channel_wavs=None):
    """Mirror of gemini_transcribe.transcribe_file for ElevenLabs."""
    model_id = model or DEFAULT_MODEL
    dur_ms = audio_utils.duration_ms(work_wav)
    if channel_wavs:
        utts = []
        for chn, cw in enumerate(channel_wavs):
            print(f"  transcribing channel {chn} (elevenlabs) ...")
            secs = audio_utils.duration_ms(cw) / 1000.0
            ch_utts = gemini_transcribe.sanity(
                to_utterances(_post(cw, "audio/wav", model_id, secs)), dur_ms)
            for u in ch_utts:
                u["speaker"] = f"C{chn}_{u['speaker']}"
                u["channel"] = chn
            utts.extend(ch_utts)
        split = True
    else:
        # the decoded work audio, not the original: an MP3's timeline can
        # differ between ElevenLabs' decoder and ours (observed: +4 s drift
        # over 2 h), and every beep is placed on OUR timeline
        utts = gemini_transcribe.sanity(
            to_utterances(_post(work_wav, "audio/wav", model_id,
                                dur_ms / 1000.0)),
            dur_ms)
        for u in utts:
            u["channel"] = None
        split = False
    utts.sort(key=lambda u: u["words"][0]["start_ms"])
    common.save_json(out_json, {"duration_ms": dur_ms,
                                "split_channels": split,
                                **gemini_transcribe.language_summary(utts),
                                "utterances": utts})
    n_words = sum(len(u["words"]) for u in utts)
    print(f"  transcript (elevenlabs): {len(utts)} utterances, {n_words} "
          f"words -> {out_json}")
    return out_json
