#!/usr/bin/env python3
"""Prompted-LLM transcription via OpenRouter audio-input chat models —
listener candidates for the judge reference (e.g. openai/gpt-audio,
mistralai/voxtral-small-24b-2507).

Same pattern as the Gemini listener: our transcription prompt + a strict
word-timestamp JSON schema, windowed to keep output under token caps, words
sanitized by gemini_transcribe.sanity() (timestamp repair applies equally).
Usage rows carry measured cost via OpenRouter usage accounting.
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audio_utils
import common
import gemini_transcribe
from adapters import env_key

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
WINDOW_S = 120           # word-JSON is output-heavy; small windows stay
STEP_S = 105             # under every candidate's completion cap
MIN_WINDOW_S = 30

# strict-mode mirror of the Gemini transcription schema
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "utterances": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "speaker": {"type": "string"},
                    "lang": {"type": "string"},
                    "overlaps_with": {"type": "string"},
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "w": {"type": "string"},
                                "start_ms": {"type": "integer"},
                                "end_ms": {"type": "integer"},
                                "conf": {"type": "string",
                                         "enum": ["high", "medium", "low"]},
                            },
                            "required": ["w", "start_ms", "end_ms", "conf"],
                        },
                    },
                },
                "required": ["speaker", "lang", "overlaps_with", "words"],
            },
        }
    },
    "required": ["utterances"],
}


# models that reject response_format json_schema (e.g. openai/gpt-audio)
# fall back to a prompt-embedded JSON demand + lenient extraction
_NO_SCHEMA = set()

JSON_DEMAND = (
    "\n\nOUTPUT FORMAT: reply with ONLY a JSON object, no prose, no markdown "
    "fences, exactly this shape: {\"utterances\":[{\"speaker\":\"S1\","
    "\"lang\":\"...\",\"overlaps_with\":\"\",\"words\":[{\"w\":\"...\","
    "\"start_ms\":0,\"end_ms\":0,\"conf\":\"high\"}]}]}")


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise RuntimeError(f"no JSON object in reply: {text[:160]!r}")
    return json.loads(text[a:b + 1])


def _call(model_id, wav_path: Path, context_text=""):
    prompt = gemini_transcribe.TRANSCRIBE_PROMPT
    if context_text:
        prompt += ("\n\nThis audio CONTINUES a longer recording. The previous "
                   "part ended with:\n" + context_text +
                   "\nKeep the same speaker labels for the same voices.")
    use_schema = model_id not in _NO_SCHEMA
    if not use_schema:
        prompt += JSON_DEMAND
    b64 = base64.b64encode(wav_path.read_bytes()).decode()
    payload = {
        "model": model_id,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 24000,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio",
             "input_audio": {"data": b64, "format": "wav"}},
        ]}],
        "usage": {"include": True},
        "provider": common.openrouter_provider(),
    }
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "transcript", "strict": True,
                            "schema": SCHEMA}}
    key = env_key("OPENROUTER_API_KEY")
    for attempt in range(4):
        status, _, body = common._request(
            ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"}, timeout=1200)
        if status == 200:
            resp = json.loads(body)
            usage = resp.get("usage", {})
            common.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            with open(common.OUTPUT_ROOT / "api_usage.jsonl", "a") as f:
                f.write(json.dumps({
                    "engine": "openrouter_audio", "model": model_id,
                    "context": common.USAGE_CONTEXT["tag"] or None,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cost_usd": usage.get("cost")}) + "\n")
            choice = resp["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("output truncated at max tokens — reduce chunk size")
            content = choice["message"].get("content") or ""
            try:
                return json.loads(content)
            except ValueError:
                return _extract_json(content)
        if (status == 400 and b"json_schema" in body
                and b"not supported" in body and use_schema):
            _NO_SCHEMA.add(model_id)
            print(f"    {model_id}: no json_schema support -> prompt-JSON mode")
            return _call(model_id, wav_path, context_text)
        if status in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print(f"    HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"openrouter audio HTTP {status}: {body[:300]!r}")
    raise RuntimeError("openrouter audio: retries exhausted")


def _window(model_id, work_wav, out_dir, start_s, length_s, dur_ms, context):
    chunk = out_dir / "or_chunk_tmp.wav"
    gemini_transcribe.cut_wav_chunk(work_wav, chunk, start_s, length_s)
    try:
        try:
            res = _call(model_id, chunk, context)
        except RuntimeError as e:
            if "truncated at max tokens" not in str(e) or length_s < MIN_WINDOW_S * 2:
                raise
            half = length_s / 2
            print(f"    window @{start_s:.0f}s truncated -> {half:.0f}s halves")
            left = _window(model_id, work_wav, out_dir, start_s, half,
                           dur_ms, context)
            right = _window(model_id, work_wav, out_dir, start_s + half,
                            length_s - half, dur_ms,
                            gemini_transcribe._tail_context(left) or context)
            return left + right
    finally:
        chunk.unlink(missing_ok=True)
    return gemini_transcribe.sanity(res.get("utterances", []), dur_ms,
                                    offset_ms=int(start_s * 1000))


def transcribe_with(model_id: str, original: Path, work_wav: Path,
                    out_json: Path, channel_wavs=None):
    dur_ms = audio_utils.duration_ms(work_wav)
    utts, start, context = [], 0.0, ""
    idx = 0
    while start * 1000 < dur_ms:
        print(f"    window {idx} @ {start:.0f}s ({model_id})")
        res = _window(model_id, work_wav, out_json.parent, start, WINDOW_S,
                      dur_ms, context)
        keep_until = (start + STEP_S + (WINDOW_S - STEP_S) / 2) * 1000
        drop_before = (start + (WINDOW_S - STEP_S) / 2) * 1000 if idx else 0
        kept = []
        for u in res:
            words = [w for w in u["words"]
                     if drop_before <= (w["start_ms"] + w["end_ms"]) / 2
                     < (keep_until if (start + WINDOW_S) * 1000 < dur_ms
                        else dur_ms + 1)]
            if words:
                kept.append({**u, "words": words})
        utts.extend(kept)
        if kept:
            context = gemini_transcribe._tail_context(kept)
        start += STEP_S
        idx += 1
    for u in utts:
        u["channel"] = None
    utts.sort(key=lambda u: u["words"][0]["start_ms"])
    common.save_json(out_json, {"duration_ms": dur_ms,
                                "split_channels": False,
                                **gemini_transcribe.language_summary(utts),
                                "utterances": utts})
    n = sum(len(u["words"]) for u in utts)
    print(f"    transcript ({model_id}): {len(utts)} utterances, {n} words")
    return out_json
