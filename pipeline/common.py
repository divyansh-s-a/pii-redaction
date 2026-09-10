#!/usr/bin/env python3
"""Shared helpers for the audio PII pipeline: config, .env, Gemini REST client.

Stdlib only (urllib) — no pip dependencies. The API key is read from .env at
the project root and is never printed or logged.
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "pipeline" / "config.json").read_text())

# Container/codec of the rendered beeped deliverable (config: beep_format;
# default FLAC lossless). One source of truth so the renderer, QC, edge
# verify, delivery copy and web download all agree on the filename.
BEEP_FORMAT = CONFIG.get("beep_format", "flac")


def beeped_name(file_id: str) -> str:
    return f"{file_id}_beeped.{BEEP_FORMAT}"


class AIStudioUnavailable(RuntimeError):
    """Raised when the Gemini AI Studio REST backend cannot serve a call
    (missing key, network failure, auth rejection, or exhausted 5xx/429), so
    generate() can transparently fall back to the OpenRouter-served Gemini."""


def _is_unavailable(status: int) -> bool:
    """HTTP statuses meaning 'backend unavailable' rather than 'bad request'.
    0 is our sentinel for a network-level failure (see _request)."""
    return status in (0, 401, 403, 408, 429, 500, 502, 503, 504)


_BACKEND = {"ai_studio_down": False}


def _read_env_value(name: str):
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith(name + "=") and line.split("=", 1)[1]:
                return line.split("=", 1)[1].strip()
    return None


def gemini_key():
    """Gemini AI Studio key or None (never raises — callers may fall back)."""
    return _read_env_value("GEMINI_API_KEY")


def openrouter_key():
    """OpenRouter key or None (powers both the judge and the Gemini fallback)."""
    return _read_env_value("OPENROUTER_API_KEY")


def api_key() -> str:
    k = gemini_key()
    if not k:
        raise SystemExit("GEMINI_API_KEY missing: put it in .env at the project root")
    return k


def _ai_studio_available() -> bool:
    return bool(gemini_key()) and not _BACKEND["ai_studio_down"]


def _mark_ai_studio_down(reason: str):
    if not _BACKEND["ai_studio_down"]:
        print(f"  ** Gemini AI Studio unavailable ({reason}) -> routing "
              "Gemini calls through OpenRouter for the rest of this run **")
    _BACKEND["ai_studio_down"] = True


def _openrouter_model(model: str) -> str:
    """Map an AI Studio model id to its OpenRouter slug. Configurable via
    config.json 'openrouter_model_map'; defaults to google/<model>."""
    mp = CONFIG.get("openrouter_model_map", {})
    if model in mp:
        return mp[model]
    return model if "/" in model else f"google/{model}"


_JSON_TYPE = {"OBJECT": "object", "STRING": "string", "INTEGER": "integer",
              "ARRAY": "array", "BOOLEAN": "boolean", "NUMBER": "number"}


def _to_json_schema(node):
    """Convert a Gemini responseSchema (UPPERCASE types) into a JSON-Schema
    dict OpenRouter accepts (lowercase types), recursively."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _JSON_TYPE.get(v, v.lower())
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: _to_json_schema(pv) for pk, pv in v.items()}
            elif k == "items":
                out[k] = _to_json_schema(v)
            else:
                out[k] = v
        return out
    if isinstance(node, list):
        return [_to_json_schema(x) for x in node]
    return node


def _request(url, data=None, headers=None, method=None, timeout=300):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # network-level failure -> status 0 sentinel (see _is_unavailable)
        return 0, {}, f"{type(e).__name__}: {e}".encode()


def upload_file(path: Path, mime_type: str, display_name: str) -> dict:
    """Resumable upload to the Gemini Files API; waits until the file is ACTIVE."""
    key = api_key()
    base = CONFIG["api_base"]
    size = path.stat().st_size
    status, headers, body = _request(
        f"{base}/upload/v1beta/files?key={key}",
        data=json.dumps({"file": {"display_name": display_name}}).encode(),
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        })
    if status != 200:
        msg = f"upload start failed HTTP {status}: {body[:400]!r}"
        raise (AIStudioUnavailable(msg) if _is_unavailable(status)
               else RuntimeError(msg))
    upload_url = headers.get("X-Goog-Upload-URL") or headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("no upload URL returned")
    status, _, body = _request(
        upload_url, data=path.read_bytes(),
        headers={"X-Goog-Upload-Command": "upload, finalize",
                 "X-Goog-Upload-Offset": "0",
                 "Content-Length": str(size)},
        timeout=1200)
    if status != 200:
        msg = f"upload failed HTTP {status}: {body[:400]!r}"
        raise (AIStudioUnavailable(msg) if _is_unavailable(status)
               else RuntimeError(msg))
    info = json.loads(body)["file"]
    name = info["name"]
    for _ in range(120):  # audio is processed server-side before it is usable
        if info.get("state") == "ACTIVE":
            return info
        time.sleep(2)
        status, _, body = _request(f"{base}/v1beta/{name}?key={key}")
        if status == 200:
            info = json.loads(body)
    raise RuntimeError(f"file {name} never became ACTIVE (state={info.get('state')})")


# Optional stage/file attribution stamped onto usage-log rows, so per-config
# cost can be computed from output/api_usage.jsonl afterwards. Callers set it
# around a pipeline stage ("transcribe:<fid>", "detect:<fid>",
# "edge_verify:<fid>"); empty = untagged (back-compatible).
USAGE_CONTEXT = {"tag": ""}


def set_usage_context(tag: str):
    USAGE_CONTEXT["tag"] = tag or ""


def _log_usage_row(row: dict):
    log = ROOT / "output" / "api_usage.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")


def generate(model: str, parts: list, response_schema=None,
             temperature: float = 0.0, max_output_tokens: int = 65536,
             thinking_budget=None) -> str:
    """generateContent returning the first candidate's text.

    Primary backend is Gemini AI Studio (generativelanguage.googleapis.com).
    If AI Studio is unavailable — no GEMINI_API_KEY, a network failure, an auth
    rejection, or exhausted 5xx/429 — the call transparently falls back to the
    SAME Gemini model served through OpenRouter (needs OPENROUTER_API_KEY),
    when one is configured. Once a run trips the fallback it stays on
    OpenRouter for the rest of the process (no per-call re-probing).

    Audio parts must be supplied as {"audio_path": <path>, "mime": <mime>} so
    both backends can carry them (AI Studio uploads to the Files API,
    OpenRouter inlines the bytes as base64). thinking_budget caps the model's
    reasoning tokens (AI Studio only); None leaves the model default. Small
    snippet jobs MUST cap it — observed ~63k thinking tokens on a 2-second
    edge-verify snippet."""
    or_key = openrouter_key()
    if _ai_studio_available():
        try:
            return _generate_ai_studio(model, parts, response_schema,
                                       temperature, max_output_tokens,
                                       thinking_budget)
        except AIStudioUnavailable as e:
            if not or_key:
                raise
            _mark_ai_studio_down(str(e)[:120])
    if or_key:
        return _generate_openrouter(model, parts, response_schema,
                                    temperature, max_output_tokens)
    raise SystemExit(
        "no Gemini backend available: set GEMINI_API_KEY (AI Studio) and/or "
        "OPENROUTER_API_KEY (fallback) in .env at the project root")


def _generate_ai_studio(model, parts, response_schema, temperature,
                        max_output_tokens, thinking_budget) -> str:
    key = api_key()
    resolved = []
    for p in parts:
        if "audio_path" in p:
            info = upload_file(Path(p["audio_path"]), p["mime"],
                               Path(p["audio_path"]).name)
            resolved.append({"file_data": {"file_uri": info["uri"],
                                           "mime_type": p["mime"]}})
        else:
            resolved.append(p)
    gen_cfg = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    if thinking_budget is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if response_schema is not None:
        gen_cfg["responseMimeType"] = "application/json"
        gen_cfg["responseSchema"] = response_schema
    payload = {"contents": [{"role": "user", "parts": resolved}],
               "generationConfig": gen_cfg}
    url = f"{CONFIG['api_base']}/v1beta/models/{model}:generateContent?key={key}"
    for attempt in range(4):
        status, _, body = _request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, timeout=1200)
        if status == 200:
            resp = json.loads(body)
            um = resp.get("usageMetadata", {})
            if um:
                _log_usage_row({
                    "engine": "gemini", "model": model,
                    "context": USAGE_CONTEXT["tag"] or None,
                    "prompt_tokens": um.get("promptTokenCount"),
                    "output_tokens": um.get("candidatesTokenCount"),
                    "thinking_tokens": um.get("thoughtsTokenCount"),
                    "total_tokens": um.get("totalTokenCount")})
            if not resp.get("candidates"):
                # 200 with no candidates: promptFeedback block or a transient
                # empty response — retry like a 5xx, then fail with the body
                fb = json.dumps(resp.get("promptFeedback", resp))[:300]
                if attempt < 3:
                    wait = 15 * (attempt + 1)
                    print(f"  empty candidates ({fb}), retrying in {wait}s ...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"no candidates in response: {fb}")
            cand = resp["candidates"][0]
            fr = cand.get("finishReason", "")
            text = "".join(p.get("text", "")
                           for p in cand.get("content", {}).get("parts", []))
            if fr == "MAX_TOKENS":
                raise RuntimeError("output truncated at max tokens — reduce chunk size")
            if fr not in ("STOP", ""):
                raise RuntimeError(f"generation stopped: {fr}")
            return text
        if status in (429, 500, 502, 503, 504):
            wait = 15 * (attempt + 1)
            print(f"  HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        if _is_unavailable(status):
            raise AIStudioUnavailable(
                f"generateContent HTTP {status}: {body[:200]!r}")
        raise RuntimeError(f"generateContent HTTP {status}: {body[:600]!r}")
    raise AIStudioUnavailable("generateContent: retries exhausted (AI Studio)")


def _strip_fences(text: str) -> str:
    """Best-effort JSON extraction from an OpenRouter reply that may arrive
    wrapped in a ```json fence or with leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        return t[a:b + 1]
    return t


def _generate_openrouter(model, parts, response_schema, temperature,
                         max_output_tokens) -> str:
    import base64
    key = openrouter_key()
    or_model = _openrouter_model(model)
    content = []
    for p in parts:
        if "audio_path" in p:
            data = Path(p["audio_path"]).read_bytes()
            mime = p.get("mime", "audio/wav")
            fmt = ("mp3" if ("mpeg" in mime or "mp3" in mime)
                   else "mp4" if ("mp4" in mime or "m4a" in mime)
                   else "wav")
            content.append({"type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(data).decode(),
                                "format": fmt}})
        elif "text" in p:
            content.append({"type": "text", "text": p["text"]})
        elif "file_data" in p:
            raise RuntimeError(
                "OpenRouter Gemini fallback needs raw audio; got a Gemini "
                "Files API URI. Pass audio as an {'audio_path','mime'} part.")
    payload = {"model": or_model, "temperature": temperature,
               "max_tokens": min(int(max_output_tokens), 32000),
               "messages": [{"role": "user", "content": content}],
               "usage": {"include": True}}
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "out", "strict": False,
                            "schema": _to_json_schema(response_schema)}}
    base = CONFIG.get("openrouter_api_base", "https://openrouter.ai/api/v1")
    url = base.rstrip("/") + "/chat/completions"
    for attempt in range(4):
        status, _, body = _request(
            url, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"}, timeout=1200)
        if status == 200:
            resp = json.loads(body)
            usage = resp.get("usage", {})
            _log_usage_row({
                "engine": "gemini_openrouter", "model": or_model,
                "context": USAGE_CONTEXT["tag"] or None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cost_usd": usage.get("cost")})
            choice = resp["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("output truncated at max tokens — reduce chunk size")
            text = choice["message"].get("content") or ""
            return _strip_fences(text) if response_schema is not None else text
        if status in (0, 429, 500, 502, 503, 504) and attempt < 3:
            wait = 15 * (attempt + 1)
            print(f"  OpenRouter HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"OpenRouter gemini HTTP {status}: {body[:400]!r}")
    raise RuntimeError("OpenRouter gemini: retries exhausted")


def resolve_model() -> str:
    """Use configured model; if unknown to the API, try fallbacks and report.
    When AI Studio is unavailable (no key / unreachable) the configured model
    is returned as-is — the OpenRouter fallback in generate() maps it."""
    key = gemini_key()
    if not key:
        print("  (no GEMINI_API_KEY; using configured model — Gemini calls "
              "will route through the OpenRouter fallback)")
        return CONFIG["model"]
    status, _, body = _request(
        f"{CONFIG['api_base']}/v1beta/models?pageSize=1000&key={key}")
    if status != 200:
        print("  (could not list models; using configured model as-is)")
        return CONFIG["model"]
    names = {m["name"].split("/")[-1] for m in json.loads(body).get("models", [])}
    for m in [CONFIG["model"], *CONFIG.get("fallback_models", [])]:
        if m in names:
            return m
    gem = sorted(n for n in names if "gemini" in n and "pro" in n)
    raise SystemExit(
        "none of the configured models exist for this key.\n"
        f"Available pro models: {gem}\nEdit pipeline/config.json accordingly.")


def load_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
