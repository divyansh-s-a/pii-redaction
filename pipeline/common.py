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


def api_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and line.split("=", 1)[1]:
                return line.split("=", 1)[1].strip()
    raise SystemExit("GEMINI_API_KEY missing: put it in .env at the project root")


def _request(url, data=None, headers=None, method=None, timeout=300):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


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
        raise RuntimeError(f"upload start failed HTTP {status}: {body[:400]!r}")
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
        raise RuntimeError(f"upload failed HTTP {status}: {body[:400]!r}")
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


def generate(model: str, parts: list, response_schema=None,
             temperature: float = 0.0, max_output_tokens: int = 65536,
             thinking_budget=None) -> str:
    """generateContent call returning the text of the first candidate.
    thinking_budget caps the model's reasoning tokens (Gemini thinkingConfig);
    None leaves the model default. Small snippet jobs MUST cap it — observed:
    ~63k thinking tokens on a 2-second edge-verify snippet."""
    key = api_key()
    gen_cfg = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    if thinking_budget is not None:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if response_schema is not None:
        gen_cfg["responseMimeType"] = "application/json"
        gen_cfg["responseSchema"] = response_schema
    payload = {"contents": [{"role": "user", "parts": parts}],
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
                log = ROOT / "output" / "api_usage.jsonl"
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, "a") as f:
                    f.write(json.dumps({
                        "engine": "gemini", "model": model,
                        "context": USAGE_CONTEXT["tag"] or None,
                        "prompt_tokens": um.get("promptTokenCount"),
                        "output_tokens": um.get("candidatesTokenCount"),
                        "thinking_tokens": um.get("thoughtsTokenCount"),
                        "total_tokens": um.get("totalTokenCount")}) + "\n")
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
        if status in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print(f"  HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"generateContent HTTP {status}: {body[:600]!r}")
    raise RuntimeError("generateContent: retries exhausted")


def resolve_model() -> str:
    """Use configured model; if unknown to the API, try fallbacks and report."""
    key = api_key()
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
