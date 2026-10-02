#!/usr/bin/env python3
"""The GPT detector's second road: OpenAI's own API, when OpenRouter is shut.

The detector reaches GPT through pipeline/judge_client.py, which is shared
with the audio arm and speaks only to OpenRouter. When that account ran out
of credit every page's GPT call failed, and the pipeline fell back to one
detector family for the whole run. This module is the document arm's own
transport to the same model at OpenAI, used only after OpenRouter has refused
for an ACCOUNT reason (no credit, bad key) -- never for a slow or flaky
request, which judge_client already retries.

The switch is sticky: once OpenRouter has said "no credit" it will say so for
every page, and asking it again costs a round trip per call.

Cost: OpenAI's reply carries tokens but no dollar figure, unlike OpenRouter's.
The usage row therefore records an ESTIMATE, labelled as one, from per-token
prices fitted to 6,698 OpenRouter-billed calls to the same model (median error
6.2%).
"""
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

ENDPOINT = "https://api.openai.com/v1/chat/completions"
PRICE_IN, PRICE_OUT = 2.35e-6, 11.26e-6     # USD per token, fitted (see above)
ACCOUNT_ERRORS = ("HTTP 401", "HTTP 402", "HTTP 403", "insufficient",
                  "credit", "quota")

STATE = {"direct": False, "announced": False}


def key():
    return common._read_env_value("OPENAI_API_KEY")


def is_account_error(exc):
    s = str(exc).lower()
    return any(k.lower() in s for k in ACCOUNT_ERRORS)


def model_id(model):
    """OpenRouter slugs carry the vendor ("openai/gpt-5.6-sol"); OpenAI's
    own API wants the bare id."""
    return str(model).split("/", 1)[-1]


def call(prompt, schema, model, images=None, log_engine="detector_gpt",
         max_tokens=16000):
    """One structured-output call to OpenAI directly -> parsed JSON object."""
    k = key()
    if not k:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    if images:
        content = [{"type": "text", "text": prompt}]
        for path in images:
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,"
                       + base64.b64encode(Path(path).read_bytes()).decode()}})
    else:
        content = prompt
    payload = {
        "model": model_id(model),
        "messages": [{"role": "user", "content": content}],
        # no temperature: this model accepts only its default, and sending 0
        # (as the OpenRouter path does) is rejected outright
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "judgement", "strict": True, "schema": schema}},
    }
    for attempt in range(4):
        status, _, body = common._request(
            ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {k}",
                     "Content-Type": "application/json"}, timeout=600)
        if status == 200:
            resp = json.loads(body)
            u = resp.get("usage", {})
            pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
            with open(common.ROOT / "output" / "api_usage.jsonl", "a") as f:
                f.write(json.dumps({
                    "engine": log_engine + "_openai_direct",
                    "model": model_id(model),
                    "context": common.USAGE_CONTEXT["tag"] or None,
                    "prompt_tokens": pt, "output_tokens": ct,
                    "total_tokens": u.get("total_tokens"),
                    "cost_usd_est": round(pt * PRICE_IN + ct * PRICE_OUT, 6),
                }) + "\n")
            text = resp["choices"][0]["message"].get("content") or ""
            try:
                return json.loads(text)
            except ValueError:
                raise RuntimeError(f"openai returned non-JSON: {text[:300]!r}")
        if status in (0, 429, 500, 502, 503):     # 0 = network drop
            time.sleep(15 * (attempt + 1))
            continue
        raise RuntimeError(f"openai HTTP {status}: {body[:300]!r}")
    raise RuntimeError("openai: retries exhausted")


def judge_or_direct(judge_call, prompt, schema, model, images=None,
                    log_engine="detector_gpt"):
    """OpenRouter first; OpenAI directly once OpenRouter refuses the account."""
    if not STATE["direct"]:
        try:
            return judge_call(prompt, schema, model=model,
                              log_engine=log_engine, images=images)
        except Exception as exc:
            if not (is_account_error(exc) and key()):
                raise
            STATE["direct"] = True
            if not STATE["announced"]:
                STATE["announced"] = True
                print(f"  ** OpenRouter refused the GPT call ({str(exc)[:70]}) "
                      f"-> calling OpenAI directly for the rest of this run **",
                      flush=True)
    return call(prompt, schema, model, images=images, log_engine=log_engine)
