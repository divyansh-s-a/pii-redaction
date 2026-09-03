#!/usr/bin/env python3
"""Independent LLM judge client — the Mindy pilot's judge models (gpt-5.6-sol /
gpt-5.6-luna) reached via OpenRouter, applying OUR taxonomy v0.2 policy.

Deliberately a different model family from the pipeline's Gemini stages so
judge-based evaluation is not self-grading. Stdlib only; usage logged to
output/api_usage.jsonl.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from adapters import env_key

DEFAULT_JUDGE = "openai/gpt-5.6-sol"
DEFAULT_VERIFIER = "openai/gpt-5.6-luna"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def judge_call(prompt: str, schema: dict, model: str = DEFAULT_JUDGE,
               max_tokens: int = 16000, log_engine: str = "judge") -> dict:
    """One structured-output judge call; returns the parsed JSON object.
    log_engine labels the usage-log row (the GPT stage-C detector reuses this
    transport and must be cost-attributable separately from the judge)."""
    key = env_key("OPENROUTER_API_KEY")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judgement", "strict": True,
                            "schema": schema},
        },
        # OpenRouter usage accounting: the response then carries the measured
        # USD cost of the call, so judge/detector spend is measured, never
        # estimated from price sheets
        "usage": {"include": True},
    }
    for attempt in range(4):
        status, _, body = common._request(
            ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"}, timeout=600)
        if status == 200:
            resp = json.loads(body)
            usage = resp.get("usage", {})
            log = common.ROOT / "output" / "api_usage.jsonl"
            with open(log, "a") as f:
                f.write(json.dumps({
                    "engine": log_engine, "model": model,
                    "context": common.USAGE_CONTEXT["tag"] or None,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cost_usd": usage.get("cost")}) + "\n")
            content = resp["choices"][0]["message"].get("content") or ""
            try:
                return json.loads(content)
            except ValueError:
                raise RuntimeError(f"judge returned non-JSON: {content[:300]!r}")
        if status in (429, 500, 502, 503):
            wait = 15 * (attempt + 1)
            print(f"  judge HTTP {status}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"judge HTTP {status}: {body[:400]!r}")
    raise RuntimeError("judge: retries exhausted")


# --------------------------------------------- adversarial verification pass

# Strict structured-output schema for the verifier (additionalProperties
# false everywhere, every property required).
VERIFY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer",
                              "description": "the finding's number in the "
                                             "numbered list (1-based)"},
                    "keep": {"type": "boolean"},
                    "rule": {"type": "string",
                             "description": "the include rule satisfied "
                                            "(category name + short phrase), "
                                            "or 'REJECT: <exclusion>'"},
                },
                "required": ["index", "keep", "rule"],
            },
        }
    },
    "required": ["verdicts"],
}

VERIFY_INSTRUCTIONS = (
    "You are an adversarial verifier auditing PII findings for OVER-REACH. "
    "For each numbered finding, decide if it satisfies a specific INCLUDE "
    "rule of the policy below. Answer per finding: the include rule it "
    "satisfies (category name + short phrase) or REJECT with the exclusion "
    "that applies. Be strict: rule 11 calibration exclusions and "
    "amount-less/product-name/generic-process/no-person-link cases must be "
    "REJECTED.")


def _finding_text(f):
    for k in ("quoted_text", "spoken_text", "text"):
        v = f.get(k)
        if v:
            return str(v)
    return ""


def _one_line(s):
    return " ".join(str(s or "").split())


def verify_findings(findings, policy_text, model=DEFAULT_VERIFIER,
                    context_texts=None):
    """Adversarial over-reach filter. ONE batched call for the whole findings
    list (never one call per finding); returns a list of booleans parallel to
    `findings` (True = keep).

    findings: dicts carrying the quoted/spoken text (quoted_text /
    spoken_text / text) and a categories list. context_texts: optional
    parallel list, one line of surrounding context per finding.

    FAIL-OPEN: verification only removes findings, so a malformed verdict
    must never silently drop one — every finding defaults to keep=True and is
    rejected only by an explicit, well-formed keep=false verdict citing a
    valid index (1-based; missing / duplicate-free out-of-range indices are
    ignored)."""
    findings = list(findings)
    if not findings:
        return []
    ctx = list(context_texts or [])
    entries = []
    for i, f in enumerate(findings):
        cats = ",".join(f.get("categories", [])) or "(none)"
        line = ctx[i] if i < len(ctx) else ""
        entries.append('%d. text: "%s"\n   categories: %s\n   context: %s'
                       % (i + 1, _one_line(_finding_text(f)), cats,
                          _one_line(line)))
    prompt = (VERIFY_INSTRUCTIONS
              + "\n\n=== POLICY ===\n" + str(policy_text)
              + "\n\n=== FINDINGS (numbered) ===\n" + "\n".join(entries)
              + "\n\nReturn JSON only, matching the schema: "
                '{"verdicts":[{"index":<finding number>,"keep":true|false,'
                '"rule":"<include rule satisfied, or REJECT: exclusion>"}]} '
                "— exactly one verdict per numbered finding.")
    # module-global lookup at call time so tests can monkeypatch judge_call
    got = judge_call(prompt, VERIFY_SCHEMA, model=model)
    keep = [True] * len(findings)
    for v in (got.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        try:
            idx = int(v.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(findings) and v.get("keep") is False:
            keep[idx - 1] = False
    return keep
