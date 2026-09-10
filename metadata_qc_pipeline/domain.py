#!/usr/bin/env python3
"""Free-form content-domain classification for a call.

Runs a small LLM (config domain_model, via common.generate so it inherits the
OpenRouter fallback) over the REDACTED transcript text — never the raw one, so
no PII enters the metadata. If the model is unavailable or errors, a keyword
heuristic supplies a coarse label so the field is always populated.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import common  # noqa: E402

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "domain": {"type": "STRING"},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "STRING"},
    },
    "required": ["domain", "confidence"],
}

_PROMPT = (
    "You classify the BUSINESS/CONTENT DOMAIN of a call-centre transcript. "
    "Reply with a concise free-form domain label of 1-3 words (examples: "
    "'health insurance', 'retail banking', 'telecom support', 'debt "
    "collection', 'e-commerce delivery', 'travel booking'). Base it only on "
    "the subject matter. Give a confidence and a rationale of at most 15 "
    "words that contains NO personal data (no names, numbers, addresses).\n\n"
    "TRANSCRIPT:\n")

# Coarse offline fallback — first domain with the most keyword hits wins.
_HEURISTICS = {
    "insurance": ["insurance", "policy", "policyholder", "claim", "premium",
                  "coverage", "cashless", "tpa", "sum insured", "nominee"],
    "healthcare": ["hospital", "patient", "doctor", "discharge", "diagnosis",
                   "prescription", "treatment", "appointment", "consult"],
    "banking / finance": ["account", "bank", "loan", "emi", "credit card",
                          "debit", "ifsc", "balance", "cheque", "interest"],
    "telecom": ["recharge", "sim", "network", "data pack", "prepaid",
                "postpaid", "roaming", "tariff", "broadband"],
    "retail / e-commerce": ["order", "delivery", "refund", "product",
                            "shipment", "return", "cart", "invoice"],
    "travel": ["flight", "booking", "ticket", "hotel", "pnr", "itinerary",
               "cab", "trip"],
    "utilities": ["electricity", "bill", "meter", "gas", "water", "outage",
                  "connection"],
    "debt collection": ["overdue", "outstanding", "repayment", "collection",
                         "recovery", "default", "settlement"],
}


def _heuristic(text: str):
    low = text.lower()
    best, score = None, 0
    for dom, kws in _HEURISTICS.items():
        n = sum(1 for k in kws if k in low)
        if n > score:
            best, score = dom, n
    return {"domain": best or "unknown",
            "confidence": "low" if best else "unknown",
            "rationale": (f"keyword heuristic ({score} term matches)"
                          if best else "no domain keywords matched"),
            "source": "heuristic"}


def _clean_transcript_text(redacted_text: str) -> str:
    """Strip the '[ts - ts] S1 - ' line prefixes, keep just spoken text."""
    lines = []
    for ln in redacted_text.splitlines():
        m = re.match(r"^\[[^\]]*\]\s*\S+\s*-\s*(.*)$", ln)
        lines.append(m.group(1) if m else ln)
    return "\n".join(l for l in lines if l.strip())


def classify_domain(redacted_text: str, cfg: dict, use_llm: bool = True) -> dict:
    text = _clean_transcript_text(redacted_text)[: int(cfg["domain_max_chars"])]
    if not text.strip():
        return {"domain": "unknown", "confidence": "unknown",
                "rationale": "empty transcript", "source": "none"}
    if use_llm:
        try:
            raw = common.generate(
                cfg["domain_model"], [{"text": _PROMPT + text}], _SCHEMA,
                thinking_budget=cfg.get("domain_thinking_budget"))
            d = json.loads(raw)
            d["source"] = "llm"
            d.setdefault("confidence", "medium")
            return d
        except Exception as e:
            h = _heuristic(text)
            h["llm_error"] = str(e)[:150]
            return h
    return _heuristic(text)
