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
        "usability": {"type": "STRING",
                      "enum": ["usable", "limited", "not usable"]},
        "language": {"type": "STRING"},
        "language_share": {"type": "STRING"},
        "rationale": {"type": "STRING"},
    },
    "required": ["domain", "confidence", "language"],
}

_PROMPT = (
    "You classify the BUSINESS/CONTENT DOMAIN of a call-centre transcript and "
    "assess how usable the conversation is as speech data.\n\n"
    "Return:\n"
    "- domain: a concise free-form label of 1-3 words (examples: 'health "
    "insurance', 'retail banking', 'telecom support', 'debt collection', "
    "'e-commerce delivery', 'travel booking'). Base it only on the subject "
    "matter.\n"
    "- confidence: high | medium | low.\n"
    "- language: the dominant spoken language of the conversation as an ISO "
    "639-1 code, e.g. 'ms', 'id', 'sw', 'en', 'hi'. Judge ONLY from the text "
    "itself. Distinguish Malay from Indonesian by usage (e.g. 'awak/nak/"
    "ringgit' vs 'anda/mau/rupiah'); if you cannot tell the language "
    "confidently, answer exactly 'unknown'.\n"
    "- language_share: a short mix description, e.g. 'sw 80% / en 20%', or "
    "'unknown'.\n"
    "- usability: 'usable', 'limited', or 'not usable' — judged on whether the "
    "call is a substantive conversation with real content, versus a trivial one "
    "(greetings only, 'hello hello' / test call, wrong number, mostly dead air, "
    "silence or repeated filler, or no meaningful exchange).\n"
    "- rationale: a VERBOSE explanation of 3-5 full sentences in plain prose "
    "that covers BOTH (a) the specific topics and subject matter discussed and "
    "why they point to this domain, AND (b) an honest assessment of the "
    "conversation's substance and usability for training or evaluation — "
    "explicitly call it out when the call is low-content, greeting-only, "
    "repetitive, dominated by silence, or otherwise not a very usable "
    "conversation. If the domain is uncertain because there is too little "
    "content, say so. Contain NO personal data (no names, numbers, addresses "
    "or IDs).\n\n"
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
            "usability": "limited" if len(low.split()) < 40 else "usable",
            "rationale": (
                f"Offline keyword heuristic: matched {score} domain term(s) "
                f"for '{best}'. This is a coarse fallback (the LLM classifier "
                f"was unavailable), so treat the domain as approximate and the "
                f"usability judgement as tentative."
                if best else
                "Offline keyword heuristic found no domain terms; the transcript "
                "may be too short or low-content to classify reliably."),
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
