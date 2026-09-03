#!/usr/bin/env python3
"""Taxonomy v0.2 categories, Action Matrix v1.0 profiles, and the detection
policy prompt. Single source of truth for the pipeline; must stay identical to
outputs/Audio_PII_Taxonomy_v0.2.md and outputs/Action_Matrix_v1.0.md.
"""
from pathlib import Path


CATEGORIES = [
    "FULL_NAME", "PHONE", "EMAIL", "ADDRESS", "DATE_OF_BIRTH", "AGE",
    "PASSPORT_NUMBER", "DRIVERS_LICENSE", "GOVERNMENT_ID", "TAX_ID",
    "INSURANCE_ID", "STUDENT_ID", "LEGAL_CASE_ID", "IMMIGRATION_STATUS",
    "PAYMENT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER", "FINANCIAL_ACCOUNT_INFO",
    "FINANCIAL_INFO", "PASSWORD_CREDENTIAL", "IP_ADDRESS", "DEVICE_IDENTIFIER",
    "VEHICLE_REGISTRATION", "HEALTH_MEDICAL_INFO", "BIOMETRIC_ID",
    "SENSITIVE_DEMOGRAPHIC", "CHILD_INFO", "NATIONALITY", "USERNAME_HANDLE",
    "ASSIGNED_ID", "ORGANIZATION_PERSON_LINKED", "OTHER",
]

# Action Matrix v1.0: profile -> categories that are annotate_only (rest beep).
ANNOTATE_ONLY = {
    "strict": set(),
    "medical_consult": {"HEALTH_MEDICAL_INFO"},
}


def action_for(categories, public_figure: bool, profile: str) -> str:
    """'beep' or 'annotate_only' for one interval under a profile."""
    if public_figure:
        return "annotate_only"
    ann_only = ANNOTATE_ONLY[profile]
    cats = set(categories)
    if cats and cats <= ann_only:
        return "annotate_only"
    return "beep"


POLICY_PROMPT_FILE = "detection_policy_prompt_v4.txt"
DETECTION_POLICY = (Path(__file__).resolve().parent.parent / "prompts"
                    / POLICY_PROMPT_FILE).read_text()

DETECTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "spans": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_word": {"type": "INTEGER"},
                    "end_word": {"type": "INTEGER"},
                    "categories": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "spoken_text": {"type": "STRING"},
                    "speaker": {"type": "STRING"},
                    "public_figure": {"type": "BOOLEAN"},
                    "confidence": {"type": "STRING",
                                   "enum": ["high", "medium", "low"]},
                    "reason": {"type": "STRING"},
                },
                "required": ["start_word", "end_word", "categories",
                             "spoken_text", "confidence"],
            },
        }
    },
    "required": ["spans"],
}
