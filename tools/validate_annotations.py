#!/usr/bin/env python3
"""Validate clip annotation files against Annotation Schema v0.2 / Taxonomy v0.2.

Usage:  python3 tools/validate_annotations.py annotations/dev/*.gt.json
Exits non-zero if any file has errors. Warnings don't fail the run.
"""
import json
import sys
from pathlib import Path

CATEGORIES = {
    "FULL_NAME", "PHONE", "EMAIL", "ADDRESS", "DATE_OF_BIRTH", "AGE",
    "PASSPORT_NUMBER", "DRIVERS_LICENSE", "GOVERNMENT_ID", "TAX_ID",
    "INSURANCE_ID", "STUDENT_ID", "LEGAL_CASE_ID", "IMMIGRATION_STATUS",
    "PAYMENT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER", "FINANCIAL_ACCOUNT_INFO",
    "FINANCIAL_INFO", "PASSWORD_CREDENTIAL", "IP_ADDRESS", "DEVICE_IDENTIFIER",
    "VEHICLE_REGISTRATION", "HEALTH_MEDICAL_INFO", "BIOMETRIC_ID",
    "SENSITIVE_DEMOGRAPHIC", "CHILD_INFO", "NATIONALITY", "USERNAME_HANDLE",
    "ASSIGNED_ID", "ORGANIZATION_PERSON_LINKED", "OTHER",
}
RETIRED = {"SSN": "folded into GOVERNMENT_ID in taxonomy v0.2"}
CLIP_FIELDS = {
    "schema_version", "taxonomy_version", "file_id", "media_file",
    "media_sha256", "duration_ms", "split", "languages_observed",
    "audio_quality", "num_speakers", "overlapping_speech", "pii_present",
    "annotations", "annotator", "annotation_pass", "annotated_at", "clip_notes",
}
ANN_FIELDS = {
    "annotation_id", "start_ms", "end_ms", "speaker", "spoken_text",
    "categories", "person_link", "confidence", "needs_review", "notes",
}
OPTIONAL_ANN_FIELDS = {"public_figure", "channel"}
AUDIO_QUALITY = {"studio", "clean", "phone", "noisy", "very_noisy"}
PERSON_LINK = {"explicit", "contextual", "unclear"}
CONFIDENCE = {"high", "medium", "low"}


def check(path: Path):
    errors, warnings = [], []
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return [f"unreadable JSON: {e}"], []

    for f in CLIP_FIELDS - d.keys():
        errors.append(f"missing clip field: {f}")
    if d.get("split") not in {"dev", "holdout"}:
        errors.append(f"split must be dev|holdout, got {d.get('split')!r}")
    if d.get("audio_quality") not in AUDIO_QUALITY:
        errors.append(f"bad audio_quality {d.get('audio_quality')!r}")
    if not d.get("media_sha256"):
        errors.append("media_sha256 is empty")
    dur = d.get("duration_ms", 0)
    anns = d.get("annotations", [])
    if d.get("pii_present") and not anns:
        errors.append("pii_present true but no annotations")
    if anns and not d.get("pii_present"):
        errors.append("annotations present but pii_present false")

    seen_ids = set()
    for i, a in enumerate(anns):
        tag = a.get("annotation_id") or f"annotations[{i}]"
        for f in ANN_FIELDS - a.keys():
            errors.append(f"{tag}: missing field {f}")
        if a.get("annotation_id") in seen_ids:
            errors.append(f"{tag}: duplicate annotation_id")
        seen_ids.add(a.get("annotation_id"))
        s, e = a.get("start_ms", -1), a.get("end_ms", -1)
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e):
            errors.append(f"{tag}: bad interval start_ms={s} end_ms={e}")
        elif dur and e > dur:
            errors.append(f"{tag}: end_ms {e} exceeds duration_ms {dur}")
        cats = set(a.get("categories", []))
        if not cats:
            errors.append(f"{tag}: categories empty")
        for c in cats - CATEGORIES:
            if c in RETIRED:
                errors.append(f"{tag}: retired category {c!r} ({RETIRED[c]})")
            else:
                errors.append(f"{tag}: unknown category {c!r}")
        if a.get("person_link") not in PERSON_LINK:
            errors.append(f"{tag}: bad person_link {a.get('person_link')!r}")
        if a.get("confidence") not in CONFIDENCE:
            errors.append(f"{tag}: bad confidence {a.get('confidence')!r}")
        if not a.get("spoken_text", "").strip():
            errors.append(f"{tag}: spoken_text empty")
        # Forced needs_review rules
        forced = []
        if a.get("confidence") == "low":
            forced.append("confidence=low")
        if a.get("person_link") == "unclear":
            forced.append("person_link=unclear")
        if "OTHER" in cats:
            forced.append("category=OTHER")
            if not a.get("notes", "").strip():
                errors.append(f"{tag}: OTHER requires a reviewer note")
        if forced and not a.get("needs_review"):
            errors.append(f"{tag}: needs_review must be true ({', '.join(forced)})")
        if "CHILD_INFO" in cats and cats == {"CHILD_INFO"}:
            warnings.append(f"{tag}: CHILD_INFO usually pairs with the specific label")
        chn = a.get("channel")
        if chn is not None and not (isinstance(chn, int) and 0 <= chn <= 7):
            errors.append(f"{tag}: channel must be a small int or null")
        pf = a.get("public_figure", False)
        if not isinstance(pf, bool):
            errors.append(f"{tag}: public_figure must be a boolean")
        elif pf and "FULL_NAME" not in cats:
            warnings.append(f"{tag}: public_figure=true on a non-FULL_NAME interval")
    return errors, warnings


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    failed = False
    for p in paths:
        errors, warnings = check(p)
        status = "FAIL" if errors else "ok"
        print(f"[{status}] {p}")
        for e in errors:
            print(f"   ERROR: {e}")
        for w in warnings:
            print(f"   warn:  {w}")
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
