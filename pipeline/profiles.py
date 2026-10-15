#!/usr/bin/env python3
"""Custom render profiles, loaded from a file — data, not code.

policy.py ships two profiles (strict, medical_consult). A project whose rules
differ (e.g. a market-research study that keeps ages, as its survey data did)
supplies a profile FILE instead of editing the taxonomy:

    name: market_research_example      # any id; recorded in every output
    annotate_only: [AGE]               # policy categories kept audible
    detection_supplement: |            # optional text appended to the
      ...                              # detection prompt

When a category changes action, the prompt must say so too: the action matrix
only decides what to do with a category the detector already emitted, so the
supplement is where a profile tells the detector what it now means.

Selection: tools/redact.py --profile-file PATH, or the PII_PROFILE_FILE env
var (inherited by the audit subprocesses). policy.CATEGORIES is never
changed; the profile is registered into policy.ANNOTATE_ONLY at run time.
"""
import hashlib
import json
import os
from pathlib import Path

import policy

ENV = "PII_PROFILE_FILE"


def _read(path: Path) -> dict:
    text = Path(path).read_text()
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    import yaml           # pyyaml: needed only when a YAML profile is used
    return yaml.safe_load(text) or {}


def load(path) -> dict:
    """Parse and validate a profile file; returns its definition."""
    d = _read(Path(path))
    name = str(d.get("name") or "").strip()
    if not name:
        raise ValueError(f"{path}: profile needs a 'name'")
    if name in ("strict", "medical_consult"):
        raise ValueError(f"{path}: '{name}' is a built-in profile name")
    keep = [str(c) for c in d.get("annotate_only") or []]
    bad = [c for c in keep if c not in policy.CATEGORIES]
    if bad:
        raise ValueError(f"{path}: annotate_only has unknown categories {bad}")
    sup = d.get("detection_supplement") or ""
    return {"name": name, "annotate_only": sorted(set(keep)),
            "detection_supplement": str(sup).strip(),
            "file": str(Path(path).resolve()),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def register(defn: dict):
    policy.ANNOTATE_ONLY[defn["name"]] = set(defn["annotate_only"])
    return defn["name"]


def active():
    """The profile definition selected via PII_PROFILE_FILE, or None."""
    f = os.environ.get(ENV, "").strip()
    if not f:
        return None
    defn = load(f)
    register(defn)
    return defn


def ensure(name: str):
    """Make `name` resolvable by policy.action_for. Built-ins are always
    there; a custom name must match the active profile file."""
    if name in policy.ANNOTATE_ONLY:
        return
    defn = active()
    if defn is None or defn["name"] != name:
        raise KeyError(f"profile '{name}' is not built in and no matching "
                       f"profile file is active ({ENV})")


def detection_supplement() -> str:
    """Text appended to the detection prompt by the active profile, or ''."""
    defn = active()
    if not defn or not defn["detection_supplement"]:
        return ""
    return ("\n\nPROFILE SUPPLEMENT (" + defn["name"] + "):\n"
            + defn["detection_supplement"] + "\n")


def _self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.json"
        f.write_text(json.dumps({"name": "example_keep_age",
                                 "annotate_only": ["AGE"],
                                 "detection_supplement": "Ages are data."}))
        os.environ[ENV] = str(f)
        defn = active()
        assert defn["annotate_only"] == ["AGE"]
        assert policy.action_for(["AGE"], False, "example_keep_age") \
            == "annotate_only"
        # a child's age still beeps: CHILD_INFO is not kept
        assert policy.action_for(["CHILD_INFO", "AGE"], False,
                                 "example_keep_age") == "beep"
        assert policy.action_for(["FULL_NAME"], False,
                                 "example_keep_age") == "beep"
        assert "Ages are data." in detection_supplement()
        f.write_text(json.dumps({"name": "x", "annotate_only": ["NOPE"]}))
        try:
            load(f)
            raise AssertionError("unknown category accepted")
        except ValueError:
            pass
        del os.environ[ENV]
        assert detection_supplement() == ""
        assert len(policy.CATEGORIES) == 31
    print("profiles self-test: OK")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _self_test()
