#!/usr/bin/env python3
"""One-off deterministic repair: clamp cached transcript word durations to 5 s
(chunk-boundary artifacts) and recompute interval times from word indices —
no API calls. Prints files whose intervals changed (need re-render)."""
import json
import sys
from pathlib import Path

changed = []
for d in sorted(Path("output").iterdir()):
    tp, ip = d / "transcript.json", d / "intervals.json"
    if not tp.exists():
        continue
    tr = json.loads(tp.read_text())
    n_fixed = 0
    for u in tr.get("utterances", []):
        for w in u["words"]:
            if w["end_ms"] - w["start_ms"] > 5000:
                w["end_ms"] = w["start_ms"] + 5000
                n_fixed += 1
    if n_fixed:
        tp.write_text(json.dumps(tr, indent=2, ensure_ascii=False))
    iv_changed = False
    if ip.exists():
        data = json.loads(ip.read_text())
        words = [w for u in tr.get("utterances", []) for w in u["words"]]
        for iv in data.get("intervals", []):
            a, b = iv.get("start_word"), iv.get("end_word")
            if a is None or b is None or b >= len(words):
                continue
            ws = words[a:b + 1]
            s, e = ws[0]["start_ms"], max(w["end_ms"] for w in ws)
            if (s, e) != (iv["start_ms"], iv["end_ms"]):
                iv["start_ms"], iv["end_ms"] = s, e
                iv_changed = True
        if iv_changed:
            ip.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    if n_fixed or iv_changed:
        changed.append(d.name)
        print(f"{d.name}: {n_fixed} word(s) clamped, intervals_updated={iv_changed}")
print("\nfiles needing re-render:", [c for c in changed])
