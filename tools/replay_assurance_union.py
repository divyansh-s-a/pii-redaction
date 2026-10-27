#!/usr/bin/env python3
"""Replay the --assurance union on CACHED runs, old logic vs aligned logic,
without any model call. Prints counts only (never transcript text).

For every work dir holding a primary transcript, its intervals and an
assurance/ second-listener run:
  old    second-listener intervals at their own times and word indices
  new    the same intervals aligned onto the primary transcript
         (pipeline/align_listeners.py), as tools/redact.py now does
and reports, per file and in total:
  - union-only intervals under each logic
  - transcript words the union hides under each logic: innocent words the
    old logic hid, value words it left visible
  - how far the aligned times moved from the second listener's own
  - values the old output left partly audible: an aligned union interval
    under 95% covered by the old render (primary beeps + old union beeps)

Coverage is approximated from interval times padded by the render's pad_ms
(the corridor can only place a beep inside that pad), so both logics are
measured the same way.

  python3 tools/replay_assurance_union.py OUTPUT_ROOT [--out DIR]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import align_listeners  # noqa: E402
import common           # noqa: E402
import detect_pii       # noqa: E402
import policy           # noqa: E402


def cover(s, e, spans):
    segs = sorted((max(s, a), min(e, b)) for a, b in spans if b > s and a < e)
    tot, cs, ce = 0, None, None
    for a, b in segs:
        if ce is None or a > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    if ce is not None:
        tot += ce - cs
    return 1.0 if e <= s else tot / (e - s)


def replay(d: Path, pad: int):
    load = lambda p: json.loads(p.read_text())
    prim_tr = load(d / "transcript.json")
    sec_tr = load(d / "assurance" / "transcript.json")
    profile = load(d / "rendered_intervals.json").get(
        "profile", common.CONFIG["profile"]) \
        if (d / "rendered_intervals.json").exists() else "medical_consult"
    if profile not in policy.ANNOTATE_ONLY:
        profile = "strict"
    beeps = lambda ivs: [iv for iv in ivs if policy.action_for(
        iv["categories"], iv.get("public_figure", False), profile) == "beep"]
    prim = beeps([iv for iv in load(d / "intervals.json")["intervals"]
                  if iv.get("source") not in ("assurance_gemini",
                                              "audit_feedback")])
    sec = beeps(load(d / "assurance" / "intervals.json")["intervals"])
    pw = detect_pii.flatten_words(prim_tr)
    sw = detect_pii.flatten_words(sec_tr)
    base = [(iv["start_ms"] - pad, iv["end_ms"] + pad) for iv in prim]

    def need(iv):
        return cover(iv["start_ms"], iv["end_ms"], base) < 0.95
    old = [iv for iv in sec if need(iv)]
    amap = align_listeners.align(pw, sw)
    aligned = [x for iv in sec
               for x in align_listeners.place(iv, amap, pw, sw)]
    new = [iv for iv in aligned if need(iv)]
    decisions = {}
    for x in aligned:
        if not str(x.get("annotation_id", "")).endswith("_own_time"):
            decisions[x["placed_by"]] = decisions.get(x["placed_by"], 0) + 1

    def hidden(ivs):
        h = set()
        for iv in ivs:
            if iv.get("start_word") is None:
                continue
            h.update(i for i in range(iv["start_word"], iv["end_word"] + 1)
                     if i < len(pw))
        return h
    h_old, h_new = hidden(old), hidden(new)
    old_spans = base + [(iv["start_ms"] - pad, iv["end_ms"] + pad)
                        for iv in old]
    partly_audible = sum(1 for iv in new
                         if cover(iv["start_ms"], iv["end_ms"],
                                  old_spans) < 0.95)
    shifts = [abs(r["start_ms"] - r["retimed_from"][0]) for r in aligned
              if "retimed_from" in r and r["placed_by"] == "aligned"]
    return {"file": d.name, "second_listener_beeps": len(sec),
            "union_old": len(old), "union_new": len(new),
            "transcript_changes": h_old != h_new,
            "innocent_words_old_hid": len(h_old - h_new),
            "value_words_old_left": len(h_new - h_old),
            "partly_audible_old": partly_audible,
            "decisions": decisions,
            "shift_ms": shifts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pad = common.CONFIG.get("pad_ms", 250)
    rows = []
    for d in sorted(Path(a.output_root).iterdir()):
        need = [d / "transcript.json", d / "intervals.json",
                d / "assurance" / "transcript.json",
                d / "assurance" / "intervals.json"]
        if not all(p.exists() for p in need):
            continue
        try:
            rows.append(replay(d, pad))
        except Exception as exc:
            print(f"  skip {d.name}: {str(exc)[:80]}")
    shifts = [s for r in rows for s in r["shift_ms"]]
    tot = {
        "files": len(rows),
        "files_with_union_old": sum(r["union_old"] > 0 for r in rows),
        "files_with_union_new": sum(r["union_new"] > 0 for r in rows),
        "union_intervals_old": sum(r["union_old"] for r in rows),
        "union_intervals_new": sum(r["union_new"] for r in rows),
        "files_whose_transcript_changes": sum(r["transcript_changes"]
                                              for r in rows),
        "innocent_words_old_hid": sum(r["innocent_words_old_hid"]
                                      for r in rows),
        "value_words_old_left_visible": sum(r["value_words_old_left"]
                                            for r in rows),
        "values_partly_audible_in_old_output": sum(r["partly_audible_old"]
                                                   for r in rows),
        "files_with_a_partly_audible_value": sum(r["partly_audible_old"] > 0
                                                 for r in rows),
        "placement_decisions": {k: sum(r["decisions"].get(k, 0)
                                       for r in rows)
                                for k in ("aligned", "own_time", "both")},
        "second_listener_shift_ms": ({
            "median": int(statistics.median(shifts)),
            "p90": int(sorted(shifts)[int(0.9 * (len(shifts) - 1))]),
            "max": max(shifts), "n": len(shifts)} if shifts else None),
    }
    for k, v in tot.items():
        print(f"  {k:<40} {v}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "replay.json").write_text(json.dumps(
            {"totals": tot, "files": [dict(r, shift_ms=None) for r in rows]},
            indent=1))


if __name__ == "__main__":
    main()
