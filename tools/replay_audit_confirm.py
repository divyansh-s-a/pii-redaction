#!/usr/bin/env python3
"""Replay the judge-audit gate on CACHED audits, old rule vs new, without
any model call. Prints counts and category names only (never quotes).

old  the confirm-on-second-read step counted first-read findings the judge
     did NOT mark profile_dependent (the judge's own assertion)
new  it counts findings the file's render profile does not keep — the
     code-side rule the audit report itself uses (tools/redact.py gating())

For every work dir with a first read (audit/) the second read is replayed
only when the report rule found a candidate (that is when the workflow runs
it); a dir whose second read is not cached is reported, not guessed.

  python3 tools/replay_audit_confirm.py OUTPUT_ROOT [--out DIR]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import common  # noqa: E402
import policy  # noqa: E402


def kept(f, profile):
    return policy.action_for(f.get("categories", []), False,
                             profile) == "annotate_only"


def key(f):
    return " ".join(str(f.get("quoted_text", "")).lower().split())


def matched(f, second):
    return any(key(f) == key(g) and abs(f.get("line", 0) - g.get("line", 0))
               <= 3 for g in second)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tot = collections.Counter()
    disagree = collections.Counter()
    rows = []
    for d in sorted(Path(a.output_root).iterdir()):
        files = [p for p in (d / "audit").glob("*.json")
                 if p.name != "leak_audit_report.json"] \
            if (d / "audit").is_dir() else []
        for pf in files:
            ri = d / "rendered_intervals.json"
            profile = (json.loads(ri.read_text()).get("profile")
                       if ri.exists() else None) or common.CONFIG["profile"]
            if profile not in policy.ANNOTATE_ONLY:
                profile = "strict"
            first = json.loads(pf.read_text()).get("findings", [])
            tot["first_reads"] += 1
            for f in first:
                flag, code = bool(f.get("profile_dependent")), kept(f, profile)
                if flag and not code:
                    disagree["judge kept, profile beeps: " + "+".join(
                        sorted(f.get("categories", [])))] += 1
                elif code and not flag:
                    disagree["judge counted, profile keeps: " + "+".join(
                        sorted(f.get("categories", [])))] += 1
            n_first = sum(not kept(f, profile) for f in first)
            if not n_first:
                tot["no_candidate_on_first_read"] += 1
                continue
            pf2 = d / "audit_confirm" / pf.name
            if not pf2.exists():
                tot["second_read_not_cached"] += 1
                continue
            second = json.loads(pf2.read_text()).get("findings", [])
            old = [f for f in first
                   if not f.get("profile_dependent") and matched(f, second)]
            new = [f for f in first
                   if not kept(f, profile) and matched(f, second)]
            tot["second_reads_replayed"] += 1
            tot["confirmed_old"] += len(old)
            tot["confirmed_new"] += len(new)
            if bool(old) != bool(new):
                tot["verdict_changes"] += 1
                tot["now_held" if new else "now_released"] += 1
            rows.append({"file": d.name, "profile": profile,
                         "confirmed_old": len(old),
                         "confirmed_new": len(new)})
    for k in ("first_reads", "no_candidate_on_first_read",
              "second_read_not_cached", "second_reads_replayed",
              "confirmed_old", "confirmed_new", "verdict_changes",
              "now_held", "now_released"):
        print(f"  {k:<32} {tot.get(k, 0)}")
    print("  findings where the judge's flag and the profile disagree:")
    for k, v in disagree.most_common():
        print(f"    {v:>4}  {k}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "replay_audit_confirm.json").write_text(json.dumps(
            {"totals": dict(tot), "disagreements": dict(disagree),
             "files": rows}, indent=1))


if __name__ == "__main__":
    main()
