#!/usr/bin/env python3
"""--assurance union under a realistic second listener: this checkout vs a
ref (default main), scored against the synthetic clips' answer keys.

Runs tools/offline_harness.py in both trees with --assurance
--assurance-drift --primary-miss LABELS: the primary detector misses those
values, the second listener finds them, and its transcript runs early and
numbers its words differently — as measured on real second-listener output.
For every value only the second listener found:

  audio covered      share of the value's span under a rendered beep
                     (>= 95% counts as removed)
  words hidden       the value's own words replaced in the redacted
                     transcript
  innocent hidden    words that are no value at all but were hidden
                     anyway (the transcript-index bug)

  python3 tools/check_assurance_union.py --media DIR [--ref main]
      [--miss PHONE,CHILD_INFO,ADDRESS] [--out DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_tree(tree, media, out, work, miss):
    env = dict(os.environ, PII_WORK_DIR=str(work))
    r = subprocess.run([sys.executable, str(ROOT / "tools" /
                                            "offline_harness.py"),
                        "--tree", str(tree), "--media", str(media),
                        "--out", str(out), "--assurance", "--assurance-drift",
                        "--primary-miss", miss],
                       capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith("HARNESS_RESULT ")]
    if r.returncode or not line:
        sys.exit(f"harness failed in {tree}:\n{r.stdout[-1500:]}\n"
                 f"{r.stderr[-2500:]}")
    return json.loads(line[-1].split(" ", 1)[1])


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
    return tot / max(1, e - s)


def score(res, media, miss):
    rows = []
    for clip, r in res.items():
        work = Path(r["work"])
        gt = json.loads((media / f"{clip}.gt.json").read_text())
        tr = json.loads((media / f"{clip}.transcript.json").read_text())
        words = [w for u in tr["utterances"] for w in u["words"]]
        iv = json.loads((work / "intervals.json").read_text())["intervals"]
        ri = json.loads((work / "rendered_intervals.json").read_text())
        beeps = [(x["start_ms"], x["end_ms"]) for x in ri["rendered_merged"]]
        hidden = set()
        for x in iv:
            if x.get("start_word") is None:
                continue
            hidden.update(range(x["start_word"], x["end_word"] + 1))
        labelled = [e for e in gt["entities"] if e["kind"] in ("P", "K", "M")]

        def words_of(e):
            return {i for i, w in enumerate(words)
                    if w["start_ms"] < e["end_ms"]
                    and e["start_ms"] < w["end_ms"]}
        value_words = set().union(*[words_of(e) for e in labelled]) \
            if labelled else set()
        targets = [e for e in gt["entities"] if e["label"] in miss]
        for e in targets:
            ws = words_of(e)
            rows.append({"clip": clip, "value": e["label"],
                         "audio_covered": round(cover(e["start_ms"],
                                                      e["end_ms"], beeps), 2),
                         "words_hidden": f"{len(ws & hidden)}/{len(ws)}"})
        rows.append({"clip": clip, "value": "(innocent words hidden)",
                     "count": len(hidden - value_words)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--ref", default="main")
    ap.add_argument("--miss", default="PHONE,CHILD_INFO,ADDRESS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    media = Path(a.media).resolve()
    out = Path(a.out or tempfile.mkdtemp(prefix="assurance_union_")).resolve()
    miss = {x.strip() for x in a.miss.split(",") if x.strip()}
    wt = out / "worktree_ref"
    if not wt.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach",
                        str(wt), a.ref], check=True, capture_output=True)
    try:
        ref = run_tree(wt, media, out / "ref", out / "ref_work", a.miss)
        here = run_tree(ROOT, media, out / "branch", out / "branch_work",
                        a.miss)
        # score before the worktree goes: a ref without PII_WORK_DIR keeps
        # its work files inside it
        sr, sh = score(ref, media, miss), score(here, media, miss)
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove",
                        "--force", str(wt)], capture_output=True)
    print(f"{'clip':<10}{'value':<26}{a.ref + ': audio / words':<28}"
          "this branch: audio / words")
    for x, y in zip(sr, sh):
        if "count" in x:
            print(f"{x['clip']:<10}{x['value']:<26}{x['count']:<28}"
                  f"{y['count']}")
        else:
            print(f"{x['clip']:<10}{x['value']:<26}"
                  f"{str(x['audio_covered']) + ' / ' + x['words_hidden']:<28}"
                  f"{y['audio_covered']} / {y['words_hidden']}")
    (out / "assurance_union.json").write_text(json.dumps(
        {"ref": sr, "branch": sh}, indent=1))


if __name__ == "__main__":
    main()
