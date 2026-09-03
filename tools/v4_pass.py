#!/usr/bin/env python3
"""Policy-v4 consistency pass — re-evaluate the corpus under detection policy
v4 WITHOUT re-transcribing (transcripts are policy-independent; keeping them
fixed attributes every metric delta purely to the policy change).

Phases:
  1  re-detect + re-render all five configs on the cached transcripts
  2  production gates on all 8 files (QC + edge listen check + judge audit)
  3  fresh judge reference under v4 (--force), pseudo-GT (openai listener
     re-judged; el/gemini reuse the just-built v4 caches), leak audit
  4  v4-pass cost accounting (separate window; transcription unchanged)

NO git operations — results are reviewed by the owner before any push.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run(tag, args):
    log("%s: %s" % (tag, " ".join(str(a) for a in args)))
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print("  [%s] %s" % (tag, line.rstrip()), flush=True)
    p.wait()
    log("%s: exit %d" % (tag, p.returncode))
    return p.returncode


if __name__ == "__main__":
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()

    log("phase 1: re-detect + re-render, all configs, cached transcripts")
    cfgs = [
        ("A:el+gemini", ["--engine", "elevenlabs"]),
        ("B:gemini", []),
        ("C:whisper+gemini", ["--engine", "openai"]),
        ("D:el+gpt", ["--engine", "elevenlabs", "--detector", "gpt"]),
        ("E:whisper+gpt", ["--engine", "openai", "--detector", "gpt"]),
    ]
    for tag, extra in cfgs:
        run(tag, [PY, "pipeline/run_pipeline.py", "--force", "detect",
                  "--force", "redact"] + extra)

    log("phase 2: production gates on all 8 files")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import run_pipeline
    files = [str(f) for f in run_pipeline.gather_files(None)]
    run("F:gates", [PY, "tools/redact.py"] + files)

    log("phase 3: judge reference + scoring under v4")
    # openai listener spans are v3-stale and pseudo_gt has no per-engine
    # force; delete them so the no-force run re-judges openai only
    for f in (ROOT / "output/judge/pseudo_gt").glob("*__openai.spans.json"):
        f.unlink()
        log("cleared stale span cache: " + f.name)
    run("G:precision_recall", [
        PY, "tools/judge_precision_recall.py", "--force",
        "--configs", "elevenlabs,gemini,openai,elevenlabs__gpt,openai__gpt"])
    run("H:pseudo_gt", [PY, "tools/judge_pseudo_gt.py"])
    run("I:leak_audit", [PY, "tools/judge_leak_audit.py", "--force",
                         "--engines", "gemini,elevenlabs,openai"])

    log("phase 4: v4-pass cost window")
    run("J:cost", [PY, "tools/cost_report.py", "--since-offset", str(offset),
                   "--out", "output/cost_v4pass"])

    log("V4 PASS COMPLETE in %.0f min" % ((time.time() - t0) / 60))
