#!/usr/bin/env python3
"""Fresh full-corpus run — NO reuse of previous results (user decision
2026-09-02: fresh numbers and analysis).

Phases:
  0  record the api_usage.jsonl byte offset (cost attribution boundary)
  1  archive every previous output dir to output/_archive_<stamp>/
     (kept in place: _report_assets, el_entity [the carried-over vendor
     eval], api_usage.jsonl, the archive itself)
  2  in parallel:
       track A  production workflow on all 8 files (EL+Gemini + edge-verify
                + judge audit, deliveries/quarantine)
       track B  gemini end-to-end corpus, then whisper corpus
  3  GPT-detector configs (reuse THIS run's fresh transcripts by design)
  4  judging: precision/recall (--force) over all 5 runnable configs,
     then the 3-listener pseudo-GT union stats (--force)
  5  transcription bench vs the human reference transcripts
  6  measured cost report from the offset recorded in phase 0

Usage:  python3 tools/fresh_run.py [--dry-run]
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import run_pipeline  # noqa: E402

PY = sys.executable
KEEP = {"_report_assets", "el_entity", "api_usage.jsonl", "fresh_run_meta.json"}


def log(msg):
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg),
          flush=True)


def run_cmd(tag, args):
    log("%s: %s" % (tag, " ".join(str(a) for a in args)))
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print("  [%s] %s" % (tag, line.rstrip()), flush=True)
    p.wait()
    log("%s: exit %d" % (tag, p.returncode))
    return p.returncode


def archive(stamp, dry):
    arch = ROOT / "output" / ("_archive_" + stamp)
    moved = []
    out = ROOT / "output"
    for item in sorted(out.iterdir()):
        if item.name in KEEP or item.name.startswith("_archive"):
            continue
        moved.append(item.name)
        if not dry:
            arch.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(arch / item.name))
    for extra in (ROOT / "deliveries", ROOT / "quarantine"):
        if extra.exists():
            moved.append(extra.name + "/")
            if not dry:
                arch.mkdir(parents=True, exist_ok=True)
                shutil.move(str(extra), str(arch / extra.name))
    log("archived %d item(s) -> %s%s" % (len(moved), arch.name,
                                         " (dry-run)" if dry else ""))
    for m in moved:
        log("  - " + m)
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without touching anything")
    args = ap.parse_args()

    files = run_pipeline.gather_files(None)
    log("corpus: %d files" % len(files))
    for f in files:
        log("  - " + f.name)

    usage = ROOT / "output" / "api_usage.jsonl"
    offset = usage.stat().st_size if usage.exists() else 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log("usage-log offset: %d" % offset)
    if not args.dry_run:
        (ROOT / "output" / "fresh_run_meta.json").write_text(json.dumps(
            {"started_at": datetime.datetime.now().isoformat(
                timespec="seconds"),
             "usage_offset": offset, "archive": "_archive_" + stamp,
             "n_files": len(files)}, indent=2))

    log("phase 1: archive previous outputs")
    archive(stamp, args.dry_run)
    if args.dry_run:
        log("dry-run complete")
        return 0

    t0 = time.time()
    log("phase 2: production track (A) + gemini/whisper track (B), parallel")
    results = {}

    def track_a():
        results["A"] = run_cmd(
            "A:production",
            [PY, "tools/redact.py"] + [str(f) for f in files])

    def track_b():
        rc = run_cmd("B:gemini", [PY, "pipeline/run_pipeline.py"])
        rc2 = run_cmd("B:whisper",
                      [PY, "pipeline/run_pipeline.py", "--engine", "openai"])
        results["B"] = rc or rc2

    ta = threading.Thread(target=track_a)
    tb = threading.Thread(target=track_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    log("phase 2 done (rc A=%s B=%s) in %.0f min"
        % (results.get("A"), results.get("B"), (time.time() - t0) / 60))

    log("phase 3: GPT-detector configs (fresh transcripts from this run)")
    run_cmd("C:el+gpt", [PY, "pipeline/run_pipeline.py",
                         "--engine", "elevenlabs", "--detector", "gpt"])
    run_cmd("C:whisper+gpt", [PY, "pipeline/run_pipeline.py",
                              "--engine", "openai", "--detector", "gpt"])

    log("phase 4: judge scoring (fresh spans, --force)")
    run_cmd("D:precision_recall", [
        PY, "tools/judge_precision_recall.py", "--force",
        "--configs", "elevenlabs,gemini,openai,elevenlabs__gpt,openai__gpt"])
    run_cmd("D:pseudo_gt", [PY, "tools/judge_pseudo_gt.py", "--force"])

    log("phase 5: transcription bench vs human references")
    run_cmd("E:bench", [PY, "tools/transcription_bench.py"])

    log("phase 6: measured cost report")
    run_cmd("F:cost", [PY, "tools/cost_report.py",
                       "--since-offset", str(offset)])

    log("FRESH RUN COMPLETE in %.0f min" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
