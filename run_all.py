#!/usr/bin/env python3
"""Driver: dedupe -> PII redact -> metadata QC, sequentially, over a folder.

  python3 run_all.py --input ../samples15 --result ../result

Stage 1  dedupe the whole input set (dedupe/run_dedupe.py) -> unique reps
Stage 2  redact ONLY the unique reps (tools/redact.py) -> result/deliveries/<fid>
Stage 3  metadata QC each delivered file (metadata_qc_pipeline) -> metadata.json

Writes result/dedupe/ (report + unique set), result/deliveries/<fid>/ (beeped
audio + redacted transcript + qc + manifest + metadata) and result/summary.json.
The three pipelines stay independent; this only orchestrates them.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent           # AudioPII-QC
sys.path.insert(0, str(ROOT / "pipeline"))
import common                                     # noqa: E402
import run_pipeline                               # noqa: E402

PY = sys.executable


def run(cmd, **kw):
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), **kw)


def find_work_dir(fid, engine):
    out = ROOT / "output"
    for d in [out / f"{fid}__{engine}", out / fid, *sorted(out.glob(f"{fid}*"))]:
        if d.exists() and (d / "transcript.json").exists():
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--input", required=True, help="folder of audio files")
    ap.add_argument("--result", default=str(ROOT.parent / "result"))
    ap.add_argument("--profile", default=common.CONFIG["profile"])
    ap.add_argument("--engine", default="elevenlabs")
    ap.add_argument("--no-domain", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input).resolve()
    result = Path(args.result).resolve()
    (result / "dedupe").mkdir(parents=True, exist_ok=True)
    (result / "deliveries").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- Stage 1: DEDUPE -------------------------------------------------
    print("\n########## [1/3] DEDUPE ##########", flush=True)
    run([PY, "dedupe/run_dedupe.py", "--dir", str(input_dir),
         "--tier", "all", "--apply", "symlink",
         "--out", str(result / "dedupe"),
         "--unique-dir", str(result / "dedupe" / "unique")], check=True)
    rep = json.loads((result / "dedupe" / "dedupe_report.json").read_text())
    uniques = [l for l in (result / "dedupe" / "unique_files.txt")
               .read_text().splitlines() if l.strip()]
    print(f"  dedupe: {rep['total_files']} files -> {len(uniques)} unique "
          f"({rep['duplicate_files']} duplicates dropped, "
          f"{rep['duplicate_clusters']} clusters)", flush=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(input_dir),
        "dedupe": {"total": rep["total_files"], "unique": len(uniques),
                   "duplicates": rep["duplicate_files"],
                   "clusters": rep["duplicate_clusters"],
                   "report": str(result / "dedupe" / "dedupe_report.md")},
        "files": [],
    }

    # ---- Stage 2 + 3: REDACT then METADATA per unique rep ----------------
    for i, f in enumerate(uniques, 1):
        src = Path(f)
        fid = run_pipeline.file_id_for(src)
        print(f"\n########## [2/3] REDACT {i}/{len(uniques)}: "
              f"{src.name} ##########", flush=True)
        rc = run([PY, "tools/redact.py", str(src), "--outdir",
                  str(result / "deliveries"), "--profile", args.profile])
        delivered = None
        quarantined = False
        if (result / "deliveries" / fid / "manifest.json").exists():
            delivered = result / "deliveries" / fid
        elif (ROOT / "quarantine" / fid / "manifest.json").exists():
            # redactor quarantines inside the project; mirror into result/
            dst = result / "quarantine" / fid
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(ROOT / "quarantine" / fid, dst)
            delivered = dst
            quarantined = True

        entry = {"fid": fid, "source": str(src),
                 "redact_rc": rc.returncode,
                 "quarantined": quarantined,
                 "delivery": str(delivered) if delivered else None}

        print(f"########## [3/3] METADATA {i}/{len(uniques)}: "
              f"{fid} ##########", flush=True)
        wd = find_work_dir(fid, args.engine)
        if wd and delivered:
            mcmd = [PY, "metadata_qc_pipeline/run_metadata_qc.py",
                    "--dir", str(wd), "--delivery", str(delivered)]
            if args.no_domain:
                mcmd.append("--no-domain")
            run(mcmd)
            try:
                meta = json.loads((delivered / "metadata.json").read_text())
                entry["domain"] = (meta.get("domain") or {}).get("domain")
                entry["qc"] = (meta.get("qc") or {}).get("qc_status")
                entry["metadata"] = str(delivered / "metadata.json")
            except Exception as e:
                entry["metadata_error"] = str(e)[:150]
        else:
            entry["metadata_error"] = f"work_dir={wd} delivered={delivered}"
        summary["files"].append(entry)

    (result / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n########## DONE in {time.time() - t_start:.1f}s ##########",
          flush=True)
    print(f"  result -> {result}")
    print(f"  {rep['total_files']} in -> {len(uniques)} unique -> "
          f"{sum(1 for e in summary['files'] if e.get('qc'))} processed")
    for e in summary["files"]:
        print(f"    {e['fid']}: qc={e.get('qc')} domain={e.get('domain')}")


if __name__ == "__main__":
    main()
