#!/usr/bin/env python3
"""Audio dataset dedupe — scalable pre-filter for the redaction pipeline.

  python3 dedupe/run_dedupe.py --dir samples/
  python3 dedupe/run_dedupe.py --dir raw/ --tier all --workers 16
  python3 dedupe/run_dedupe.py --dir raw/ --apply symlink   # materialize uniques

Pipeline:
  1. gather audio files (--dir + --glob, recursive)
  2. fingerprint each (parallel, resumable cache in output/dedupe/
     fingerprints.jsonl — re-runs skip files already recorded, so a 100k-hour
     corpus can be fingerprinted incrementally or sharded across machines)
  3. attach transcript shingles when a matching transcript.json exists
     (from a prior redaction run) and the transcript tier is enabled
  4. match (exact + decoded hash groups, acoustic + transcript MinHash/LSH)
  5. union-find cluster + canonical representative per cluster
  6. write report + unique_files.txt; optionally materialize the unique set

Scale notes: matching keeps only compact MinHash signatures in memory and uses
LSH bucketing, so it is near-linear rather than O(n^2). Install Chromaprint
(`fpcalc`) on the worker boxes for the fastest fingerprinting; the numpy
backend is the automatic fallback.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import common                                       # noqa: E402
import run_pipeline                                 # noqa: E402
import fingerprint as fp                            # noqa: E402
import match as matcher                             # noqa: E402
import cluster as clusterer                         # noqa: E402

CFG = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".mp4", ".mov", ".m4v", ".aac",
              ".flac", ".ogg", ".opus", ".wma", ".avi", ".mpeg", ".mpga",
              ".mp2", ".webm"}

# worker globals (set once per process; avoids re-pickling cfg per task)
_W = {}


def _winit(cfg, backend):
    _W["cfg"], _W["backend"] = cfg, backend


def _work(path):
    try:
        return fp.fingerprint_file(path, _W["cfg"], _W["backend"])
    except Exception as e:                           # pragma: no cover
        return {"path": str(path), "name": Path(path).name,
                "size": Path(path).stat().st_size if Path(path).exists() else 0,
                "error": str(e)[:200], "audio_tokens": []}


def gather(dirs, glob):
    files = []
    for d in dirs:
        base = Path(d)
        if base.is_file():
            files.append(base.resolve())
            continue
        for p in base.rglob(glob):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p.resolve())
    # de-dup the path list itself, deterministic order
    return sorted({str(p): p for p in files}.values(), key=lambda p: str(p))


def load_cache(cache_path: Path):
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                cache[r["path"]] = r
            except Exception:
                continue
    return cache


def fingerprint_all(files, cache, cache_path: Path, backend, workers):
    """Fingerprint everything not already cached (same path + size), appending
    new records to the JSONL cache as they complete. Returns records list."""
    todo = [f for f in files
            if not (str(f) in cache
                    and cache[str(f)].get("size") == f.stat().st_size
                    and "audio_tokens" in cache[str(f)])]
    print(f"  {len(files)} file(s); {len(cache)} cached; "
          f"{len(todo)} to fingerprint (backend={backend}, workers={workers})")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        done = 0
        t0 = time.time()
        with open(cache_path, "a") as sink:
            if workers == 1:
                _winit(CFG, backend)
                results = (_work(str(f)) for f in todo)
                for rec in results:
                    cache[rec["path"]] = rec
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
                    done += 1
                    if done % 50 == 0 or done == len(todo):
                        print(f"    fingerprinted {done}/{len(todo)} "
                              f"({done / (time.time() - t0 + 1e-9):.1f}/s)")
            else:
                import multiprocessing as mp
                with mp.Pool(workers, initializer=_winit,
                             initargs=(CFG, backend)) as pool:
                    for rec in pool.imap_unordered(
                            _work, [str(f) for f in todo], chunksize=4):
                        cache[rec["path"]] = rec
                        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        sink.flush()
                        done += 1
                        if done % 50 == 0 or done == len(todo):
                            print(f"    fingerprinted {done}/{len(todo)} "
                                  f"({done / (time.time() - t0 + 1e-9):.1f}/s)")
    return [cache[str(f)] for f in files if str(f) in cache]


def _find_transcript(path: Path):
    fid = run_pipeline.file_id_for(path)
    outroot = common.ROOT / "output"
    if not outroot.exists():
        return None
    for d in sorted(outroot.glob(fid + "*")):
        t = d / "transcript.json"
        if t.exists():
            return t
    return None


def attach_transcripts(records):
    n = 0
    for r in records:
        t = _find_transcript(Path(r["path"]))
        if t:
            try:
                r["transcript_tokens"] = fp.transcript_tokens(t, CFG)
                n += 1
            except Exception:
                r["transcript_tokens"] = []
    return n


def best_edge_per_pair(edges):
    """Collapse multi-tier edges to one per pair, keeping the highest score;
    remember every tier that fired (exact > decoded > audio > transcript)."""
    order = {"exact": 3, "decoded": 2, "audio": 1, "transcript": 0}
    best = {}
    for a, b, score, tier in edges:
        key = (a, b)
        cur = best.get(key)
        if cur is None or (score, order[tier]) > (cur["score"],
                                                  order[cur["tier"]]):
            best[key] = {"score": score, "tier": tier, "tiers": set()}
        best[key]["tiers"].add(tier)
    return best


def build_report(files, records, edges, recmap):
    paths = [str(f) for f in files]
    clusters = clusterer.cluster(paths, edges)
    best = best_edge_per_pair(edges)
    dup_clusters, keep, drop = [], [], []
    for members in clusters:
        rep = clusterer.pick_representative(members, recmap)
        keep.append(rep)
        if len(members) == 1:
            continue
        pair_info = []
        for (a, b), info in best.items():
            if a in members and b in members:
                pair_info.append({"a": a, "b": b, "score": info["score"],
                                  "tier": info["tier"],
                                  "tiers": sorted(info["tiers"])})
        for m in members:
            if m != rep:
                drop.append(m)
        dup_clusters.append({
            "representative": rep,
            "members": members,
            "size": len(members),
            "matches": sorted(pair_info, key=lambda x: -x["score"]),
        })
    dup_clusters.sort(key=lambda c: -c["size"])
    errored = [r["path"] for r in records if r.get("error")]
    report = {
        "total_files": len(paths),
        "unique_files": len(keep),
        "duplicate_files": len(drop),
        "duplicate_clusters": len(dup_clusters),
        "errored_files": errored,
        "clusters": dup_clusters,
        "keep": sorted(keep),
        "drop": sorted(drop),
    }
    return report


def write_markdown(report, path: Path):
    lines = ["# Audio dedupe report", ""]
    lines.append(f"- files scanned: **{report['total_files']}**")
    lines.append(f"- unique after dedupe: **{report['unique_files']}**")
    lines.append(f"- duplicates to drop: **{report['duplicate_files']}**")
    lines.append(f"- duplicate clusters: **{report['duplicate_clusters']}**")
    if report["errored_files"]:
        lines.append(f"- files with errors: **{len(report['errored_files'])}**")
    lines.append("")
    if report["clusters"]:
        lines.append("## Duplicate clusters")
        for i, c in enumerate(report["clusters"], 1):
            lines.append("")
            lines.append(f"### Cluster {i} ({c['size']} files)")
            lines.append(f"- keep: `{Path(c['representative']).name}`")
            for m in c["members"]:
                if m != c["representative"]:
                    lines.append(f"- drop: `{Path(m).name}`")
            for mt in c["matches"]:
                lines.append(f"  - {Path(mt['a']).name} ~ {Path(mt['b']).name}"
                             f"  ({mt['tier']}, score={mt['score']})")
    else:
        lines.append("_No duplicates found._")
    lines.append("")
    path.write_text("\n".join(lines))


def apply_action(report, action, unique_dir: Path, dup_dir: Path):
    def link(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if action == "copy":
            shutil.copy2(src, dst)
        elif action == "symlink":
            dst.symlink_to(src.resolve())
        elif action == "move":
            shutil.move(str(src), str(dst))
    for rep in report["keep"]:
        src = Path(rep)
        link(src, unique_dir / src.name)
    if action == "move":
        for d in report["drop"]:
            src = Path(d)
            if src.exists():
                dst = dup_dir / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
    print(f"  applied '{action}': {len(report['keep'])} uniques -> {unique_dir}"
          + (f"; {len(report['drop'])} dupes -> {dup_dir}"
             if action == "move" else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--dir", action="append", required=True,
                    help="directory (recursive) or file to scan; repeatable")
    ap.add_argument("--glob", default="*", help="filename glob (default: all)")
    ap.add_argument("--tier", default="all",
                    choices=["exact", "audio", "transcript", "all"],
                    help="matching tiers to run (default: all)")
    ap.add_argument("--backend", default=CFG.get("fp_backend", "auto"),
                    choices=["auto", "fpcalc", "numpy"],
                    help="acoustic fingerprint backend (auto uses fpcalc if "
                         "installed, else numpy)")
    ap.add_argument("--workers", type=int, default=0,
                    help="fingerprint worker processes (0 = all cores)")
    ap.add_argument("--audio-threshold", type=float, default=None)
    ap.add_argument("--transcript-threshold", type=float, default=None)
    ap.add_argument("--apply", choices=["copy", "symlink", "move"],
                    default=None, help="materialize the unique set")
    ap.add_argument("--out", default=str(common.ROOT / "output" / "dedupe"),
                    help="output directory for cache + report")
    ap.add_argument("--unique-dir", default=None,
                    help="where --apply writes uniques (default: <out>/unique)")
    ap.add_argument("--limit", type=int, default=0, help="scan first N files")
    args = ap.parse_args()

    if args.audio_threshold is not None:
        CFG["audio_threshold"] = args.audio_threshold
    if args.transcript_threshold is not None:
        CFG["transcript_threshold"] = args.transcript_threshold
    tiers = ({"exact", "audio", "transcript"} if args.tier == "all"
             else {args.tier})
    workers = args.workers or os.cpu_count() or 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "fingerprints.jsonl"

    files = gather(args.dir, args.glob)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("no audio files found")
        return
    print(f"=== dedupe: {len(files)} file(s) | tiers={sorted(tiers)} ===")

    want_audio = bool({"audio", "exact"} & tiers)
    cache = load_cache(cache_path)
    records = fingerprint_all(files, cache, cache_path, args.backend,
                              workers if want_audio else 1)

    if "transcript" in tiers:
        n = attach_transcripts(records)
        print(f"  transcripts attached: {n}/{len(records)}"
              + ("" if n else " (none found — transcript tier idle)"))

    recmap = {r["path"]: r for r in records}
    edges = matcher.all_edges(records, CFG, tiers)
    report = build_report(files, records, edges, recmap)

    (out / "dedupe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report, out / "dedupe_report.md")
    (out / "unique_files.txt").write_text("\n".join(report["keep"]) + "\n")

    print(f"\n=== result ===")
    print(f"  unique: {report['unique_files']} | duplicates: "
          f"{report['duplicate_files']} | clusters: "
          f"{report['duplicate_clusters']}")
    if report["errored_files"]:
        print(f"  errors: {len(report['errored_files'])} file(s)")
    print(f"  report -> {out / 'dedupe_report.md'}")
    print(f"  unique list -> {out / 'unique_files.txt'}")

    if args.apply:
        unique_dir = Path(args.unique_dir) if args.unique_dir else out / "unique"
        apply_action(report, args.apply, unique_dir, out / "duplicates")


if __name__ == "__main__":
    main()
