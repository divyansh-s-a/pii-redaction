#!/usr/bin/env python3
"""Metadata / QC sidecar — describe one redacted deliverable.

  python3 metadata_qc_pipeline/run_metadata_qc.py --dir output/<fid>__elevenlabs
  python3 metadata_qc_pipeline/run_metadata_qc.py --dir output/<fid>__elevenlabs \
        --delivery deliveries/<fid>          # also drop metadata.json there
  python3 metadata_qc_pipeline/run_metadata_qc.py --dir <d> --no-domain

Reads a processed file's working directory (prepare.json, transcript.json,
intervals.json, rendered_intervals.json, the redacted transcript and the beeped
deliverable) and writes metadata.json. Standalone by design — a top-level
driver will later chain dedupe -> redact -> metadata.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import common  # noqa: E402
import probe  # noqa: E402
import audio_stats  # noqa: E402
import diarization  # noqa: E402
import domain as domain_mod  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parent / "config.json").read_text())


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_fid(work_dir: Path, prepare, arg_fid):
    if arg_fid:
        return arg_fid
    if prepare and prepare.get("file_id"):
        return prepare["file_id"]
    for f in work_dir.glob("*_beeped.*"):
        return f.name.rsplit("_beeped.", 1)[0]
    return work_dir.name.split("__")[0]


def _find_deliverable(work_dir: Path, fid: str):
    p = work_dir / common.beeped_name(fid)
    if p.exists():
        return p
    hits = sorted(work_dir.glob(f"{fid}_beeped.*")) or sorted(
        work_dir.glob("*_beeped.*"))
    return hits[0] if hits else None


def redaction_stats(intervals, rendered, duration_ms):
    ivs = (intervals or {}).get("intervals", [])
    cats = {}
    needs_review = 0
    for iv in ivs:
        for c in iv.get("categories", []):
            cats[c] = cats.get(c, 0) + 1
        if iv.get("needs_review"):
            needs_review += 1
    beeped_ms = 0
    if rendered:
        for r in rendered.get("rendered_merged", []):
            beeped_ms += max(0, int(r["end_ms"]) - int(r["start_ms"]))
    return {
        "pii_count": len(ivs),
        "pii_categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "needs_review_count": needs_review,
        "beeped_intervals": (rendered or {}).get("beeped_intervals",
                                                 len(ivs)),
        "beeped_duration_ms": beeped_ms,
        "beeped_pct": round(100.0 * beeped_ms / duration_ms, 2)
        if duration_ms else None,
    }


def _norm_lang(code):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import score_run
    return score_run.norm_lang(code)


def language_block(engine_lang: dict, dom: dict):
    """Two-signal language reconciliation (owner decision 2026-09-10):
    assert a primary language ONLY when the transcription engine's LID and
    the text-based check agree; otherwise primary_language='unknown' with the
    detected components listed, both raw signals kept with provenance, and a
    review flag set. The flag NEVER quarantines — language is not a
    redaction-safety concern."""
    eng_primary = _norm_lang((engine_lang or {}).get("primary_language"))
    eng_langs = [_norm_lang(l.get("lang"))
                 for l in (engine_lang or {}).get("languages", [])]
    raw_txt = str(dom.get("language") or "unknown").strip().lower()
    txt = _norm_lang(raw_txt.split()[0].split("(")[0] if raw_txt else "unknown")
    agree = (txt not in ("unknown", "mixed") and eng_primary == txt)
    detected = []
    for cand in [txt] + [eng_primary] + eng_langs:
        if cand not in ("unknown", "mixed") and cand not in detected:
            detected.append(cand)
    block = {
        "primary_language": txt if agree else "unknown",
        "languages_detected": detected,
        "signals_agree": agree,
        "engine_lid": {"primary_language": eng_primary,
                       "languages": eng_langs, "source": "transcription engine"},
        "text_check": {"language": txt,
                       "share": dom.get("language_share"),
                       "source": "LLM read of the redacted transcript"},
    }
    if not agree:
        block["review_flag"] = ("engine LID and text check disagree or are "
                                "unsure — primary left 'unknown' pending "
                                "human confirmation (never a quarantine "
                                "reason)")
    return block


def build_metadata(work_dir: Path, delivery_dir, fid, use_domain=True):
    prepare = _load(work_dir / "prepare.json")
    transcript = _load(work_dir / "transcript.json") or {}
    intervals = _load(work_dir / "intervals.json")
    rendered = _load(work_dir / "rendered_intervals.json")
    manifest = None
    for cand in ([delivery_dir] if delivery_dir else []) + [work_dir]:
        manifest = manifest or _load(Path(cand) / "manifest.json")

    deliverable = _find_deliverable(work_dir, fid)

    # domain (from redacted transcript — privacy-safe)
    redacted = work_dir / f"{fid}_redacted.txt"
    dom = {"domain": "unknown", "source": "none"}
    if redacted.exists():
        dom = domain_mod.classify_domain(redacted.read_text(), CFG,
                                         use_llm=use_domain)

    tech = probe.probe(deliverable) if deliverable else {}
    signal = (audio_stats.signal_stats(deliverable, CFG)
              if deliverable else {"error": "no deliverable found"})
    diar = diarization.diarization_stats(transcript) if transcript else {}
    duration_ms = (tech.get("duration_ms") or signal.get("duration_ms")
                   or (prepare or {}).get("duration_ms"))
    redaction = redaction_stats(intervals, rendered, duration_ms)

    qc = {}
    if manifest:
        qc = {"qc_status": manifest.get("qc_status"),
              "audit_leak_candidates": (manifest.get("judge_audit")
                                        or {}).get("leak_candidates"),
              "quarantined": bool(manifest.get("quarantined_reason")),
              "quarantined_reason": manifest.get("quarantined_reason")}

    lang = language_block((diar or {}).get("language"), dom)

    return {
        "file_id": fid,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "language": lang,
        "domain": dom,
        "audio": {
            "deliverable": deliverable.name if deliverable else None,
            "codec": tech.get("codec"),
            "container": tech.get("container"),
            "sample_rate_hz": tech.get("sample_rate_hz")
            or signal.get("sample_rate_hz"),
            "channel_count": tech.get("channel_count")
            or signal.get("channel_count"),
            "bit_depth": tech.get("bit_depth"),
            "bit_rate": tech.get("bit_rate"),
            "duration_ms": duration_ms,
            "file_size_bytes": tech.get("file_size_bytes"),
            "source_file": (prepare or {}).get("source"),
            "source_sha256": (prepare or {}).get("sha256"),
            "source_channels": ((prepare or {}).get("channel_analysis")
                                or {}).get("channels"),
            "probe": tech.get("probe"),
        },
        "signal": signal,
        "diarization": diar,
        "redaction": redaction,
        "qc": qc,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--dir", required=True,
                    help="working dir of a processed file (output/<fid>__...)")
    ap.add_argument("--delivery", default=None,
                    help="delivery dir to also receive metadata.json")
    ap.add_argument("--fid", default=None, help="override file_id")
    ap.add_argument("--no-domain", dest="domain", action="store_false",
                    default=True, help="skip the domain LLM call")
    ap.add_argument("--out", default=None,
                    help="metadata.json path (default: <dir>/metadata.json)")
    args = ap.parse_args()

    work_dir = Path(args.dir).resolve()
    if not work_dir.exists():
        raise SystemExit(f"dir not found: {work_dir}")
    prepare = _load(work_dir / "prepare.json")
    fid = _resolve_fid(work_dir, prepare, args.fid)

    t0 = time.time()
    meta = build_metadata(work_dir, args.delivery, fid, use_domain=args.domain)
    out = Path(args.out) if args.out else work_dir / "metadata.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if args.delivery:
        Path(args.delivery).mkdir(parents=True, exist_ok=True)
        (Path(args.delivery) / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))

    d = meta["domain"]
    a = meta["audio"]
    s = meta["signal"]
    dia = meta["diarization"]
    print(f"=== metadata: {fid}  ({time.time() - t0:.2f}s) ===")
    print(f"  domain: {d.get('domain')} ({d.get('confidence')}, "
          f"{d.get('source')})")
    print(f"  audio: {a.get('codec')} {a.get('sample_rate_hz')}Hz "
          f"{a.get('channel_count')}ch {a.get('duration_ms')}ms")
    print(f"  signal: peak {s.get('peak_dbfs')}dBFS rms {s.get('rms_dbfs')}"
          f"dBFS silence {s.get('silence_pct')}% snr {s.get('estimated_snr_db')}dB")
    print(f"  speakers: {dia.get('speaker_count')} | turns "
          f"{dia.get('turn_count')} | wpm {dia.get('words_per_minute')} | "
          f"lang {meta.get('language', {}).get('primary_language')} "
          f"(engine LID {(dia.get('language') or {}).get('primary_language')})")
    print(f"  redaction: {meta['redaction']['pii_count']} PII, "
          f"{meta['redaction']['beeped_pct']}% beeped")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
