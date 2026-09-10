#!/usr/bin/env python3
"""Standalone final-report generator for a delivered audio dataset.

Scans the per-file ``metadata.json`` files produced by metadata_qc_pipeline and
writes a single markdown report describing the DATASET: domain mix, audio
format, duration, signal quality, speaker/conversation shape and language
distribution. It intentionally says nothing about PII redaction outcomes.

  python3 generate_final_report.py --result ../result
  python3 generate_final_report.py --deliveries ../result/deliveries --out report.md

Stdlib only (json, statistics, pathlib, argparse, collections, datetime).
"""
import argparse
import json
import statistics as st
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------- helpers
def _load_all(dirs):
    out = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("metadata.json")):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                pass
    return out


def _nums(values):
    return [v for v in values if isinstance(v, (int, float))]


def _hms(ms):
    if not ms:
        return "0:00"
    s = int(ms) // 1000
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _stat_row(label, values, unit="", nd=2):
    v = _nums(values)
    if not v:
        return f"| {label} | — | — | — | — |"
    return (f"| {label} | {min(v):.{nd}f}{unit} | {st.mean(v):.{nd}f}{unit} | "
            f"{st.median(v):.{nd}f}{unit} | {max(v):.{nd}f}{unit} |")


def _dist_table(counter, header, total):
    lines = [f"| {header} | Files | Share |", "|---|---:|---:|"]
    for key, n in counter.most_common():
        pct = 100.0 * n / total if total else 0
        lines.append(f"| {key} | {n} | {pct:.1f}% |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- report
def build_report(metas, dedupe, title):
    n = len(metas)
    total_ms = sum(_nums(m.get("audio", {}).get("duration_ms") for m in metas))
    total_bytes = sum(_nums(m.get("audio", {}).get("file_size_bytes")
                            for m in metas))
    L = []
    A = L.append

    A(f"# {title}\n")
    A(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
      f"{n} deliverable file(s)_\n")

    # ---- Overview ----
    A("## 1. Dataset overview\n")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| Deliverable files | {n} |")
    A(f"| Total audio | {_hms(total_ms)} ({total_ms / 3_600_000:.2f} h) |")
    A(f"| Mean file length | {_hms(total_ms / n) if n else '—'} |")
    A(f"| Total size | {total_bytes / 1_048_576:.1f} MiB |")
    if dedupe:
        A(f"| Source files scanned | {dedupe.get('total_files')} |")
        A(f"| Duplicates removed | {dedupe.get('duplicate_files')} "
          f"({dedupe.get('duplicate_clusters')} clusters) |")
        A(f"| Unique after dedupe | {dedupe.get('unique_files')} |")
    A("")

    # ---- Domain ----
    dom = Counter((m.get("domain") or {}).get("domain") or "unknown"
                  for m in metas)
    conf = Counter((m.get("domain") or {}).get("confidence") or "n/a"
                   for m in metas)
    A("## 2. Domain distribution\n")
    A(_dist_table(dom, "Domain", n))
    A("")
    A(f"Domain confidence: " +
      ", ".join(f"{k} {v}" for k, v in conf.most_common()) + "\n")

    # ---- Audio format ----
    aud = [m.get("audio", {}) for m in metas]
    cont = Counter(a.get("container") or "—" for a in aud)
    codec = Counter(a.get("codec") or "unspecified" for a in aud)
    sr = Counter(f"{a.get('sample_rate_hz')} Hz" for a in aud)
    ch = Counter(f"{a.get('channel_count')} ch" for a in aud)
    brs = _nums(a.get("bit_rate") for a in aud)
    A("## 3. Audio format\n")
    A(_dist_table(cont, "Container", n))
    A("")
    A(_dist_table(sr, "Sample rate", n))
    A("")
    A(_dist_table(ch, "Channels", n))
    A("")
    A("Codec: " + ", ".join(f"{k} ({v})" for k, v in codec.most_common()))
    if brs:
        A(f"  \nBit rate: {min(brs) / 1000:.0f}–{max(brs) / 1000:.0f} kbps "
          f"(mean {st.mean(brs) / 1000:.0f} kbps)")
    A("")

    # ---- Duration ----
    durs = _nums(a.get("duration_ms") for a in aud)
    buckets = Counter()
    for d in durs:
        mn = d / 60000.0
        buckets["< 1 min" if mn < 1 else "1–3 min" if mn < 3 else
                "3–5 min" if mn < 5 else "5–10 min" if mn < 10 else
                "> 10 min"] += 1
    order = ["< 1 min", "1–3 min", "3–5 min", "5–10 min", "> 10 min"]
    A("## 4. Duration distribution\n")
    A("| Metric | Min | Mean | Median | Max |")
    A("|---|---|---|---|---|")
    if durs:
        A(f"| Duration | {_hms(min(durs))} | {_hms(st.mean(durs))} | "
          f"{_hms(st.median(durs))} | {_hms(max(durs))} |")
    A("")
    A("| Length bucket | Files | Share |")
    A("|---|---:|---:|")
    for b in order:
        if buckets.get(b):
            A(f"| {b} | {buckets[b]} | {100.0 * buckets[b] / n:.1f}% |")
    A("")

    # ---- Signal quality ----
    sig = [m.get("signal", {}) for m in metas]
    A("## 5. Signal quality\n")
    A("| Metric | Min | Mean | Median | Max |")
    A("|---|---|---|---|---|")
    A(_stat_row("Peak (dBFS)", [s.get("peak_dbfs") for s in sig]))
    A(_stat_row("RMS (dBFS)", [s.get("rms_dbfs") for s in sig]))
    A(_stat_row("Crest factor (dB)", [s.get("crest_factor_db") for s in sig]))
    A(_stat_row("Est. SNR (dB)", [s.get("estimated_snr_db") for s in sig]))
    A(_stat_row("Noise floor (dBFS)",
                [s.get("noise_floor_dbfs") for s in sig]))
    A(_stat_row("Silence (%)", [s.get("silence_pct") for s in sig], nd=1))
    A(_stat_row("Speech (%)", [s.get("speech_pct") for s in sig], nd=1))
    A(_stat_row("Clipping (%)", [s.get("clipping_pct") for s in sig], nd=3))
    A("")

    # ---- Speakers / conversation ----
    dia = [m.get("diarization", {}) for m in metas]
    spk = Counter(f"{d.get('speaker_count')} speakers" for d in dia)
    dominant = _nums((d.get("speaker_distribution") or [{}])[0].get("talk_pct")
                     for d in dia)
    A("## 6. Speakers & conversation\n")
    A(_dist_table(spk, "Speaker count", n))
    A("")
    A("| Metric | Min | Mean | Median | Max |")
    A("|---|---|---|---|---|")
    A(_stat_row("Words / minute", [d.get("words_per_minute") for d in dia],
                nd=1))
    A(_stat_row("Turns", [d.get("turn_count") for d in dia], nd=0))
    A(_stat_row("Avg turn (s)",
                [(d.get("avg_turn_ms") or 0) / 1000 for d in dia], nd=1))
    A(_stat_row("Longest monologue (s)",
                [(d.get("longest_monologue_ms") or 0) / 1000 for d in dia],
                nd=1))
    A(_stat_row("Speech coverage (%)",
                [d.get("speech_coverage_pct") for d in dia], nd=1))
    if dominant:
        A(f"\nDominant-speaker talk share: mean {st.mean(dominant):.1f}%, "
          f"range {min(dominant):.1f}–{max(dominant):.1f}%")
    A("")

    # ---- Language ----
    lang = Counter((d.get("language") or {}).get("primary_language") or "n/a"
                   for d in dia)
    cs = sum(1 for d in dia if (d.get("language") or {}).get("code_switching"))
    nl = Counter(str((d.get("language") or {}).get("num_languages") or 0)
                 for d in dia)
    A("## 7. Language distribution\n")
    A(_dist_table(lang, "Primary language", n))
    A("")
    A(f"Multilingual (code-switching) files: {cs}/{n}  ·  "
      f"languages-per-file: " +
      ", ".join(f"{k}→{v}" for k, v in sorted(nl.items())) + "\n")

    # ---- Per-file appendix ----
    A("## 8. Per-file detail\n")
    A("| File | Domain | Duration | Rate | Ch | Speakers | Turns | WPM | "
      "Lang | SNR dB | Silence % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(metas, key=lambda x: -(x.get("audio", {})
                                           .get("duration_ms") or 0)):
        a, s, d = m.get("audio", {}), m.get("signal", {}), \
            m.get("diarization", {})
        fid = m.get("file_id", "?")
        A(f"| {fid} | {(m.get('domain') or {}).get('domain', '—')} | "
          f"{_hms(a.get('duration_ms'))} | {a.get('sample_rate_hz')} | "
          f"{a.get('channel_count')} | {d.get('speaker_count')} | "
          f"{d.get('turn_count')} | {d.get('words_per_minute')} | "
          f"{(d.get('language') or {}).get('primary_language', '—')} | "
          f"{s.get('estimated_snr_db')} | {s.get('silence_pct')} |")
    A("")
    A("---")
    A(f"_Report covers {n} deliverable file(s); "
      f"{total_ms / 3_600_000:.2f} h of audio._")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", default=str(ROOT.parent / "result"),
                    help="result/ base (finds deliveries/ + dedupe/)")
    ap.add_argument("--deliveries", default=None,
                    help="override folder to scan for metadata.json")
    ap.add_argument("--include-quarantine", action="store_true",
                    help="also include result/quarantine files in the dataset")
    ap.add_argument("--title", default="Audio Dataset — Final Delivery Report")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = Path(args.result).resolve()
    scan = [Path(args.deliveries).resolve()] if args.deliveries \
        else [result / "deliveries"]
    if args.include_quarantine:
        scan.append(result / "quarantine")

    metas = _load_all(scan)
    if not metas:
        raise SystemExit(f"no metadata.json found under: "
                         f"{', '.join(str(s) for s in scan)}")

    dedupe = None
    dr = result / "dedupe" / "dedupe_report.json"
    if dr.exists():
        try:
            dedupe = json.loads(dr.read_text())
        except Exception:
            pass

    report = build_report(metas, dedupe, args.title)
    out = Path(args.out).resolve() if args.out else result / "FINAL_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}  ({len(metas)} files, "
          f"{sum(_nums(m.get('audio', {}).get('duration_ms') for m in metas)) / 3_600_000:.2f} h)")


if __name__ == "__main__":
    main()
