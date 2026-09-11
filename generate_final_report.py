#!/usr/bin/env python3
"""Standalone final-report generator for a delivered audio dataset.

Two modes:
  * default (deterministic) — compiles the per-file metadata.json files into a
    measured markdown report (dataset distribution + a Deduplication section
    showing which clips were duplicates, which were kept and which dropped).
  * --llm — sends the SAME measured facts (plus an optional .docx style
    template) to Claude Opus 4.8 via OpenRouter and writes the polished,
    buyer-facing narrative report it returns.

Both modes describe the DATASET only; neither reports PII-redaction
success/failure, QC pass/fail, quarantine or leak-audit outcomes.

  python3 generate_final_report.py --result ../result
  python3 generate_final_report.py --result ../result --llm
  python3 generate_final_report.py --llm --template ../Health-Insurance-Call-Audio-Dataset-Report.docx

Stdlib only for the deterministic path; --llm reuses pipeline/common.py for the
OpenRouter call (needs OPENROUTER_API_KEY in AudioPII-QC/.env).
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

OPENROUTER_MODEL = "anthropic/claude-opus-4.8"
DEFAULT_TEMPLATE = ROOT.parent / "Health-Insurance-Call-Audio-Dataset-Report.docx"
PRIVACY_HANDLING = ("Per-asset redaction policy applied; removal verified by an "
                    "independent ASR pass")


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


_REASON = {
    "exact": "Byte-for-byte / exact decoded-audio duplicate",
    "audio": "Perceptual audio-fingerprint match (re-encoded copy of the "
             "same recording)",
    "transcript": "Near-identical transcript",
}


def _dedupe_facts(dedupe):
    if not dedupe:
        return None
    clusters = []
    for c in dedupe.get("clusters", []):
        rep = Path(c["representative"]).stem
        members = [Path(m).stem for m in c.get("members", [])]
        dropped = [m for m in members if m != rep]
        m0 = (c.get("matches") or [{}])[0]
        tier = m0.get("tier") or "?"
        clusters.append({
            "kept": rep,
            "dropped": dropped,
            "match_type": tier,
            "similarity_score": m0.get("score"),
            "reason": _REASON.get(tier, tier),
        })
    return {
        "source_files_scanned": dedupe.get("total_files"),
        "duplicates_removed": dedupe.get("duplicate_files"),
        "duplicate_clusters": dedupe.get("duplicate_clusters"),
        "unique_after_dedupe": dedupe.get("unique_files"),
        "clusters": clusters,
    }


# ------------------------------------------------------------ facts (for LLM)
def build_facts(metas, dedupe, title):
    n = len(metas)
    aud = [m.get("audio", {}) for m in metas]
    sig = [m.get("signal", {}) for m in metas]
    dia = [m.get("diarization", {}) for m in metas]
    total_ms = sum(_nums(a.get("duration_ms") for a in aud))
    durs = _nums(a.get("duration_ms") for a in aud)
    brs = _nums(a.get("bit_rate") for a in aud)

    def _agg(vals, nd=2):
        v = _nums(vals)
        if not v:
            return None
        return {"min": round(min(v), nd), "mean": round(st.mean(v), nd),
                "median": round(st.median(v), nd), "max": round(max(v), nd)}

    # duration length bands (mirrors the reference appendix)
    bands = Counter()
    for d in durs:
        mn = d / 60000.0
        bands["Under 1 minute" if mn < 1 else "1–3 minutes" if mn < 3 else
              "3–5 minutes" if mn < 5 else "5–10 minutes" if mn < 10 else
              "Over 10 minutes"] += 1

    dd = _dedupe_facts(dedupe)
    delivered_ids = {m.get("file_id") for m in metas}
    reencoded_delivered = []
    if dd:
        for c in dd["clusters"]:
            if c["match_type"] == "audio" and c["kept"] in delivered_ids:
                reencoded_delivered.append(c["kept"])

    return {
        "title": title,
        "generated": datetime.now().strftime("%d %B %Y"),
        "privacy_handling": PRIVACY_HANDLING,
        "overview": {
            "deliverable_files": n,
            "total_audio_hms": _hms(total_ms),
            "total_audio_hours": round(total_ms / 3_600_000, 2),
            "mean_file_length_hms": _hms(total_ms / n) if n else None,
            "shortest_hms": _hms(min(durs)) if durs else None,
            "longest_hms": _hms(max(durs)) if durs else None,
            "total_size_mib": round(
                sum(_nums(a.get("file_size_bytes") for a in aud)) / 1_048_576, 1),
        },
        "domain_distribution": dict(Counter(
            (m.get("domain") or {}).get("domain") or "unknown" for m in metas)),
        "domain_confidence": dict(Counter(
            (m.get("domain") or {}).get("confidence") or "n/a" for m in metas)),
        "usability_distribution": dict(Counter(
            (m.get("domain") or {}).get("usability") or "n/a" for m in metas)),
        "low_usability_files": [
            {"file_id": m.get("file_id"),
             "usability": (m.get("domain") or {}).get("usability"),
             "note": (m.get("domain") or {}).get("rationale")}
            for m in metas
            if (m.get("domain") or {}).get("usability") in ("limited", "not usable")],
        "audio_format": {
            "containers": dict(Counter(a.get("container") or "—" for a in aud)),
            "codecs": dict(Counter(a.get("codec") or "unspecified" for a in aud)),
            "sample_rates_hz": dict(Counter(a.get("sample_rate_hz") for a in aud)),
            "channels": dict(Counter(a.get("channel_count") for a in aud)),
            "bit_rate_kbps": ({"min": round(min(brs) / 1000),
                               "max": round(max(brs) / 1000),
                               "mean": round(st.mean(brs) / 1000)} if brs else None),
            "source_codec_note": "Unspecified in source metadata; container, "
                                 "sample rate, channels and bit rate are measured "
                                 "directly.",
        },
        "duration": {
            "min_hms": _hms(min(durs)) if durs else None,
            "mean_hms": _hms(st.mean(durs)) if durs else None,
            "median_hms": _hms(st.median(durs)) if durs else None,
            "max_hms": _hms(max(durs)) if durs else None,
            "length_bands": dict(bands),
        },
        "signal_quality": {
            "peak_dbfs": _agg(s.get("peak_dbfs") for s in sig),
            "rms_dbfs": _agg(s.get("rms_dbfs") for s in sig),
            "crest_factor_db": _agg(s.get("crest_factor_db") for s in sig),
            "estimated_snr_db": _agg(s.get("estimated_snr_db") for s in sig),
            "noise_floor_dbfs": _agg(s.get("noise_floor_dbfs") for s in sig),
            "silence_pct": _agg((s.get("silence_pct") for s in sig), nd=1),
            "speech_pct": _agg((s.get("speech_pct") for s in sig), nd=1),
            "clipping_pct": _agg((s.get("clipping_pct") for s in sig), nd=3),
        },
        "conversation": {
            "speaker_counts": dict(Counter(d.get("speaker_count") for d in dia)),
            "words_per_minute": _agg((d.get("words_per_minute") for d in dia), 1),
            "turns": _agg((d.get("turn_count") for d in dia), 0),
            "avg_turn_s": _agg(((d.get("avg_turn_ms") or 0) / 1000 for d in dia), 1),
            "longest_monologue_s": _agg(
                ((d.get("longest_monologue_ms") or 0) / 1000 for d in dia), 1),
            "speech_coverage_pct": _agg(
                (d.get("speech_coverage_pct") for d in dia), 1),
            "total_turns": sum(_nums(d.get("turn_count") for d in dia)),
            "dominant_speaker_talk_pct": _agg(
                ((d.get("speaker_distribution") or [{}])[0].get("talk_pct")
                 for d in dia), 1),
        },
        "language_distribution": dict(Counter(
            (m.get("language") or d.get("language") or {}).get("primary_language")
            or "n/a" for m, d in zip(metas, dia))),
        "code_switching_files": sum(
            1 for d in dia if (d.get("language") or {}).get("code_switching")),
        "deduplication": dd,
        "reencoded_delivered_files": reencoded_delivered,
        "per_file": [{
            "file_id": m.get("file_id"),
            "domain": (m.get("domain") or {}).get("domain"),
            "duration_hms": _hms(m.get("audio", {}).get("duration_ms")),
            "sample_rate_hz": m.get("audio", {}).get("sample_rate_hz"),
            "channels": m.get("audio", {}).get("channel_count"),
            "speakers": m.get("diarization", {}).get("speaker_count"),
            "turns": m.get("diarization", {}).get("turn_count"),
            "wpm": m.get("diarization", {}).get("words_per_minute"),
            "language": (m.get("language")
                         or m.get("diarization", {}).get("language")
                         or {}).get("primary_language"),
            "usability": (m.get("domain") or {}).get("usability"),
            "peak_dbfs": m.get("signal", {}).get("peak_dbfs"),
            "rms_dbfs": m.get("signal", {}).get("rms_dbfs"),
            "snr_db": m.get("signal", {}).get("estimated_snr_db"),
            "silence_pct": m.get("signal", {}).get("silence_pct"),
            "speech_coverage_pct": m.get("diarization", {})
            .get("speech_coverage_pct"),
        } for m in sorted(metas, key=lambda x: -(x.get("audio", {})
                                                 .get("duration_ms") or 0))],
    }


# ------------------------------------------------------------ deterministic md
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

    # ---- Deduplication ----
    df = _dedupe_facts(dedupe)
    if df:
        A("## 2. Deduplication\n")
        A(f"{df['source_files_scanned']} source files were scanned; "
          f"{df['duplicates_removed']} were removed as duplicates across "
          f"{df['duplicate_clusters']} clusters, leaving "
          f"{df['unique_after_dedupe']} unique recordings.\n")
        A("| Cluster | Kept | Dropped | Match | Score | Reason |")
        A("|---:|---|---|---|---:|---|")
        for i, c in enumerate(df["clusters"], 1):
            sc = c["similarity_score"]
            A(f"| {i} | {c['kept']} | {', '.join(c['dropped'])} | "
              f"{c['match_type']} | "
              f"{sc if sc is not None else '—'} | {c['reason']} |")
        A("")

    dom = Counter((m.get("domain") or {}).get("domain") or "unknown"
                  for m in metas)
    A("## 3. Domain distribution\n")
    A(_dist_table(dom, "Domain", n))
    A("")
    use = Counter((m.get("domain") or {}).get("usability") or "n/a"
                  for m in metas)
    A(_dist_table(use, "Conversation usability", n))
    low = [m.get("file_id") for m in metas
           if (m.get("domain") or {}).get("usability") in ("limited",
                                                           "not usable")]
    if low:
        A(f"\nLower-usability calls (low content / greeting-only / dead air): "
          f"{', '.join(low)}.")
    A("")

    aud = [m.get("audio", {}) for m in metas]
    cont = Counter(a.get("container") or "—" for a in aud)
    sr = Counter(f"{a.get('sample_rate_hz')} Hz" for a in aud)
    ch = Counter(f"{a.get('channel_count')} ch" for a in aud)
    brs = _nums(a.get("bit_rate") for a in aud)
    A("## 4. Audio format\n")
    A(_dist_table(cont, "Container", n))
    A("")
    A(_dist_table(sr, "Sample rate", n))
    A("")
    A(_dist_table(ch, "Channels", n))
    if brs:
        A(f"\nBit rate: {min(brs) / 1000:.0f}–{max(brs) / 1000:.0f} kbps "
          f"(mean {st.mean(brs) / 1000:.0f} kbps)")
    A("")

    durs = _nums(a.get("duration_ms") for a in aud)
    A("## 5. Duration distribution\n")
    A("| Metric | Min | Mean | Median | Max |")
    A("|---|---|---|---|---|")
    if durs:
        A(f"| Duration | {_hms(min(durs))} | {_hms(st.mean(durs))} | "
          f"{_hms(st.median(durs))} | {_hms(max(durs))} |")
    A("")

    sig = [m.get("signal", {}) for m in metas]
    A("## 6. Signal quality\n")
    A("| Metric | Min | Mean | Median | Max |")
    A("|---|---|---|---|---|")
    A(_stat_row("Peak (dBFS)", [s.get("peak_dbfs") for s in sig]))
    A(_stat_row("RMS (dBFS)", [s.get("rms_dbfs") for s in sig]))
    A(_stat_row("Crest factor (dB)", [s.get("crest_factor_db") for s in sig]))
    A(_stat_row("Est. SNR (dB)", [s.get("estimated_snr_db") for s in sig]))
    A(_stat_row("Noise floor (dBFS)", [s.get("noise_floor_dbfs") for s in sig]))
    A(_stat_row("Silence (%)", [s.get("silence_pct") for s in sig], nd=1))
    A(_stat_row("Clipping (%)", [s.get("clipping_pct") for s in sig], nd=3))
    A("")

    dia = [m.get("diarization", {}) for m in metas]
    spk = Counter(f"{d.get('speaker_count')} speakers" for d in dia)
    A("## 7. Speakers & conversation\n")
    A(_dist_table(spk, "Speaker count", n))
    A("")
    A("| Metric | Min | Mean | Median | Max |")
    A("|---|---|---|---|---|")
    A(_stat_row("Words / minute", [d.get("words_per_minute") for d in dia], nd=1))
    A(_stat_row("Turns", [d.get("turn_count") for d in dia], nd=0))
    A(_stat_row("Longest monologue (s)",
                [(d.get("longest_monologue_ms") or 0) / 1000 for d in dia], nd=1))
    A("")

    lang = Counter((m.get("language") or m.get("diarization", {})
                    .get("language") or {}).get("primary_language") or "n/a"
                   for m in metas)
    A("## 8. Language distribution\n")
    A(_dist_table(lang, "Primary language", n))
    A("")

    A("## 9. Per-file detail\n")
    A("| File | Domain | Usability | Duration | Rate | Ch | Speakers | Turns | "
      "WPM | Lang | SNR dB | Silence % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(metas, key=lambda x: -(x.get("audio", {})
                                           .get("duration_ms") or 0)):
        a, s, d = m.get("audio", {}), m.get("signal", {}), \
            m.get("diarization", {})
        A(f"| {m.get('file_id', '?')} | "
          f"{(m.get('domain') or {}).get('domain', '—')} | "
          f"{(m.get('domain') or {}).get('usability', '—')} | "
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


# ------------------------------------------------------------------- LLM path
def _docx_text(path: Path) -> str:
    """Extract readable text from a .docx (or return a .md/.txt as-is)."""
    import re
    import html
    import zipfile
    if path.suffix.lower() in (".md", ".txt"):
        return path.read_text()
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = xml.replace("</w:tc>", " \x01 ").replace("</w:tr>", "\n") \
             .replace("</w:p>", "\n")
    text = html.unescape(re.sub(r"<[^>]+>", "", xml)).replace("\x01", " | ")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip(" |").strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _openrouter_chat(model, system, user, max_tokens, temperature=0.4):
    import common
    key = common.openrouter_key()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in AudioPII-QC/.env "
                         "(required for --llm)")
    base = common.CONFIG.get("openrouter_api_base",
                             "https://openrouter.ai/api/v1")
    url = base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "temperature": temperature,
               "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    status, _, body = common._request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"}, timeout=1200)
    if status != 200:
        raise SystemExit(f"OpenRouter HTTP {status}: {body[:500]!r}")
    resp = json.loads(body)
    ch = resp["choices"][0]
    text = ch["message"].get("content") or ""
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    print(f"  Opus tokens: prompt={resp.get('usage', {}).get('prompt_tokens')} "
          f"output={resp.get('usage', {}).get('completion_tokens')} "
          f"cost=${resp.get('usage', {}).get('cost')}")
    return text.strip() + "\n"


_SYSTEM = (
    "You are a senior data-licensing analyst writing a detailed, buyer-facing "
    "delivery report for an audio/speech dataset pack. Write in clear, measured, "
    "professional English, and be VERBOSE and thorough — match the depth, length "
    "and section structure of the TEMPLATE provided, section for section. "
    "Every quantitative claim (counts, durations, hertz, dB, percentages, "
    "scores) MUST come from the FACTS JSON — never invent, estimate or round "
    "beyond what is given; numbers are measured from the delivered files, not "
    "declared. Qualitative framing that carries NO measured numbers (licensing "
    "posture, classification-tag controlled vocabulary, sample/access process, "
    "use-case narratives) should follow the TEMPLATE's house style. You must "
    "NOT mention PII-redaction success or failure, QC pass/fail, quarantine, "
    "leak audits, or per-file PII counts; you MAY describe privacy handling only "
    "at the generic level of FACTS.privacy_handling and the template's "
    "sample-preparation narrative. Output GitHub-flavoured Markdown only (rich "
    "headings, subheadings and tables); do not wrap the whole document in a code "
    "fence."
)


def _user_prompt(facts, template_text):
    outline = (
        "Produce a COMPLETE, detailed dataset delivery report that mirrors the "
        "TEMPLATE section-for-section and is at least as long and thorough. Use "
        "all of these sections, with rich prose (not just tables):\n\n"
        "- Title line + one-line subtitle describing the pack.\n"
        "- Cover table: Asset class, Volume in this pack, Languages, Domains, "
        "Provenance (naturally occurring real calls), Privacy handling, Report "
        "generated. Follow with the standard one-line 'numbers are measured, not "
        "declared' note.\n"
        "1. Description — a rich multi-sentence paragraph grounded in the facts.\n"
        "2. Readme — subsections: 'The gap this fills' (bulleted, explaining why "
        "narrowband/accented/in-situ/turn-taking/long-form matters), 'How it can "
        "help' (a use-case table with 5–6 rows), and 'Overview of the pack' "
        "(Audio, Manifest, Envelope, Annotation layers, Source handling).\n"
        "3. Classification tags — a three-column table (Plane | Value | What it "
        "means for you) using controlled-vocabulary values in the template's "
        "style (asset class, modality, domain, generation provenance, "
        "sensitivity tier, PII class, rights basis, permitted use, exclusivity, "
        "buyer archetype, deal shape, lifecycle stage). Keep the measured domain "
        "split accurate to FACTS.\n"
        "4. Scale — a Measure|Value table INCLUDING source files scanned, "
        "duplicates removed, unique-after-dedupe, speakers per file and total "
        "conversational turns, plus a short paragraph on scaling beyond the "
        "pack.\n"
        "5. Deduplication — REQUIRED. One summary sentence, then a table with "
        "one row per cluster: Kept file, Dropped file(s), Match type, Similarity "
        "score, Reason (use FACTS.deduplication.clusters exactly). Add a sentence "
        "noting which delivered files derive from re-encoded copies "
        "(FACTS.reencoded_delivered_files).\n"
        "6. Samples and access — describe the two-stage sample preparation (PII "
        "handled first via the generic policy; watermarking second) and the "
        "controlled access path, with a Stage|What you get|Turnaround table.\n"
        "7. Known limitations — bulleted and candid (scale, language balance, "
        "bandwidth, transcription is machine-generated, source codec metadata "
        "unspecified, re-encoded pairs, rights under review). ALSO include a "
        "'Conversation usability' bullet: give the usability mix from "
        "FACTS.usability_distribution and explicitly name any file(s) in "
        "FACTS.low_usability_files as low-content / greeting-only / dead-air "
        "calls that a buyer may wish to exclude.\n"
        "8. Appendix A — Full metadata report. Reproduce EVERY sub-table with "
        "min/mean/median/max where applicable: A.1 Domain distribution (+ "
        "confidence), A.2 Audio format, A.3 Duration distribution + length "
        "bands, A.4 Signal quality (peak/RMS/crest/SNR/noise floor/silence/"
        "speech/clipping), A.5 Speakers & conversation dynamics (speaker-count "
        "mix; WPM/turns/avg turn/longest monologue/speech coverage; dominant-"
        "speaker share), A.6 Language distribution (+ code-switching note), A.7 "
        "Per-file detail table from FACTS.per_file — INCLUDE a Usability column.\n\n"
        "Rules: match the TEMPLATE's tone and section shape, but every number "
        "must come from FACTS — never copy a figure from the TEMPLATE that is "
        "not in FACTS. Do not shorten the Appendix.\n"
    )
    tmpl = (f"\n\nTEMPLATE (authoritative for STYLE, STRUCTURE and VERBOSITY — "
            f"reproduce this depth; numbers come from FACTS only):\n\"\"\"\n"
            f"{template_text[:80000]}\n\"\"\"\n") if template_text else ""
    return (outline + tmpl +
            "\n\nFACTS (authoritative JSON — the ONLY source of numbers):\n"
            + json.dumps(facts, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", default=str(ROOT.parent / "result"))
    ap.add_argument("--deliveries", default=None)
    ap.add_argument("--include-quarantine", action="store_true")
    ap.add_argument("--title", default="Audio Dataset — Final Delivery Report")
    ap.add_argument("--out", default=None)
    ap.add_argument("--llm", action="store_true",
                    help="generate the narrative report via Claude Opus 4.8")
    ap.add_argument("--model", default=OPENROUTER_MODEL)
    ap.add_argument("--template", default=None,
                    help="style exemplar (.docx/.md); defaults to the bundled "
                         "reference if present")
    ap.add_argument("--max-tokens", type=int, default=32000)
    args = ap.parse_args()

    result = Path(args.result).resolve()
    scan = [Path(args.deliveries).resolve()] if args.deliveries \
        else [result / "deliveries"]
    if args.include_quarantine:
        scan.append(result / "quarantine")

    metas = _load_all(scan)
    if not metas:
        raise SystemExit("no metadata.json found under: "
                         + ", ".join(str(s) for s in scan))

    dedupe = None
    dr = result / "dedupe" / "dedupe_report.json"
    if dr.exists():
        try:
            dedupe = json.loads(dr.read_text())
        except Exception:
            pass

    if args.llm:
        template_text = ""
        tpath = Path(args.template).resolve() if args.template else DEFAULT_TEMPLATE
        if tpath and Path(tpath).exists():
            try:
                template_text = _docx_text(Path(tpath))
                print(f"  using style template: {tpath}")
            except Exception as e:
                print(f"  (template unreadable, ignoring: {e})")
        facts = build_facts(metas, dedupe, args.title)
        print(f"  calling {args.model} via OpenRouter ...")
        report = _openrouter_chat(args.model, _SYSTEM,
                                  _user_prompt(facts, template_text),
                                  args.max_tokens)
        default_out = result / "FINAL_REPORT_llm.md"
    else:
        report = build_report(metas, dedupe, args.title)
        default_out = result / "FINAL_REPORT.md"

    out = Path(args.out).resolve() if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out}  ({len(metas)} files, "
          f"{sum(_nums(m.get('audio', {}).get('duration_ms') for m in metas)) / 3_600_000:.2f} h)")


if __name__ == "__main__":
    main()
