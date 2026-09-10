#!/usr/bin/env python3
"""Assemble gitUpload/ — the shareable, latest-only snapshot of the workflow.

Rules (user decisions 2026-09-02):
  - latest versions only (no v0.1 docs, no old zips)
  - sample = REDACTED deliverables only (no original audio, no raw transcript
    — nothing PII-bearing enters the repo)
  - no keys: .env is never copied; a .env.example with placeholder names is
    generated; users bring their own keys
Idempotent: wipes and rebuilds gitUpload/ on every run.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "gitUpload"

README = """# AudioPII-QC — Audio PII Redaction Workflow (Deccan.ai pilot)

> **Private repository.** Sample outputs are redacted recordings of real
> calls; voices remain recognizable even with PII beeped. Do not mirror or
> re-share outside the team.

Removes spoken personal information (PII) from audio: every detected PII value
is replaced by a 1 kHz beep in the audio and a `<CATEGORY REDACTED>` marker in
the transcript. Handles any language/accent (validated on Indian English,
Hindi, Bengali, Odia, Telugu incl. code-switching), multiple speakers, and
telephone-quality recordings — plus initial validation on Malay/Indonesian
and Swahili calls (2026-09-10: spoken-digit safety net and QC digit tripwire
cover both; see System notes in the report). Full results:
`outputs/Audio_PII_Project_Report_v2.docx`.

## Requirements

- Python >= 3.9. The redaction workflow and evaluation harness are standard
  library only — no pip installs. EXCEPTION: the dataset dedupe pipeline
  (dedupe/) needs `numpy` and `scipy` (`pip install numpy scipy`), or
  Chromaprint's `fpcalc` on PATH for its fastest fingerprint backend.
- An audio converter: macOS ships `afconvert` (nothing to install);
  Linux/Windows need `ffmpeg` on PATH. NOTE: the ffmpeg branch is written to
  produce identical formats but has not yet been exercised on a Linux box —
  run the self-tests below on first use.
- YOUR OWN API keys (see Setup). Keys are personal; never commit them.

## Setup

    cp .env.example .env
    # edit .env and paste YOUR OWN keys (never share or commit this file):
    #   GEMINI_API_KEY        Google AI Studio
    #   ELEVEN_LABS_API_KEY   ElevenLabs
    #   OPENROUTER_API_KEY    OpenRouter (judges / GPT detector / whisper)

## Redact a file (the production workflow)

    python3 tools/redact.py path/to/audio.mp3

One command per file (multiple paths allowed). Stages: prepare -> transcribe
(ElevenLabs scribe_v2 + keyterms; automatic Gemini fallback) -> detect (Gemini,
versioned policy, 31 categories + spoken-digit safety net) -> redact (word-aware
1 kHz beeps + redacted transcript from ONE interval list) -> gates: signal QC,
beep-edge listen check, independent judge audit (all default ON; --no-audit
opts out of the audit).

- PASS -> `deliveries/<file_id>/`: beeped audio (lossless FLAC by default;
  config `beep_format`), redacted .txt, qc_report.md,
  manifest.json (models, prompt hashes, source sha, gate outcomes).
- Any unresolved gate -> `quarantine/<file_id>/` for human review. Nothing
  ships silently.
- Profiles: `--profile strict|medical_consult` (Action Matrix; extensible).
- High-risk content: `--assurance` unions a second listener's detections into
  the beeps.

## Local web app (upload + quarantine review)

    python3 tools/webapp.py          # -> http://127.0.0.1:8788 (localhost only)

Two views: **Upload & Redact** (drop an audio file, the full gated workflow
runs, download the redacted outputs) and **Review quarantine** (per held
file: every flagged beep edge with an audio snippet player, an Extend-beep
button that re-renders the audio, and Approve-&-release / Keep-held with a
mandatory reviewer note — every action is recorded in the manifest).
CLI equivalent: `python3 tools/release.py extend|release|keep ...`.

![Upload & Redact](docs/screenshots/upload_redact.png)
![Review quarantine](docs/screenshots/review_quarantine.png)

Triage tips: cross-script mishearings (the listener writing Devanagari where
the transcript romanized) are the most common false flag; spoken digit words
or value-token hits at an amount boundary deserve real ears — see the report.

## Watermark evaluation samples (before sharing with a buyer)

    python3 tools/watermark.py --make-assets     # once (ElevenLabs TTS, cached
                                                 # in prompts/watermark/)
    python3 tools/watermark.py deliveries/<fid> [...]

Prepends the full spoken evaluation notice, then overlays a short
"Deccan AI evaluation sample" tag at length-scaled intervals — placed
transcript-aware (favours silence gaps, never touches a beep) with the
program ducked 6 dB under the tag. Writes `*_watermarked.flac` + a
`*_watermark.json` sidecar (schedule, levels, source sha); the original
deliverable is never modified. Deterrent + provenance notice, not DRM.
Config: the `watermark` block in pipeline/config.json.

## Evaluation harness (rerunnable on any corpus, no human annotation needed)

    python3 pipeline/run_pipeline.py --engine elevenlabs|gemini|openai \\
                                     [--detector gemini|gpt]      # any config
    python3 tools/judge_precision_recall.py \\
        --configs elevenlabs,gemini,openai,elevenlabs__gpt,openai__gpt
    python3 tools/judge_leak_audit.py       # text-leak audit of redacted outputs
    python3 tools/transcription_bench.py    # WER/CER vs human references (if present)
    python3 tools/cost_report.py            # measured $/audio-hour per config
    python3 tools/fresh_run.py [--dry-run]  # archive + rerun EVERYTHING from scratch

Self-tests (run after install, especially on Linux):

    python3 tools/judge_precision_recall.py --self-test
    python3 tools/judge_pseudo_gt.py --self-test
    python3 tools/judge_leak_audit.py --self-test
    python3 tools/cost_report.py --self-test
    python3 pipeline/detect_pii.py

## Changing policy (no code changes)

Behavior lives in versioned files: the detection policy prompt in `prompts/`
(current version named in `pipeline/policy.py` and prompts/README; shared by
detector AND judges), `outputs/Audio_PII_Taxonomy_v0.2.md`,
`pipeline/policy.py` (profiles / Action Matrix), `prompts/keyterms_india_v1.txt`.
Edit, bump the version, re-run with `--force detect`.

## Repo layout

    pipeline/   the 5-stage pipeline (orchestrator, engines, detector, render,
                QC, edge listen check, judge client)
    tools/      production workflow, evaluation harness, report generators
    prompts/    versioned prompts + keyterms (see prompts/README.md)
    dedupe/     dataset dedupe (exact + acoustic + transcript tiers)
    metadata_qc_pipeline/  per-delivery metadata.json (domain, usability,
                signal + conversation analytics; classifies the REDACTED
                transcript only)
    run_all.py  driver: dedupe -> redact -> metadata over a folder
    generate_final_report.py  dataset report (deterministic, or --llm
                buyer-facing narrative via OpenRouter)
    docs/       screenshots used by this README
    outputs/    deliverable documents (taxonomy, schema, reports, figures)
    results/    fresh-run evaluation reports (markdown + json)
    sample/     one REDACTED example output (no original audio by policy)

## Privacy & security notes

- Deliverables of real calls are still sensitive recordings — handle
  `deliveries/` under the same access rules as the source audio.
- Raw transcripts and detection intervals contain PII and stay in `output/`
  (gitignored); they never enter deliveries or this repo.
- `.env` is gitignored; each user brings their own keys.
"""

ENV_EXAMPLE = """# Copy to .env and fill with YOUR OWN keys. Never commit .env.
GEMINI_API_KEY=
ELEVEN_LABS_API_KEY=
ELEVENLABS_API_KEY=
OPENROUTER_API_KEY=
"""

GITIGNORE = """.env
.DS_Store
__pycache__/
*.pyc
output/
deliveries/
quarantine/
media/
"""


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    # preserve the git history across rebuilds (the folder is a live repo)
    git_keep = None
    if (G / ".git").exists():
        git_keep = ROOT / "_gitupload_git_keep"
        if git_keep.exists():
            shutil.rmtree(git_keep)
        shutil.move(str(G / ".git"), str(git_keep))
    if G.exists():
        shutil.rmtree(G)
    G.mkdir()
    if git_keep is not None:
        shutil.move(str(git_keep), str(G / ".git"))
    (G / "README.md").write_text(README)
    (G / ".env.example").write_text(ENV_EXAMPLE)
    (G / ".gitignore").write_text(GITIGNORE)

    for d in ("pipeline", "tools", "prompts",
              # dataset pipelines (collaborator, 2026-09):
              "dedupe", "metadata_qc_pipeline"):
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_dir() or "__pycache__" in str(f) or f.suffix == ".pyc":
                continue
            cp(f, G / f.relative_to(ROOT))
    for f in ("run_all.py", "generate_final_report.py"):
        cp(ROOT / f, G / f)
    # docs/System_Design.md is the owner's personal design log — NEVER copied
    # into the repo (owner decision 2026-09-03)

    # deliverable docs — latest versions only
    for name in ("Audio_PII_Taxonomy_v0.2.md", "Audio_PII_Taxonomy_v0.2.docx",
                 "Action_Matrix_v1.0.md", "Annotation_Schema_v0.2.md",
                 "Evaluation_Spec_v0.2.md",
                 "Transcription_Benchmark_Report_v3.docx",
                 "Audio_PII_Project_Report_v2.docx"):
        p = ROOT / "outputs" / name
        if p.exists():
            cp(p, G / "outputs" / name)
    tpl = ROOT / "outputs/templates/clip_annotation.template.json"
    if tpl.exists():
        cp(tpl, G / "outputs/templates/clip_annotation.template.json")
    for name in ("workflow_diagram.png", "evaluation_diagram.png",
                 "diagrams.html"):
        cp(ROOT / "output/_report_assets" / name, G / "outputs/figures" / name)

    # fresh-run evaluation results (small md/json only)
    results = [
        "output/judge/precision_recall/precision_recall_report.md",
        "output/judge/precision_recall/precision_recall_report.json",
        "output/judge/pseudo_gt/pseudo_gt_report.md",
        "output/judge/pseudo_gt/pseudo_gt_report.json",
        "output/judge/leak_audit/leak_audit_report.md",
        "output/judge/leak_audit/leak_audit_report.json",
        "output/cost/cost_report.md", "output/cost/cost_report.json",
        "output/bench/transcription_bench.json",
        "output/fresh_run_meta.json",
    ]
    for r in results:
        p = ROOT / r
        if p.exists():
            cp(p, G / "results" / p.name)

    # sample: REDACTED deliverables ONLY (user decision — no original audio,
    # no raw transcript, nothing PII-bearing beyond the redacted outputs)
    fid = "025354f09d2dcc71517beb7553441a85"
    src = ROOT / "deliveries" / fid
    if src.exists():
        for f in sorted(src.iterdir()):
            cp(f, G / "sample" / f.name)
        m = json.loads((src / "manifest.json").read_text())
        ivs = json.loads((ROOT / "output" / (fid + "__elevenlabs")
                          / "intervals.json").read_text())["intervals"]
        def ts(ms):
            return "%d:%05.2f" % (ms // 60000, (ms % 60000) / 1000)
        lines = [
            "# Sample: redacted output of the production workflow",
            "",
            "Source: a 2m29s English-language insurance/hospital telephone",
            "call (8 kHz, mono, two speakers). This folder contains ONLY the",
            "redacted deliverables - by policy the original audio and raw",
            "transcript never enter version control.",
            "",
            "## What the workflow did to this file",
            "",
            "1. Transcribed with ElevenLabs scribe_v2 (+ keyterms vocabulary),",
            "   word-level timestamps, diarization.",
            "2. Gemini applied the versioned detection policy (Taxonomy v0.2)",
            "   to the",
            "   canonicalized word stream and found %d PII intervals:" % len(ivs),
        ]
        for iv in ivs:
            lines.append("   - %s at %s-%s" % ("/".join(iv["categories"]),
                                               ts(iv["start_ms"]),
                                               ts(iv["end_ms"])))
        lines += [
            "3. Rendered word-aware 1 kHz beeps: each beep edge sits at the",
            "   quietest moment between the value and the nearest kept word",
            "   (person names get a 300 ms onset guard), so neighbouring words",
            "   like 'from' between the name and the organization stay",
            "   audible while the values are fully covered. The transcript",
            "   gets <CATEGORY REDACTED> markers from the SAME intervals.",
            "4. Gates (all default): signal QC %s; beep-edge listen check"
            % m.get("qc_status"),
            "   clean (%d rounds, no unresolved or unchecked edges); judge"
            % (m.get("edge_verification") or {}).get("rounds", 0),
            "   audit %s leak candidates. Full provenance in manifest.json"
            % (m.get("judge_audit") or {}).get("leak_candidates"),
            "   (models, prompt hashes, source sha, gate outcomes).",
            "",
            "Listen for: the name at ~0:12 (two separate beeps with 'from'",
            "audible between name and organization) and the amounts at",
            "~1:36-1:45 (neighbouring words 'Okay/right/Yeah/sir' intact).",
        ]
        (G / "sample" / "SAMPLE_README.md").write_text("\n".join(lines) + "\n")

    # screenshots slot (user drops their two captures here before pushing)
    shots = G / "docs" / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    for name in ("upload_redact.png", "review_quarantine.png"):
        src_shot = ROOT / "docs" / "screenshots" / name
        if src_shot.exists():
            cp(src_shot, shots / name)
    if not any(shots.glob("*.png")):
        (shots / "ADD_SCREENSHOTS.txt").write_text(
            "Drop the two web-app captures here as upload_redact.png and\n"
            "review_quarantine.png (referenced from the README), or place\n"
            "them in docs/screenshots/ of the project and rebuild.\n")

    # verify: no key material anywhere in the tree
    import re
    hits = []
    for f in G.rglob("*"):
        if f.is_dir() or f.suffix in (".png", ".docx", ".m4a"):
            continue
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        if re.search(r"sk_[a-f0-9]{16,}|sk-or-v1-[a-f0-9]{16,}|AIza[A-Za-z0-9_\-]{20,}", txt):
            hits.append(str(f))
    if hits:
        print("!! KEY MATERIAL FOUND — aborting:", hits)
        shutil.rmtree(G)
        return 1
    n = sum(1 for f in G.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in G.rglob("*") if f.is_file())
    print(f"gitUpload/ built: {n} files, {size/1e6:.1f} MB, key scan clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
