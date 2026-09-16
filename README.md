# AudioPII-QC — Audio PII Redaction Workflow (Deccan.ai pilot)

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
- High-risk content / non-English audio: use union mode (next section).

## Union mode (`--assurance`): high recall across languages

    python3 tools/redact.py path/to/audio.mp3 --assurance

Union mode transcribes the audio a SECOND time with an independent engine
(Gemini), runs the same detection policy over that transcript too, and adds
every union-only detection to the beeps. Everything else — gates, quarantine,
manifests — is unchanged.

**Why it exists:** a single-engine pipeline can only redact what its own
transcript contains, and on many languages the transcript silently drops
exactly the tokens PII lives in (ages, spoken digit strings, names) while
overall word accuracy still looks fine. The 2026-09 cross-language evaluation
(134 files: English, Malay, Swahili, Hindi, Bengali, Odia) found that the
MAJORITY of missed PII items in every language were absent from the primary
transcript — undetectable by any downstream improvement. Measured effect on
the full 134-file corpus: overall must-beep leak rate fell from 35.9% to
11.3% with union mode alone (English 13%->3.7%, Malay 28.6%->9.5%, Odia
53.5%->4.7%), and detection recall rose to 87-96% for English/Malay/Swahili.
On Indic languages union fixes coverage but single-pass detection still
caps recall (~66-68%) — pair it with per-language packs there.

**When to use it:** whenever high recall matters and the audio is not
well-served English — multilingual corpora, code-switched speech, telephony
audio, or any language the pipeline has no proven track record on. For a new
language, treat union mode as the default until per-language evaluation shows
the single-engine path meets your recall bar.

**Cost:** roughly one extra transcription plus one extra detection pass per
file (the second listener can be a lighter model). Expect approximately 1.5-2x
the per-audio-hour cost of the single-engine path — the price of not being
blind to what one engine cannot hear.

Notes: works with every other flag (`--mono`, `--langpack`, `--profile`);
union-only intervals are marked `source: "assurance_gemini"` in
intervals.json and the manifest records that assurance ran.

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

    python3 pipeline/run_pipeline.py --engine elevenlabs|gemini|openai \
                                     [--detector gemini|gpt]      # any config
    python3 tools/judge_precision_recall.py \
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
