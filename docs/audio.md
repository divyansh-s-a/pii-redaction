# Audio arm — spoken PII redaction

Part of the PII redaction toolkit; see the [root README](../README.md) for how
this relates to the [multimodal document arm](../multimodal/README.md). Paths in
this document are relative to the repository root.

> **Private repository.** Code only: no data of any kind (audio, transcripts
> or documents, redacted or not) is kept here. Do not mirror or re-share
> outside the team.

Removes spoken personal information (PII) from audio: every detected PII value
is replaced by a 1 kHz beep in the audio and a `<CATEGORY REDACTED>` marker in
the transcript. Handles any language/accent (validated on Indian English,
Hindi, Bengali, Odia, Telugu incl. code-switching), multiple speakers, and
telephone-quality recordings — plus initial validation on Malay/Indonesian
and Swahili calls (spoken-digit safety net and QC digit tripwire
cover both).

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

## Relationship to the document arm

The audio taxonomy is **not shared**. `pipeline/policy.py` — 31 categories, two
actions (`beep`, `annotate_only`) — is imported only by the audio pipeline, and
the document arm has never modified it. It is asserted byte-identical by
`multimodal/synth/regress.py`, which also checks that `judge_call`'s audio code
path is unchanged (`images=None` by default) on every run.

The document arm's `generic_document` pack deliberately **mirrors these labels**
— FULL_NAME, CONTACT_CHANNEL, ADDRESS, DATE_OF_BIRTH, AGE_AND_SEX,
SENSITIVE_DEMOGRAPHIC, CHILD_INFO, GOVERNMENT_ID, ASSIGNED_ID,
PAYMENT_CARD_NUMBER, PASSWORD_CREDENTIAL, DEVICE_VEHICLE_NETWORK_ID,
HEALTH_MEDICAL_INFO, ORGANIZATION_PERSON_LINKED, OTHER — so both arms describe
PII in one vocabulary. It is a mirror, not a dependency: the two files are
edited independently and neither imports the other.

What the document arm adds is what audio explicitly scopes out (faces,
signatures, machine-readable codes, burned-in identifiers, handwriting,
container metadata) and a set of KEEP rows protecting page utility, which a
two-action beep/no-beep matrix has no way to express.

## Which command do I run?

| | `tools/redact.py` | `run_all.py` |
|---|---|---|
| scope | one file (or a few, listed) | a folder |
| transcribe → detect → beep | yes | yes, by calling `redact.py` |
| QC, beep-edge check, judge audit | yes | yes, by calling `redact.py` |
| `--sweep` / `--assurance` | pass them yourself | **on by default** |
| dedupe across the set | impossible — it sees one file | yes |
| dataset summary + per-delivery metadata | no | yes |

Redaction and verification are **identical** either way, because `run_all.py`
stage 2 *is* `tools/redact.py`. The difference is only what surrounds it.

    # one file
    python3 tools/redact.py path/to/audio.mp3 --sweep --assurance

    # a dataset — sweep and assurance already on
    python3 run_all.py --input path/to/folder --result path/to/result

Reach for `run_all.py` whenever you have more than a handful of files: the
dedupe pass means a duplicated call is transcribed and detected once, not
twice, which is the main cost saving in a delivery.

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
overall word accuracy still looks fine. Cross-language evaluation
(English, Malay, Swahili, Hindi, Bengali, Odia) found that the
MAJORITY of missed PII items in every language were absent from the primary
transcript — undetectable by any downstream improvement. Measured effect on
the full corpus: the must-beep leak rate fell substantially with union mode
alone, with the largest gains on the languages that started weakest.
**When to use it:** whenever high recall matters and the audio is not
well-served English — multilingual corpora, code-switched speech, telephony
audio, or any language the pipeline has no proven track record on. For a new
language, treat union mode as the default until per-language evaluation shows
the single-engine path meets your recall bar.

**Cost:** roughly one extra transcription plus one extra detection pass per
file (the second listener can be a lighter model). Expect approximately 1.5-2x
the per-audio-hour cost of the single-engine path — the price of not being
blind to what one engine cannot hear.

**The second listener's detections are placed on the primary's
timeline.** Its word numbering is its own, and its clock is not the
primary's — on long files its timestamps can be seconds off. So before a
union detection is beeped, the two transcripts are aligned
(`pipeline/align_listeners.py`) and the detection is moved to the primary's
words and times; each one is then checked — the position whose primary words
match the detection's own text wins, and when neither wins both stretches are
beeped, so the audio is never covered less than at the second listener's own
times. A union detection with no primary word under it is beeped in the
audio and leaves the transcript as it is. `tools/replay_assurance_union.py`
replays this on cached runs, old placement against new, without model calls.

Notes: works with every other flag (`--mono`, `--langpack`, `--profile`);
union-only intervals are marked `source: "assurance_gemini"` in
intervals.json (with `placed_by` and `retimed_from` recording the placement)
and the manifest records that assurance ran. A re-run strips the previous
run's union-only intervals and re-renders the primary-only baseline before
uniting again.

## Category-sweep detection (`--sweep`)

    python3 tools/redact.py path/to/audio.mp3 --sweep --assurance

Instead of one free-form detection pass over the transcript, `--sweep` runs
four NARROW passes — identity, numbers/IDs, personal facts, health/child —
each carrying the full policy plus a directive to be exhaustive for that
group only. It exists because a single pass lets categories compete for
attention: on Indic consultations the detector filed nearly everything under
HEALTH_MEDICAL_INFO, and under the `medical_consult` profile that category is
annotate-only, so identity values were detected and then never beeped.

**Use `--sweep --assurance` together for delivery work.** They fix different
failures and compose:

| flag | what it adds | what it fixes |
|---|---|---|
| `--assurance` | a second, independent listener; its findings are unioned with the primary's | values the primary transcript never contained. The larger effect on values that must be beeped and were not, because a second listener fails differently from the first |
| `--sweep` | four narrow detection passes instead of one free-form pass | mislabelled values — the detector filing everything under one category, where the action matrix then declines to beep it |
| both | union supplies the content, sweep labels it correctly | the strongest configuration; use it whenever recall matters |

Neither flag can remove a beep the default run would have made — both only add
— so the cost is compute and latency, not extra destruction.

Detection quality varies by language. English, Malay and Swahili are strongest;
Hindi and Bengali are weaker, and Bengali responds to neither mechanism well —
pair those with per-language keyterm packs and expect to review output.
**Cost:** measured ~1.5x the single-pass detection cost on the primary
transcript (the narrow passes emit far fewer output tokens; only the repeated
policy prompt grows). Combined with `--assurance`, budget roughly $3 per
audio-hour against $1.64 for the default path — that combined figure is an
estimate, since the assurance stage's detection calls were unlabelled in the
usage log until now; they are attributed as `detect_assurance:<fid>` from
this change on, so the next run measures it exactly.

**Note on CHILD_INFO:** it stacks additively onto any category and is what
lifts a child's clinical detail out of `medical_consult`'s annotate-only
exemption, so it is permitted in every sweep pass — never restricted to its
own group.

## Named entities: `--mapping` and `--replace pseudonym`

Brands, product and model names, dealers, organisations and the research
agency are not personal data, so the taxonomy never targets them. A project
that must hide them (e.g. a market-research study whose material becomes a
benchmark) supplies a **mapping** of real names to invented ones:

    python3 tools/redact.py group1.mp3 --sweep --assurance \
        --mapping study_mapping.yaml --replace pseudonym \
        --profile-file study_profile.yaml

The mapping uses the same YAML as the project's text pseudonymiser — sections
of `{pattern: case-insensitive regex, replace: invented name}` — with three
optional fields per entry:

    brands:
      - {pattern: "Zorvex(?:\\s+Vision)?", replace: Norvale,
         aliases: [ज़ोरवेक्स, zor vex], category: BRAND}
    agency:
      - {pattern: "Brightpath Research", replace: Insightwell}
    people:            # replace mode: spoken as the invented name
      - {pattern: "\\bRavindra\\b", replace: colleague}
    keep:              # notes; never detected, never changed
      - Pune / Nashik (fieldwork locations)

`aliases` are spoken forms the regex cannot express (Devanagari spellings,
transliterations, common mis-hearings); `tools/expand_mapping.py` generates
them once with a model into `<mapping>.aliases.yaml` beside the mapping.
Replace the **distinctive** name only and leave generic words to be spoken
(`Plumora` → `Tandel`, not `Plumora Bank` → `Tandel Bank`): a mention with or
without "Bank" then reads right, and short replacements fit their slot.

**What `--mapping` turns on** (in either mode):

| stage | added |
|---|---|
| transcribe | the real names lead ElevenLabs' keyterms, so they are spelled consistently |
| detect | one extra model pass over the same word stream listing the entities (any language or script, inflected forms, slogans and other hints), plus a deterministic net: the regexes, aliases, Devanagari→Latin keys and a sound-alike match |
| render | `beep`: mapped mentions are beeped. `pseudonym`: see below |
| audit | the mapped-entity audit — matcher + confirmed judge read over the redacted transcript, and the **listen check**: the delivered audio is transcribed again and scanned for real names |
| package | output file names are passed through the mapping (study folders often carry the brand); the manifest keeps the source hash, not its path |

**Pseudonym mode** replaces a mapped mention with a stock synthetic voice
saying the invented name (ElevenLabs premade voices, or `--tts say` for macOS
voices offline). A speaker's voice is never cloned: their median pitch only
picks between a low and a high stock voice. The clip is trimmed, time-fitted
without changing pitch (widening into neighbouring silence first), matched in
loudness and bandwidth to the speech it replaces, and laid over room tone.
Only **confirmed** mentions are voiced — a regex or alias match, or a model
mention that also resembles the entity; a sound-alike match alone, a slogan
or other hint, an overlap with personal data, or a clip that cannot fit is
**beeped** instead, and the manifest says how many and why. A wrongly voiced
brand would put words in a speaker's mouth; a beep cannot.

The transcript shows the invented name where the audio speaks it and the usual
`<CATEGORY REDACTED>` everywhere else (`<BRAND REDACTED>` for a beeped mapped
mention). QC checks every voiced slot on the output signal: no tone, the
original speech gone (it would correlate with the source), loudness within
4 dB of what it replaced.

Default runs (no `--mapping`, `--replace beep`) are unchanged:
`tools/check_beep_identity.py` re-runs the full workflow offline on synthetic
clips in this checkout and in a worktree of `main` and compares every output.

### Custom profiles (`--profile-file`)

    name: market_research_example
    annotate_only: [AGE]
    detection_supplement: |
      A respondent's own age is analysis data in this project; still mark it
      as AGE. A child's age is CHILD_INFO + AGE as usual.

A profile file names the policy categories a project keeps audible and,
because a category's meaning must change in the prompt too, an optional
supplement for the detector. `pipeline/policy.py` is not edited; the profile
is registered at run time and recorded (name and hash) in every output.

### Keeping work files outside the checkout

Stage caches (`output/<file>/…`: transcripts, intervals, renders) hold
unredacted text. `PII_WORK_DIR=/path` moves them, and `--quarantine-dir`
moves held bundles; `PII_ENV_FILE=/path/.env` reads keys from a file outside
the checkout.

## Processing a whole folder (`run_all.py`)

    python3 run_all.py --input path/to/folder --result path/to/result

Use this for a dataset. It adds the two things a single-file run cannot do,
then calls `tools/redact.py` for each file — so redaction, QC, the beep-edge
check and the judge audit are identical either way:

1. **Dedupe first** (exact + acoustic + transcript tiers) and redact only the
   unique representatives, so a duplicated call is not transcribed and
   detected twice.
2. **Metadata** per delivery (domain, language, signal, usability), computed
   over the REDACTED transcript only.

`--sweep` and `--assurance` are **on by default here**, matching the
recommendation above. They can be turned off for evaluation runs with
`--no-sweep` / `--no-assurance`, and any other redactor flag is passed through
verbatim after `--`:

    python3 run_all.py --input in/ -- --langpack ms --edge-verify

Output:

    result/summary.json                dataset run summary
    result/dedupe/                     report, fingerprints, unique reps
    result/deliveries/<fid>/           beeped audio, redacted transcript,
                                       qc_report.md, manifest.json, metadata.json

The working directory `output/<fid>__<engine>/` holds the evidence for a run —
transcript with word timestamps, detected intervals, what was actually beeped,
the second listener's independent read, and the judge audit. It contains
UNREDACTED material and is gitignored; it is not part of a delivery.

The watermark is a separate, deliberate step (see below) — not every delivery
is shared externally.

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
detector AND judges),
`pipeline/policy.py` (profiles / Action Matrix), `prompts/keyterms_india_v1.txt`.
Edit, bump the version, re-run with `--force detect`.

## Repo layout

    pipeline/   the 5-stage pipeline (orchestrator, engines, detector, render,
                QC, edge listen check, judge client; entity_map,
                mapped_detect, pseudonym_render, profiles for --mapping /
                --replace pseudonym / --profile-file)
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

## Privacy & security notes

- Deliverables of real calls are still sensitive recordings — handle
  `deliveries/` under the same access rules as the source audio.
- Raw transcripts and detection intervals contain PII and stay in `output/`
  (gitignored); they never enter deliveries or this repo.
- `.env` is gitignored; each user brings their own keys.
