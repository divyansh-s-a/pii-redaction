# Prompts used by the pipeline (versioned)

The code loads these files at import time — editing a prompt here changes the
pipeline (bump the filename version and pipeline reference together; record
the change in docs/System_Design.md).

| File | Used by | Purpose |
|---|---|---|
| `transcription_prompt_v1.txt` | Stage B, Gemini engine only | Transcription instructions: diarization (S1/S2…), keep original language/script, word-level start/end ms, overlap marking, fillers-are-words |
| `transcription_response_schema_v1.json` | Stage B, Gemini engine | Strict JSON output schema (utterances → words with times) enforced via responseSchema |
| `detection_policy_prompt_v1.txt` | Stage C, ALL engines' transcripts | The Taxonomy v0.2 PII detection policy: 31 categories, global rules (masked-value exclusion, fiction exclusion, public figures, repetition rule, spoken-form handling) |
| `detection_response_schema_v1.json` | Stage C | Strict JSON output schema (spans with word indices, categories, confidence, public_figure) |
| `judge_leak_audit_prompt_v1.txt` | `tools/judge_leak_audit.py` (Phase A judge evaluation) | Independent leak audit of REDACTED transcripts: find PII that survived redaction; embeds the v2 detection policy; profile-dependent findings (e.g. HEALTH under medical_consult) flagged separately. The tool appends the numbered redacted transcript and enforces a strict findings JSON schema at call time |

Notes:
- Chunked transcription appends a continuation suffix at call time ("This audio
  CONTINUES a longer recording… keep the same speaker labels") — see
  `pipeline/gemini_transcribe.py::transcribe_bytes`.
- ElevenLabs and OpenAI-Whisper are parameter-driven APIs, not prompted:
  their "prompt equivalents" are request parameters (`diarize=true`,
  `timestamps_granularity=word` for ElevenLabs; `response_format=verbose_json`,
  `timestamp_granularities[]=word` for Whisper) — see `pipeline/adapters/`.
- Stage C's detector is pluggable since 2026-09-02 (`--detector gemini|gpt` on
  run_pipeline.py): both backends read the SAME policy file and word-index
  schema — gemini via responseSchema, gpt (openai/gpt-5.6-luna via OpenRouter)
  via a strict-mode mirror of it in `pipeline/detect_pii.py`. Whatever the
  transcription engine, one detector model is held constant within a
  comparison so results isolate the intended variable.

**v2 (2026-09-01)**: `detection_policy_prompt_v2.txt` — fiction rule extended to
narrated audio dramas (episode framing, narrator voice, outros); real production
credits still count. Trigger: MP EP - 1 case study (see System_Design.md).

**v3 (2026-09-01)**: `detection_policy_prompt_v3.txt` — D8 clarification
(person's insurer/TPA/lender is ORGANIZATION_PERSON_LINKED), multilingual
financial values, and negative-calibration rule 11 harvested from 20 observed
judge over-reaches. Shared by the stage-C detector and both judge tools.

| `keyterms_india_v1.txt` | Stage B, ElevenLabs engine | Recognition-biasing vocabulary (generic Indian given names + major insurers/TPAs/hospitals) sent as repeated `keyterms` multipart fields; fixes dropped proper nouns. Not corpus-derived (see System_Design). |

**v4 (2026-09-03)**: `detection_policy_prompt_v4.txt` — fiction rule redefined
by REALITY OF REFERENTS, not format: true-crime/documentary/testimonial
narratives are non-fiction and their personal details are PII; the fiction
exclusion needs CLEAR fiction signals; uncertain realness defaults to REAL
(recall-first). Trigger: user question + synthetic genre experiment showing
an ambiguous crime narrative was treated as fiction (only the digit safety
net caught the phone number). Shared by detector and judges.
