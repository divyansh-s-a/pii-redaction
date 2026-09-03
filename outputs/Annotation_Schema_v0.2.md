# Audio PII Annotation Schema v0.2

Companion to `Audio_PII_Taxonomy_v0.2.md` (31 categories) and
`Action_Matrix_v1.0.md`. The taxonomy defines *what counts as PII*; this
document defines *how an interval is recorded* so results are scorable across
the Gemini pipeline, AssemblyAI, ElevenLabs, and any future system. Annotation
is profile-independent: reviewers never consider whether a category will be
beeped in some profile.

Changes from v0.1: `public_figure` field added to interval records;
masked/partial values are now excluded entirely (never annotated); category
set updated to taxonomy v0.2 (SSN removed; USERNAME_HANDLE, ASSIGNED_ID,
ORGANIZATION_PERSON_LINKED added); version strings bumped.

## File layout

One JSON file per media clip, named `<file_id>.gt.json`:

```
annotations/
  dev/       clip_001.gt.json, ...
  holdout/   clip_101.gt.json, ...   <- created before any vendor output is seen, never tuned on
```

Pipeline outputs use the same interval shape in `output/<file_id>/intervals.json`
so GT and predictions are directly comparable.

## Clip-level record

```json
{
  "schema_version": "0.2",
  "taxonomy_version": "0.2",
  "file_id": "clip_001",
  "media_file": "media/incoming/raw/clip_001.mp3",
  "media_sha256": "<hash of the exact file annotated>",
  "duration_ms": 183000,
  "split": "dev",
  "languages_observed": ["en-IN", "hi"],
  "audio_quality": "phone",
  "num_speakers": 2,
  "overlapping_speech": true,
  "pii_present": true,
  "annotations": [ ... ],
  "annotator": "shivangi",
  "annotation_pass": "first",
  "annotated_at": "2026-09-01",
  "clip_notes": ""
}
```

Field rules (unchanged from v0.1): hash the exact file annotated; re-encoded
media must be re-verified. `languages_observed` is breakdown metadata only.
`audio_quality` ∈ studio|clean|phone|noisy|very_noisy. `pii_present: false`
clips are required corpus members (they measure false beeps).
`annotation_pass` ∈ first|self_review.

## Interval record

```json
{
  "annotation_id": "clip_001_03",
  "start_ms": 12400,
  "end_ms": 15100,
  "speaker": "S2",
  "spoken_text": "nine eight seven six five four three two one zero",
  "categories": ["PHONE"],
  "person_link": "explicit",
  "confidence": "high",
  "public_figure": false,
  "needs_review": false,
  "notes": ""
}
```

Field rules:

- `start_ms`/`end_ms` — first and last audible phoneme of the value in the
  original media; no lead-ins, no self-added pad (pad is a renderer setting).
- `speaker` — `S1`, `S2`, … in order of first appearance, or `unknown`. With
  overlapping speech, attribute the interval to the speaker who UTTERED the
  value. On mixed audio (mono/dual-mono — voices summed into one track) the
  beep unavoidably mutes ALL voices in that window; the speaker field is for
  attribution and analysis, not selective muting. On split-channel recordings
  (one speaker per channel) the pipeline beeps only that speaker's channel.
- `channel` — optional int. Only for split-channel media: the source channel
  (0, 1, …) the value was spoken on; null/omitted for mixed audio. Pipeline
  outputs set it automatically from per-channel transcription.
- `spoken_text` — verbatim words as spoken; never normalized to digits/symbols.
- `categories` — one or more codes from the **31-label taxonomy v0.2**.
  Multi-label goes on ONE interval (e.g. `["CHILD_INFO","FULL_NAME"]`).
- `person_link` — explicit | contextual | unclear (unclear forces needs_review).
- `confidence` — high | medium | low (low forces needs_review).
- `public_figure` — optional, default false. `true` only on `FULL_NAME`
  intervals per the Action Matrix rule; forces nothing but is always audited.
- `needs_review` — forced true for: low confidence, unclear person_link, every
  `OTHER`, lone first names, unintelligible surroundings.
- **Do NOT annotate masked/partial values** ("ending 4832") — excluded by
  policy D5. A complete value spoken with pauses/fillers is complete.
- Repetitions are separate records; an echo by another speaker is a new record.

## Annotation procedure (single-reviewer protocol)

Unchanged from v0.1: (1) full listen; (2) annotate with ms-precision player;
(3) never view model output before/during GT annotation; (4) metadata last;
(5) `self_review` pass ≥24 h later over all `needs_review`, all `OTHER`, and a
random 20% of clear intervals; (6) unresolved policy calls go to the policy
owner — until adjudicated this corpus is "reviewed ground truth (single
annotator)", not a gold set.

## Validation

Run `tools/validate_annotations.py` before a file enters the corpus. v0.2
enforces: schema fields, interval sanity, **taxonomy v0.2 codes** (`SSN` is
rejected with a pointer to `GOVERNMENT_ID`), forced needs_review rules, notes
on `OTHER` (not required for `ASSIGNED_ID`), unique ids, and warns when
`public_figure` is set on a non-FULL_NAME interval.
