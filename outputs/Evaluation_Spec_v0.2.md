# Evaluation Spec v0.2 — scoring engines on spoken-PII beeping

Defines how every engine is scored **before** its output is seen, so the scoring cannot drift toward any system. Implemented by `tools/score_run.py`; this document and that tool must stay in agreement.

Changes from v0.1: comparison restructured around swappable **transcription** engines (Gemini / ElevenLabs / OpenAI-Whisper) sharing the same detection + rendering, with vendor-native detection as optional later scenarios; must-beep set defined per profile via Action Matrix v1.0; coverage threshold is now >= 95% of the GT interval (was 1.0 with a 50 ms tolerance); public-figure and annotate-only intervals explicitly excluded from the leak denominator and reported separately; detection-level recall/precision added (padding-independent); engine quality metrics added (span transcription WER, language ID, speaker count); `tools/score_run.py` now exists (was "to be written").

## Comparison scenarios

Primary scenario — **stage-B swap**. Each engine runs transcription only; everything downstream is byte-identical:

```
original media -> ENGINE transcription (stage B)        <- only this differs
               -> SHARED stage C detection (LLM policy pass + safety net)
               -> SHARED stage D renderer (same pad, same tone, same code)
               -> beeped media + rendered_intervals.json
               -> tools/score_run.py vs the same GT
```

Engines under comparison: **Gemini**, **ElevenLabs**, **OpenAI-Whisper**. Each engine writes `output/<file_id><engine_suffix>/` in the standard artifact shapes (`transcript.json`, `intervals.json`, `rendered_intervals.json`), so the scorer needs only `--engine-suffix`. This isolates transcription quality: any leak difference is attributable to stage B.

Optional later scenarios — **vendor-native detection**: a vendor's own PII entity detection replaces stage C, with its categories mapped to the 31-label taxonomy through a fixed mapping table committed before the first run (`outputs/vendor_category_map.md`, to be written when API configs are set). Rendering and scoring stay shared. A vendor's own beeped-audio export is never scored directly — that would compare renderers, not models.

## The must-beep set (per profile)

Scoring is profile-relative. A GT annotation is **must-beep** under a profile iff `action_for(categories, public_figure, profile) == "beep"` in `pipeline/policy.py` — the same Action Matrix v1.0 call the renderer uses. Concretely:

- `public_figure: true` intervals are `annotate_only` in **every** profile (decision D1) — excluded from the leak denominator, reported separately (including how many were beeped anyway, which is a policy violation).
- An interval is `annotate_only` when **all** its categories are annotate-only under the profile (`medical_consult`: `HEALTH_MEDICAL_INFO`); one must-beep category on a multi-label interval makes the whole interval must-beep. Excluded from the leak denominator, reported separately.
- Everything else is must-beep.

## Matching predictions to ground truth

The leak metric scores the **rendered beep intervals after padding and merging** (`rendered_intervals.json`, `rendered_merged`) — i.e., what a listener would actually hear — against GT intervals.

For each must-beep GT interval `g`:

- `coverage(g)` = |g ∩ union(channel-compatible rendered beep intervals)| / |g|. Channel rule: a rendered interval with `channel: null` covers every channel; channel `k` covers a GT interval on channel `k`, or one with no channel recorded.
- `g` is **covered** if `coverage(g) >= 0.95`.
- `g` is a **leak** otherwise. Partial coverage is still a leak — a phone number with 3 digits audible is exposed. (v0.1 required coverage 1.0 with 50 ms tolerance; 95% of a typical multi-second value interval is a comparable allowance, defined without a second constant.)

**Leak rate** = leaks / must-beep GT intervals; also reported per audio-hour using GT `duration_ms`.

Detection quality is measured separately, on `intervals.json` **before** padding/snap/merge: GT annotations are matched one-to-one to predicted intervals, greedily by temporal IoU; a pair is eligible when IoU >= 0.3 OR the prediction covers >= 70% of the GT interval. This separates "found the value" from "beeped enough of it".

A rendered segment that overlaps no GT interval (GT padded by 250 ms tolerance) is a **false beep event**; all rendered time outside padded GT counts toward over-redaction.

## Metrics

Primary (safety):

| Metric | Definition |
|---|---|
| **Leak rate** | leaked / must-beep GT intervals (the headline number; lower is better) |
| Leaks per audio-hour | leaked / GT audio-hours |
| Worst-category leak rate | max per-category leak rate among categories with >= 5 must-beep GT intervals |

Secondary (detection + usability):

| Metric | Definition |
|---|---|
| Detection recall | matched GT / all GT, reported separately for must-beep and annotate-only |
| Detection precision | matched predictions / all predictions |
| Category accuracy | among matched pairs, predicted categories ∩ GT categories non-empty |
| Over-redaction | beeped seconds per audio-hour outside padded GT |
| False beep events | rendered segments touching no padded GT, count per audio-hour |
| Timing tightness | mean \|start delta\| and \|end delta\| over matched pairs, ms |
| Cost & latency | $ per audio-hour at list price; wall-clock turnaround per clip |

## Engine quality metrics

Beyond leak/detection, three dimensions compare the engines as *transcribers* directly against reviewer ground truth. All are computed by `tools/score_run.py` from `transcript.json` + GT; none affect the decision rule below (they explain leak differences, they don't override them).

**Span transcription accuracy.** For each GT interval, WER between GT `spoken_text` and the engine's words falling inside the GT interval (±150 ms slack for timestamp drift). Both sides are normalized first: lowercase, punctuation stripped, numbers canonicalized — number-words become digits (an English number-phrase parser plus single-digit Hindi and Bengali word maps, Latin and Devanagari script), and adjacent digit tokens merge into one token — so "ninety-eight seven six", "98 7 6" and "9876" compare equal regardless of each engine's formatting convention. Reported: mean span WER, **% exact spans** (WER = 0), and a **span not transcribed** tally (no engine words in the window — the ASR-miss signal, counted separately rather than as WER 1.0). Caveat: GT `spoken_text` is seeded from Gemini's transcript wording and reviewer-corrected by ear, so Gemini's span WER is anchored low; treat cross-engine deltas, not absolute values, as the signal (same anchoring caveat as detection recall).

**Language ID accuracy.** Engine `primary_language` and `languages[]` (from `transcript.json`) vs the reviewer's clip-level `languages_observed`. Codes are normalized before comparison (`en`/`eng`/`english`/`en-IN` → `en`, etc.). Reported: **primary-language match rate** (engine primary ∈ reviewer set) and **language-set Jaccard** (|engine set ∩ reviewer set| / |union|) per clip and aggregate.

**Speaker/channel consistency.** Engine distinct-speaker count vs reviewer `num_speakers`. Reported: **match rate** (exact count agreement) and **MAE**, with the file's `channel_analysis` verdict (from `prepare.json`) shown for context — split-channel files get diarization partly for free. Structural caveat: OpenAI Whisper does not diarize mixed audio (always reports 1 speaker), so this dimension inherently favors diarizing engines (Gemini, ElevenLabs); it is reported as a capability difference, not a tunable error.

## Error attribution

Every leak is classified into exactly one stage, by inspecting the engine transcript and intervals (manual step; the scorer's per-leak table is the worklist):

1. `asr_miss` — value absent/garbled in transcript (words not recoverable)
2. `detection_miss` — value present in transcript, not flagged as PII
3. `category_error` — flagged, but the category is out of the beep policy so it wasn't beeped
4. `timing_error` — flagged, but the interval didn't cover the audio (late start / early end)
5. `render_error` — correct interval in, beep wrong out (shared-renderer bug; fix and re-run every engine)

Every false beep is classified as: `asr_hallucination`, `overdetection` (non-PII flagged), or `overspan` (correct hit, excessive interval).

## Breakdowns

Automated by the scorer: PII category, clip language (`languages_observed`), GT confidence. Manual/on-demand: audio quality, spoken form (read digits vs "double/triple" vs spelled vs natural speech), single vs multi speaker.

## Decision rule

On the frozen holdout, run each configured scenario once. Winner:

1. Lowest **leak rate**. A difference of fewer than 2 leaks total on the holdout is a tie — move to rule 2.
2. If tied: lower over-redaction time.
3. If still tied: cost, then latency, then operational fit (language coverage roadmap, data-processing terms).

No configuration changes after the holdout run. If every leak rate is unacceptable for the product, the outcome is "none alone is sufficient" — the next iteration (ensembling, secondary digit/pattern pass, human review queue) is designed on the dev set only.

## Reporting

`tools/score_run.py` emits a markdown report (per-file table + aggregate + breakdowns) and a machine-readable `score.json` next to it. Every leak gets a row: file_id, annotation_id, category, interval, coverage, spoken text — this table, not the aggregate score, is the main artifact for the review meeting. The scorer always exits 0: scores are information, not gates.

## Addendum (2026-09-01): judge-based interim evaluation and gate

With human GT deferred, an independent LLM judge (gpt-5.6-sol via OpenRouter,
applying detection policy v3) provides interim evaluation: (a) leak audit of
redacted transcripts (findings must literally quote surviving text; leak vs
kept-by-profile computed in code from each file's recorded render profile);
(b) cross-engine pseudo-GT (judge detections on each engine's raw transcript,
time-anchored, unioned; upper-bound caveat applies). An adversarial verifier
(gpt-5.6-luna) must cite the include-rule per finding or reject it (fail-open).
Vendor-native detection scenario (ElevenLabs entity_detection) was run and
CLOSED: union coverage 35.4%, precision 3.8%, fiction flooded — see
System_Design. Engine decision gate (all met 2026-09-01 for the EL+Gemini
hybrid, two residual listen-and-confirm items logged): judge-detector
agreement >= 80%, zero text-leak candidates, QC all green, unheard-leak
candidates resolved. Human GT remains the final arbiter and additionally
scores the judge itself.


## Addendum (2026-09-02): detection precision/recall of record — human GT dropped

The human GT pass was dropped (reviewer time). The evaluation of record is
`tools/judge_precision_recall.py`; all numbers in Report v2 come from it on
the fresh 2026-09-02 full re-run.

**Reference per file:** the judge (GPT-5.6-sol, same policy v3 file as the
detector) detects on BOTH listener transcripts (ElevenLabs + Gemini); every
span passes the adversarial rule-citing verifier (GPT-5.6-luna: name the
include rule or reject); verified spans cluster into unique items (temporal
IoU >= 0.3 / 70% cover, OR same canonicalized value within +/-2.5 s —
Indic-folded, number-canonicalized). Constituent spans are kept per item.

**Metrics (per configuration; per file, per category, micro-averaged):**
- DETECTION RECALL — reference items matched by the config's detected
  intervals / all items (match = the temporal cluster rule OR same value
  within +/-2.5 s).
- DETECTION PRECISION — detected intervals matching a reference item / all
  detected intervals (same match rule, same reference).
- RENDER LEAK RATE — must-beep items < 95% covered by the rendered beeps /
  all must-beep items. Coverage is OCCURRENCE-BASED (duration-weighted over
  an item's constituent spans, never its hull) and SKEW-AWARE (a span counts
  covered when a different listener's span of the same occurrence is covered
  on the render timeline). Measures the render; production gates decide what
  ships.
- POLICY-VERIFIED PRECISION (supplementary) — intervals the rule-citing
  verifier confirms against a named include rule / all intervals. An LLM
  judgment with observed run-to-run variance; reported alongside detection
  precision, never instead of it.
- TIMESTAMP AGREEMENT — start-offset between the two listeners' spans of the
  same value (median/p90 per file); relative clock skew, not absolute timing
  (that is verified by the beep-edge listen check and human spot-listens).

**Scope:** the reference is a shared benchmark for comparing configurations
fairly; values neither listener transcribed sit outside it, and both
listeners can mis-hear the same proper name (observed) — human spot-checks
complement these numbers. Configurations using a GPT detector share a model
family with the judge; the cross-family read (Gemini detector, GPT judge) is
the cleaner comparison.
