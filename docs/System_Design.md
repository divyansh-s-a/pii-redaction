# Audio PII Redaction Pipeline — System Design

Living document. Updated with every pipeline change so a final report or
presentation can be assembled directly from it.
Last updated: 2026-09-01.

## 1. Problem

Given user-uploaded audio (or the audio track of video) in any language or
accent — including Indian languages and Hindi/English code-switching, multiple
speakers, and telephone-quality recordings — produce:

1. a redacted audio file where every spoken PII value is replaced by a 1 kHz beep,
2. a redacted transcript in the incumbent team's format
   (`[HH:MM:SS.ms - HH:MM:SS.ms] speaker - text` with `<CATEGORY REDACTED>` markers),

under a versioned policy (Taxonomy v0.2, Action Matrix v1.0) whose category
actions are domain-configurable (profiles).

## 2. Architecture

```
media file (.mp3/.mp4 …)
   │
   ▼
A prepare      hash • 16 kHz mono work WAV (afconvert; original untouched)
   │           stereo: channel_analysis -> split-channel mode (per-channel WAVs)
   │           chunk plan for files > 8 min
   ▼
B transcribe   pluggable engine → transcript.json
   │           gemini-3.1-pro-preview | elevenlabs scribe_v2 (+keyterms
   │           vocabulary) | whisper-1; recommended: elevenlabs
   │           utterances{speaker, lang, channel, overlaps_with, words[{w,start_ms,end_ms,conf}]}
   │           + primary_language and languages[] (word-share weighted)
   │           split-channel files: each channel transcribed separately
   ▼
C detect       canonicalized word stream → Gemini policy pass (prompt v3)
   │           + deterministic safety net (Indic-aware)
   │           → intervals.json  ←—— SINGLE SOURCE OF TRUTH
   ▼
D redact       action per interval from Action Matrix profile
   │           audio: edge-snap → pad 250 ms → merge (per channel) → mute+tone → m4a
   │           split-channel: beep only the offending speaker's channel
   │           text:  same intervals → <CATEGORY REDACTED> transcript
   ▼
E qc           signal-level verification of the rendered file (Goertzel):
               every planned beep is pure tone ≥90% (on its target channel);
               no orphan tones; other channel untouched (split mode);
               speakers-vs-channels sanity (split files: one speaker per
               channel); marker/beep reconciliation; digit-near-edge leak
               heuristic; language report (primary + all, word-share)
               → qc_report.md (PASS / FLAGGED — nothing passes silently)
```

Artifacts per file in `output/<file_id>/`: `prepare.json`, `work_16k.wav`,
`transcript.json`, `intervals.json`, `<id>_beeped.m4a`, `<id>_redacted.txt`,
`rendered_intervals.json`, `qc_report.md`. Every stage is cached and
individually re-runnable (`--force <stage>`), so a policy change re-runs
detection without paying for transcription again.

## 3. Design decisions and reasoning

### D-arch-1: transcription and detection are separate model calls
One call ("transcribe and find PII") couples failures and makes policy changes
expensive. Separated: stage B output is cached; taxonomy/prompt iterations only
re-run stage C (cheap, text-only); the detector is swappable (Gemini today,
Claude or a vendor API tomorrow) without touching transcription.

### D-arch-2: one intervals.json drives BOTH audio and transcript
In the incumbent's ODI_005 file we found the transcript redacting one utterance
while the audio beeped a different one — audio and text were redacted
independently and disagreed, leaking one spoken instance. Structurally
impossible here: both renderers consume the same intervals.

### D-arch-3: LLM timestamps are treated as untrusted
LLM word timestamps come from model judgment, not a forced-alignment head; they
drift on long audio and can hallucinate. Mitigations, in order:
1. **Chunking** — files > 480 s are transcribed in 360 s chunks stepped every
   330 s (30 s overlap), so drift cannot accumulate; overlap-zone duplicate
   words are dropped at stitch time; chunk N+1 receives the tail of chunk N as
   context to keep speaker labels stable.
2. **Edge snapping** — each beep edge moves to the quietest 20 ms window within
   ±150 ms (never shrinking the interval), so beeps start/end in speech gaps
   rather than mid-phoneme.
3. **Safety pad** — 250 ms each side after snapping (renderer setting, not GT).
4. **QC gate (stage E)** — the rendered file is signal-verified: the same
   Goertzel pure-tone analysis that exposed the incumbent's defect confirms
   every planned interval is ≥90% pure tone and no unplanned tones exist.
Residual risk: a *detection-time* timestamp so wrong the beep lands on the
wrong words entirely; mitigated by the leak heuristic (digit words adjacent to
beep edges are flagged) and, later, human review of flagged files.
Upgrade path: local forced alignment (e.g. MMS/wav2vec2 aligner) between B and C.

### D-arch-4: deterministic safety net under the LLM detector
Recall-first: runs of ≥7 spoken digits (English words, Hindi words in Latin or
Devanagari, Bengali, digit tokens, "double/triple" multipliers, fillers inside
the run tolerated) that the LLM did not mark become intervals anyway —
category guessed from length (10–11 → PHONE, 12 → GOVERNMENT_ID/Aadhaar-shaped,
15–16 → PAYMENT_CARD_NUMBER, else OTHER), confidence low, `needs_review: true`,
`source: "safety_net"`. A missed phone number is worse than an extra beep.

### D-arch-4a: safety-net language coverage extended (2026-09-03)
User question ("what if the language is outside DIGIT_WORDS?") led to a
coverage review. Findings: numerals of ANY script were already counted
(Python's Unicode isdigit) — only digit WORDS were allowlist-bound; and on
the fresh corpus run the net contributed 0 of 358 intervals (the LLM detector
caught every digit sequence itself — the net is genuine belt-and-braces).
Extension: DIGIT_WORDS now covers Tamil, Malayalam, Kannada, Odia, Marathi,
Gujarati and Punjabi (native + romanized) alongside the original
English/Hindi/Bengali/Telugu; one combined 9-script numeral table replaces
the three per-script ones; Devanagari double/triple multipliers and
multi-language yes-fillers added. Deliberate exclusion, documented in code:
romanized Gujarati "be" (2) — a ubiquitous English word that would corrupt
digit runs in code-switched speech. Inline self-test asserts one word and one
numeral per new language plus the "be" exclusion.

### D-arch-5: overlapping speakers — the mono-mix limitation, and the split-channel mode

**Limitation (mixed audio).** When all voices share one mixed track (mono, or
stereo whose channels carry the same mix — true of the current corpus), the
voices are summed into the same samples and cannot be separated afterwards. A
beep therefore mutes ALL voices in its window: if speaker B talks over speaker
A's phone number, B's words in that second are lost too. This is the safe
failure mode — the alternative (trying to preserve B) risks audible PII. AI
source separation could in principle split voices, but imperfect separation can
leave PII residue in the "kept" audio, which is unacceptable for redaction, so
it is out of scope. The `speaker` field attributes the disclosure for analysis
and per-speaker leak statistics, not for selective muting.

**Split-channel mode (implemented 2026-09-01).** Some sources record one
speaker per channel (e.g. call systems: agent left, customer right). The
pipeline detects and exploits this automatically:

1. *prepare* runs `channel_analysis` on stereo files: zero-lag correlation of
   L/R plus per-channel energy ratio. `split = corr < 0.5 AND energy_ratio >
   0.15` (a near-identical dual-mono mix has corr ≈ 1; distinct speakers have
   corr ≈ 0). Result is recorded in `prepare.json`.
2. When split, each channel is transcribed SEPARATELY (better diarization for
   free); utterances carry `channel: 0|1` and speakers are namespaced
   `C0_S1`, `C1_S1`, …
3. Intervals inherit the channel of their words; edge-snapping runs against
   that channel's work WAV.
4. The renderer beeps ONLY the offending channel — the other speaker's audio
   in that window is preserved bit-for-bit. Per-channel intervals never merge
   across channels.
5. QC verifies per channel: the target channel is ≥90% pure tone AND the other
   channel is NOT beeped (coverage > 0.5 there is flagged as
   "other speaker's audio lost"); orphan-tone scan runs per channel.

Verified by self-test: synthetic two-voice stereo → split=True detected
(corr −0.001), dual-mono → split=False (corr 1.0); channel-targeted beep gave
tone coverage 1.00 on the target channel, 0.00 on the other, with the other
channel's samples unchanged bit-for-bit. The current corpus has no split files
(all mono or dual-mono), so this mode awaits real two-channel material.

### D-arch-6: macOS-native rendering (no ffmpeg dependency) — AMENDED 2026-09-02
Update (user requirement: support must not be Mac-restricted): audio decode/
encode now auto-selects afconvert (macOS) or ffmpeg (Linux/Windows) in
audio_utils._converter(); everything else was already pure Python/stdlib.
The afconvert path is what the whole pilot was validated on; the ffmpeg
branch is written to produce the same 16-bit PCM WAV and AAC m4a but has NOT
been executed on a Linux box yet — first task for whoever deploys off-Mac is
to run the tool self-tests there. Original decision below.
This machine has no ffmpeg/Homebrew. Rendering: afconvert decodes the ORIGINAL
file at native rate/channels → pure-Python PCM mixing replaces each interval
with a −10 dB 1 kHz sine (matching the incumbent's replace-not-overlay style,
verified by signal analysis of their files) → afconvert encodes m4a/AAC.
`read_wav` parses RIFF manually because afconvert emits WAVE_FORMAT_EXTENSIBLE
headers Python's `wave` module rejects. ffmpeg path (tools/beep_renderer.py)
remains for video remux later.

### D-arch-7: model choice
`gemini-3.1-pro-preview` (config-driven; resolve_model() lists the key's models
and falls back visibly, never silently). Observed on the 2:29 telephone file:
3.1-pro preserved spoken number words verbatim ("thirty thousand six hundred
and thirty five rupees") where 2.5-pro normalized to digits, and caught the
person-linked organization 2.5-pro missed. Candidate to benchmark:
`gemini-3.5-transcribe` (dedicated ASR model) for stage B word timing.

### D-arch-8a: language reporting
Each utterance carries the model's language tag; `language_summary` aggregates
them (weighted by word count) into `primary_language` plus a full `languages`
list with word-shares, stored in `transcript.json`, echoed in every QC report,
and rolled up in `corpus_summary.json`. QC patches older cached transcripts
that predate the field, so re-transcription is never needed for it.

### D-arch-8b: speakers-vs-channels QC
For split-channel files, each channel should carry exactly one speaker. QC
flags (a) total distinct speakers != channel count and (b) any single channel
whose transcription reports more than one speaker (speakerphone, diarization
noise, or wrong split verdict). On mixed audio the speaker count is reported
as a note only — there is no expected relationship to the single channel.

### D-arch-9: preprocessing policy — deliberately minimal
What the pipeline DOES before transcription: decode a 16 kHz mono work WAV
(analysis copy only — Gemini receives the ORIGINAL file for best quality),
channel analysis + per-channel splits, and chunking of long files. What it
deliberately does NOT do: denoising, loudness normalization, silence trimming,
filler removal, or any text normalization ahead of detection. Rationale:
(1) the transcription model is more robust to raw audio than to artifacts
introduced by cleanup; (2) the Mindy pilot measured preprocessing helping only
its weakest detector (Presidio) while its own docs flag normalization as
edge-case-risky ("email to vishal at google dot com"); (3) every destructive
transform distorts the timeline or the wording that ground truth must match.
The safety net normalizes COPIES of words for pattern-matching only — original
timestamps are always used for beeps. If error analysis later shows repeated
misses on a specific audio class (e.g. very noisy telephone), targeted
preprocessing gets added as an A/B-tested option, never as a silent default.

### D-arch-8: REST + stdlib only
All API calls via urllib against the Files API (resumable upload, poll to
ACTIVE) and generateContent with a strict responseSchema (JSON mode,
temperature 0). No pip dependencies — the pipeline runs on a stock macOS
Python 3.9.

## 4. Policy stack (inputs to stage C/D)

- `outputs/Audio_PII_Taxonomy_v0.2.md` — 31 jurisdiction-neutral categories;
  routing table for country-specific IDs (Aadhaar→GOVERNMENT_ID, PAN→TAX_ID,
  UPI→FINANCIAL_ACCOUNT_INFO, patient/NHS numbers→ASSIGNED_ID, IFSC alone
  excluded); audio/India edge cases.
- `outputs/Action_Matrix_v1.0.md` — profiles: `strict` (all beeped),
  `medical_consult` (= strict minus HEALTH_MEDICAL_INFO, which is annotate-only
  — clinical content is the recording's purpose; matches incumbent behavior).
  Public figures: annotated `public_figure: true`, never beeped, always audited.
  Masked/partial values: excluded entirely.
- `pipeline/policy.py` — the same 31 codes + matrix, machine-readable; the
  detection prompt embeds the taxonomy. Consistency across the four sources is
  verified by script (31 == 31 == 31 == 31 as of 2026-09-01).

## 5. Verification status (2026-09-01)

| Check | Result |
|---|---|
| Renderer self-test (synthetic WAV, tone coverage in/out, m4a roundtrip) | PASS (coverage 1.00 / 0.00) |
| pad/merge, edge-snap never-shrink | PASS |
| Validator: retired SSN rejected, new categories accepted, public_figure rules | PASS |
| 31-code consistency (taxonomy / validator / policy.py / matrix) | PASS |
| End-to-end dry run, 2:29 telephone file, gemini-3.1-pro-preview | PASS — 5 intervals (name, org, 3× amount), QC 0 problems |
| Split-channel self-test (detection, targeted beep, other channel untouched, per-channel merge) | PASS |
| Mono regression after channel changes (telephone file re-render + QC) | PASS |
| Full corpus run (8 raw files) | paused mid-run by user; resumable (stages cached) |

### D-eval-1: correction-based GT (2026-09-01) — DROPPED 2026-09-02
Original decision: ground truth would be created by reviewing/correcting the
Gemini pipeline's output (a self-contained review.html per file), not by
from-scratch annotation — a time-saving trade-off with a recorded anchoring
risk (Gemini's misses stay missed, inflating its measured recall).
2026-09-02, user decision: the human GT pass is dropped for lack of reviewer
time. The review-page generator (tools/make_review_page.py) and the generated
review pages were removed; tools/validate_annotations.py stays as the
Annotation Schema v0.2 reference validator, and tools/score_run.py stays both
as a shared library (the judge tools import its normalizers and coverage
math) and as the ready-made scorer should GT ever be produced. Consequence:
the judge-based evaluation (D-eval-3, tools/judge_precision_recall.py) is the
evaluation of record for this pilot, and every judge-based number keeps its
upper-bound caveat permanently — there is no human calibration pass to
retire it.

### D-eval-2: engine scorecard dimensions (2026-09-01)
The engine comparison scores four axes (Evaluation Spec v0.2,
`tools/score_run.py`): (1) end-to-end leak/detection metrics — leak rate,
detection recall/precision, over-redaction; (2) span transcription accuracy —
WER between GT `spoken_text` and the engine's words inside each GT interval,
after lowercasing, punctuation stripping, and number canonicalization; (3)
language ID — primary-language match rate and language-set Jaccard vs the
reviewer's `languages_observed`; (4) speaker count — engine distinct speakers
vs reviewer `num_speakers` (match rate, MAE). Span-WER replaces full-clip WER
deliberately: there are no reference transcripts for the full clips (GT
annotates PII intervals only), the PII spans are the part of the transcript
that matters for redaction, and number canonicalization (number-words →
digits, adjacent digits merged) is needed for fairness because the engines
format spoken numbers differently ("ninety eight" vs "98" vs "9 8"). Known
caveats: GT wording is Gemini-seeded (D-eval-1), and Whisper cannot diarize
mixed audio (always 1 speaker), so the speaker axis structurally favors
diarizing engines.

### Observed: detection is sensitive to transcript formatting (2026-09-01)
Same detector, same policy, same audio — different stage-B transcripts led to
different detections (ElevenLabs' transcript contained the spoken bill amounts
verbatim, yet the detector flagged them only on Gemini's and Whisper's
transcripts). Segmentation/punctuation/casing of the input transcript affects
LLM detection. Candidate mitigations if GT confirms materiality: light
canonicalization of transcripts before detection (same normalizer for all
engines), or self-consistency (N detection samples, union/vote).

### First benchmark results — reference-transcript bench (2026-09-01)
Vendor engines vs the three human reference transcripts (Hindi/Odia/Bengali
telephone consultations, redacted utterances skipped, beeped words masked):

| engine | global WER hi/or/bn | lang ID | speakers |
|---|---|---|---|
| ElevenLabs scribe_v2 | 0.24 / 0.56 / 0.25 | 3/3 | 3/3 |
| OpenAI whisper-1 (via OpenRouter) | 0.94 / 1.00 / 1.00 | 1/3 | 0/3 |

whisper-1 failure is qualitative, not marginal: full-duration coverage but
degenerate sparse output (87/28/51 words vs ~700-800 real), repetition-loop
hallucinations, and wrong-script output (Odia audio -> Telugu script; Bengali
audio -> Gujarati script) — the documented whisper low-resource +
telephone-quality failure mode. whisper-1 (2022) is the only transcription
model OpenRouter's /audio/transcriptions exposes; OpenAI's newer audio models
(gpt-audio chat models) would need a different integration and remain an
option. Gemini's bench pass runs after the corpus transcriptions finish.
Caveat: mean per-utterance WER is inflated by window-edge contamination
(documented in the bench tool); global WER/CER is the headline number.

### Case study: MP EP - 1 (2026-09-01) — three defenses added
The fourth fiction episode initially produced 3 intervals its siblings correctly
did not: a character's age and name from the story's narration — and the LLM
returned word ranges so wide that one single-word value would have beeped 4.5
minutes to end-of-file. QC passed it (over-beeping was not then a QC concern).
Three layers were added, each catching a different failure independently:
1. **Span-trim guard** (detect): a returned word range far wider than the quoted
   spoken_text (>3x + 5 words) is trimmed to the best-matching window,
   confidence dropped to low, needs_review forced, note recorded.
2. **Giant-beep tripwire** (QC): any rendered beep > 45 s flags the file.
3. **Detection prompt v2**: the fiction rule now explicitly covers narrated
   audio dramas (episode framing, narrator voice, next-episode outros), while
   real production credits still count. Change-controlled: real example
   (MP EP - 1), expected impact (fewer fiction false positives), version bump
   (prompts/detection_policy_prompt_v2.txt). Re-run result: 0 intervals,
   consistent with sibling episodes; corpus 8/8 PASS.
For comparison fairness, detection is re-run corpus-wide with v2 before any
engine scoring (all engines share the same detection prompt version).

### Defect class 3: chunk-boundary word timestamps (2026-09-01)
Gemini's chunked transcriptions carried hundreds of words with impossible
durations (a single word "lasting" 3+ minutes, ending at chunk/file end) —
caught when the giant-beep tripwire flagged a 208 s beep whose span was one
word wide (so the span-trim guard correctly passed it: three independent
defect classes now, three independent defenses). Fix: sanity() clamps any
word duration to 5 s; a deterministic repair pass fixed all cached transcripts
and recomputed interval times from word indices with zero API cost. Only one
file's rendered output changed. Reinforces D-arch-3: LLM word timestamps are
untrusted and every downstream layer must bound them.

### Defect class 4: engine clock skew at value onsets (2026-09-02, user-reported)
A colleague listening to the first shared sample heard the first syllable of a
name before the beep started ("this is <name> from …" with the name's onset
audible). Root cause: the engine timestamps the value's start LATE (~300 ms
observed), and no transcript-side check can see it — the transcript contains
the full word; only the audio knows the word actually began earlier. Defense:
pipeline/edge_verify.py listens to the rendered file itself. It cuts ~2 s
around every beep edge FROM THE BEEPED AUDIO (covered speech is already tone,
so only genuine leftovers survive), transcribes the snippets, and
fuzzy-matches heard tokens against the beeped values; a match extends that
edge 400 ms and re-renders, up to 2 rounds; anything still audible
quarantines the file. First iteration of the matcher was too permissive
(similarity 0.5 fuzzy-matched function words — "tell"~"Shailesh" — and
positional "last 3 words" adjacency flagged neighbors seconds away, so the
loop extended forever). Tightened 2026-09-02: similarity ≥ 0.66, minimum
token length 3, a function-word stoplist (En + Hinglish), and adjacency by
the snippet's own word timestamps (≤ 500 ms from the boundary). The digit
rule is unchanged: any digit-bearing token at an edge always flags.

Planted-defect testing (deliberately shifting the name beep to expose
150-500 ms of the name) then falsified the matcher's core premise: the
snippet ASR heard the exposed fragment as "Vidya"/"Vidyut" while both
production listeners wrote "Ritika" — two ASRs render the same ambiguous
Indian name so differently that similarity-to-the-value CANNOT be the primary
signal (ratio ≈ 0.36, far under any usable threshold; notably the
pre-keyterms ElevenLabs run had also heard "Vidya", so the audio itself is
ambiguous). Redesign: the primary check is now EXPECTED-AUDIBLE matching —
the transcript + rendered beeps determine exactly which non-PII words should
be audible in each snippet, and a content word heard near the boundary that
matches NO expected word (fuzzy ≥ 0.6, Indic-folded, stopwords/short tokens
ignored) flags the edge regardless of what the ASR thinks the word was.
Value-token and digit matching remain as secondary signals. Two operational
fixes from the same tests: snippet ASR runs on Flash with a 256-token
thinking cap (3.1 Pro burned ~63k thinking tokens per 2-second snippet —
minutes of latency and real cost per edge), and a snippet that still fails
after 3 retries is recorded as an UNCHECKED edge in the manifest instead of
silently passing (a failed check is a hole, not a skip).

The expected-audible check then over-fired on the CLEAN render — three FP
mechanisms, each with its own guard (all observed, none hypothetical):
(1) the snippet ASR writes neighbours in another script ("Okay" came back as
Devanagari ओके; folded 'oke' vs 'okay' scored 0.571) — the expected-match
threshold is 0.55 with Indic folding, and real leaked fragments score ~0.3;
(2) it hallucinates words from the 1 kHz tone itself (heard "zero" on pure
tone) — heard tokens whose timestamps sit ≥50% on a rendered beep are
discarded; (3) one-off phonetic mishearings ("This is" → "jaise") — a
non-digit flag must be CONFIRMED by a second independent listen (real
fragments persist across listens, hallucinations do not); digit hits still
flag immediately. Final validation (2026-09-02): planted exposures of 150 ms
and 206 ms of the name were both caught (heard as "Vidyut" / "Vidhya" —
unexpected-at-edge), the beeps were extended, and the clean render verified
with zero false flags. Sensitivity floor, stated honestly: convergence can
leave up to ~60 ms of onset residue — below what the snippet ASR reports,
about a fraction of one phoneme.

User-ear evidence (2026-09-02, v3/v4 listen-throughs) then CALIBRATED the
floor: with the corridor renderer's 20 ms graze the user heard "Vi" of the
name before the beep, and with a 150 ms guard still heard "Vi/Vid" — the
name's true onset precedes BOTH listeners' timestamps by 200-300 ms (the
same magnitude as defect class 4's original observation), and a fragment
that small is below the snippet ASR's audibility but not below a human's.
Three data points fixed the number: beep start at stamped-onset -20 ms
leaked, -150 ms leaked, -300 ms (the v2 geometry) was clean to the listener.
Render-side fix: FULL_NAME intervals carry a hard 300 ms start-side floor
(ONSET_GUARD_MS), even when it eats the preceding word(s) — losing "This is"
is nothing next to leaking a name's first syllable. Names only: FINANCIAL and
other categories keep the tight corridors (user-verified correct), and
digit-bearing exposures have a stronger verifier net anyway (digits flag
immediately). Ends stay word-graze-tight (no tail complaints across three
listen-throughs). Division of labour, made explicit: the render guards
protect against SYSTEMATIC onset lateness; the edge verifier catches GROSS
failures (150 ms+); the human ear remains the arbiter below that.

### Observed: cross-listener agreement can still mis-hear a name (2026-09-02)
The sample call's caller name is humanly heard as "Vidya"; ElevenLabs (WITH
"Vidya" in its keyterms vocabulary) and Gemini (no keyterms) both transcribed
"Ritika", while three independent snippet-ASR listens heard Vidya / Vidyut /
Vidhya. Lessons recorded: (1) two-listener agreement is not truth for proper
names — the judge-based reference inherits this (its caveat already says so);
(2) keyterms bias recognition but do not guarantee a listed name is chosen,
so keyterms cannot be treated as a correctness mechanism, only a recall aid;
(3) REDACTION is unaffected — the name is beeped in audio and <NAME REDACTED>
in text regardless of which spelling the engine wrote; the error lives only
in the raw (unredacted) transcript, i.e. it is a transcription-accuracy
issue, not a privacy issue.

### Defect class 5: degenerate bunched word timestamps (2026-09-02, user-reported)
Same sample, second symptom: neighbors of a value ("right", "sir") were
beeped while parts of the value itself weren't. Root cause: ElevenLabs
sometimes emits whole utterances whose words all share one instant (dozens of
~1 ms "words") — intervals built from those times land arbitrarily. Defense in
sanity(): if ≥ ⅓ of an utterance's words are < 20 ms long, the utterance's
total span is redistributed across its words proportionally to word length
(minimum 40 ms/word). A retroactive cache repair fixed 48 words on the shared
sample with zero API cost. Defect classes 4 and 5 are the two halves of the
same lesson as D-arch-3, now empirical: engine word times fail in both
directions (skewed onsets, collapsed spans), so the renderer needs BOTH a
statistical repair on the way in and an audio-side verification on the way
out.

### Defect class 6: blind pads swallow neighbour words (2026-09-02, user-reported)
Second listen-through of the fixed sample: the name beep was correct but
"from" between <NAME> and <ORGANIZATION> was also beeped (the two 250 ms pads
merged straight across the 191 ms gap holding it), and the FINANCIAL_INFO
beeps clipped neighbours ("Okay", "right", "Yeah", "sir") and overran into
silence (snap-to-quiet could move an edge 150 ms outward, pad another 250 ms).
Fix: stage D now places every beep edge inside a WORD-AWARE CORRIDOR — the
edge lands on the quietest moment between the value's own timestamp and the
nearest non-redacted word, padding up to pad_ms into silence but never across
a transcribed neighbour word. The privacy floor is min_pad_ms (50 ms) into
silence and WORD_GRAZE_MS (20 ms) against a transcribed neighbour (the engine
segmented both words there, and the beep-edge listen check is the empirical
backstop). Result on the sample: PII words 0% audible, "from" back to 79%
audible, "Okay"/"right"/"Yeah"/"sir" 100% audible, QC PASS. Side effect worth
keeping: tight corridors EXPOSE bad engine timestamps instead of masking them
with fat pads — on the same file the Whisper-transcript configs now show
would-be leaks (whisper heard the name as "Redya" ending 146 ms early), which
is exactly the kind of comparative evidence the engine scorecard needs.
snap_edges/edge_snap_ms are retired from stage D (kept as a legacy helper).

### Defect class 7: extension-bridged beeps break the QC plan (2026-09-03)
Found by the post-change validation run, not by the change being validated
(the language-table A/B was clean on all 40 transcripts). The beep-edge
listen check is nondeterministic across days — on this pass it heard
fragments at two edges of the sample and extended both beeps +400 ms, which
made adjacent beeps TOUCH; the audio then physically carries one continuous
tone, but edge_verify wrote the two unmerged spans as the plan, so QC's
orphan check saw a tone matching no planned beep and quarantined a clean
file. Failure direction was safe (hold, never ship), but recurring. Fix:
after extensions, edge_verify merges touching/overlapping same-channel beeps
before re-rendering and records the MERGED spans in rendered_intervals — the
plan now always describes the physical audio. Lesson appended to D-arch-2's
principle: every writer of rendered_intervals must emit what the audio
actually contains, not what the step intended.

### D-eval-3: LLM-judge evaluation (2026-09-01) — GT on hold, judges interim
With human GT deferred, the Mindy pilot's judge models (gpt-5.6-sol / luna,
reached via OpenRouter — Deccan's internal proxy is VPN-only) evaluate the
engines as an interim signal. The judges apply OUR taxonomy v0.2 policy (not
Mindy's 29-category one); what is borrowed is the independent model family —
an OpenAI-family judge grading a Gemini-family detector avoids self-grading.
Two tools:
1. tools/judge_leak_audit.py — judge reads each engine's REDACTED transcript
   and reports surviving PII (quotes must literally appear in the text:
   hallucination guard). Findings on gemini outputs double as stage-C detector
   misses (human review queue).
2. tools/judge_pseudo_gt.py — judge detects PII on each engine's RAW
   transcript; findings are time-anchored and UNIONed across engines into
   pseudo-GT; every engine's rendered beeps are scored against it. Explicit
   limit: PII missed by all engines is invisible — upper-bound numbers.
Profile correctness (user-caught defect, fixed before first run): neither tool
assumes a domain. Leak-vs-kept classification is computed IN CODE from the
profile recorded in each file's own rendered_intervals.json (the judge knows
nothing about profiles); --profile exists only for explicit counterfactual
scoring. When human GT lands, the judge itself gets scored against it —
calibrating how much weight the judge-based numbers deserve.

### v3 results — before/after (2026-09-01)
Policy prompt v3 (D8 org clarification + multilingual values + calibration
rule 11) + adversarial verification + Indic canonicalization, re-run corpus-wide:
- Judge leak audit: 0 leak candidates on all 24 outputs (was 18/30/7 per
  engine); the only surviving findings are profile-consistent health mentions.
  Verifier rejected 9/10/3 over-reach findings per engine.
- Detector v2 -> v3 (real recall gains, not filtering): Telugu call 5 -> 11
  intervals — now catches the phone number, both person-linked orgs, and all
  five bill amounts it previously missed; Hinglish call 11 -> 20 (person-linked
  insurer/TPA/lender orgs now included); all QC PASS.
- Judge-detector agreement (gemini pipeline): 36% -> 81.2%; union coverage
  33% -> 57.4%. Residual pseudo-leak numbers are dominated by cross-engine
  timestamp offsets in the union (an engine cannot cover another engine's
  timeline) — the agreement metric is the detection-quality read.
Remaining known gaps: cross-engine union timeline artifacts; judge calibration
against human GT still pending (GT on hold).

### Vendor-native detection scenario — ElevenLabs entity detection (2026-09-01)
Ran scribe_v2 with entity_detection=["pii"] over the full corpus and scored it
against the judge union and head-to-head against our detector on the identical
ElevenLabs transcripts (tools/elevenlabs_entity_eval.py). Result: 1,036
entities corpus-wide, 757 mappable to our taxonomy; union coverage 35.4%
(our pipeline: 57.4%), precision vs union **3.8%** (our detector: ~89%).
Decisive failure mode: it is generic NER with no policy layer — the Bengali
FICTION episodes got 157-275 entities each (character names, story amounts)
where the correct count is ~0; our fiction rule produces exactly 0 there.
No person-link requirement, no masked-value exclusion, no profiles.
Conclusion: vendor-native detection is not a viable substitute for the policy
detector; it would over-beep fiction into uselessness while missing a third of
real PII. The stage-B(EL)+stage-C(Gemini) hybrid remains the target
configuration; this scenario is closed.

### Hybrid-config fixes (2026-09-01, user-approved)
Fix 1 — detection-input canonicalization (`detect_pii.canon_token`): every
engine's words are presented to the detector in one house style (fancy quotes/
dandas normalized, attached punctuation stripped, internal @ . - , preserved).
Targets the 5 formatting-sensitivity misses on ElevenLabs transcripts. Applied
identically to all engines so comparisons stay fair; indices/timestamps untouched.
Fix 2 — ElevenLabs `keyterms` recognition biasing: 93-term GENERIC vocabulary
(common Indian given names + major insurers/TPAs/hospitals/banks,
prompts/keyterms_india_v1.txt). Provenance note: the list is general domain
vocabulary, deliberately not harvested from corpus transcripts; several corpus
names (Shailesh, Keerthana) are simply common names that any such list contains
— the benchmark-leakage consideration is acknowledged and the true validation
remains human GT. API detail: keyterms must be repeated multipart fields, each
term < 50 chars (JSON-array-as-string is rejected). Live verification on the
Hinglish call: 'Shailesh' 0x -> 2x, 'Keerthana' garbled -> 1x, 'Medi Assist'
variants -> 4x consistent, 'Orix' misspelt -> 1x correct.
Decision gate re-run in progress: EL corpus re-transcribed with keyterms; all
engines re-detected on canonicalized input; judge eval re-run. Gate: EL+Gemini
must reach agreement >= 80%, zero unheard-leak candidates, all QC green.

### Decision-gate outcome (2026-09-01)
Post-fix re-run (EL keyterms + canonicalized detection input, all engines):
- Judge-detector agreement: ElevenLabs pipeline 65% -> **81.2%** (now highest;
  gemini 77.1%). Gate threshold >= 80%: PASSED.
- Leak audit: 0 leak candidates, all 24 outputs. QC 24/24 PASS. PASSED.
- Unheard-leak candidates: ~10 -> 4 raw; item-level verification resolves 2 as
  cross-engine spelling mirror-images (KPMG/PTMP org — both pipelines beep the
  same moment; Namitha/Nutan same-moment name), leaving TWO residual human-ear
  items on the EL pipeline: (1) the Hinglish phone number's leading "double
  nine" — EL transcribed the number from "nine eight zero..." onward, so its
  beep may start ~0.5-1 s late and expose the first two digits (Hinglish call
  ~219-222 s); (2) Telugu name 'Archana' (gemini hears it at ~13.9 s of the
  Telugu call; EL transcribed no name there). Gate: SUBSTANTIALLY PASSED —
  residuals are exactly 2 listen-and-confirm moments, logged for the GT pass.
Recommendation stands: EL+Gemini hybrid at ~1/40th transcription cost, with
spot-listening of the 2 residual moments and the assurance-mode option
(dual-transcribe union) for high-risk content.

### Observed: thinking-driven truncation + romanization under caps (2026-09-02, fresh run)
Gemini 3.1 Pro's internal thinking can consume the ENTIRE output-token budget
on dense Indic telephone audio — observed truncations at 240 s, 120 s, 60 s
and even 30 s windows (window size barely matters when thinking, not word
JSON, is the consumer). Fix: on truncation the transcriber retries once with
thinking capped at 4,096 tokens (leaving ~57k for words) before bisecting.
Second-order finding: under the capped retry the model ROMANIZED Devanagari
speech despite explicit script instructions (a Devanagari human reference
then scores ~0.79 WER purely from script). Both behaviours are recorded as
measured in Benchmark v3; they are accuracy/cost risks specific to the
Gemini-transcription configuration and further support the hybrid choice.

### D-eval-4: judge-based detection precision/recall + six-config comparison (2026-09-02)
With the human GT pass dropped (D-eval-1), the pilot's precision/recall
numbers come from tools/judge_precision_recall.py. Reference per file: the
verified judge (sol) detections on BOTH listener transcripts (elevenlabs +
gemini; luna adversarial verification, span caches shared with
judge_pseudo_gt), clustered into unique items. Metrics per configuration —
DETECTION RECALL (reference items matched by intervals.json / all items;
match = temporal cluster rule OR same canonicalized value within ±2.5 s),
DETECTION PRECISION (intervals matching a reference item / all intervals),
REDACTION RECALL / LEAK RATE (must-beep items ≥95% covered by rendered
beeps), and supplementary POLICY-VERIFIED PRECISION (luna confirms each
interval satisfies an include rule). Two coverage-semantics fixes found on
the first smoke run: (1) value-match clustering chains REPEATED mentions of
one value into an item whose hull spans minutes of non-PII speech — coverage
is now duration-weighted over the constituent spans, not the hull; (2)
listeners timestamp the same occurrence 1–2 s apart, so a span also counts
covered when a different listener's span of the same occurrence is covered
(the engine beeped it on its own clock; the residue is skew, not a leak).
Both fixes live in judge_pseudo_gt.item_coverage and improve that tool too.

Stage C is now PLUGGABLE (--detector gemini|gpt on run_pipeline.py): the gpt
backend sends the SAME policy v3 prompt and word-index schema to
openai/gpt-5.6-luna via the judge_client transport (usage rows tagged
detector_gpt). Output dirs gain a __gpt suffix; cached transcripts are
seeded from the plain-engine dir so no audio is re-transcribed to swap
detectors. Six configurations compared end to end (cost × performance):
EL-inbuilt (carried over from the closed vendor eval), EL+Gemini
(production), Gemini, Whisper+Gemini, EL+GPT, Whisper+GPT. Fairness caveat
recorded everywhere the numbers appear: the judge is GPT-family, so the
__gpt detector configs share a family with it (detector luna, judge sol);
the cross-family read (Gemini detector, GPT judge) is the cleaner
comparison. Usage-log rows now carry a context tag
(transcribe:/detect:/edge_verify:<fid>) so per-stage cost per audio-hour is
computed from output/api_usage.jsonl, not estimated.

## 6. Known limitations / open questions

- Word timestamps are model-estimated; no forced-alignment stage yet.
- Speaker labels may swap between chunk boundaries despite context priming
  (QC does not yet check cross-chunk speaker consistency).
- Beep mutes all voices in the window on MIXED (mono/dual-mono) audio — see
  D-arch-5; split-channel recordings avoid this by beeping only the offending
  channel, but no current corpus file is split-channel.
- `FINANCIAL_INFO` boundary (bill amounts vs incidental prices) is the fuzziest
  category — same finding as the Mindy pilot; watch in review.
- The QC leak heuristic only sees digits the ASR transcribed; a digit the ASR
  missed entirely is invisible to every automated layer (human spot-listening
  still required before any external use).
- No gold set yet — pipeline outputs are unmeasured against ground truth; the
  eval spec defines the scoring once GT exists.
- Video files: rendering path is audio-only today (m4a out); video remux needs
  ffmpeg.

## 7. Production workflow (tools/redact.py)

One command runs the decided configuration end to end:

```
python3 tools/redact.py MEDIA... [--profile strict|medical_consult]
                        [--outdir deliveries] [--assurance] [--no-audit]
```

Delivery folder per file: beeped m4a + redacted transcript + qc_report.md +
manifest.json (source sha, engine/model/keyterms hash, policy prompt version +
hash, profile, QC/audit outcomes, languages, fallback flag). PII-bearing
artifacts (transcript.json, intervals.json) never enter deliveries.
Gates and handling (user decisions 2026-09-01): QC FLAGGED or audit findings ->
quarantine/<id>/ for human review, never delivered; ElevenLabs outage ->
automatic Gemini stage-B fallback, prominently recorded; video input ->
audio-only delivery with manifest note (remux needs ffmpeg); 0-PII outcome is
a valid, verified delivery; same-name different-content collisions get
sha-suffixed folders. Since 2026-09-02 (user decision) the workflow also runs,
per delivery and BY DEFAULT: empirical beep-edge verification (defect class 4
defense — unresolved audible fragments quarantine) and the independent
LLM-judge leak audit (findings quarantine; --no-audit skips it, not
recommended). The remaining optional feature is --assurance
(dual-transcription union into beeps, for high-risk content). Verified: happy
path, no-PII path, quarantine, fallback and collision tests all pass.

## 8. Run book

```
python3 tools/redact.py <media...>                # PRODUCTION: hybrid workflow (edge-verify + judge audit by default) -> deliveries/
python3 tools/redact.py <media...> --no-audit     # skip the judge audit (not recommended for deliveries)
python3 pipeline/run_pipeline.py                  # experiments: all raw files, cached stages skipped
python3 pipeline/run_pipeline.py --file <path>    # one file
python3 pipeline/run_pipeline.py --profile strict # strict profile render
python3 pipeline/run_pipeline.py --force detect   # re-run detection after a policy change
python3 pipeline/run_pipeline.py --engine elevenlabs --detector gpt   # config 5: EL transcript + GPT detector (dirs gain __gpt)
python3 tools/judge_precision_recall.py           # detection precision/recall vs the judge reference (all configs)
python3 tools/judge_precision_recall.py --configs elevenlabs,gemini,openai,elevenlabs__gpt,openai__gpt
python3 tools/fresh_run.py [--dry-run]          # archive old outputs + rerun EVERYTHING fresh (all engines, judges, bench, costs)
python3 tools/cost_report.py --since-offset N    # measured $/audio-hour per config from usage-log rows after byte offset N
python3 tools/webapp.py                          # local web app: upload & redact + quarantine review/release (127.0.0.1:8788)
python3 tools/release.py extend|release|keep ... # quarantine review actions from the CLI (decisions recorded in manifest)
```

Config: `pipeline/config.json`. Key: `.env` (`GEMINI_API_KEY=…`, chmod 600,
gitignored; rotate after the pilot — it transited chat once).
