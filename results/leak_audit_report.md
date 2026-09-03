# Judge leak audit — redacted transcripts (Phase A)

Generated 2026-09-02 by `tools/judge_leak_audit.py` | judge model `openai/gpt-5.6-sol` | verifier `openai/gpt-5.6-luna` (adversarial over-reach filter ON) | out `/Users/shivangi/Documents/Claude/Audio_PII_Project/output/judge/leak_audit`

The judge reads each engine's REDACTED transcript with the full detection policy and reports PII values that survived redaction. Findings failing the hallucination guard (quote not found in the cited line +/-1) or citing unknown categories were dropped and are counted below. With verification ON, validated findings must also survive an adversarial verification pass (each must satisfy a specific INCLUDE rule of the policy or be rejected as over-reach); rejections are counted per engine. `profile_dependent` findings (may legitimately stay audible under a lenient profile, e.g. HEALTH_MEDICAL_INFO under medical_consult) are reported separately; the rest are the true **leak candidates**.

## Aggregate — engine comparison

| engine | files audited | validated findings | leak candidates | profile-dependent | dropped (hallucinated) | dropped (bad category) | rejected by verifier |
|---|---|---|---|---|---|---|---|
| gemini | 8 | 1 | **0** | 1 | 0 | 0 | 12 |
| elevenlabs | 8 | 0 | **0** | 0 | 0 | 0 | 6 |
| openai | 8 | 0 | **0** | 0 | 0 | 0 | 2 |

## Engine `gemini`

- files audited: 8
- validated findings: 1 | leak candidates (excl. profile-dependent): 0 | profile-dependent: 1
- hallucination guard: 0 finding(s) dropped (quote not in cited line +/-1); 0 dropped for no valid category
- findings_rejected_by_verifier: 12 (adversarial over-reach filter)

### Findings per category

| category | findings | leak candidates |
|---|---|---|
| HEALTH_MEDICAL_INFO | 1 | 0 |

### Per file

| file_id | lines | findings | leak cand. | dropped | lang ok | judge notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 34 | 0 | 0 | 0 | yes | No unredacted PII values survived. References to a claim ID contain no identifie |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 94 | 0 | 0 | 0 | yes | No unredacted concrete PII values survived. Remaining references such as Aadhaar |
| 5252dea2cb5c97b14aa7dcf365211a85 | 59 | 1 | 0 | 0 | yes | Only surviving health/medical information was found. These findings may legitima |
| EP-3 | 50 | 0 | 0 | 0 | yes | The transcript is coherent Bengali narrated fiction with episode framing and cha |
| EP_3 | 156 | 0 | 0 | 0 | yes | The transcript is coherent Bengali narrated fiction with episode framing, charac |
| MP_EP_-_1 | 86 | 0 | 0 | 0 | yes | The transcript is coherent narrated fiction with character dialogue and a next-e |
| MP_EP_-_2 | 27 | 0 | 0 | 0 | yes | The transcript is clearly narrated fiction with episode framing, character dialo |
| c34897552fdcca8ae3f8d5a7015d1a85 | 50 | 0 | 0 | 0 | yes | The Telugu transcript is coherent enough to audit. Redaction markers and non-val |

### Findings (validated)

| file_id | line | categories | profile-dep | quote | reason |
|---|---|---|---|---|---|
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | HEALTH_MEDICAL_INFO | yes | medications post-surgery | This discloses the speaker's medication and surgical-treatment context. |

## Engine `elevenlabs`

- files audited: 8
- validated findings: 0 | leak candidates (excl. profile-dependent): 0 | profile-dependent: 0
- hallucination guard: 0 finding(s) dropped (quote not in cited line +/-1); 0 dropped for no valid category
- findings_rejected_by_verifier: 6 (adversarial over-reach filter)

### Per file

| file_id | lines | findings | leak cand. | dropped | lang ok | judge notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 40 | 0 | 0 | 0 | yes | No readable values remaining in the transcript meet the detection policy. Redact |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 104 | 0 | 0 | 0 | yes | No readable PII values survived redaction. References to Aadhaar, bank accounts, |
| 5252dea2cb5c97b14aa7dcf365211a85 | 71 | 0 | 0 | 0 | yes | Only surviving health/medical information was found; these findings are profile- |
| EP-3 | 60 | 0 | 0 | 0 | yes | The transcript is framed as a fictional episodic story (including an episode ope |
| EP_3 | 56 | 0 | 0 | 0 | yes | The transcript is coherent Bengali narrated fiction with episode framing and cha |
| MP_EP_-_1 | 70 | 0 | 0 | 0 | yes | No surviving PII findings. The transcript is clearly narrated fictional/story co |
| MP_EP_-_2 | 51 | 0 | 0 | 0 | yes | The transcript is coherent Bengali narrated fiction with episode framing and a n |
| c34897552fdcca8ae3f8d5a7015d1a85 | 54 | 0 | 0 | 0 | yes | The Telugu-English code-switched transcript is coherent enough to audit. Redacti |

### Findings (validated)

None.

## Engine `openai`

- files audited: 8
- validated findings: 0 | leak candidates (excl. profile-dependent): 0 | profile-dependent: 0
- hallucination guard: 0 finding(s) dropped (quote not in cited line +/-1); 0 dropped for no valid category
- findings_rejected_by_verifier: 2 (adversarial over-reach filter)

### Per file

| file_id | lines | findings | leak cand. | dropped | lang ok | judge notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 8 | 0 | 0 | 0 | yes | No readable values meet the detection policy. The visible payment-method referen |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 27 | 0 | 0 | 0 | yes | No readable PII values survived redaction. References to Aadhaar, bank accounts, |
| 5252dea2cb5c97b14aa7dcf365211a85 | 6 | 0 | 0 | 0 | yes | The transcript contains some awkward or garbled phrasing but remains coherent en |
| EP-3 | 18 | 0 | 0 | 0 | yes | The transcript is narrated fictional/story content with character dialogue and a |
| EP_3 | 29 | 0 | 0 | 0 | yes | No surviving PII. The transcript is narrated fictional/story content, indicated  |
| MP_EP_-_1 | 28 | 0 | 0 | 0 | yes | No surviving PII qualifies. The transcript is narrated fictional/story content w |
| MP_EP_-_2 | 27 | 0 | 0 | 0 | yes | No surviving PII values are visible. The repeated phrases appear to be episode/s |
| c34897552fdcca8ae3f8d5a7015d1a85 | 9 | 0 | 0 | 0 | NO | The transcript is largely incoherent/garbled and contains no clearly readable su |

### Findings (validated)

None.

## Judge disagreement — detector misses (review queue)

Stage-C detection ran on the GEMINI engine's own transcript, so a leak candidate the judge can still read on a gemini redacted transcript is PII our detector missed (or under-covered). These findings are queued for human review:

None — no non-profile-dependent leak candidates on gemini transcripts.

