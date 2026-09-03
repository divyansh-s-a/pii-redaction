# Detection precision/recall report — judge `openai/gpt-5.6-sol`, verifier `openai/gpt-5.6-luna`, profile `medical_consult`

Generated 2026-09-02 by `tools/judge_precision_recall.py` | reference listeners: elevenlabs, gemini | 8 file(s) | predictions: `output`

## Metric definitions

- **DETECTION RECALL** — reference items matched by the config's intervals.json / all reference items (match = temporal IoU >= 0.3 or 70% cover, OR same canonicalized value within +/-2.5 s)
- **DETECTION PRECISION** — intervals matching a reference item / all intervals in intervals.json (same match rule, same reference)
- **REDACTION RECALL** — must-beep reference items >= 95% covered by the rendered beeps / all must-beep items (channel-aware coverage)
- **LEAK RATE** — 1 - redaction recall — the headline safety number
- **POLICY-VERIFIED PRECISION (supplementary)** — intervals the adversarial verifier confirms satisfy a policy include rule / all intervals; reported alongside detection precision, never instead of it

## Scope of these numbers

> **The reference is a UNION of what the two listener transcripts (elevenlabs, gemini) surfaced to the judge — a shared benchmark for comparing configurations fairly; values neither listener transcribed sit outside its scope, so human spot-checks complement these numbers.**
>
> **The judge and verifier are LLMs applying the written policy; their reference is unvalidated. Human ground truth remains the calibrator and will later score the judge itself.**
>
> **Configs with a __gpt detector share a model family with the judge (detector luna, judge sol). Rule-verified precision especially may flatter them; the cross-family read (Gemini detector, GPT judge) is the cleaner comparison.**
>
> **Value-based matching requires a shared category between reference item and interval; purely temporal matches do not.**

## Aggregate per configuration (micro-averaged)

| config | files | detection recall | detection precision | redaction recall | leak rate | policy-verified precision (suppl.) |
|---|---|---|---|---|---|---|
| ElevenLabs + Gemini (production) | 8 | 35/40 = 87.5% | 42/50 = 84.0% | 33/39 = 84.6% | 6/39 = 15.4% | 38/50 = 76.0% |
| Gemini end-to-end | 8 | 37/40 = 92.5% | 45/52 = 86.5% | 36/39 = 92.3% | 3/39 = 7.7% | 45/52 = 86.5% |
| Whisper + Gemini | 8 | 21/40 = 52.5% | 25/31 = 80.6% | 13/39 = 33.3% | 26/39 = 66.7% | 27/31 = 87.1% |
| ElevenLabs + GPT detector | 8 | 34/40 = 85.0% | 38/44 = 86.4% | 32/39 = 82.1% | 7/39 = 17.9% | 36/44 = 81.8% |
| Whisper + GPT detector | 8 | 19/40 = 47.5% | 23/28 = 82.1% | 11/39 = 28.2% | 28/39 = 71.8% | 26/28 = 92.9% |

## Per file

### ElevenLabs + Gemini (production)

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 3/4 = 75.0% | 5/5 = 100.0% | 4 | 1 | missed: 025354f09d2dcc71517beb7553441a85_R002 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 16/16 = 100.0% | 21/21 = 100.0% | 16 | 1 |  |
| 5252dea2cb5c97b14aa7dcf365211a85 | 10 | 8/10 = 80.0% | 8/14 = 57.1% | 9 | 1 | missed: 5252dea2cb5c97b14aa7dcf365211a85_R004,5252dea2cb5c97b14aa7dcf365211a85_R006 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 10 | 8/10 = 80.0% | 8/10 = 80.0% | 10 | 3 | missed: c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R004 |

### Gemini end-to-end

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 3/4 = 75.0% | 5/5 = 100.0% | 4 | 1 | missed: 025354f09d2dcc71517beb7553441a85_R001 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 15/16 = 93.8% | 21/22 = 95.5% | 16 | 1 | missed: 06e8274b603dfddc4ff0a84ed09f1a85_R015 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 10 | 9/10 = 90.0% | 9/15 = 60.0% | 9 | 1 | missed: 5252dea2cb5c97b14aa7dcf365211a85_R005 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 10 | 10/10 = 100.0% | 10/10 = 100.0% | 10 | 0 |  |

### Whisper + Gemini

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 3/4 = 75.0% | 5/5 = 100.0% | 4 | 2 | missed: 025354f09d2dcc71517beb7553441a85_R001 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 10/16 = 62.5% | 12/16 = 75.0% | 16 | 10 | missed: 06e8274b603dfddc4ff0a84ed09f1a85_R001,06e8274b603dfddc4ff0a84ed09f1a85_R004,06e8274b603dfddc4ff0a84ed09f1a85_R011,06e8274b603dfddc4ff0a84ed09f1a85_R013,06e8274b603dfddc4ff0a84ed09f1a85_R014,06e8274b603dfddc4ff0a84ed09f1a85_R015 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 10 | 8/10 = 80.0% | 8/10 = 80.0% | 9 | 4 | missed: 5252dea2cb5c97b14aa7dcf365211a85_R005,5252dea2cb5c97b14aa7dcf365211a85_R006 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 10 | 0/10 = 0.0% | n/a | 10 | 10 | missed: c34897552fdcca8ae3f8d5a7015d1a85_R001,c34897552fdcca8ae3f8d5a7015d1a85_R002,c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R004,c34897552fdcca8ae3f8d5a7015d1a85_R005,c34897552fdcca8ae3f8d5a7015d1a85_R006,c34897552fdcca8ae3f8d5a7015d1a85_R007,c34897552fdcca8ae3f8d5a7015d1a85_R008,c34897552fdcca8ae3f8d5a7015d1a85_R009,c34897552fdcca8ae3f8d5a7015d1a85_R010 |

### ElevenLabs + GPT detector

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 4/4 = 100.0% | 5/5 = 100.0% | 4 | 0 |  |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 13/16 = 81.2% | 16/16 = 100.0% | 16 | 3 | missed: 06e8274b603dfddc4ff0a84ed09f1a85_R008,06e8274b603dfddc4ff0a84ed09f1a85_R010,06e8274b603dfddc4ff0a84ed09f1a85_R012 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 10 | 9/10 = 90.0% | 9/12 = 75.0% | 9 | 1 | missed: 5252dea2cb5c97b14aa7dcf365211a85_R004 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 10 | 8/10 = 80.0% | 8/11 = 72.7% | 10 | 3 | missed: c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R004 |

### Whisper + GPT detector

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 3/4 = 75.0% | 5/5 = 100.0% | 4 | 2 | missed: 025354f09d2dcc71517beb7553441a85_R001 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 8/16 = 50.0% | 10/14 = 71.4% | 16 | 12 | missed: 06e8274b603dfddc4ff0a84ed09f1a85_R001,06e8274b603dfddc4ff0a84ed09f1a85_R002,06e8274b603dfddc4ff0a84ed09f1a85_R004,06e8274b603dfddc4ff0a84ed09f1a85_R008,06e8274b603dfddc4ff0a84ed09f1a85_R011,06e8274b603dfddc4ff0a84ed09f1a85_R013,06e8274b603dfddc4ff0a84ed09f1a85_R014,06e8274b603dfddc4ff0a84ed09f1a85_R015 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 10 | 8/10 = 80.0% | 8/9 = 88.9% | 9 | 4 | missed: 5252dea2cb5c97b14aa7dcf365211a85_R005,5252dea2cb5c97b14aa7dcf365211a85_R006 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 10 | 0/10 = 0.0% | n/a | 10 | 10 | missed: c34897552fdcca8ae3f8d5a7015d1a85_R001,c34897552fdcca8ae3f8d5a7015d1a85_R002,c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R004,c34897552fdcca8ae3f8d5a7015d1a85_R005,c34897552fdcca8ae3f8d5a7015d1a85_R006,c34897552fdcca8ae3f8d5a7015d1a85_R007,c34897552fdcca8ae3f8d5a7015d1a85_R008,c34897552fdcca8ae3f8d5a7015d1a85_R009,c34897552fdcca8ae3f8d5a7015d1a85_R010 |

## Per category (aggregate)

### ElevenLabs + Gemini (production)

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 6/6 = 100.0% | 8/11 = 72.7% |
| FULL_NAME | 9/13 = 69.2% | 11/12 = 91.7% |
| HEALTH_MEDICAL_INFO | 0/1 = 0.0% | 0/2 = 0.0% |
| ORGANIZATION_PERSON_LINKED | 16/16 = 100.0% | 18/20 = 90.0% |
| PHONE | 5/5 = 100.0% | 5/5 = 100.0% |

### Gemini end-to-end

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 6/6 = 100.0% | 8/11 = 72.7% |
| FULL_NAME | 11/13 = 84.6% | 14/14 = 100.0% |
| HEALTH_MEDICAL_INFO | 1/1 = 100.0% | 1/4 = 25.0% |
| ORGANIZATION_PERSON_LINKED | 15/16 = 93.8% | 17/18 = 94.4% |
| PHONE | 5/5 = 100.0% | 5/5 = 100.0% |

### Whisper + Gemini

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 1/6 = 16.7% | 3/4 = 75.0% |
| FULL_NAME | 7/13 = 53.8% | 9/12 = 75.0% |
| HEALTH_MEDICAL_INFO | 0/1 = 0.0% | n/a |
| ORGANIZATION_PERSON_LINKED | 10/16 = 62.5% | 9/11 = 81.8% |
| PHONE | 4/5 = 80.0% | 4/4 = 100.0% |

### ElevenLabs + GPT detector

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 6/6 = 100.0% | 8/11 = 72.7% |
| FULL_NAME | 9/13 = 69.2% | 9/10 = 90.0% |
| HEALTH_MEDICAL_INFO | 1/1 = 100.0% | 1/1 = 100.0% |
| ORGANIZATION_PERSON_LINKED | 14/16 = 87.5% | 15/17 = 88.2% |
| PHONE | 5/5 = 100.0% | 5/5 = 100.0% |

### Whisper + GPT detector

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 1/6 = 16.7% | 3/4 = 75.0% |
| FULL_NAME | 7/13 = 53.8% | 9/12 = 75.0% |
| HEALTH_MEDICAL_INFO | 0/1 = 0.0% | n/a |
| ORGANIZATION_PERSON_LINKED | 8/16 = 50.0% | 7/8 = 87.5% |
| PHONE | 4/5 = 80.0% | 4/4 = 100.0% |

## Speakers and languages (per file, per listener)

| file_id | EL speakers | Gemini speakers | agree | EL primary lang | Gemini primary lang | agree | all languages (union) |
|---|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 2 | 2 | True | en | en | True | en |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 3 | 3 | True | en | en | True | en, mixed |
| 5252dea2cb5c97b14aa7dcf365211a85 | 2 | 2 | True | en | en | True | en |
| EP-3 | 3 | 11 | False | bn | bn | True | bn |
| EP_3 | 2 | 5 | False | bn | bn | True | bn, mixed, en |
| MP_EP_-_1 | 2 | 5 | False | bn | bn | True | bn, en |
| MP_EP_-_2 | 2 | 4 | False | bn | bn | True | bn |
| c34897552fdcca8ae3f8d5a7015d1a85 | 2 | 2 | True | te | te | True | te |

## Cross-listener timestamp agreement

Start-time offset between the two listeners' spans of the SAME value (their relative clock skew — the reason beep coverage is scored skew-aware and value clustering allows +/-2.5 s).

| file_id | value pairs | median offset ms | p90 offset ms |
|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 680 | 860 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 17 | 520 | 1270 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 7 | 270 | 740 |
| EP-3 | 0 | n/a | n/a |
| EP_3 | 0 | n/a | n/a |
| MP_EP_-_1 | 0 | n/a | n/a |
| MP_EP_-_2 | 0 | n/a | n/a |
| c34897552fdcca8ae3f8d5a7015d1a85 | 6 | 420 | 800 |
| **all files** | 34 | 420 | 1440 |

## Reference items per file

### 025354f09d2dcc71517beb7553441a85 — 4 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85_R001 | 12690-12760 | FULL_NAME | elevenlabs | beep | Ritika |
| 025354f09d2dcc71517beb7553441a85_R002 | 13000-13400 | FULL_NAME | gemini | beep | Vidya |
| 025354f09d2dcc71517beb7553441a85_R003 | 13380-14400 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | KIMS Hospital |
| 025354f09d2dcc71517beb7553441a85_R004 | 95940-106800 | FINANCIAL_INFO | elevenlabs,gemini | beep | thirty thousand six hundred and thirty-five rupees |

### 06e8274b603dfddc4ff0a84ed09f1a85 — 16 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| 06e8274b603dfddc4ff0a84ed09f1a85_R001 | 3400-4280 | FULL_NAME | elevenlabs,gemini | beep | Keerthana |
| 06e8274b603dfddc4ff0a84ed09f1a85_R002 | 4700-6040 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Vikram Hospital |
| 06e8274b603dfddc4ff0a84ed09f1a85_R003 | 10200-10940 | FULL_NAME | elevenlabs,gemini | beep | Praveen |
| 06e8274b603dfddc4ff0a84ed09f1a85_R004 | 51300-52460 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Orix |
| 06e8274b603dfddc4ff0a84ed09f1a85_R005 | 76700-81900 | FULL_NAME | elevenlabs,gemini | beep | Shailesh |
| 06e8274b603dfddc4ff0a84ed09f1a85_R006 | 221180-229300 | PHONE | elevenlabs,gemini | beep | double nine eight zero one six one double three one |
| 06e8274b603dfddc4ff0a84ed09f1a85_R007 | 226800-237900 | ORGANIZATION_PERSON_LINKED,PHONE | elevenlabs,gemini | beep | nine double zero eight three zero five six four eight |
| 06e8274b603dfddc4ff0a84ed09f1a85_R008 | 239100-244900 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | H D F C bank |
| 06e8274b603dfddc4ff0a84ed09f1a85_R009 | 252300-257700 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Ayush Health |
| 06e8274b603dfddc4ff0a84ed09f1a85_R010 | 321080-321720 | ORGANIZATION_PERSON_LINKED | elevenlabs | beep | Ayu Health |
| 06e8274b603dfddc4ff0a84ed09f1a85_R011 | 323460-324140 | ORGANIZATION_PERSON_LINKED | elevenlabs | beep | Medi Assist |
| 06e8274b603dfddc4ff0a84ed09f1a85_R012 | 381120-387120 | FULL_NAME | elevenlabs,gemini | beep | ಸುಧೀರ್ |
| 06e8274b603dfddc4ff0a84ed09f1a85_R013 | 447008-447558 | ORGANIZATION_PERSON_LINKED | elevenlabs | beep | Medi Assist |
| 06e8274b603dfddc4ff0a84ed09f1a85_R014 | 449228-457500 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | media assist |
| 06e8274b603dfddc4ff0a84ed09f1a85_R015 | 457878-458248 | ORGANIZATION_PERSON_LINKED | elevenlabs | beep | Ayu Health |
| 06e8274b603dfddc4ff0a84ed09f1a85_R016 | 458000-466500 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | ವಿಕ್ರಮ್ |

### 5252dea2cb5c97b14aa7dcf365211a85 — 10 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| 5252dea2cb5c97b14aa7dcf365211a85_R001 | 2960-3500 | FULL_NAME | elevenlabs,gemini | beep | Keerthana |
| 5252dea2cb5c97b14aa7dcf365211a85_R002 | 4500-5600 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Dr Agarwal Hospital |
| 5252dea2cb5c97b14aa7dcf365211a85_R003 | 10200-11020 | FULL_NAME | elevenlabs,gemini | beep | Noor Jahan |
| 5252dea2cb5c97b14aa7dcf365211a85_R004 | 29600-30000 | FULL_NAME | gemini | beep | Anish |
| 5252dea2cb5c97b14aa7dcf365211a85_R005 | 30620-30940 | FULL_NAME | elevenlabs | beep | Haneef |
| 5252dea2cb5c97b14aa7dcf365211a85_R006 | 46700-48000 | HEALTH_MEDICAL_INFO | gemini | annotate_only | medications post-surgery |
| 5252dea2cb5c97b14aa7dcf365211a85_R007 | 56500-58280 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Medi Assist |
| 5252dea2cb5c97b14aa7dcf365211a85_R008 | 168225-172401 | PHONE | elevenlabs,gemini | beep | nine double nine eight double zero one nine zero double seve |
| 5252dea2cb5c97b14aa7dcf365211a85_R009 | 173960-177531 | PHONE | elevenlabs,gemini | beep | nine seven three one five four double nine zero five |
| 5252dea2cb5c97b14aa7dcf365211a85_R010 | 179640-182500 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Model Solutions India Private Limited Company |

### EP-3 — 0 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|

### EP_3 — 0 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|

### MP_EP_-_1 — 0 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|

### MP_EP_-_2 — 0 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|

### c34897552fdcca8ae3f8d5a7015d1a85 — 10 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| c34897552fdcca8ae3f8d5a7015d1a85_R001 | 7200-7600 | FULL_NAME | gemini | beep | నూతన్ |
| c34897552fdcca8ae3f8d5a7015d1a85_R002 | 7500-8500 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | beep | Medicover Hospital |
| c34897552fdcca8ae3f8d5a7015d1a85_R003 | 13500-14400 | FULL_NAME | gemini | beep | ఎల్ పి సావిత్రి |
| c34897552fdcca8ae3f8d5a7015d1a85_R004 | 21300-22100 | FULL_NAME | gemini | beep | రాజేంద్ర చారి |
| c34897552fdcca8ae3f8d5a7015d1a85_R005 | 81300-84400 | PHONE | gemini | beep | ఎయిట్ వన్ టూ త్రీ సిక్స్ త్రీ ఎయిట్ జీరో ఫోర్ టూ |
| c34897552fdcca8ae3f8d5a7015d1a85_R006 | 121800-123600 | FINANCIAL_INFO | elevenlabs,gemini | beep | ఫిఫ్టీన్ థౌసండ్ లో కట్టాము |
| c34897552fdcca8ae3f8d5a7015d1a85_R007 | 166380-171500 | FINANCIAL_INFO | elevenlabs,gemini | beep | రెండు లక్షల నలభై తొమ్మిది వేల నాలుగు వందల తొంభై ఐదు రూపాయలు |
| c34897552fdcca8ae3f8d5a7015d1a85_R008 | 176500-180900 | FINANCIAL_INFO | elevenlabs,gemini | beep | ఇరవై ఒక్క వేల రెండు వందల యాభై రూపాయలు డిస్కౌంట్ ఇచ్చారు |
| c34897552fdcca8ae3f8d5a7015d1a85_R009 | 183000-186000 | FINANCIAL_INFO | elevenlabs,gemini | beep | లక్ష యాభై వేలు క్యాష్ ద్వారా పే చేసాము |
| c34897552fdcca8ae3f8d5a7015d1a85_R010 | 188200-193600 | FINANCIAL_INFO | elevenlabs,gemini | beep | డెబ్బై ఎనిమిది వేల రెండు వందల నలభై ఐదు రూపాయలు నెఫ్ట్ ద్వారా |

