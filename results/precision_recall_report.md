# Detection precision/recall report — judge `openai/gpt-5.6-sol`, verifier `openai/gpt-5.6-luna`, profile `medical_consult`

Generated 2026-09-03 by `tools/judge_precision_recall.py` | reference listeners: el_scribe_v1, gem_38_flash | 8 file(s) | predictions: `output`

## Metric definitions

- **DETECTION RECALL** — reference items matched by the config's intervals.json / all reference items (match = temporal IoU >= 0.3 or 70% cover, OR same canonicalized value within +/-2.5 s)
- **DETECTION PRECISION** — intervals matching a reference item / all intervals in intervals.json (same match rule, same reference)
- **REDACTION RECALL** — must-beep reference items covered by the rendered beeps / all must-beep items. An item counts as covered when a config interval matching it (detection match rule) is >= 95% beep-covered on the config's own clock, or when the reference spans themselves are >= 95% covered (channel-aware). The own-clock rule keeps cross-engine timestamp skew between the reference listeners and the config's engine from counting as leaks; items the config never detected are judged on reference-span coverage alone.
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

| config | files | detection recall | detection precision | redaction recall | leak rate (all files) | leak rate (delivered files) | policy-verified precision (suppl.) |
|---|---|---|---|---|---|---|---|
| ElevenLabs + Gemini (production) | 8 | 41/44 = 93.2% | 48/50 = 96.0% | 39/42 = 92.9% | 3/42 = 7.1% | 0/4 = 0.0% | 45/50 = 90.0% |
| Gemini end-to-end | 8 | 40/44 = 90.9% | 48/51 = 94.1% | 39/42 = 92.9% | 3/42 = 7.1% | 0/4 = 0.0% | 47/51 = 92.2% |
| Whisper + Gemini | 8 | 25/44 = 56.8% | 29/31 = 93.5% | 26/42 = 61.9% | 16/42 = 38.1% | 0/4 = 0.0% | 26/31 = 83.9% |
| ElevenLabs + GPT detector | 8 | 38/44 = 86.4% | 42/45 = 93.3% | 37/42 = 88.1% | 5/42 = 11.9% | 0/4 = 0.0% | 42/45 = 93.3% |
| Whisper + GPT detector | 8 | 24/44 = 54.5% | 27/128 = 21.1% | 25/42 = 59.5% | 17/42 = 40.5% | 0/4 = 0.0% | 104/128 = 81.2% |

Leak rate (all files) counts every corpus file including quarantined ones; the production gates withhold quarantined files from delivery, so leak rate (delivered files) is the shipping-relevant view. A file with no gate output counts as delivered.

## Per file

### ElevenLabs + Gemini (production)

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 3/4 = 75.0% | 4/5 = 80.0% | 4 | 0 | missed: 025354f09d2dcc71517beb7553441a85_R001 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 16/16 = 100.0% | 21/21 = 100.0% | 16 | 1 | QUARANTINED — withheld from delivery |
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | 12/12 = 100.0% | 13/14 = 92.9% | 10 | 0 | QUARANTINED — withheld from delivery |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 12 | 10/12 = 83.3% | 10/10 = 100.0% | 12 | 2 | QUARANTINED — withheld from delivery; missed: c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R007 |

### Gemini end-to-end

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 4/4 = 100.0% | 5/5 = 100.0% | 4 | 0 |  |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 15/16 = 93.8% | 21/22 = 95.5% | 16 | 1 | QUARANTINED — withheld from delivery; missed: 06e8274b603dfddc4ff0a84ed09f1a85_R014 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | 11/12 = 91.7% | 12/14 = 85.7% | 10 | 0 | QUARANTINED — withheld from delivery; missed: 5252dea2cb5c97b14aa7dcf365211a85_R005 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 12 | 10/12 = 83.3% | 10/10 = 100.0% | 12 | 2 | QUARANTINED — withheld from delivery; missed: c34897552fdcca8ae3f8d5a7015d1a85_R006,c34897552fdcca8ae3f8d5a7015d1a85_R007 |

### Whisper + Gemini

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 4/4 = 100.0% | 5/5 = 100.0% | 4 | 0 |  |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 11/16 = 68.8% | 14/16 = 87.5% | 16 | 4 | QUARANTINED — withheld from delivery; missed: 06e8274b603dfddc4ff0a84ed09f1a85_R001,06e8274b603dfddc4ff0a84ed09f1a85_R004,06e8274b603dfddc4ff0a84ed09f1a85_R013,06e8274b603dfddc4ff0a84ed09f1a85_R014,06e8274b603dfddc4ff0a84ed09f1a85_R015 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | 10/12 = 83.3% | 10/10 = 100.0% | 10 | 0 | QUARANTINED — withheld from delivery; missed: 5252dea2cb5c97b14aa7dcf365211a85_R005,5252dea2cb5c97b14aa7dcf365211a85_R012 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 12 | 0/12 = 0.0% | n/a | 12 | 12 | QUARANTINED — withheld from delivery; missed: c34897552fdcca8ae3f8d5a7015d1a85_R001,c34897552fdcca8ae3f8d5a7015d1a85_R002,c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R004,c34897552fdcca8ae3f8d5a7015d1a85_R005,c34897552fdcca8ae3f8d5a7015d1a85_R006,c34897552fdcca8ae3f8d5a7015d1a85_R007,c34897552fdcca8ae3f8d5a7015d1a85_R008,c34897552fdcca8ae3f8d5a7015d1a85_R009,c34897552fdcca8ae3f8d5a7015d1a85_R010,c34897552fdcca8ae3f8d5a7015d1a85_R011,c34897552fdcca8ae3f8d5a7015d1a85_R012 |

### ElevenLabs + GPT detector

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 4/4 = 100.0% | 3/4 = 75.0% | 4 | 0 |  |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 15/16 = 93.8% | 20/21 = 95.2% | 16 | 1 | QUARANTINED — withheld from delivery; missed: 06e8274b603dfddc4ff0a84ed09f1a85_R008 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | 9/12 = 75.0% | 9/10 = 90.0% | 10 | 2 | QUARANTINED — withheld from delivery; missed: 5252dea2cb5c97b14aa7dcf365211a85_R002,5252dea2cb5c97b14aa7dcf365211a85_R011,5252dea2cb5c97b14aa7dcf365211a85_R012 |
| EP-3 | 0 | n/a | n/a | 0 | 0 |  |
| EP_3 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | n/a | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 12 | 10/12 = 83.3% | 10/10 = 100.0% | 12 | 2 | QUARANTINED — withheld from delivery; missed: c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R007 |

### Whisper + GPT detector

| file_id | ref items | detection recall | detection precision | must-beep | leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 4/4 = 100.0% | 5/5 = 100.0% | 4 | 0 |  |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 11/16 = 68.8% | 13/15 = 86.7% | 16 | 4 | QUARANTINED — withheld from delivery; missed: 06e8274b603dfddc4ff0a84ed09f1a85_R001,06e8274b603dfddc4ff0a84ed09f1a85_R004,06e8274b603dfddc4ff0a84ed09f1a85_R013,06e8274b603dfddc4ff0a84ed09f1a85_R014,06e8274b603dfddc4ff0a84ed09f1a85_R015 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | 9/12 = 75.0% | 9/9 = 100.0% | 10 | 1 | QUARANTINED — withheld from delivery; missed: 5252dea2cb5c97b14aa7dcf365211a85_R005,5252dea2cb5c97b14aa7dcf365211a85_R011,5252dea2cb5c97b14aa7dcf365211a85_R012 |
| EP-3 | 0 | n/a | 0/23 = 0.0% | 0 | 0 |  |
| EP_3 | 0 | n/a | 0/65 = 0.0% | 0 | 0 |  |
| MP_EP_-_1 | 0 | n/a | 0/11 = 0.0% | 0 | 0 |  |
| MP_EP_-_2 | 0 | n/a | n/a | 0 | 0 |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 12 | 0/12 = 0.0% | n/a | 12 | 12 | QUARANTINED — withheld from delivery; missed: c34897552fdcca8ae3f8d5a7015d1a85_R001,c34897552fdcca8ae3f8d5a7015d1a85_R002,c34897552fdcca8ae3f8d5a7015d1a85_R003,c34897552fdcca8ae3f8d5a7015d1a85_R004,c34897552fdcca8ae3f8d5a7015d1a85_R005,c34897552fdcca8ae3f8d5a7015d1a85_R006,c34897552fdcca8ae3f8d5a7015d1a85_R007,c34897552fdcca8ae3f8d5a7015d1a85_R008,c34897552fdcca8ae3f8d5a7015d1a85_R009,c34897552fdcca8ae3f8d5a7015d1a85_R010,c34897552fdcca8ae3f8d5a7015d1a85_R011,c34897552fdcca8ae3f8d5a7015d1a85_R012 |

## Per category (aggregate)

### ElevenLabs + Gemini (production)

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 8/8 = 100.0% | 9/10 = 90.0% |
| FULL_NAME | 11/13 = 84.6% | 11/12 = 91.7% |
| HEALTH_MEDICAL_INFO | 2/2 = 100.0% | 3/3 = 100.0% |
| ORGANIZATION_PERSON_LINKED | 17/19 = 89.5% | 20/20 = 100.0% |
| PHONE | 5/5 = 100.0% | 5/5 = 100.0% |

### Gemini end-to-end

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 8/8 = 100.0% | 9/11 = 81.8% |
| FULL_NAME | 13/13 = 100.0% | 14/14 = 100.0% |
| HEALTH_MEDICAL_INFO | 1/2 = 50.0% | 2/3 = 66.7% |
| ORGANIZATION_PERSON_LINKED | 16/19 = 84.2% | 18/18 = 100.0% |
| PHONE | 5/5 = 100.0% | 5/5 = 100.0% |

### Whisper + Gemini

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 3/8 = 37.5% | 4/4 = 100.0% |
| FULL_NAME | 9/13 = 69.2% | 10/12 = 83.3% |
| HEALTH_MEDICAL_INFO | 0/2 = 0.0% | n/a |
| ORGANIZATION_PERSON_LINKED | 12/19 = 63.2% | 11/11 = 100.0% |
| PHONE | 4/5 = 80.0% | 4/4 = 100.0% |

### ElevenLabs + GPT detector

| category | detection recall | detection precision |
|---|---|---|
| ASSIGNED_ID | n/a | 0/1 = 0.0% |
| FINANCIAL_INFO | 8/8 = 100.0% | 8/9 = 88.9% |
| FULL_NAME | 12/13 = 92.3% | 11/12 = 91.7% |
| HEALTH_MEDICAL_INFO | 1/2 = 50.0% | 1/1 = 100.0% |
| ORGANIZATION_PERSON_LINKED | 15/19 = 78.9% | 17/17 = 100.0% |
| PHONE | 5/5 = 100.0% | 5/5 = 100.0% |

### Whisper + GPT detector

| category | detection recall | detection precision |
|---|---|---|
| FINANCIAL_INFO | 3/8 = 37.5% | 4/4 = 100.0% |
| FULL_NAME | 9/13 = 69.2% | 10/107 = 9.3% |
| HEALTH_MEDICAL_INFO | 0/2 = 0.0% | 0/3 = 0.0% |
| ORGANIZATION_PERSON_LINKED | 11/19 = 57.9% | 9/10 = 90.0% |
| PHONE | 4/5 = 80.0% | 4/4 = 100.0% |

## Speakers and languages (per file, per listener)

| file_id | el_scribe_v1 speakers | gem_38_flash speakers | agree | el_scribe_v1 primary lang | gem_38_flash primary lang | agree | all languages (union) |
|---|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 2 | 2 | True | en | en | True | en |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 3 | 2 | False | en | en | True | en, kn |
| 5252dea2cb5c97b14aa7dcf365211a85 | 2 | 2 | True | en | en | True | en, hi |
| EP-3 | 2 | 10 | False | bn | bn | True | bn, mixed |
| EP_3 | 2 | 5 | False | bn | bn | True | bn, mixed, en |
| MP_EP_-_1 | 2 | 6 | False | bn | bn | True | bn, mixed, en |
| MP_EP_-_2 | 2 | 5 | False | bn | bn | True | bn, mixed |
| c34897552fdcca8ae3f8d5a7015d1a85 | 2 | 2 | True | te | te | True | te |

## Cross-listener timestamp agreement

Start-time offset between the two listeners' spans of the SAME value (their relative clock skew — the reason beep coverage is scored skew-aware and value clustering allows +/-2.5 s).

| file_id | value pairs | median offset ms | p90 offset ms |
|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 310 | 520 |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 16 | 218 | 550 |
| 5252dea2cb5c97b14aa7dcf365211a85 | 8 | 210 | 330 |
| EP-3 | 0 | n/a | n/a |
| EP_3 | 0 | n/a | n/a |
| MP_EP_-_1 | 0 | n/a | n/a |
| MP_EP_-_2 | 0 | n/a | n/a |
| c34897552fdcca8ae3f8d5a7015d1a85 | 9 | 190 | 740 |
| **all files** | 37 | 210 | 550 |

## Reference items per file

### 025354f09d2dcc71517beb7553441a85 — 4 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85_R001 | 12720-13763 | FULL_NAME,ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Finfortune |
| 025354f09d2dcc71517beb7553441a85_R002 | 14013-14764 | ORGANIZATION_PERSON_LINKED | el_scribe_v1 | beep | KIMS Hospital |
| 025354f09d2dcc71517beb7553441a85_R003 | 95940-98020 | FINANCIAL_INFO | el_scribe_v1,gem_38_flash | beep | thirty thousand six hundred and thirty-five rupees |
| 025354f09d2dcc71517beb7553441a85_R004 | 100900-105700 | FINANCIAL_INFO | el_scribe_v1,gem_38_flash | beep | Thirty thousand six hundred and thirty-five rupees |

### 06e8274b603dfddc4ff0a84ed09f1a85 — 16 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| 06e8274b603dfddc4ff0a84ed09f1a85_R001 | 3760-4280 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | Keerthana |
| 06e8274b603dfddc4ff0a84ed09f1a85_R002 | 4660-6040 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Mitramitra hospital |
| 06e8274b603dfddc4ff0a84ed09f1a85_R003 | 10070-11020 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | Praveen |
| 06e8274b603dfddc4ff0a84ed09f1a85_R004 | 52180-52580 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Orix |
| 06e8274b603dfddc4ff0a84ed09f1a85_R005 | 77280-81900 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | Shailesh |
| 06e8274b603dfddc4ff0a84ed09f1a85_R006 | 221180-228320 | PHONE | el_scribe_v1,gem_38_flash | beep | nine nine eight zero one six one double three one |
| 06e8274b603dfddc4ff0a84ed09f1a85_R007 | 227380-238260 | ORGANIZATION_PERSON_LINKED,PHONE | el_scribe_v1,gem_38_flash | beep | nine double zero eight three zero five six four eight |
| 06e8274b603dfddc4ff0a84ed09f1a85_R008 | 240040-247680 | ORGANIZATION_PERSON_LINKED | gem_38_flash | beep | HDFC Bank |
| 06e8274b603dfddc4ff0a84ed09f1a85_R009 | 252960-258760 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Ayu Health team |
| 06e8274b603dfddc4ff0a84ed09f1a85_R010 | 321080-328430 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Medi Assist |
| 06e8274b603dfddc4ff0a84ed09f1a85_R011 | 381208-381508 | FULL_NAME | el_scribe_v1 | beep | ಸುಧಾ |
| 06e8274b603dfddc4ff0a84ed09f1a85_R012 | 382508-382738 | FULL_NAME | el_scribe_v1 | beep | ಸುಧಾ |
| 06e8274b603dfddc4ff0a84ed09f1a85_R013 | 446730-457070 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | ಮೀಡಿ ಅಸಿಸ್ಟ್ |
| 06e8274b603dfddc4ff0a84ed09f1a85_R014 | 457878-458248 | ORGANIZATION_PERSON_LINKED | el_scribe_v1 | beep | Ayu Health |
| 06e8274b603dfddc4ff0a84ed09f1a85_R015 | 459228-459528 | ORGANIZATION_PERSON_LINKED | el_scribe_v1 | beep | Vikram |
| 06e8274b603dfddc4ff0a84ed09f1a85_R016 | 459390-466490 | FULL_NAME,ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Vikram |

### 5252dea2cb5c97b14aa7dcf365211a85 — 12 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| 5252dea2cb5c97b14aa7dcf365211a85_R001 | 2960-3400 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | Keerthana |
| 5252dea2cb5c97b14aa7dcf365211a85_R002 | 4090-5540 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Dr Aakarul Hospital |
| 5252dea2cb5c97b14aa7dcf365211a85_R003 | 10240-10990 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | Noor Jahan |
| 5252dea2cb5c97b14aa7dcf365211a85_R004 | 30620-30960 | FULL_NAME | el_scribe_v1 | beep | Manish |
| 5252dea2cb5c97b14aa7dcf365211a85_R005 | 46470-48140 | HEALTH_MEDICAL_INFO | gem_38_flash | annotate_only | medications post surgery |
| 5252dea2cb5c97b14aa7dcf365211a85_R006 | 57070-58280 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Medi Assist |
| 5252dea2cb5c97b14aa7dcf365211a85_R007 | 111890-116890 | FINANCIAL_INFO | gem_38_flash | beep | 42,000 |
| 5252dea2cb5c97b14aa7dcf365211a85_R008 | 168520-173520 | PHONE | el_scribe_v1,gem_38_flash | beep | nine double nine eight double zero one nine zero double seve |
| 5252dea2cb5c97b14aa7dcf365211a85_R009 | 173940-179250 | PHONE | el_scribe_v1,gem_38_flash | beep | nine seven three one five four double nine zero five |
| 5252dea2cb5c97b14aa7dcf365211a85_R010 | 179640-186320 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Model Solutions India Private Limited Company |
| 5252dea2cb5c97b14aa7dcf365211a85_R011 | 186486-192060 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | State Bank of India |
| 5252dea2cb5c97b14aa7dcf365211a85_R012 | 284260-285880 | HEALTH_MEDICAL_INFO | el_scribe_v1 | annotate_only | ECG X-rays |

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

### c34897552fdcca8ae3f8d5a7015d1a85 — 12 item(s)

| item | interval ms | categories | heard by | action | spoken_text |
|---|---|---|---|---|---|
| c34897552fdcca8ae3f8d5a7015d1a85_R001 | 7090-7460 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | నమిత |
| c34897552fdcca8ae3f8d5a7015d1a85_R002 | 7450-8400 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Medicover Hospital |
| c34897552fdcca8ae3f8d5a7015d1a85_R003 | 13620-14410 | FULL_NAME | gem_38_flash | beep | ఎంపిసి అర్జున్ |
| c34897552fdcca8ae3f8d5a7015d1a85_R004 | 20380-21650 | FULL_NAME | el_scribe_v1,gem_38_flash | beep | Rajendar Chari |
| c34897552fdcca8ae3f8d5a7015d1a85_R005 | 82280-84320 | PHONE | el_scribe_v1 | beep | eight one two three six eight zero four two |
| c34897552fdcca8ae3f8d5a7015d1a85_R006 | 101410-102280 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | Ayu Health |
| c34897552fdcca8ae3f8d5a7015d1a85_R007 | 115820-121390 | ORGANIZATION_PERSON_LINKED | el_scribe_v1,gem_38_flash | beep | ఆరోగ్య ఫైనాన్స్ |
| c34897552fdcca8ae3f8d5a7015d1a85_R008 | 121960-126960 | FINANCIAL_INFO | el_scribe_v1,gem_38_flash | beep | fifteen thousand కట్టాము |
| c34897552fdcca8ae3f8d5a7015d1a85_R009 | 166350-174320 | FINANCIAL_INFO | el_scribe_v1,gem_38_flash | beep | 2,49,495 రూపాయలు |
| c34897552fdcca8ae3f8d5a7015d1a85_R010 | 177710-180700 | FINANCIAL_INFO | el_scribe_v1 | beep | 21,250 రూపాయలు discount ఇచ్చారని |
| c34897552fdcca8ae3f8d5a7015d1a85_R011 | 183060-188600 | FINANCIAL_INFO | el_scribe_v1,gem_38_flash | beep | 1,50,000 cash ద్వారా pay చేశామని |
| c34897552fdcca8ae3f8d5a7015d1a85_R012 | 187700-194740 | FINANCIAL_INFO | el_scribe_v1,gem_38_flash | beep | Remaining amount 78,245 రూపాయలు NEFT ద్వారా pay చేశామని |

