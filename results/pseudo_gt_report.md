# Pseudo-ground-truth report — judge `openai/gpt-5.6-sol`, profile `medical_consult`

Generated 2026-09-02 by `tools/judge_pseudo_gt.py` | engines: gemini, elevenlabs, openai | 8 file(s) | predictions: `output` | verifier `openai/gpt-5.6-luna` (adversarial over-reach filter ON)

## CAVEAT — read before using these numbers

> **Pseudo-GT is a UNION of what the engines' transcripts surfaced. PII that EVERY engine failed to transcribe (or the judge missed on all of them) is INVISIBLE here — it appears in no denominator.**
>
> **All rates are measured against this reference: a 0% pseudo leak rate means nothing the reference contains leaked; human spot-checks complement it for values outside the reference.**
>
> **The judge applies the same written policy as stage C but is itself an LLM; its spans are unvalidated. Human ground truth remains the arbiter and will later be used to score the judge itself.**
>
> **Items surfaced by only one engine are transcription-dependent: the other engines never wrote the value down, so their detectors never had a chance — leaks on such items measure the whole engine pipeline, not the detector alone.**

## Aggregate per engine

Pseudo-GT: 43 item(s) total — 42 must-beep under `medical_consult`, 1 annotate-only (excluded from the leak denominator).

| engine | must-beep | leaked | pseudo leak rate | covered >=95% | agreement (detector vs judge) | spans rejected by verifier |
|---|---|---|---|---|---|---|
| gemini | 42 | 6 | 6/42 = 14.3% | 36/42 = 85.7% | 37/43 = 86.0% | 11 |
| elevenlabs | 42 | 8 | 8/42 = 19.0% | 34/42 = 81.0% | 36/43 = 83.7% | 7 |
| openai | 42 | 17 | 17/42 = 40.5% | 25/42 = 59.5% | 23/43 = 53.5% | 4 |

## Surfaced-by breakdown

Items surfaced by a single engine are the transcription-dependent ones — the other engines never heard the value.

| surfaced by N engine(s) | items | must-beep | leaked by >=1 engine |
|---|---|---|---|
| 1 | 15 | 14 | 12 |
| 2 | 15 | 15 | 11 |
| 3 | 13 | 13 | 0 |

| engine | items surfaced ONLY by it | of which must-beep |
|---|---|---|
| gemini | 5 | 4 |
| elevenlabs | 6 | 6 |
| openai | 4 | 4 |

## Per file

| file_id | items | must-beep | ann-only | gemini leaked | elevenlabs leaked | openai leaked | notes |
|---|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 4 | 4 | 0 | 1/4 = 25.0% | 1/4 = 25.0% | 0/4 = 0.0% |  |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 18 | 18 | 0 | 4/18 = 22.2% | 4/18 = 22.2% | 6/18 = 33.3% | gemini: 4 span(s) rejected by verifier (over-reach); openai: 3 span(s) rejected by verifier (over-reach) |
| 5252dea2cb5c97b14aa7dcf365211a85 | 11 | 10 | 1 | 1/10 = 10.0% | 0/10 = 0.0% | 1/10 = 10.0% | gemini: 6 span(s) rejected by verifier (over-reach); elevenlabs: 6 span(s) rejected by verifier (over-reach); openai: 1 span(s) rejected by verifier (over-reach) |
| EP-3 | 0 | 0 | 0 | n/a | n/a | n/a |  |
| EP_3 | 0 | 0 | 0 | n/a | n/a | n/a |  |
| MP_EP_-_1 | 0 | 0 | 0 | n/a | n/a | n/a |  |
| MP_EP_-_2 | 0 | 0 | 0 | n/a | n/a | n/a |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 10 | 10 | 0 | 0/10 = 0.0% | 3/10 = 30.0% | 10/10 = 100.0% | gemini: 1 span(s) rejected by verifier (over-reach); elevenlabs: 1 span(s) rejected by verifier (over-reach) |

## Leaked pseudo-GT items (leaked by at least one engine)

| file_id | item | interval ms | categories | surfaced by | leaked by | spoken_text |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 025354f09d2dcc71517beb7553441a85_P001 | 12690-12760 | FULL_NAME | elevenlabs | gemini | Ritika |
| 025354f09d2dcc71517beb7553441a85 | 025354f09d2dcc71517beb7553441a85_P002 | 12960-13400 | FULL_NAME | gemini,openai | elevenlabs | Vidya |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P002 | 4200-4520 | FULL_NAME | openai | gemini,elevenlabs | Seetha Nagyar |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P005 | 51300-52460 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | openai | Orix |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P009 | 245480-245820 | ORGANIZATION_PERSON_LINKED | openai | gemini,elevenlabs | KPMC |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P010 | 252300-257700 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | openai | Ayush Health |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P012 | 323460-324140 | ORGANIZATION_PERSON_LINKED | elevenlabs | openai | Medi Assist |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P014 | 447008-447558 | ORGANIZATION_PERSON_LINKED | elevenlabs | openai | Medi Assist |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P015 | 449228-457500 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | openai | media assist |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P016 | 457878-458248 | ORGANIZATION_PERSON_LINKED | elevenlabs | gemini,elevenlabs,openai | Ayu Health |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P018 | 485395-486660 | FULL_NAME | openai | gemini,elevenlabs | Vikram |
| 5252dea2cb5c97b14aa7dcf365211a85 | 5252dea2cb5c97b14aa7dcf365211a85_P003 | 10200-11020 | FULL_NAME | elevenlabs,gemini | openai | Noor Jahan |
| 5252dea2cb5c97b14aa7dcf365211a85 | 5252dea2cb5c97b14aa7dcf365211a85_P005 | 30620-30940 | FULL_NAME | elevenlabs | gemini | Haneef |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P001 | 7200-7600 | FULL_NAME | gemini | openai | నూతన్ |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P002 | 7500-8500 | ORGANIZATION_PERSON_LINKED | elevenlabs,gemini | openai | మెడికవర్ హాస్పిటల్ |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P003 | 13500-14400 | FULL_NAME | gemini | elevenlabs,openai | ఎల్ పి సావిత్రి |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P004 | 21300-22100 | FULL_NAME | gemini | elevenlabs,openai | రాజేంద్ర చారి |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P005 | 81300-84400 | PHONE | gemini | elevenlabs,openai | ఎయిట్ వన్ టూ త్రీ సిక్స్ త్రీ ఎయిట్ జీరో ఫోర్ టూ |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P006 | 121800-123600 | FINANCIAL_INFO | elevenlabs,gemini | openai | ఫిఫ్టీన్ థౌసండ్ లో కట్టాము |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P007 | 166380-171500 | FINANCIAL_INFO | elevenlabs,gemini | openai | రెండు లక్షల నలభై తొమ్మిది వేల నాలుగు వందల తొంభై ఐదు రూపాయలు |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P008 | 176500-180900 | FINANCIAL_INFO | elevenlabs,gemini | openai | ఇరవై ఒక్క వేల రెండు వందల యాభై రూపాయలు డిస్కౌంట్ ఇచ్చారు |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P009 | 183000-186000 | FINANCIAL_INFO | elevenlabs,gemini | openai | లక్ష యాభై వేలు క్యాష్ ద్వారా పే చేసాము |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P010 | 188200-193600 | FINANCIAL_INFO | elevenlabs,gemini | openai | డెబ్బై ఎనిమిది వేల రెండు వందల నలభై ఐదు రూపాయలు నెఫ్ట్ ద్వారా |

