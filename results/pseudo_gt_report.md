# Pseudo-ground-truth report — judge `openai/gpt-5.6-sol`, profile `medical_consult`

Generated 2026-09-03 by `tools/judge_pseudo_gt.py` | engines: gemini, elevenlabs | 8 file(s) | predictions: `output` | verifier `openai/gpt-5.6-luna` (adversarial over-reach filter ON)

## CAVEAT — read before using these numbers

> **Pseudo-GT is a UNION of what the engines' transcripts surfaced. PII that EVERY engine failed to transcribe (or the judge missed on all of them) is INVISIBLE here — it appears in no denominator.**
>
> **All rates are measured against this reference: a 0% pseudo leak rate means nothing the reference contains leaked; human spot-checks complement it for values outside the reference.**
>
> **The judge applies the same written policy as stage C but is itself an LLM; its spans are unvalidated. Human ground truth remains the arbiter and will later be used to score the judge itself.**
>
> **Items surfaced by only one engine are transcription-dependent: the other engines never wrote the value down, so their detectors never had a chance — leaks on such items measure the whole engine pipeline, not the detector alone.**

## Aggregate per engine

Pseudo-GT: 39 item(s) total — 38 must-beep under `medical_consult`, 1 annotate-only (excluded from the leak denominator).

| engine | must-beep | leaked | pseudo leak rate | covered >=95% | agreement (detector vs judge) | spans rejected by verifier |
|---|---|---|---|---|---|---|
| gemini | 38 | 6 | 6/38 = 15.8% | 32/38 = 84.2% | 36/39 = 92.3% | 5 |
| elevenlabs | 38 | 7 | 7/38 = 18.4% | 31/38 = 81.6% | 36/39 = 92.3% | 11 |

## Surfaced-by breakdown

Items surfaced by a single engine are the transcription-dependent ones — the other engines never heard the value.

| surfaced by N engine(s) | items | must-beep | leaked by >=1 engine |
|---|---|---|---|
| 1 | 10 | 10 | 9 |
| 2 | 29 | 28 | 3 |

| engine | items surfaced ONLY by it | of which must-beep |
|---|---|---|
| gemini | 8 | 8 |
| elevenlabs | 2 | 2 |

## Per file

| file_id | items | must-beep | ann-only | gemini leaked | elevenlabs leaked | notes |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 3 | 3 | 0 | 0/3 = 0.0% | 1/3 = 33.3% | gemini: 1 span(s) rejected by verifier (over-reach); elevenlabs: 1 span(s) rejected by verifier (over-reach) |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 13 | 13 | 0 | 0/13 = 0.0% | 2/13 = 15.4% | gemini: 1 span(s) rejected by verifier (over-reach); elevenlabs: 4 span(s) rejected by verifier (over-reach) |
| 5252dea2cb5c97b14aa7dcf365211a85 | 12 | 11 | 1 | 1/11 = 9.1% | 2/11 = 18.2% | gemini: 3 span(s) rejected by verifier (over-reach); elevenlabs: 4 span(s) rejected by verifier (over-reach) |
| EP-3 | 0 | 0 | 0 | n/a | n/a |  |
| EP_3 | 0 | 0 | 0 | n/a | n/a |  |
| MP_EP_-_1 | 0 | 0 | 0 | n/a | n/a |  |
| MP_EP_-_2 | 0 | 0 | 0 | n/a | n/a |  |
| c34897552fdcca8ae3f8d5a7015d1a85 | 11 | 11 | 0 | 5/11 = 45.5% | 2/11 = 18.2% | elevenlabs: 2 span(s) rejected by verifier (over-reach) |

## Leaked pseudo-GT items (leaked by at least one engine)

| file_id | item | interval ms | categories | surfaced by | leaked by | spoken_text |
|---|---|---|---|---|---|---|
| 025354f09d2dcc71517beb7553441a85 | 025354f09d2dcc71517beb7553441a85_P001 | 13000-13400 | FULL_NAME | gemini | elevenlabs | Vidya |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P009 | 252300-257700 | ORGANIZATION_PERSON_LINKED | gemini | elevenlabs | Ayush Health |
| 06e8274b603dfddc4ff0a84ed09f1a85 | 06e8274b603dfddc4ff0a84ed09f1a85_P013 | 459000-466500 | ORGANIZATION_PERSON_LINKED | gemini | elevenlabs | ವಿಕ್ರಮ್ |
| 5252dea2cb5c97b14aa7dcf365211a85 | 5252dea2cb5c97b14aa7dcf365211a85_P004 | 29600-30000 | FULL_NAME | gemini | elevenlabs | Anish |
| 5252dea2cb5c97b14aa7dcf365211a85 | 5252dea2cb5c97b14aa7dcf365211a85_P005 | 30620-30940 | FULL_NAME | elevenlabs | gemini | Haneef |
| 5252dea2cb5c97b14aa7dcf365211a85 | 5252dea2cb5c97b14aa7dcf365211a85_P012 | 285700-293000 | FINANCIAL_INFO | gemini | elevenlabs | three thousand three hundred |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P003 | 13500-14400 | FULL_NAME | gemini | elevenlabs | ఎల్ పి సావిత్రి |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P006 | 101600-102070 | ORGANIZATION_PERSON_LINKED | elevenlabs | gemini | Ayu Health |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P007 | 121800-123600 | FINANCIAL_INFO | elevenlabs,gemini | gemini | ఫిఫ్టీన్ థౌసండ్ లో కట్టాము |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P009 | 176500-180900 | FINANCIAL_INFO | gemini | gemini,elevenlabs | ఇరవై ఒక్క వేల రెండు వందల యాభై రూపాయలు డిస్కౌంట్ ఇచ్చారు |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P010 | 183000-186340 | FINANCIAL_INFO | elevenlabs,gemini | gemini | లక్ష యాభై వేలు క్యాష్ ద్వారా పే చేసాము |
| c34897552fdcca8ae3f8d5a7015d1a85 | c34897552fdcca8ae3f8d5a7015d1a85_P011 | 188200-193600 | FINANCIAL_INFO | elevenlabs,gemini | gemini | డెబ్బై ఎనిమిది వేల రెండు వందల నలభై ఐదు రూపాయలు నెఫ్ట్ ద్వారా |

