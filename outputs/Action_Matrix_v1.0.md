# Action Matrix v1.0 — what gets beeped, per domain profile

Companion to `Audio_PII_Taxonomy_v0.2.md`. The taxonomy defines *what is
annotated*; this document defines *what the renderer does about it*. Annotation
is profile-independent — one ground truth serves every profile; only the beep
action changes.

## Profile mechanism

A profile is a mapping `category -> action` where action is `beep` or
`annotate_only`. Profiles never change annotation rules, interval boundaries,
or the taxonomy. Adding a domain later (banking support, education,
podcast/media…) = adding one column here, with evidence, and a version bump.

Interval-level override: an interval flagged `public_figure: true` is
`annotate_only` in **every** profile (see decision D1).

## The matrix

| # | Category | `strict` (default) | `medical_consult` |
|---|---|---|---|
| 1 | FULL_NAME | beep (unless public_figure) | beep (unless public_figure) |
| 2 | PHONE | beep | beep |
| 3 | EMAIL | beep | beep |
| 4 | ADDRESS | beep | beep |
| 5 | DATE_OF_BIRTH | beep | beep |
| 6 | AGE | beep | beep |
| 7 | PASSPORT_NUMBER | beep | beep |
| 8 | DRIVERS_LICENSE | beep | beep |
| 9 | GOVERNMENT_ID | beep | beep |
| 10 | TAX_ID | beep | beep |
| 11 | INSURANCE_ID | beep | beep |
| 12 | STUDENT_ID | beep | beep |
| 13 | LEGAL_CASE_ID | beep | beep |
| 14 | IMMIGRATION_STATUS | beep | beep |
| 15 | PAYMENT_CARD_NUMBER | beep | beep |
| 16 | BANK_ACCOUNT_NUMBER | beep | beep |
| 17 | FINANCIAL_ACCOUNT_INFO | beep | beep |
| 18 | FINANCIAL_INFO | beep | beep |
| 19 | PASSWORD_CREDENTIAL | beep | beep |
| 20 | IP_ADDRESS | beep | beep |
| 21 | DEVICE_IDENTIFIER | beep | beep |
| 22 | VEHICLE_REGISTRATION | beep | beep |
| 23 | HEALTH_MEDICAL_INFO | beep | **annotate_only** |
| 24 | BIOMETRIC_ID | beep | beep |
| 25 | SENSITIVE_DEMOGRAPHIC | beep | beep |
| 26 | CHILD_INFO | beep | beep |
| 27 | NATIONALITY | beep | beep |
| 28 | USERNAME_HANDLE | beep | beep |
| 29 | ASSIGNED_ID | beep | beep |
| 30 | ORGANIZATION_PERSON_LINKED | beep | beep |
| 31 | OTHER | beep | beep |

`medical_consult` rationale: in doctor–patient recordings the clinical content
(disease, symptoms, medication) is the purpose of the recording. The incumbent
process confirms this: it beeps NAME/ADDRESS/AGE/ORG but leaves all 35+
disease/medicine mentions audible (Set-2 evidence,
`media/Set2_Beeped_vs_Transcript_Findings.md`). Identity is removed, medicine
stays — classic de-identification.

## Annotator-facing public-figure rule

`public_figure: true` only when the person is an established public figure
referred to in their public capacity (politician in a policy discussion,
celebrity's public work). Their private details (phone, address, health,
accounts) are normal PII. A person merely well known locally, or a public
figure appearing as a data subject (patient, customer), is NOT exempt.
Doubt → `public_figure: false` + `needs_review: true`. Every exempt interval is
logged and auditable (eval reports an exempt-names table).

## Decision log

| ID | Date | Decision | Decider | Rationale / evidence |
|---|---|---|---|---|
| D1 | 2026-09-01 | Public figures NOT beeped; flagged + audited; private details still beeped; doubt→beep | Shivangi | Matches Mindy client policy + stage-3 judge rule 23; avoids constant beeping of news/podcast content |
| D2 | 2026-09-01 | Social handles are a must-beep category (`USERNAME_HANDLE`) | Shivangi | Handles are persistent personal identifiers; Mindy left the question unresolved |
| D3 | 2026-09-01 | Domain-configurable profiles; launch with `strict` + `medical_consult` | Shivangi | Set-2 evidence: incumbent keeps clinical content audible; one GT serves all profiles |
| D4 | 2026-09-01 | SSN folded into GOVERNMENT_ID; canonical routing table added | Shivangi | Jurisdiction neutrality (Aadhaar vs SSN); Mindy's national-ID routing was unstable |
| D5 | 2026-09-01 | Masked/partial identifiers excluded entirely | Shivangi | Mindy stage-3 stance; avoids beeping routine "ending 4832" read-backs |
| D6 | 2026-09-01 | Added ASSIGNED_ID and ORGANIZATION_PERSON_LINKED | Shivangi | Mindy gold misfiled person-assigned IDs into OTHER; incumbent redacts `<ORG>` |
| D7 | 2026-09-01 | GT via correction of Gemini output (time-saving), from-scratch subset later to measure anchoring | Shivangi | speed over purity for round 1 |
| D8 | 2026-09-01 | Person's insurer/TPA/lender IS ORGANIZATION_PERSON_LINKED (clarifying examples added; definition unchanged); financial values count in any language/script | Shivangi | judge-evaluation evidence: 8 detector misses were in-policy orgs; 7 misses were Telugu values |

## Open parameters

- Beep pad duration: renderer default 250 ms/side; revisit after listening tests.
- Confidence thresholds / review routing: deferred until human-review workflow.
- Additional profiles: added by evidence + version bump only.