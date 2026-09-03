# Audio PII Taxonomy v0.2

Supersedes v0.1 (kept for history). Changelog at the end. Companion documents:
`Action_Matrix_v1.0.md` (what gets beeped per profile), `Annotation_Schema_v0.2.md`
(how intervals are recorded), `docs/System_Design.md` (pipeline design).

## Purpose and scope

Detection and annotation policy for **spoken PII in user-uploaded audio or the
audio track of video**, producing time intervals that can be replaced with a beep.

Out of scope: visual PII; identifying someone from voice/accent (biometric
inference); legal-compliance conclusions for any specific country or contract.

The taxonomy is **jurisdiction-neutral by design**: no category is specific to
one country. Country-specific identifiers (SSN, Aadhaar, NINO, PAN, NRIC, …)
are *examples routed into* neutral categories via Appendix A. The taxonomy is
also **profile-independent**: annotation never depends on whether a category
will be beeped in a given domain profile — that action lives in the Action Matrix.

## Taxonomy versus edge-case rules versus actions

- **Taxonomy:** the controlled labels (`PHONE`, `GOVERNMENT_ID`, …). Changes rarely.
- **Edge-case rules:** inclusion/exclusion/boundary rules deciding whether a label
  applies and exactly what audio is covered. Evolve with evidence, versioned.
- **Actions:** beep vs annotate-only, decided per category per domain profile in
  the Action Matrix — never inside this document.

## Annotation unit

One annotation = one contiguous spoken value (or one combined disclosure block)
that must be handled together.

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

Rules for every annotation:

1. `start_ms`/`end_ms` refer to the original audio, never a normalized transcript.
2. `spoken_text` preserves the words actually spoken; never rewritten as digits/symbols.
3. Multi-label goes on ONE interval (e.g. `["CHILD_INFO","FULL_NAME"]`); never
   nested/duplicate intervals for the same audio.
4. The interval covers the entire sensitive value; the safety pad is a renderer
   setting, not ground truth.
5. Every repetition of a value — by the same or another speaker — is its own
   interval. A leak of any one repetition is a leak.
6. If evidence is unclear, set `needs_review: true` and explain; never invent a value.

## Global decision rules

### Evidence and person link

- Detect explicit, intelligible values. Do not infer PII from accent, voice,
  name origin, ethnicity, religion, or background context.
- First-person disclosures establish a person link. Direct contact details,
  addresses, credentials, and personal identifiers may qualify without a nearby
  name unless clearly organization-only. *(Mindy stage-3 rule 4)*
- Exclude values explicitly described as fictional, dummy, sample, test,
  fabricated, or placeholder, and names clearly assigned to fictional
  characters. Do not automatically exclude separate concrete PII-shaped values
  merely because they occur inside a story. On a clearly marked specimen or
  template document read aloud, sample numbers/credentials are excluded; other
  explicitly stated values are evaluated independently. *(stage-3 rules 19–22)*
- **Public figures:** an established public figure's name, referred to in their
  public capacity, is annotated `FULL_NAME` with `public_figure: true`. Their
  private contact details, credentials, private addresses, account data,
  document numbers, and sensitive records still qualify as normal PII.
  Doubt about public status → `public_figure: false` + `needs_review`. *(stage-3 rule 23; action decided in the matrix — currently: not beeped)*

### Exact values and boundaries

- Detect concrete values, not field labels, topics, document names, or
  surrounding explanations. "My passport number", "a PIN", "my address" alone
  are not annotations.
- Beep the smallest spoken interval that fully communicates the value; exclude
  lead-ins ("my number is", "email me at", "she lives at") and trailing
  unrelated words.
- **Masked/partial/suffix-only values are excluded entirely** — not annotated,
  not beeped: "card ending 4832", "last four of my Aadhaar", a masked read-back.
  A complete value spoken with pauses/fillers is NOT partial — it is complete
  and annotated. *(policy decision 2026-09-01; aligns with stage-3 rule 5)*
- A complete combined disclosure block (card number + expiry + CVV spoken
  consecutively) is ONE interval, including intervening labels; never nested
  card/expiry/CVV intervals. *(stage-3 rules 64/75)*
- Prefer the most specific category. Never add a broad category
  (`FINANCIAL_INFO`, `HEALTH_MEDICAL_INFO`, `OTHER`) solely because a specific
  one is present. Exception: `CHILD_INFO` stacks additively (see cat. 26).

### Spoken forms, languages, and transcription uncertainty

- Spoken digits, spelled letters, punctuation words ("at", "dot", "dash",
  "slash", "underscore"), and repetition constructions ("double", "triple";
  Hindi equivalents such as "do baar chhe") count when the complete identifier
  is intelligible.
- Do not normalize ground-truth wording. Detection may normalize a *copy* to
  find patterns, but beeps always come from original word timestamps.
- Language switching is expected and never changes PII status. A value spoken
  half in one language, half in another, is one value.
- If the audio itself is unintelligible, flag for review; label what is
  intelligible, not what a model guessed.

### General exclusions

- Organization-only identifiers, product models, document labels, order
  references, generic serials, routine prices, broad locations, ordinary dates.
- Bank infrastructure values (IFSC, sort code, routing number, SWIFT/BIC) alone;
  inside a transfer block with a personal account number they join that block's
  single interval.
- Category words are not values ("a phone number", "my bank account").

## Controlled taxonomy — 31 categories

### 1. `FULL_NAME`
Identifying natural-person name (normally multi-part; a clearly identifying lone
first name qualifies with context). **Include:** names in personal context;
professionals' names identifying the individual (exclude separable titles
"Dr"/"Professor"). **Exclude/review:** initials, honorifics, generic roles,
organizations, fictional characters. Public figures: annotate + `public_figure: true`.

### 2. `PHONE`
Complete personal phone/fax number, written or spoken form, incl. country/trunk
prefixes and "double/triple" constructions. **Exclude/review:** partial suffixes,
extensions alone, generic service lines, obviously fake numbers.

### 3. `EMAIL`
Complete conventional personal email address incl. unambiguous spoken form
("priya dot anand at outlook dot com"). **Exclude/review:** domain-only,
incomplete, example addresses; a username without a domain (→ `USERNAME_HANDLE`).

### 4. `ADDRESS`
Residence/street address or comparably precise person-linked location.
Core street/box value only; normally exclude city/state/postal code when a core
street address is present. A PIN/postal code spoken as part of a full personal
address joins the interval; alone it is excluded. **Exclude/review:** city,
country, workplace, venue, broad area, place of birth.

### 5. `DATE_OF_BIRTH`
Explicitly stated/labeled birth date. **Exclude/review:** ordinary dates,
issue/expiry dates, age (→ `AGE`), birth year alone.

### 6. `AGE`
Explicit numerical age of a person; numeric expression only, excluding "aged",
"turning", "years old". **Exclude/review:** durations ("3–4 months of pain" is
NOT age — Set-2 evidence), years, grade levels, unattached numbers.

### 7. `PASSPORT_NUMBER`
Complete passport number, any country, person-linked. **Exclude/review:** labels
without values, indistinct document numbers.

### 8. `DRIVERS_LICENSE`
Complete driving licence number, any jurisdiction. **Exclude/review:** vehicle
registration (→ 22), licence type, fragments.

### 9. `GOVERNMENT_ID`
**Any national identity, citizen, voter, social-security, or comparable
government-issued personal identifier, any jurisdiction.** Examples: US SSN,
Aadhaar, UK NINO, Singapore NRIC, voter ID, ration card ID. *(v0.2: absorbs the
former `SSN` category.)* **Exclude/review:** corporate registrations, country
codes, unlabeled values of unclear type, health-service/patient numbers (→ 29).

### 10. `TAX_ID`
Personal tax identifier (personal PAN, TIN, ITIN, UTR, TFN…). **Exclude/review:**
corporate tax numbers, GSTIN and other business registrations.

### 11. `INSURANCE_ID`
Insurance policy/member/subscriber identifier (health or other). **Exclude/review:**
insurer name, plan type, claim topic, unlabeled numbers.

### 12. `STUDENT_ID`
Student/pupil/enrolment identifier. **Exclude/review:** school name (→ 30 if
person-linked), class, grade.

### 13. `LEGAL_CASE_ID`
Court/docket/case identifier clearly linked to a natural person. **Exclude/review:**
court names, legal topics, hearing dates, unlinked case numbers.

### 14. `IMMIGRATION_STATUS`
Explicit personal visa/asylum/refugee/residency/undocumented status. A stated
status only; an immigration *reference number* is `ASSIGNED_ID`. **Exclude/review:**
applications, agencies, visa mentions without a stated status.

### 15. `PAYMENT_CARD_NUMBER`
Complete raw card number; a consecutive card+expiry+CVV disclosure is one
combined interval. **Exclude:** masked values, last-four, expiry alone, CVV
alone, card possession statements.

### 16. `BANK_ACCOUNT_NUMBER`
Complete personal domestic account number or IBAN, incl. unambiguous spoken
form. **Exclude:** masked suffixes, bank names, transfer instructions,
IFSC/sort/routing/SWIFT alone.

### 17. `FINANCIAL_ACCOUNT_INFO`
Concrete personal brokerage/retirement/loan/pension or comparable
financial-account information not better classified as a bank account number.
Includes **UPI IDs / payment-app VPAs** ("ramesh at oksbi"). **Exclude:** bank
infrastructure beside a complete account number (no redundant intervals).

### 18. `FINANCIAL_INFO`
Clearly personal substantive financial fact: salary, income, debt, arrears, tax
liability, assets, balance, completed transaction. **Exclude:** incidental
prices, requested transfers, limits, organization balances, generic money talk.

### 19. `PASSWORD_CREDENTIAL`
Password, PIN, OTP, authentication/recovery code, or other secret value.
**Exclude:** credential labels, reset instructions, placeholder examples; CVV
inside a card block stays in that block.

### 20. `IP_ADDRESS`
Complete IPv4/IPv6 address linked to a person/account/device in context.
**Exclude:** examples, network-range discussion.

### 21. `DEVICE_IDENTIFIER`
MAC, IMEI, hardware serial, advertising ID, UDID with a personal-device link.
**Exclude:** product models, OS versions, org inventory labels.

### 22. `VEHICLE_REGISTRATION`
Licence plate / registration number / VIN linked to a person. **Exclude:**
vehicle models, incomplete fragments.

### 23. `HEALTH_MEDICAL_INFO`
Identified person's explicit diagnosis, condition, significant symptom,
disability, medication, treatment, clinical test/result, genetic fact, or blood
type. First-person disclosures and patient-specific records establish the link.
Span the shortest phrase expressing the medical fact; exclude lead-ins ("I
have", "diagnosed with"). **Exclude:** generic words (patient, clinic, doctor,
appointment, treatment as a topic); casual diet/stress/feeling-unwell talk;
bare allergens or food items outside genuinely clinical context. *(stage-3
rules 37–44 + rule 12.)* Action is profile-dependent (annotate-only in
`medical_consult`; beeped in `strict`).

### 24. `BIOMETRIC_ID`
Biometric or genetic-kit identifier value. **Exclude:** the sound of a voice,
generic biometrics discussion, non-identifying traits.

### 25. `SENSITIVE_DEMOGRAPHIC`
Explicitly stated religion, ethnicity, caste, sexual orientation, gender
identity, political opinion, or union membership of a person. **Exclude:**
inference from name/accent/pronouns/location; isolated form markers ("M"/"F").

### 26. `CHILD_INFO`
Concrete personal information explicitly linked to an identifiable child/minor.
**Stacks additively:** apply alongside the specific label on the same interval
(`["CHILD_INFO","FULL_NAME"]`). **Exclude:** generic mentions of children,
classes, schools, custody without a child-specific value.

### 27. `NATIONALITY`
Explicit person-linked nationality/citizenship/origin statement. **Exclude:**
country codes, generic country references, places of birth, document boilerplate.

### 28. `USERNAME_HANDLE` *(new in v0.2)*
Complete personal social-media/platform handle or username, written or spoken
("at priya underscore anand", "my insta is …"). **Exclude:** org/brand accounts,
incomplete fragments, display names that are just the person's name (→ FULL_NAME
rules), an email address (→ EMAIL).

### 29. `ASSIGNED_ID` *(new in v0.2)*
Person-assigned organizational identifier: patient ID, medical-record number,
health-service number (NHS-type), employee ID, customer ID, membership number,
application or immigration-reference number. Must be a concrete value clearly
assigned to a person. **Exclude:** claim/order/workflow/document numbers,
org-only IDs, generic serials. *(Takes the load Mindy's gold misfiled into
OTHER; no reviewer note required.)*

### 30. `ORGANIZATION_PERSON_LINKED` *(new in v0.2)*
Named clinic, hospital, employer, school, or provider **when context ties it to
a person's identity** ("I work at …", "my doctor at Apollo Jayanagar", the
patient's school). Explicitly includes the person's insurer, TPA (third-party
administrator), lender, or service company handling THEIR case ("my claim with
Medi Assist", "the loan from Orix") — clarified 2026-09-01 (D8) after detector
under-application; the definition itself is unchanged. Matches the incumbent
process's `<ORG REDACTED>` behavior.
**Exclude:** organizations discussed generically, brands, public institutions
mentioned without a person link, and bare bank/company names with no person
link in context.

### 31. `OTHER`
A concrete, persistent, person-linked identifier or sensitive datum with **no**
more specific category — now rare by design (most former uses belong in
`ASSIGNED_ID`). Requires a reviewer note; repeated patterns must be proposed as
a new versioned category.

## Appendix A — Canonical ID routing table

Fixes the instability observed in the Mindy pilot (same UK NI number coded three
different ways). This table is normative; extend it by version bump only.

| Identifier (examples) | Category |
|---|---|
| US SSN, Aadhaar, UK NINO, SG NRIC, voter ID, ration card ID, national ID | `GOVERNMENT_ID` |
| Personal PAN, TIN, ITIN, UTR, TFN | `TAX_ID` |
| GSTIN, EIN, corporate tax/registration numbers | excluded (organization-only) |
| Passport number (any country) | `PASSPORT_NUMBER` |
| Driving licence (any jurisdiction) | `DRIVERS_LICENSE` |
| NHS number, patient/medical-record number, hospital UHID | `ASSIGNED_ID` |
| Insurance policy/member/subscriber number | `INSURANCE_ID` |
| Employee/customer/membership/application number | `ASSIGNED_ID` |
| Immigration/asylum reference number | `ASSIGNED_ID` (+ `IMMIGRATION_STATUS` only if a status is stated) |
| Bank account number, IBAN | `BANK_ACCOUNT_NUMBER` |
| UPI ID / VPA ("name@oksbi"), brokerage/retirement/loan/pension refs | `FINANCIAL_ACCOUNT_INFO` |
| IFSC, sort code, routing number, SWIFT/BIC — alone | excluded (bank infrastructure) |
| IMEI, MAC, serial, UDID, advertising ID | `DEVICE_IDENTIFIER` |
| Licence plate, VIN | `VEHICLE_REGISTRATION` |
| Social handle, gamer tag, platform username | `USERNAME_HANDLE` |

## Appendix B — Audio and India-specific edge cases (required test set)

All v0.1 audio test cases remain (digit-by-digit numbers, double/triple,
spelled letters, spoken email punctuation, natural dates/addresses,
language switching, accents/noise/phone quality, overlapping speakers,
ASR near-misses, beep boundary errors, deliberately fake/partial values,
multi-category values, PII-free clips full of ordinary numbers). Added:

1. Hindi/Indic spoken numbers: "ek do teen chaar…", Devanagari digits in
   transcripts, mixed-language digit strings ("nine eight सात six").
2. Hindi repetition constructions ("double aath", "do baar chhe").
3. +91 and 10-digit mobile forms; landlines with STD codes.
4. 6-digit PIN codes — inside a spoken full address (annotated as part of it)
   vs alone (excluded).
5. Spoken UPI addresses ("ramesh at oksbi", "priya at ybl").
6. Aadhaar spoken in 4-4-4 groups; PAN spoken letter-by-letter (ABCDE1234F).
7. Code-switched value split across languages mid-value.
8. A value interrupted by fillers/pauses and resumed ("nine eight … haan …
   seven six") — one value, one interval.
9. Same value repeated by another speaker (echo) — two intervals.
10. Overlapping speech where only one speaker utters PII — on mixed audio
    (mono/dual-mono) the interval unavoidably beeps every voice in the window
    (voices are summed into one track and cannot be separated); speaker
    attribution is still recorded. On split-channel recordings (one speaker
    per channel) only the offending speaker's channel is beeped and the other
    channel is preserved — both behaviors are required test cases.
11. Medical consultations: disease/medicine mentions annotated
    `HEALTH_MEDICAL_INFO` (action depends on profile), patient IDs `ASSIGNED_ID`,
    clinic names `ORGANIZATION_PERSON_LINKED`.

## Policy decisions — resolved and open

Resolved (see Action_Matrix_v1.0.md for the full log): public figures not
beeped (flagged + audited); handles are `USERNAME_HANDLE`, beeped; masked/partial
values excluded entirely; SSN folded into `GOVERNMENT_ID`; profiles mechanism
with `strict` and `medical_consult`; sensitive-tier actions are per-profile.

Open: beep pad duration (renderer default 250 ms pending listening tests);
confidence thresholds and review routing; additional domain profiles.

## Change-control rule

Unchanged from v0.1: no taxonomy/rule change solely because a model performs
poorly. A change requires a real example, the proposed rule, expected FP/FN
impact, reviewer approval, updated test cases, and a version bump.

## Clarifications (post-v0.2, same version — examples only, no rule changes)

- 2026-09-01 (D8): `ORGANIZATION_PERSON_LINKED` examples extended to the
  person's insurer/TPA/lender (judge-evaluation evidence: detector caught
  hospitals but missed Orix / Medi Assist / Ayu Health, all person-linked
  under the existing definition). `FINANCIAL_INFO` example note: values count
  in any language or script (Telugu word-form amounts were being missed).

## Changelog v0.1 → v0.2 (2026-09-01)

1. **Removed `SSN` as a top-level category; `GOVERNMENT_ID` now explicitly
   covers social-security-type national IDs.** Rationale: jurisdiction
   neutrality (user decision); SSN was the only country-specific code; Mindy's
   own data showed per-country routing was unstable.
2. **Added `USERNAME_HANDLE`.** Rationale: user decision (must-beep); Mindy left
   handles unresolved (gold flagged the question, judges disagreed).
3. **Added `ASSIGNED_ID`.** Rationale: Mindy's gold misfiled patient/member/
   immigration-reference numbers into OTHER; medical-consult data makes patient
   IDs common; OTHER returns to a true discovery label.
4. **Added `ORGANIZATION_PERSON_LINKED`.** Rationale: incumbent process
   redacts `<ORG REDACTED>` in medical calls; person-linked orgs are
   quasi-identifying.
5. **Masked/partial values excluded entirely** (was: annotate-with-flag
   ambiguity). User decision, matching Mindy stage-3 rule 5.
6. **Public-figure handling made explicit** with `public_figure` flag
   (annotate, don't beep; private details still beeped; doubt → beep).
7. **Imported Mindy stage-3 refinements**: fictional/specimen carve-outs,
   combined card blocks, health span boundaries + non-clinical exclusions,
   additive CHILD_INFO, no financial double-coding, OTHER constraints.
8. **Added Appendix A routing table and Appendix B India/audio edge cases.**
9. **Actions moved out of the taxonomy** into the Action Matrix with domain
   profiles (`strict`, `medical_consult`).
