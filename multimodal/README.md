# Multimodal document PII redaction

Removes personal information from **documents** — PDFs, scans, images, and the
page-based formats that behave like them. Sibling to the audio arm
(see [`../docs/audio.md`](../docs/audio.md)); the two share transport and model
plumbing and nothing else.

> **The output is redaction only.** Nothing is ever written into the page — no
> pseudonym tokens, no coarsened values, no placeholders. A redacted document
> that contains text the original never had is making claims of its own, and
> every such claim is a new way to be wrong. This is enforced as an invariant,
> not a convention: `actions.added_text(before, after)` must be empty for every
> page, and the verifier fails the run otherwise. The one exception is opt-in:
> `--replace` (below) writes a short code on each removed value's box and a legend
> after the last page; those codes are then the only text the output gains, and
> none of them holds a value.

## Why it is separate from the audio arm

The audio pipeline's action matrix has exactly two outcomes — beep it, or leave
it. That is the right vocabulary for a phone number in a call recording. Applied
to a 2,000-page claim bundle it can only destroy or ignore, so every judgement
collapses to "black box it", and the first document deliverable was rejected for
being simultaneously leaky and destroyed — whole pages blacked out, a lab
report with every test result gone, and identity still visible.

So this arm owns its own taxonomy layer. `pipeline/policy.py` — the audio
taxonomy — is not touched by anything here.

## Two questions, answered separately

**Which engine can read this file?** — `router/registry.py` maps 60 extensions
plus magic bytes onto 11 modalities (audio, video, document, image, spreadsheet,
presentation, cad, email, text, archive, unknown). Seven of those route here.

**What counts as PII in it?** — `taxonomies/`, a registry of self-describing
packs. These are orthogonal: the same engine serves a claim bundle and a CAD
drawing, and the same taxonomy could arrive as a PDF, a spreadsheet or an email.
Conflating them is what produced the original failure.

## Quick start

    # the dispatcher: routes, or takes an explicit arm/taxonomy
    python3 router/dispatch.py INPUT.pdf                   # auto
    python3 router/dispatch.py INPUT.pdf --insurance       # force a pack
    python3 router/dispatch.py INPUT.pdf --generic
    python3 router/dispatch.py INPUT.mp3 --audio           # hand off to the audio arm
    python3 router/dispatch.py INPUT.pdf --dry-run         # show the decision only

    # or call the document pipeline directly
    python3 document_pipeline/run_document.py INPUT.pdf --out OUT.pdf

    # force a taxonomy
    python3 document_pipeline/run_document.py INPUT.pdf --out OUT.pdf \
            --taxonomy generic_document

    # what packs exist
    python3 taxonomies/registry.py --list

Outputs beside `OUT.pdf`:

| file | contents |
|---|---|
| `OUT.pdf` | the redacted document |
| `OUT.audit.jsonl` | every box: page, action requested vs applied, category, source |

## How a page is processed

1. **Read** — real text layer where there is one, Apple Vision OCR otherwise
   (`readers.py`). No install: Vision is driven through `swift -`.
2. **Detect** — the pack's own prompt, over a numbered token stream. The model
   returns token indices, never coordinates; geometry always comes from the
   stream (`detect.py`).
3. **Localise** — one box per *contiguous* run. A span whose tokens sit on
   different lines unions into a rectangle that swallows everything between
   them, which can black out most of a page (`geometry.py`).
4. **Claim what no label claims** — a letter's recipient block carries no
   colon and no keyword, so nothing anchors it. `unlabelled_identity_blocks()`
   finds short contiguous line groups that read as a name and address, and
   vetoes the issuer's letterhead by testing the **first line only** (an
   organisation leads with its own name; "opp. City Hospital" in an address's
   third line must not veto anything).
5. **Anchor what OCR cannot read** — at 109 dpi a handwritten name arrives as
   `S OP N A I`. Two fallbacks: printed **labels** say where a value must be
   (`formfields.py`), and **chromatic ink** segments blue pen from the printed
   template (`handwriting.py`). Ink governs where it works; labels are the
   fallback for greyscale, bitonal and born-digital pages.
6. **Objective pass** — QR by Apple Vision decode *and* by finder-pattern shape
   (the two fail differently), faces, ID portraits, cheque signature zones
   (`objective.py`).
7. **Propagate** — a value learned anywhere in a bundle is struck everywhere,
   matched OCR-tolerantly (`propagate.py`, `matching.py`).
8. **Redact** — pixels destroyed and glyphs deleted in one pass (`redact.py`).
   Verification is a separate, opt-in step (below), not part of a default run.

Scanned pages also get passes that find candidates from **pixels** and let a
model **classify** them — the model still never emits a coordinate:

- **Stamps, seals and signatures** (`carriers.py`): coloured-ink candidates,
  classified before anything is cut, so blue-pen clinical notes survive. A
  date inside a stamp is carved back out.
- **Residual marks** (`inkmarks.py`): dark marks no reader explained (black
  signatures, greyscale logos), shown to the classifier with a red outline,
  their size and position, in context.
- **Charting grids** (`inkmarks.py`): medication and nursing charts are found
  from their ruled lattice; each cell's pen entry (a name over a staff number)
  is judged whole, twice, by a chart-specific prompt.
- **Sign-off columns** (`formfields.identity_columns`): table columns headed
  "Signature", "Administered by", "Verified by" … on leaning, faint or wrapped
  headers; a column of printed sentences is never one.
- **Text no reader read** (`unread.py`): Vision text regions plus lines found
  in the pixels (Indic scripts Vision cannot read), classified as personal or
  not; a region inside a field the pack keeps (bank name, IFSC) defers to it.
- **Small embedded marks** (`marks.py`): logos placed as their own images on
  born-digital pages; bank marks are kept.

The letterhead band stops at a table's first ruled border — unless an identity
label ("Name of the Patient", "Age") sits in the freed area.

If the GPT detector's provider refuses the account (no credit), detection
switches to OpenAI directly (`openai_direct.py`, needs `OPENAI_API_KEY`); if
GPT is down altogether the second detector carries the page, marked
single-detector. If both fail, the fixed rules (ID patterns, labels) still run.

## Writing a custom taxonomy

A pack is a **directory, not code**. Adding one is a data change.

    taxonomies/
      my_pack/
        pack.json        categories, action matrix, selection signals
        prompt.txt       detection guidance (the category table is generated)
        hooks.py         optional, only if the pack needs custom logic

### `pack.json`

```json
{
  "id": "hr_records",
  "version": "1.0.0",
  "title": "HR and personnel files",
  "extends": null,
  "conservative": false,
  "actions": ["REDACT", "PSEUDONYMIZE", "GENERALIZE", "MASK_PARTIAL", "KEEP"],
  "signals": {
    "hr_vocabulary": {
      "weight": 2.0, "cap": 4,
      "any": ["offer of employment", "payroll", "national insurance",
              "probation", "line manager", "grievance"]
    }
  },
  "categories": [
    {
      "id": "H01",
      "name": "EMPLOYEE_NAME",
      "action": "REDACT",
      "enforcement": "label_anchored",
      "role": "EMPLOYEE",
      "definition": "IN: the employee's name wherever it appears. OUT: the line manager (H02).",
      "corpus_examples": "Candidate name; Employee name; Signed by",
      "owner_scope": "EXPLICIT",
      "confidence": "SETTLED",
      "note": "one line on why this action, and what it costs"
    }
  ]
}
```

**Actions** — `REDACT`, `PSEUDONYMIZE`, `GENERALIZE`, `MASK_PARTIAL`, `KEEP`.
All four destructive ones currently render identically (an opaque black box);
they differ in the audit record and the token map, not on the page. When a span
carries several categories the **most protective** action wins, and a category
the pack does not define does **not** fall through to `KEEP` — it fails closed to
`REDACT`, because over-redaction is visible and recoverable while a leak is
neither.

**Enforcement** — name the mechanism that will actually carry the row:
`checksum`, `label_anchored`, `handwriting_ink`, `value_propagated`,
`objective_visual`, `model_judged`. Prefer anything over `model_judged`: it is
the only one that varies page to page, and it is the mechanism behind every
"why was this redacted here but not there" complaint. In `insurance_medical`
exactly one row of forty is model-judged.

**`extends`** lets a tenant pack inherit another and override a few rows rather
than restate forty:

```json
{ "id": "insurance_acme", "extends": "insurance_medical",
  "categories": [ { "id": "T34", "name": "CLAIM_FINANCIAL_AMOUNT",
                    "action": "REDACT", "enforcement": "label_anchored",
                    "definition": "Acme keeps no amounts." } ] }
```

### Selection

Deterministic and logged at every step:

1. `--taxonomy <id>` always wins.
2. Otherwise each pack scores its declared `signals` against text sampled from
   the document (a page is OCR'd if there is no text layer).
3. Ambiguity — nothing scores, or the top two are within a margin — falls to the
   pack flagged `"conservative": true`, **never** to the more permissive one.
   Guessing permissive leaks; guessing conservative over-redacts, which shows up
   in the over-redaction metric and is fixed by a re-run.

The chosen pack, its score and its basis are printed and recorded.

### Mechanisms speak semantics, not your ids

The label lexicon, ink zoning and the objective visual pass emit **semantic
keys** (`sem:person_name`, `sem:clinical`, `sem:event_date`), and each pack maps
them to its own rows through a `"semantics"` block in `pack.json`. Hard-coding
one pack's ids in a mechanism is a correctness bug, not a style one: under any
other pack those ids are unknown, `action_for` fails closed to `REDACT`, and
every KEEP rule **inverts** — clinical content, dates, bank names and amounts
all destroyed the moment selection falls back to the conservative pack.

### Adding label vocabulary — and why it is pack-scoped

Label vocabulary lives in **two** places, and the split is the isolation
boundary between packs:

| where | scope | holds |
|---|---|---|
| `document_pipeline/formfields.py` → `LEXICON` | **shared, 29 rows** | labels that mean the same on any document: name, address, geo, contact, DOB, age/sex, bank rows, government IDs, signature |
| `pack.json` → `"lexicon"` | **that pack only** | domain vocabulary — 11 rows for `insurance_medical`, 9 for `generic_document` |

```json
{ "id": "hr_records",
  "lexicon": [
    ["^(joining|probation\\s*end)\\s*date$", "sem:event_date", false],
    ["^(gross|net)\\s*salary$",                "sem:person_money", false]
  ] }
```

A pack's rows are compiled **only while that pack is selected**, and they are
matched *before* the shared base so a domain pack can override a base reading.
`extends` **appends** a parent's lexicon rather than replacing it.

This was not the original design, and the reason it changed is worth stating.
`LEXICON` used to be global. A bare `patient` row added to help a *generic*
clinic letter then matched the end of charge-line descriptions on a real
hospital bill — `Amount Received From Patient  0.00`, `Patient Payable
156559.00` — and because a label's answer area is whatever sits to its right,
three bill totals were destroyed by a row that had nothing to do with bills.
Scoping makes that impossible rather than merely unlikely: the row does not
exist during another pack's run. `synth/regress.py` asserts it.

OCR repair (`_fix_token`) is scoped the same way, so `Dicherge → Discharge`
works under `insurance_medical` and is correctly unknown to the base.

Three traps worth knowing, all learned the hard way:

* **Bare words are not field labels.** `cell` matched "Red Blood Cell",
  "Packed Cell" and "Red Cell (RDW)" on a haematology report and blacked out
  three results rows. `surgeon` and `GST` matched *charge lines* on a bill and
  took the amount beside them. `patient` matched a bill total. Require a
  disambiguating suffix.
* **A one-word label inside a sentence is a coincidence.** `IN THE CITY CIVIL
  COURT` yielded a `City` field whose answer area ate the court's identity and
  the hearing date. `_is_prose_line` rejects single-word matches on lines that
  read as prose; a colon anywhere on the line vetoes that rejection, because a
  badly-scanned form row still names a field.
* **Regions must not enter data columns.** `numeric_columns()` finds x-ranges
  where figures stack across three or more rows; `numeric_rows()` finds the
  rows those columns share. No field region may enter a column, and no *money*
  or *content* reading on a figures row survives — that is how a bill keeps
  both its amounts and its line-item descriptions.

### OCR repair must not rewrite real words

`_fix_token` snaps a token to the nearest lexicon word within one or two
edits. Left unguarded it rewrites ordinary English into field labels, and each
rewrite invents a field that eats the line beside it — `case → care`,
`Estate → State`, `Other → Mother`, `Password → Passport`. The system
dictionary is consulted first, because a correctly-spelled word of the
language is not an OCR misread. `NEVER_FIX` holds the domain words it must leave
alone, and a candidate that is merely a **prefix or suffix** of the input is
rejected outright — real OCR damage substitutes characters in the middle, it
does not cleanly truncate. Vocabulary is harvested from the patterns'
*literal* words, never from fragments a character class splices (`licen[cs]e`
contains no word `licen`).

## `--second-detector` — on by default

    # the default: two model families, findings unioned
    python3 document_pipeline/run_document.py IN.pdf --out OUT.pdf

    # one detector only (evaluation, or no credentials for a second family)
    python3 document_pipeline/run_document.py IN.pdf --out OUT.pdf \
            --second-detector none

**A single detector over a scanned page can return nothing at all** — no
exception, no log line, and `--verify` still passes, because verification
checks that *detected* values were removed and cannot know about values never
detected. The page ships unredacted and nothing says so.

Measured on one real 29-page claim bundle:

| | one detector | union |
|---|---|---|
| pages with **zero** model findings | 2 of 29 | **0** |
| identity values still readable in the output | 5 of 10 | **1 of 10** |

Every one of those five leaks sat on a single page that happened to get no
detection that run. Two model families do not fail on the same page, which is
why the union closes it and a re-run does not.

It only ever ADDS findings, so it cannot remove a box the primary detector
drew and does not increase over-redaction. It roughly doubles detection spend
and needs credentials for a second family. If that detector is unavailable the
page falls back to the primary alone, says so loudly, and records
`second_detector_degraded` in the audit — it can never take the primary's
findings down with it.

## Verification, and the correction loop

Both are **opt-in**, like the audio arm's `--assurance` and `--remediate-audit`.
A default run redacts and stops.

    # re-read the finished PDF and report anything that survived
    python3 document_pipeline/run_document.py IN.pdf --out OUT.pdf --verify

    # ...and re-redact whatever survived, then check again
    python3 document_pipeline/run_document.py IN.pdf --out OUT.pdf \
            --remediate --max-rounds 3

`--verify` re-opens the output and looks for every value the run claimed to
destroy — inside its own box, in the raw object streams, and in the pixels. It
exits **non-zero** when the mechanism failed, so a script or a batch job sees
what a person reading the output sees.

`--remediate` closes the loop: survivors are handed back as findings, the
document is redacted again, and the check repeats until it is clean or
`--max-rounds` is reached. This is the document counterpart of `edge_verify`,
which extends a beep and re-renders until the value is no longer audible.

Two properties worth knowing:

* **It can only add.** A correction pass draws more boxes; it never removes
  one. The worst case is over-redaction, which the column and row guards
  bound — not a new leak.
* **It is bounded.** A round that fixes nothing stops immediately. If
  re-redacting the same box did not remove the value, a third attempt will not
  either, and the honest outcome is to report it rather than spin.

## Code-based replacement (`--replace`)

Opt-in. The redaction is unchanged and finishes first (with `--verify` /
`--remediate` if given); then every value the run removed gets ONE short code
for the whole file, written on a small white label over its black box:

    python3 document_pipeline/run_document.py IN.pdf --out OUT.pdf --replace \
            --codes-sidecar KEYS/IN.codes.json

| Code | Stands for |
|---|---|
| `Pt1`, `Prop1`, `Per1`, `Dr1` | the patient, the insured or proposer, another person, a clinician |
| `Hosp1`, `Ins1` | a provider; an insurer or TPA |
| `HPID1 ...4821`, `Pol1 ...4410`, `Aad1 ...3312`, `PAN1 ...674P`, `ID1`, `Acct1`, `Reg1`, `Emp1`, `Doc1`, `Dev1` | identifiers by kind, with only the last four characters |
| `Ph1`, `Em1`, `Addr1`, `DOB1` | a phone, an email, an address, a date of birth |
| `(v2)`, `(v3)` after a code | the same value written another way (initials, word order, a misread letter or digit) |
| `[sig]`, `[nm]`, `[age/sex]` | a signature, a name in another script, an age / sex / relationship field |

* **One code per value.** The same person, organisation or identifier gets the
  same code on every page; which boxes share a code is decided by the token
  store (OCR-tolerant for people and identifiers, by core word for
  organisations; a name in running prose takes the code of the person it
  matches). Codes are numbered per family in order of first appearance.
* **Nothing readable is written.** A label shows no value beyond an
  identifier's last four characters; a label that would is not written. A box
  too small for a 7 pt label, a face, QR code, stamp, logo, pen-ink region,
  any box whose value no detector read, and every box on a rotated page stay
  plain black.
* **A legend** (code = the kind of value it stands for, never the value) is
  appended after the last page, so page numbers do not change.
* **The key stays apart.** The code -> real value key is written to
  `--codes-sidecar` (default `<workdir>/<file>.codes.json`). It is a
  re-identification key: store it apart from the output, never ship it with it.

## Visual leftover pass (`--visual-verify`)

Opt-in. `--verify` reads the TEXT of the output; it cannot see a signature,
stamp, logo, seal or ID photo only partly covered, handwriting that runs past a
box, or a name the detector never read. This pass looks at every page of the
finished output as a person would and blacks out what is still readable:

    python3 document_pipeline/run_document.py IN.pdf --out OUT.pdf \
            --visual-verify --visual-rounds 2 --visual-spend-cap 100

| Step | What happens |
|---|---|
| find | a vision model (`--visual-model`, default Claude Opus 5.5 via OpenRouter with data collection off) sees the whole page on a lettered grid and lists every item the pack's rules say must go that is still readable |
| fine | per item, a zoomed crop on a fine grid; the model lists every cell the item touches |
| cover | each connected run of cells becomes one box, tightened to the ink inside (faint stamps and show-through count), so a stroke across several cells is covered whole |
| verify | the model sees the proposed boxes and vetoes any that would also destroy kept content (clinical text, amounts, episode dates, printed labels) or that miss the item; a vetoed run falls back to per-cell boxes, verified the same way |
| check | find again on the fixed page; at most `--visual-rounds` fix rounds, then one last look that only reports |

New boxes are true redactions placed in the page's unrotated space, so they
land correctly on rotated pages; the pass can only add boxes. A per-page report
(`<workdir>/<file>.visual_verify.jsonl`) lists every box and anything still
seen after the last round. It runs before `--replace`, so its boxes stay plain
black. Cost: about one vision call per page, plus a crop and a check per item
found; it reduces what survives, it does not prove zero.

Every claim is checked against the artifact, never asserted:

| check | what it proves |
|---|---|
| residual text inside its own box | the value is gone from the text layer |
| box renders ≥90% black | the pixels were destroyed, not covered |
| `added_text(before, after)` empty | nothing was written into the page |
| QR re-detected by decode **and** by shape | no symbol survives |
| faces re-detected | no portrait survives |
| cosmetic control **must fail** | the verifier is measuring something real |

That last one matters most: a `--cosmetic` run draws opaque rectangles and
changes nothing underneath. It looks identical in every viewer. If it ever
passes verification, the verifier is broken and no result from it means
anything.

## Regression suite

    python3 synth/edge_cases.py --out out/edge        # edge corpus
    python3 synth/generic_cases.py --out out/generic --scanned
    python3 synth/regress.py --quick                 # no model calls
    python3 synth/regress.py                         # full
    python3 synth/score_generic.py --dir out/generic --run

34 checks across both arms: module self-tests, the audio taxonomy and shared
modules, pack behaviour for **both** packs (KEEP rules, precedence,
fail-closed, `extends`, conservative fallback, **lexicon isolation**, no
unmapped semantic key), eleven edge-case documents, and the negative control.
Cases that genuinely cannot be decided report **INCONCLUSIVE** rather than
green — an assertion that cannot fail is worse than no assertion.

The edge corpus covers what the insurance data never produces: pages rotated 90
and 270 degrees, an invisible OCR layer over a scan, a bitonal fax, black
ballpoint (invisible to the chromatic channel), an empty page, a blank form, a
guidance sheet, a bill with numeric columns, a diagonal watermark, and a 3400pt
page.

## Synthetic corpus

`synth/` generates documents **with ground truth**, because we place every value
and therefore know its category and exact box. That is the only way to get TRUE
recall rather than a judge's estimate.

    python3 synth/make_corpus.py --out out/synth
    python3 synth/score.py --dir out/synth

Identifiers are generated through the pipeline's **own** validators, so a
synthetic Aadhaar passes Verhoeff and a synthetic card passes Luhn — otherwise
the checksum path never fires and its measured recall is fiction. Each document
ships born-digital *and* rasterised to a degraded scan, targeting the real
the resolution range real scans arrive at.

`synth/assets/` is **gitignored on purpose**: the QR and portrait used there are
real rasters lifted from a source document, and the QR decodes to a real
identity. Harvest your own locally; the generator runs without them.

## The two packs

Same engine, same page, different answers — decided entirely by which pack the
run binds. Where they differ:

| value | `insurance_medical` | `generic_document` |
|---|---|---|
| clinical content | **KEEP** | **REDACT** |
| date of birth | KEEP | GENERALIZE |
| organisation contact (`info@`, front desk) | REDACT | **KEEP** |
| organisation registration (GSTIN, CIN) | REDACT | KEEP |
| personal money (salary, balance) | REDACT | MASK_PARTIAL |
| minor / child flag | GENERALIZE | REDACT |
| document serial | MASK_PARTIAL | KEEP |

An adjudicator cannot work without the clinical record, so `insurance_medical`
keeps it and destroys every carrier of the treating provider's identity. A
general-purpose document gives no such justification, so `generic_document`
redacts health content and keeps the issuing organisation, which is how a
reader knows where a document came from.

Everything else agrees: names pseudonymised in every role, addresses and
contact destroyed, government IDs and cards destroyed, branch banking details
kept under both, event dates kept under both, unzoned pen ink kept under both.

`generic_document` mirrors the audio taxonomy's labels and adds the visual
categories audio scopes out — faces, signatures, machine-readable codes,
burned-in identifiers, handwriting, container metadata — plus KEEP rows that
exist to protect document utility: numeric tables, empty fields, instructional
text, page furniture, issuing organisation.

## Known limits

A page-spanning **watermark** is invisible to OCR and survives. **ID photos**
are only found on cards whose QR decodes as an identity document.
**Handwriting on greyscale or black-pen pages** has no chromatic signal and
falls back to labelled fields only. Detection on low-resolution scans varies
between runs; `--second-detector` reduces that.

From a blind visual audit of 96 pages (Sep 2026), still open:
- a logo **emblem** beside a wordmark that was cut;
- some doctors' handwritten **sign-offs** on notes pages;
- **mirror-image text** showing through from the back of a scanned sheet;
- blurry **1D barcodes** on patient stickers (Vision cannot decode them; the
  shape detector finds QR codes only);
- text burned into **ultrasound images**; a stamp only partly covered;
- **over-redaction** of dates, clinical values and amounts on bank statements
  and handwritten charts.

Cost, measured: about $0.19 per page per pack (Gemini ~80%, GPT ~20%).
