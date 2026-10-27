# Dataset route — one run, every modality, redact or replace

`multimodal/router/run_dataset.py` takes a whole dataset (a folder of studies,
or one file) and delivers a cleaned copy of it.

```
python3 multimodal/router/run_dataset.py INPUT --mode redact|replace \
    --taxonomy market_research --out DELIVERY --work WORK
```

* `DELIVERY` receives only redacted files, in a mirror of the input tree; folder
  and file names are cleaned too.
* `WORK` holds the ledgers, the mapping, the code book, caches and reports. They
  contain real names and values: never share `WORK`, never put it in a
  repository. Both folders must be outside the repository and outside `INPUT`.
* The insurance route (`router/dispatch.py --insurance`) is separate and
  unchanged.

## Stages

| Stage | What happens |
|---|---|
| 0 inventory | every file gets a route by its type (below); unreadable files are **held** and listed, never copied through |
| 1 detect | the taxonomy pack decides what counts; one ledger per file |
| 2 mapping | replace mode only: ONE mapping for the whole dataset, from every ledger, before anything is written |
| 3 render | the dataset net plus each file's own spans; black boxes / beeps / blur, or invented names |
| 4 verify | every output read back; leftovers or a file that does not open → **held**; a broken control per route |
| 4b audit | a second model family reads a sample of delivered files; findings join the net (and the mapping) and the dataset is rendered again |
| 5 report | `WORK/report.json`, `WORK/report.md` |

## Routes

| Input | Route | Notes |
|---|---|---|
| mp3, wav, m4a, … | audio | the audio arm (`tools/redact.py`) with the pack's `audio_profile.yaml`, `--sweep` and `--assurance` on by default as in `run_all.py` (`--no-sweep` / `--no-assurance` turn them off) |
| mp4, mov, … | video | the video arm (`run_video.py`): sound through the audio arm, frames blurred |
| docx, pptx, xlsx, xlsm | office | body, tables, comments and their authors, tracked deletions, notes, headers/footers, charts and their data, sheet names, cells (shared, inline, numbers), pivot caches, properties, `app.xml` titles, hyperlinks, embedded office files (routed again), images (classified; brand / product / screenshot / person images greyed), thumbnails (always greyed) |
| doc, rtf | office | converted to docx by macOS `textutil`; delivered as `.docx` |
| csv, tsv | text | cell by cell; survey column rules |
| txt, qdf, qtf, stpn, qpsmran | text | line by line; encoding kept (UTF-16 stays UTF-16, CRLF stays CRLF) |
| asc | text | fixed-width data: the runs of text inside each record (open answers, models) are detected; known forms are also searched without word boundaries (fields are glued together); every replacement padded or cut to the column width, removed values become spaces |
| pdf | pdf | text layer read with a box per CHARACTER (a brand glued into another word is boxed alone), or OCR (Apple Vision) on a scanned page; true removal (PyMuPDF redaction); a scanned page is re-rasterised with its boxes; a page whose text sits where redaction cannot reach is flattened; images and faces greyed; metadata blanked |
| png, jpg, … | image | OCR; a brand image or person photo greyed entirely, otherwise boxes |
| zip | zip | members routed, then zipped again under cleaned names |

## Detection (stage 1)

* **Model pass** — the pack's prompt over each DISTINCT text piece once per
  dataset (cached by content in `WORK/text_cache.jsonl`), batched in file order so
  the model sees neighbouring text. A batch whose call fails is retried; a file
  with an undetected piece is held.
* **Patterns** — emails and phone numbers (also phones stored as numbers in a
  sheet).
* **Survey column rules** (`document_pipeline/survey.py`) — a short header that
  names the column puts every value below it under that category: name →
  person; phone / mobile / email → contact (removed); address / pincode / GPS →
  address (removed); device id / uuid → removed; respondent / panel id → code
  `R0001`; interviewer / user / login / `_submitted_by` → code `INT001`. A plain
  running serial (`SERIAL`, 10001, 10002 …) is a row key and is kept.
* **File and folder names** — detected like text (`_` and `.` count as word
  breaks).
* **Audio / video** — the audio arm's detection plus entity discovery over the
  transcripts (both listeners) and on-screen text.

## Mapping (stage 2, replace mode)

Every person and brand-like form in every ledger is grouped into entities
(same phonetic key, Devanagari folded to Latin) and named in batches by
`prompts/dataset_naming_v1.txt`: people get invented Indian names of the same
shape and gender; brands, products, dealers, the agency and slogans get
invented names that fit their category; the invented name replaces the
distinctive CORE only (`Zorvex Electronics Hub` → `Kelmora Electronics Hub`).
`entity_map` refuses an invented name that a real form matches or sounds like;
the refused token is swapped (people: a pool name from
`prompts/name_pool_india_v1.json` that occurs nowhere among the dataset's real
names, the same swap everywhere; brands: a changed ending) until the mapping
loads. Two different entities never share an invented name. Naming sees the
study each entity appears in, so a retailer that competes with the client is
replaced while a mere marketplace is kept.

`--stop-after mapping` pauses after writing `WORK/mapping.yaml`; a later run
with `--mapping FILE` renders without detecting again. The mapping only ever
grows: the leak audit appends names, existing invented names never change.

**One mapping, before any replacement.** A single run over the dataset builds
the mapping from every ledger (documents, recordings, names) before anything is
rendered. A dataset may be split into two runs (`--skip-media`, then
`--media-only`) only SEQUENTIALLY: the second run takes the first run's final
mapping (`--mapping WORK1/mapping.yaml`) and only appends names the first never
saw. Two split runs growing the same base in parallel can give one real name
two invented names.

## Render (stage 3)

| | redact | replace |
|---|---|---|
| people, brands, products, dealers, agency | black box (█ in office/text; filled box in PDF/images); beep in audio; blur in video | the mapping's invented name (spoken by a stock voice in audio) |
| linking ids | black box | consistent codes per study (`R0001`, `INT001`, `ID0001`) |
| phones, emails, addresses, other ids | black box / beep | removed (black box; an empty cell in a survey column) |
| file and folder names | `[BRAND]`-style tags | invented names |

In office files the original characters are deleted from the file and the █
characters written in their place: nothing remains underneath. The **dataset
net** applies every removable form found in ANY file to EVERY file.

## Verify (stage 4)

Each output is read back — joined segment text AND the raw text of every XML
part (namespaces and typography parts excepted), the PDF text layer, OCR of
images and scans, and binary members of office files (names only) — and must
hold no removable form and no email or phone; it must open (zip and XML parse,
PDF and image load, fixed-width line lengths unchanged). In replace mode a
brand is checked by the part the mapping replaces (generic words around it
stay by design). An output with leftovers first gets an automated **repair
pass** (the net applied to the output itself; PDF pages flattened); only if
that fails is the file held. A **broken control** per route (the original
file) must fail the same check.

## Leak audit (stage 4b)

A second model family (`--audit-model`, default GPT) reads a stratified sample
of delivered files (`--audit-files`, default 40) and lists anything that
should have been removed. Findings that really occur in the output join the
net; in replace mode names join the mapping (even ones naming had kept), and
the dataset is rendered again — up to two rounds. Invented names, codes,
boxes, platforms, fonts and software names are not leaks.

## Testing

* `tools/synth_documents.py OUT` — an invented two-study dataset in every
  document format with known answers; `tools/score_dataset.py` scores a run
  (leak rate by kind, values that must survive, mapping coverage, invented
  name present in every file, codes agreeing across files, ledger spans outside
  the answers).
* Audio and video: the synthetic clips of `tools/synth_media.py` /
  `tools/synth_video.py` through the same entry point.
