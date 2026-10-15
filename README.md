# PII redaction toolkit

> **Private repository.** Code only: no data of any kind (audio, transcripts
> or documents, redacted or not) is kept here. Do not mirror or re-share
> outside the team. This code processes recordings and documents that contain
> personal information: handle inputs, outputs and any sidecar files under the
> same rules as the source data.

Two arms over shared transport and model plumbing. They handle different media
and answer to different rules about what counts as PII.

| | [Audio](docs/audio.md) | [Video](docs/video.md) | [Documents](multimodal/README.md) |
|---|---|---|---|
| input | calls, recordings | ads and other video | PDFs, scans, images, page-based formats |
| output | 1 kHz beep (or, opt-in, a stock voice saying an invented name for a mapped brand) + redacted transcript | redacted mp4 (sound via the audio arm; logos, brand text, faces, numbers blurred) + shot-by-shot script | redacted PDF; **nothing written into the page** |
| taxonomy | fixed: 31 categories, 2 actions; optional custom profile file and name mapping | as audio, plus every logo and face | pluggable **packs**, 5 actions |

## One entry point

    python3 multimodal/router/dispatch.py INPUT            # route automatically

Or name the pipeline directly — routing is inference, and when you already know
what the file is, saying so is better:

    python3 multimodal/router/dispatch.py call.mp3  --audio
    python3 multimodal/router/dispatch.py claim.pdf --insurance
    python3 multimodal/router/dispatch.py notes.pdf --generic

    # long forms
    --pipeline audio
    --pipeline document --taxonomy insurance_medical
    --pipeline document --taxonomy generic_document

    # print the decision and the exact command, run nothing
    python3 multimodal/router/dispatch.py INPUT --dry-run

    # arm-specific flags pass through after `--`
    python3 multimodal/router/dispatch.py call.mp3 --audio -- --sweep --assurance

A forced choice is recorded as `explicit_override` rather than a signal score,
so a later reader can tell which decisions a person made and which the router
made. Auto-routing sends audio to the audio arm and video to the video arm
(`--video`; `--audio` on a video file delivers its sound only); documents, images,
spreadsheets, presentations, CAD, email and text to the document arm, where a
taxonomy pack is selected from the document's own signals.

To inspect a routing decision alone:

    python3 multimodal/router/analyze.py path/to/input

## Audio — [`docs/audio.md`](docs/audio.md)

    python3 tools/redact.py path/to/audio.mp3 --sweep --assurance   # one file
    python3 run_all.py --input folder/ --result result/              # a dataset

Transcribe → detect → beep, behind signal QC, a beep-edge listen check and an
independent judge audit. `--assurance` adds a second independent listener;
`--sweep` runs four narrow detection passes instead of one. `run_all.py` adds
dataset dedupe and per-delivery metadata around the same pipeline, with both
flags on by default. The audible watermark is a separate step
(`tools/watermark.py`) applied to bundles before they are shared.

### Named entities and invented names (opt-in)

    python3 tools/redact.py group1.mp3 --sweep --assurance \
        --mapping study_mapping.yaml --replace pseudonym --profile-file p.yaml

A mapping of real names (brands, products, dealers, organisations, an agency)
to invented ones turns on their detection in any language or script, a
mapped-entity audit with a listen check on the delivered audio, and
pseudonymised output names. `--replace pseudonym` speaks the invented name in
a stock synthetic voice (never a clone) where a mention is confirmed; personal
data stays beeped. Default runs are unchanged. See
[docs/audio.md](docs/audio.md#named-entities---mapping-and---replace-pseudonym).

## Video — [`docs/video.md`](docs/video.md)

    python3 multimodal/video_pipeline/run_video.py AD.mp4 --mapping m.yaml \
        --replace pseudonym -- --sweep --assurance

Sound through the audio arm; on screen, every logo, brand mark, brand text,
face, phone number and URL is blurred with tracking across frames (invented
text plates on static shots in pseudonym mode); verified on unsampled frames
with a broken control that must fail; plus a shot-by-shot script.

## Documents — [`multimodal/README.md`](multimodal/README.md)

    python3 multimodal/document_pipeline/run_document.py INPUT.pdf --out OUT.pdf

Read (text layer or Apple Vision OCR) → detect against a taxonomy pack →
localise on a deterministic token stream → redact pixels and glyphs.
Handwriting OCR cannot read is anchored by printed
labels and by chromatic pen ink; QR codes are found by decode *and* by
finder-pattern shape.

Three opt-in flags, each the document counterpart of an audio one:

    --second-detector none     turn OFF the second model family (it is ON by
                               default: one detector can silently return
                               nothing on a scanned page, shipping it
                               unredacted with a clean verify)
    --verify                   re-read the finished PDF and report anything
                               that survived; exits non-zero if it did
    --remediate                as --verify, then re-redact the survivors and
                               check again, up to --max-rounds

`--remediate` is the document equivalent of the audio arm's beep-edge loop: it
can only add boxes, never remove one, and it stops as soon as a round fixes
nothing.

**The taxonomy is data, not code.** A pack is a directory — `pack.json` plus a
prompt — so a new domain, jurisdiction, tenant or downstream task is a new
pack, not a rewrite. Two ship: `insurance_medical` and `generic_document`.
Writing your own is documented in the
[document README](multimodal/README.md#writing-a-custom-taxonomy).

**One run binds one pack.** The selection happens once per document, is printed,
and is recorded in the manifest. Packs are isolated: rules cannot cross, and
pack-owned label vocabulary is compiled only during that pack's own run.

## Two design rules that run through both arms

**The model never emits coordinates that matter.** It returns an index into a
deterministic token stream; geometry comes from the stream. A model's box is a
hint, never truth.

**Detected is not removed.** Every removal claim is verified by re-reading the
output, and a deliberately broken control — a cosmetic redactor that draws
opaque rectangles and changes nothing underneath — is required to **fail** that
verification. If the control ever passes, the verifier is broken.

## Requirements

- Python ≥ 3.9. The default audio workflow is standard library only;
  `--mapping` needs `pyyaml`, `--replace pseudonym` needs `numpy` (and uses
  `scipy` when present). The document arm additionally needs `PyMuPDF`,
  `numpy` and `scipy`; the video arm needs `numpy`, `opencv-python-headless`,
  `Pillow` and ffmpeg (on PATH, or the `imageio-ffmpeg` wheel). Apple Vision
  (macOS) is driven through `swift -` with nothing to install.
- An audio converter: macOS ships `afconvert`; Linux/Windows need `ffmpeg`.
- Your own API keys — see [Setup](docs/audio.md#setup). Keys are personal and
  never committed.
- Optional: `PII_CORPUS_DIR` pointing at a local corpus, which enables the
  corpus-backed self-tests. Without it they skip and say so.

## Layout

    pipeline/      audio: orchestrator, engines, detector, render, QC, judges
    tools/         audio: production workflow, evaluation harness, reports
    prompts/       audio: versioned detection prompts and keyterms
    dedupe/        dataset dedupe (exact, acoustic, transcript tiers)
    metadata_qc_pipeline/   per-delivery metadata over REDACTED transcripts
    multimodal/    the document arm — router, pipeline, taxonomies, tests;
                   video_pipeline/ is the video arm
    docs/          audio documentation

## Privacy notes

- Deliverables of real calls remain sensitive recordings; handle them under the
  same rules as the source audio.
- The document arm writes **no re-identification key**. It used to mint a
  `*.tokens.json` mapping pseudonyms to real values, for a `[PERSON_1]` label
  that was written into the vacated box; that write-back was removed, and the
  minting with it. `PSEUDONYMIZE` destroys exactly as `REDACT` does.
- Outputs default to a `redacted/` subfolder beside the input, never into the
  input directory itself — originals and redactions do not share a folder, and
  a batch that globs `*.pdf` cannot pick up its own output as an input.
- Raw transcripts, detection intervals, redaction outputs, evaluation reports
  and sample media stay in gitignored working directories and never enter this
  repository.
