# Video arm — redacting an advertisement or other video

Part of the PII redaction toolkit; see the [root README](../README.md). Paths
are relative to the repository root.

> **Private repository.** Code only: no data of any kind (video, audio,
> frames, transcripts, redacted or not) is kept here.

    python3 multimodal/video_pipeline/run_video.py AD.mp4 \
        --mapping study_mapping.yaml --replace pseudonym \
        --profile-file study_profile.yaml \
        --outdir OUT --quarantine-dir HELD -- --sweep --assurance

or through the router, which sends `.mp4/.mov/.m4v/.avi/.mkv/.webm` here:

    python3 multimodal/router/dispatch.py AD.mp4 --video -- --mapping m.yaml

The **sound** goes through the [audio arm](audio.md) unchanged — the same
flags after `--`, the same beep or invented-name replacement, audits and
listen check. The **picture** is handled here:

| step | what happens |
|---|---|
| shots | cuts by colour-histogram change; a shot is *static* when neither the picture nor any region in it moves |
| sample | `--sample-fps` (default 2) plus every shot's first and last frame |
| detect | Apple Vision reads text (with per-word boxes), finds unreadable text regions (Devanagari and other Indic scripts) and faces; the mapping matcher runs over the OCR words directly; vision models (Gemini, plus GPT as a second detector — `--detectors`) say what each word or region *is* by id, and draw boxes only for what no reader locates (logos without text, packaging, mascots, missed faces) |
| track | every region is carried forward and backward to the neighbouring samples by template matching; where the match fails the box holds and grows — over-blur is the safe direction; tracking never crosses a cut |
| render | every region pixelated and blurred: listed brands in any form, **every** logo or brand mark, **every** face, phone numbers, URLs, handles. With `--replace pseudonym`, a static shot's mapped text or logo also gets a plain plate with the invented name; moving shots are blurred only |
| mux | redacted picture + redacted sound (H.264 CRF 17, AAC 192k) |
| verify | on frames the detector did *not* sample: OCR + matcher (readable real names, URLs containing them, phone numbers), face sharpness against the original (a blurred face can still be *found*, but it has lost its detail), and a vision-model re-check with the invented names as an allowlist. A deliberately cosmetic redaction of a few original frames (a translucent box) must **fail** these checks; if it passes, the verifier is broken and the run is held |
| script | one frame per shot from the *redacted* video and the redacted transcript become a shot-by-shot script (visual, camera, on-screen text, sound); it is then scanned like a transcript — a real name in the script would mean the redaction leaked |

Deliverables in `OUT/<id>/`: `<id>_redacted.mp4`, `<id>_script.md`,
`<id>_redacted.txt` (the audio transcript), `qc_report.md`, `manifest.json`.
Any problem — an audio-arm quarantine, a detector that failed on a frame, a
verification finding, a control that passed, a real name in the script —
sends the bundle to the quarantine directory instead. With a mapping, `<id>`
is the source name passed through the mapping, and the manifest stores the
source hash, not its path.

**Faces.** Every face is blurred, real or depicted, foreground or background.
Apple Vision's face detector is the deterministic source; it does not fire on
drawn faces, so the synthetic test exercises the model path for faces, and the
verification re-check counts any face left with detail on real material.

**Known limits.** Text the OCR cannot read is located by Vision's text-region
detector and identified by the model; its geometry is exact, its reading is
not. A replaced name in the sound removes the music under it too (there is no
voice/music separation), softened by short fades. A logo that changes shape
mid-shot (an animated morph) is followed by the tracker's hold-and-grow, not
re-detected, between samples — raise `--sample-fps` for heavily animated ads.

**Tests.** `tools/synth_video.py OUT_DIR` draws a 12 s ad with known per-frame
ground truth (panning wordmark and a drawn face, a sliding Devanagari
wordmark, a static packshot, a static end card with tagline, URL and
toll-free number, and a voice-over naming the brands);
`tools/score_video.py` reports blur recall by shot and element kind, blur
precision, static-shot calls, plate placement and the run's own verification.
