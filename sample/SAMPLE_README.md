# Sample: redacted output of the production workflow

Source: a 2m29s English-language insurance/hospital telephone
call (8 kHz, mono, two speakers). This folder contains ONLY the
redacted deliverables - by policy the original audio and raw
transcript never enter version control.

## What the workflow did to this file

1. Transcribed with ElevenLabs scribe_v2 (+ keyterms vocabulary),
   word-level timestamps, diarization.
2. Gemini applied detection policy v3 (Taxonomy v0.2) to the
   canonicalized word stream and found 5 PII intervals:
   - FULL_NAME at 0:12.69-0:12.76
   - ORGANIZATION_PERSON_LINKED at 0:13.38-0:13.91
   - FINANCIAL_INFO at 1:35.94-1:38.02
   - FINANCIAL_INFO at 1:41.18-1:42.20
   - FINANCIAL_INFO at 1:43.32-1:45.24
3. Rendered word-aware 1 kHz beeps: each beep edge sits at the
   quietest moment between the value and the nearest kept word
   (person names get a 300 ms onset guard), so neighbouring words
   like 'from' between the name and the organization stay
   audible while the values are fully covered. The transcript
   gets <CATEGORY REDACTED> markers from the SAME intervals.
4. Gates (all default): signal QC PASS; beep-edge listen check
   clean (1 rounds, no unresolved or unchecked edges); judge
   audit 0 leak candidates. Full provenance in manifest.json
   (models, prompt hashes, source sha, gate outcomes).

Listen for: the name at ~0:12 (two separate beeps with 'from'
audible between name and organization) and the amounts at
~1:36-1:45 (neighbouring words 'Okay/right/Yeah/sir' intact).
