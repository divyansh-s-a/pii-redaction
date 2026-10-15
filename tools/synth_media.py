#!/usr/bin/env python3
"""Synthetic test audio for the replacement modes (--mapping, --replace).

Everything here is INVENTED: brands, agency, people, numbers, places that
are not on the project's keep list. Nothing is drawn from any corpus. Media
is generated locally (macOS `say` voices) into an output folder OUTSIDE the
repository (the repo ignores media by extension anyway). The video arm's
synthetic ad is tools/synth_video.py.

  python3 tools/synth_media.py audio OUT_DIR      # 3 focus-group clips

audio writes, per clip: <clip>.m4a, <clip>.gt.json (exact entity times: each
entity is synthesised as its own chunk, so its edges are known to the
sample), <clip>.transcript.json (pipeline format, word times spread evenly
inside each chunk — for offline render tests only), plus mapping.yaml and
profile.yaml for the whole set.
"""
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

# ------------------------------------------------------------- the set
# mapping: invented "real" names -> invented replacements
MAPPING_YAML = """\
# Synthetic test mapping: every name here is invented.
# Replace the DISTINCTIVE name only; generic words after it (Bank,
# Electronics) stay spoken, so a mention with or without them reads right.
brands:
  - {pattern: "Zorvex(?:\\\\s+Vision)?", replace: Kelmora, aliases: ["ज़ोरवेक्स", "झोरवेक्स", "zor vex"]}
  - {pattern: "Plumora", replace: Tandel, aliases: ["प्लुमोरा", "प्लमोरा"]}
  - {pattern: "Hexlin", replace: Varnet, category: DEALER, aliases: ["हेक्सलिन"]}
  - {pattern: "Quillfin", replace: Ostrava, aliases: ["क्विलफिन"]}
brand_facts:
  - {pattern: "see more,? live more", replace: "watch it all"}
agency:
  - {pattern: "Brightpath(?:\\\\s+Research)?", replace: Insightwell, aliases: ["ब्राइटपाथ"]}
people:
  - {pattern: "\\\\bRavindra\\\\b", replace: colleague}
keep:
  - Pune / Nashik / पुणे (fieldwork locations)
"""
PROFILE_YAML = """\
name: synthetic_market_research
# keep what a survey dataset keeps: a respondent's age; everything else that
# identifies a person is redacted as under strict
annotate_only: [AGE]
detection_supplement: |
  A respondent's own age is kept (it is analysis data in this project);
  still mark it as AGE. A child's age is CHILD_INFO + AGE as usual.
  Brands, banks, shops, dealers and agencies a respondent buys from, banks
  with or talks about are research content handled by the project's
  mapping: do NOT mark them ORGANIZATION_PERSON_LINKED. Still mark a
  person's employer, workplace or school, and a specific branch or clinic
  that would identify them.
"""
# speaker -> say voice (test speech; the pseudonym `say` voices are
# Aman/Lekha, so test speakers avoid Aman)
VOICES = {"MOD": "Rishi", "R1": "Tara", "R2": "Daniel", "R3": "Karen",
          "H1": "Lekha", "H2": "Lekha"}

# chunk = "plain text" | (text, kind, label)
#   kind M = mapped entity (label = mapping entry id)
#   kind P = personal data (label = policy category)
#   kind K = kept by profile / keep list (label = why)
#   kind H = brand hint not in the mapping (label = entry it reveals)
CLIPS = {
    "synth_en": ("en", [
        ("MOD", ["Good evening everyone. I am", ("Ravindra", "P", "FULL_NAME"),
                 "from", ("Brightpath Research", "M", "agency:0"),
                 "and today we will talk about televisions."]),
        ("R1", ["Hi, I am", ("Meera", "P", "FULL_NAME"), "and I am",
                ("thirty four", "K", "AGE"), "years old. I live in",
                ("Pune", "K", "keep"), "."]),
        ("R1", ["Last year I bought a", ("Zorvex Vision", "M", "brands:0"),
                "TV from", ("Hexlin", "M", "brands:2"), "Electronics",
                "near my house."]),
        ("R2", ["My name is", ("Arjun Kale", "P", "FULL_NAME"),
                ". Honestly the", ("Zorvex", "M", "brands:0"),
                "picture is good but I paid with my", ("Plumora", "M", "brands:1"), "Bank",
                "card and the EMI was confusing."]),
        ("R2", ["You can call me on", ("nine eight two two four one five "
                                       "seven three six", "P", "PHONE"),
                "if you want the bill."]),
        ("R3", ["I prefer", ("Quillfin", "M", "brands:3"),
                ". Their ad says", ("see more, live more", "M",
                                    "brand_facts:0"),
                "and my son, he is", ("seven", "P", "CHILD_INFO"),
                ", loves it."]),
        ("MOD", ["Thank you. Anyone else who went to", ("Hexlin", "M",
                                                        "brands:2"), "?"]),
    ]),
    "synth_hi": ("hi", [
        ("H1", ["नमस्ते, मेरा नाम", ("सुनीता", "P", "FULL_NAME"), "है।"]),
        ("H1", ["मैंने पिछले साल", ("ज़ोरवेक्स", "M", "brands:0"),
                "का टीवी लिया था, और", ("प्लुमोरा", "M", "brands:1"),
                "बैंक से लोन लिया।"]),
        ("H2", ["मेरी उम्र", ("बत्तीस", "K", "AGE"), "साल है, मैं",
                ("पुणे", "K", "keep"), "में रहती हूँ।"]),
        ("H2", ["मैं", ("हेक्सलिन", "M", "brands:2"),
                "की दुकान पर गई थी, पर मुझे", ("क्विलफिन", "M", "brands:3"),
                "ज़्यादा पसंद है।"]),
        ("H1", ["मेरा नंबर है", ("नौ आठ सात छह पांच चार तीन दो एक शून्य", "P",
                                 "PHONE"), "।"]),
    ]),
    "synth_mr": ("mr", [
        ("H1", ["नमस्कार, मी", ("प्रिया", "P", "FULL_NAME"), "आहे."]),
        ("H1", ["मी", ("झोरवेक्स", "M", "brands:0"),
                "चा टीव्ही घेतला, पण", ("प्लमोरा", "M", "brands:1"),
                "बँकेचा अनुभव चांगला नव्हता."]),
        ("H2", ["आम्ही", ("नाशिक", "K", "keep"), "मध्ये राहतो, आणि",
                ("हेक्सलिन", "M", "brands:2"), "मधून खरेदी केली."]),
        ("H2", ["माझा पत्ता", ("बारा, गुलमोहर सोसायटी, कोथरूड", "P",
                               "ADDRESS"), "आहे."]),
    ]),
}
GAP_MS = 70          # between chunks of one utterance
TURN_MS = 450        # between utterances
FS = 16000


def _say(text, voice, out_wav):
    subprocess.run(["say", "-v", voice, "-o", str(out_wav),
                    "--file-format=WAVE", f"--data-format=LEI16@{FS}", text],
                   check=True, capture_output=True)


def _frames(path):
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def _trim(pcm: bytes, thr=300):
    """Trim leading/trailing near-silence from 16-bit mono PCM."""
    import array
    a = array.array("h")
    a.frombytes(pcm)
    i, j = 0, len(a)
    while i < j and abs(a[i]) < thr:
        i += 1
    while j > i and abs(a[j - 1]) < thr:
        j -= 1
    i = max(0, i - FS // 100)
    j = min(len(a), j + FS // 100)
    return a[i:j].tobytes()


def make_audio(out_dir: Path):
    import random
    rnd = random.Random(7)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mapping.yaml").write_text(MAPPING_YAML)
    (out_dir / "profile.yaml").write_text(PROFILE_YAML)
    for clip, (lang, utts) in CLIPS.items():
        pcm = bytearray()
        gt = {"clip": clip, "language": lang, "entities": [],
              "utterances": []}
        tr_utts = []
        with tempfile.TemporaryDirectory() as td:
            for ui, (spk, chunks) in enumerate(utts):
                u_start = len(pcm) // 2 * 1000 // FS
                words = []
                for ci, ch in enumerate(chunks):
                    text, kind, label = (ch, None, None) if isinstance(
                        ch, str) else ch
                    if not text.strip(" .?,।"):
                        continue
                    wav = Path(td) / f"{ui}_{ci}.wav"
                    _say(text, VOICES[spk], wav)
                    seg = _trim(_frames(wav))
                    s_ms = len(pcm) // 2 * 1000 // FS
                    pcm += seg
                    e_ms = len(pcm) // 2 * 1000 // FS
                    if kind:
                        gt["entities"].append({"text": text, "kind": kind,
                                               "label": label, "speaker": spk,
                                               "start_ms": s_ms,
                                               "end_ms": e_ms})
                    toks = text.split()
                    total = sum(len(t) for t in toks) or 1
                    t0 = s_ms
                    for t in toks:
                        d = (e_ms - s_ms) * len(t) / total
                        words.append({"w": t, "start_ms": int(t0),
                                      "end_ms": int(t0 + d)})
                        t0 += d
                    pcm += b"\x00\x00" * (FS * GAP_MS // 1000)
                pcm += b"\x00\x00" * (FS * (TURN_MS + rnd.randint(0, 200))
                                      // 1000)
                u_end = len(pcm) // 2 * 1000 // FS
                gt["utterances"].append({"speaker": spk, "start_ms": u_start,
                                         "end_ms": u_end})
                tr_utts.append({"speaker": f"speaker_{spk}", "words": words,
                                "lang": lang, "overlaps_with": ""})
        # light room noise so silence is not digital zero
        import array
        a = array.array("h")
        a.frombytes(bytes(pcm))
        for k in range(len(a)):
            a[k] = max(-32768, min(32767, a[k] + rnd.randint(-40, 40)))
        wav_out = out_dir / f"{clip}.wav"
        with wave.open(str(wav_out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes(a.tobytes())
        m4a = out_dir / f"{clip}.m4a"
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-q", "127",
                        str(wav_out), str(m4a)], check=True,
                       capture_output=True)
        wav_out.unlink()
        gt["duration_ms"] = len(a) * 1000 // FS
        (out_dir / f"{clip}.gt.json").write_text(
            json.dumps(gt, indent=1, ensure_ascii=False))
        (out_dir / f"{clip}.transcript.json").write_text(json.dumps(
            {"utterances": tr_utts, "primary_language": lang,
             "languages": [{"lang": lang, "word_share": 1.0}]},
            indent=1, ensure_ascii=False))
        print(f"  {clip}: {gt['duration_ms'] / 1000:.1f}s, "
              f"{len(gt['entities'])} labelled spans -> {m4a}")


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "audio":
        sys.exit(__doc__)
    out = Path(sys.argv[2]).resolve()
    if ROOT in out.parents or out == ROOT:
        sys.exit("refusing to write media inside the repository")
    make_audio(out)


if __name__ == "__main__":
    main()
