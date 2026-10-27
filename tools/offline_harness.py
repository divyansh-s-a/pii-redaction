#!/usr/bin/env python3
"""Offline harness: the FULL delivery workflow (tools/redact.py deliver())
on synthetic media, with every network stage replaced by the clip's answer
key — no API calls, no cost, fully deterministic.

Stubbed: ElevenLabs/Gemini transcription (returns <clip>.transcript.json),
Gemini detection (returns the answer key's spans as word indices), edge
verification (clean). Everything else — prepare, render, transcript, QC,
digit-edge auto-extension, manifest, delivery packaging — is the real code.

Uses:
  * byte-identity of the default beep mode against another checkout
    (tools/check_beep_identity.py drives this once per tree);
  * the --assurance union under a realistic second listener
    (--assurance-drift: its clock runs early and its word numbering is
    shifted; --primary-miss: values only it finds), scored by
    tools/check_assurance_union.py;
  * offline checks of pseudonym rendering (--replace pseudonym --tts say).

  python3 tools/offline_harness.py --tree TREE --media DIR --out DIR
      [--mapping M --replace pseudonym --tts say --profile-file P]
Prints one JSON object: sha256 of every deliverable and work artifact.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--media", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default="synth_en,synth_hi,synth_mr")
    ap.add_argument("--profile", default="medical_consult")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--replace", default=None)
    ap.add_argument("--tts", default=None)
    ap.add_argument("--profile-file", default=None)
    ap.add_argument("--phone-tail-miss", action="store_true",
                    help="the stub detector stops one word short on PHONE "
                         "spans, so QC's digit-edge check fires and the "
                         "auto-extension RE-RENDER path runs")
    ap.add_argument("--assurance", action="store_true")
    ap.add_argument("--assurance-drift", action="store_true",
                    help="the second listener's transcript runs early "
                         "(150 ms + 3%% of elapsed time, at most 1 s) and "
                         "joins its first two words, so its word indices "
                         "and times both differ from the primary's")
    ap.add_argument("--primary-miss", default="",
                    help="comma-separated labels (e.g. PHONE,ADDRESS) the "
                         "PRIMARY detector misses; only the second "
                         "listener finds them")
    a = ap.parse_args()
    tree, media, out = (Path(a.tree).resolve(), Path(a.media).resolve(),
                        Path(a.out).resolve())
    out.mkdir(parents=True, exist_ok=True)
    if a.mapping:
        os.environ["PII_MAPPING_FILE"] = str(Path(a.mapping).resolve())
    if a.replace:
        os.environ["PII_REPLACE_MODE"] = a.replace
    if a.tts:
        os.environ["PII_TTS_ENGINE"] = a.tts
    if a.profile_file:
        os.environ["PII_PROFILE_FILE"] = str(Path(a.profile_file).resolve())
    sys.path.insert(0, str(tree / "pipeline"))
    import common
    import detect_pii
    import gemini_transcribe
    import edge_verify
    from adapters import elevenlabs_transcribe

    state = {"clip": None, "phase": "primary"}
    missed = {x.strip() for x in a.primary_miss.split(",") if x.strip()}

    def drift(t):
        return max(0, int(t - min(1000, 150 + 0.03 * t)))

    def second_transcript():
        tr = json.loads((media / f"{state['clip']}.transcript.json")
                        .read_text())
        if not a.assurance_drift:
            return tr
        for u in tr["utterances"]:
            for w in u["words"]:
                w["start_ms"], w["end_ms"] = (drift(w["start_ms"]),
                                              drift(w["end_ms"]))
        ws = tr["utterances"][0]["words"]
        if len(ws) >= 2:
            ws[0:2] = [{"w": ws[0]["w"] + ws[1]["w"],
                        "start_ms": ws[0]["start_ms"],
                        "end_ms": ws[1]["end_ms"]}]
        return tr

    def fake_transcribe(model, original, work_wav, out_json, channel_wavs=None):
        shutil.copy2(media / f"{state['clip']}.transcript.json", out_json)

    def fake_transcribe_second(model, original, work_wav, out_json,
                               channel_wavs=None):
        Path(out_json).write_text(json.dumps(second_transcript(),
                                             ensure_ascii=False))

    real_detect_file = detect_pii.detect_file

    def detect_file_phase(model, file_id, transcript_path, *x, **k):
        # the second listener's run lives in <work>/assurance/; test the
        # folder's own name, never the whole path (a run folder may itself
        # be called "assurance_...")
        state["phase"] = ("assurance"
                          if Path(transcript_path).parent.name == "assurance"
                          else "primary")
        try:
            return real_detect_file(model, file_id, transcript_path, *x, **k)
        finally:
            state["phase"] = "primary"

    def spans_for(kinds):
        gt = json.loads((media / f"{state['clip']}.gt.json").read_text())
        second = state["phase"] == "assurance"
        tr = (second_transcript() if second else json.loads(
            (media / f"{state['clip']}.transcript.json").read_text()))
        shift = drift if (second and a.assurance_drift) else (lambda t: t)
        words = detect_pii.flatten_words(tr)
        out = []
        for e in gt["entities"]:
            if e["kind"] not in kinds:
                continue
            if not second and e["label"] in missed:
                continue
            es, ee = shift(e["start_ms"]), shift(e["end_ms"])
            idx = [w["i"] for w in words
                   if w["start_ms"] < ee and es < w["end_ms"]]
            if not idx:
                continue
            out.append((idx[0], idx[-1], e))
        return out

    def fake_generate(model, parts, response_schema=None, **kw):
        prompt = parts[0]["text"]
        if "ENTITIES (id, category" in prompt:
            import entity_map
            em = entity_map.active()
            ids = {e["id"]: f"E{k}" for k, e in enumerate(em.entries, 1)}
            res = [{"start_word": a_, "end_word": b_,
                    "entity_id": ids.get(e["label"], ""),
                    "kind": "hint" if e["kind"] == "H" else "mention",
                    "spoken_text": e["text"], "confidence": "high"}
                   for a_, b_, e in spans_for({"M", "H"})]
            return json.dumps({"spans": res})
        cats = {"FULL_NAME", "PHONE", "ADDRESS", "AGE", "CHILD_INFO"}
        res = []
        for a_, b_, e in spans_for({"P", "K"}):
            if e["label"] not in cats:
                continue
            c = (["CHILD_INFO", "AGE"] if e["label"] == "CHILD_INFO"
                 else [e["label"]])
            if a.phone_tail_miss and e["label"] == "PHONE" and b_ > a_:
                b_ -= 1
            res.append({"start_word": a_, "end_word": b_, "categories": c,
                        "spoken_text": e["text"], "speaker": e["speaker"],
                        "confidence": "high", "reason": "answer key"})
        return json.dumps({"spans": res})

    elevenlabs_transcribe.transcribe_file = fake_transcribe
    gemini_transcribe.transcribe_file = fake_transcribe_second
    detect_pii.detect_file = detect_file_phase
    common.generate = fake_generate
    common.resolve_model = lambda: common.CONFIG["model"]
    edge_verify.verify_and_fix = lambda *x, **k: {"clean": True,
                                                  "extensions": [],
                                                  "stub": True}
    spec = importlib.util.spec_from_file_location("tools_redact",
                                                  tree / "tools" / "redact.py")
    tr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr_mod)
    if a.profile_file:
        import profiles
        a.profile = profiles.active()["name"]

    result = {}
    work_root = getattr(common, "OUTPUT_ROOT", tree / "output")
    for clip in a.clips.split(","):
        state["clip"] = clip
        # a clean start per clip: the second listener's run is cached in the
        # work dir and is NOT redone by the forced stages, so a previous
        # harness run's (e.g. drifted) assurance/ would leak into this one
        for stale in (work_root / f"{clip}__elevenlabs", work_root / clip):
            if stale.exists() and clip.startswith("synth_"):
                shutil.rmtree(stale)
        src = out / "in" / f"{clip}.m4a"
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(media / f"{clip}.m4a", src)
        kw = {}
        if "listen_check" in tr_mod.deliver.__code__.co_varnames:
            kw = {"listen_check": False,
                  "quarantine_dir": out / "quarantine"}
        man = tr_mod.deliver(src, a.profile, out / "deliveries",
                             a.assurance, False, common.CONFIG["model"],
                             auto_extend=True, force_stages=["prepare",
                                                             "transcribe",
                                                             "detect",
                                                             "redact"],
                             **kw)
        fid = clip
        root = getattr(common, "OUTPUT_ROOT", tree / "output")
        work = root / f"{fid}__elevenlabs"
        bundle = None
        for base in (out / "deliveries", out / "quarantine",
                     tree / "quarantine"):
            for d in sorted(base.glob("*")) if base.exists() else []:
                if (d / "manifest.json").exists() and json.loads(
                        (d / "manifest.json").read_text()).get(
                        "source_sha256") == man["source_sha256"]:
                    bundle = d
        files = {}
        for f in sorted(bundle.iterdir()):
            if f.name != "manifest.json":
                files[f.name] = sha(f)
        m = json.loads((bundle / "manifest.json").read_text())
        for k in ("generated_at", "source_file"):
            m.pop(k, None)
        files["manifest(no time/path)"] = hashlib.sha256(json.dumps(
            m, sort_keys=True).encode()).hexdigest()
        for n in ("intervals.json", "rendered_intervals.json"):
            files["work/" + n] = sha(work / n)
        result[clip] = {"verdict": bundle.parent.name, "qc": m["qc_status"],
                        "files": files, "bundle": str(bundle),
                        "work": str(work)}
    print("HARNESS_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
