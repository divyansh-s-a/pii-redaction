#!/usr/bin/env python3
"""THE production workflow — one command, the decided configuration.

    python3 tools/redact.py MEDIA [MEDIA ...] [--profile strict|medical_consult]
                            [--outdir deliveries] [--assurance] [--no-audit]
                            [--edge-verify]

Configuration (fixed by the 2026-09-01 decision gate):
  stage B  ElevenLabs scribe_v2 + keyterms vocabulary  (best ear, ~$0.22/hr)
  stage C  Gemini + the versioned detection policy on canonicalized words
  stage D  1 kHz beep render + redacted transcript from ONE interval list
  stage E  signal-verified QC gate
          (+ empirical beep-edge verification, OFF by default: --edge-verify;
           it re-transcribes the audio around every beep edge and is slow)
  stage F  independent LLM-judge leak audit (default ON; --no-audit skips)

--assurance : additionally transcribe with Gemini and UNION the detected
  intervals into the audio beeps (recall-first; for high-risk content).
  Audio gets the union; the transcript redacts the ElevenLabs-mapped intervals
  and the manifest lists any assurance-only beeps (audio is the primary
  privacy artifact).
The independent LLM-judge leak audit runs on the redacted transcript BY
DEFAULT (part of the main workflow since 2026-09-02); its result is recorded
in the manifest and any leak candidate quarantines the file. --no-audit skips
it (cost-sensitive batch runs only — not recommended for deliveries).

Gates & handling (user decisions 2026-09-01):
  QC FLAGGED or audit leak candidates  -> quarantine/<id>/ (never deliveries/)
  ElevenLabs unavailable               -> automatic Gemini stage-B fallback,
                                          recorded prominently in the manifest
  Video input                          -> beeped AUDIO delivered + manifest
                                          note (video remux needs ffmpeg)
  0 PII found                          -> valid delivery, noted as verified

Every delivery folder contains: beeped audio, redacted transcript, qc report,
and manifest.json recording the models, prompt versions, keyterms hash, source
hash and QC/audit outcomes — full provenance per file.
"""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import audio_utils
import common
import detect_pii
import audio_utils
import edge_verify
import qc_report
import redact as redact_mod
import run_pipeline

ROOT = common.ROOT
ENGINE = "elevenlabs"           # decided 2026-09-01 (decision gate; see docs)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def union_assurance(fid, el_dir: Path, media: Path, work: Path, model,
                    channel_wavs):
    """Assurance mode: gemini-transcribe + detect the same audio; return the
    gemini intervals (same audio timeline, so their times are valid beeps)."""
    import gemini_transcribe
    adir = el_dir / "assurance"
    adir.mkdir(exist_ok=True)
    tr = adir / "transcript.json"
    iv = adir / "intervals.json"
    if not tr.exists():
        print("  [assurance] gemini transcription ...")
        gemini_transcribe.transcribe_file(model, media, work, tr,
                                          channel_wavs=channel_wavs)
    if not iv.exists():
        detect_pii.detect_file(model, fid + "_assurance", tr, iv,
                               common.CONFIG["value_gap_merge_ms"])
    return common.load_json(iv)["intervals"]


def deliver(media: Path, profile: str, outdir: Path, assurance: bool,
            audit: bool, model: str, edge_verify_on: bool = False,
            auto_extend: bool = True):
    fid = run_pipeline.file_id_for(media)
    engine_used = ENGINE
    fallback_note = None
    try:
        res = run_pipeline.run_file(media, model, profile, set(), ENGINE, None)
        src_dir = ROOT / "output" / f"{fid}__{ENGINE}"
    except (RuntimeError, SystemExit) as e:
        print(f"  !! elevenlabs stage failed ({str(e)[:120]}) — "
              "FALLING BACK to Gemini transcription (higher cost)")
        engine_used = "gemini"
        fallback_note = f"elevenlabs unavailable: {str(e)[:200]}"
        res = run_pipeline.run_file(media, model, profile, set(), "gemini", None)
        src_dir = ROOT / "output" / fid
    is_video = media.suffix.lower().lstrip(".") in ("mp4", "mov", "m4v", "avi")

    extra_beeps = []
    if assurance:
        prep = common.load_json(src_dir / "prepare.json")
        channel_wavs = [src_dir / n for n in prep.get("channel_wavs", [])] or None
        a_ivs = union_assurance(fid, src_dir, media, src_dir / "work_16k.wav",
                                model, channel_wavs)
        el_ivs = common.load_json(src_dir / "intervals.json")["intervals"]
        rendered = common.load_json(src_dir / "rendered_intervals.json")
        merged = [(r["start_ms"], r["end_ms"]) for r in rendered["rendered_merged"]]

        def covered(s, e):
            got = sum(max(0, min(e, b) - max(s, a)) for a, b in merged)
            return e <= s or got / (e - s) >= 0.95

        import policy
        for ivn in a_ivs:
            if policy.action_for(ivn["categories"], ivn.get("public_figure", False),
                                 profile) != "beep":
                continue
            if not covered(ivn["start_ms"], ivn["end_ms"]):
                extra_beeps.append(ivn)
        if extra_beeps:
            print(f"  [assurance] {len(extra_beeps)} union-only interval(s) "
                  "added to audio beeps")
            all_ivs = el_ivs + [dict(iv, source="assurance_gemini")
                                for iv in extra_beeps]
            data = {"file_id": fid, "taxonomy_version": "0.2",
                    "intervals": all_ivs}
            common.save_json(src_dir / "intervals.json", data)
            redact_mod.redact_file(media, src_dir / "work_16k.wav",
                                   src_dir / "intervals.json",
                                   src_dir / "transcript.json",
                                   src_dir, fid, profile,
                                   channel_wavs=channel_wavs)
            qc_report.qc_file(src_dir, fid, src_dir / "transcript.json",
                              src_dir / "intervals.json",
                              common.CONFIG["beep_freq_hz"])

    # ---- digit-boundary auto-extension (owner decision 2026-09-10) ----
    # A digit word at a beep boundary almost always belongs to the beeped
    # number: extend the beep over the word (over-redaction, the safe
    # direction), re-render, re-QC — then LISTEN to exactly the extended
    # beeps (targeted edge check). The empirical listen stays the gate; the
    # transcript-level flag alone no longer quarantines. --no-auto-extend
    # restores the old hold-everything behavior.
    auto_ext = []
    ev = None
    de_path = src_dir / "qc_digit_edges.json"
    if auto_extend and res["qc"] != "PASS" and de_path.exists():
        edges = json.loads(de_path.read_text())
        if edges:
            rendered = common.load_json(src_dir / "rendered_intervals.json")
            spans = [(r["start_ms"], r["end_ms"], r.get("channel"))
                     for r in rendered["rendered_merged"]]
            pad = common.CONFIG.get("min_pad_ms", 50)
            for ed in edges:
                a = max(0, min(ed["w_start"], ed["iv_start"]) - pad)
                b = max(ed["w_end"], ed["iv_end"]) + pad
                spans.append((a, b, ed.get("channel")))
                auto_ext.append({"word": ed["word"],
                                 "w_start": ed["w_start"],
                                 "w_end": ed["w_end"],
                                 "interval": ed["interval_id"]})
            dur = common.load_json(src_dir / "prepare.json")["duration_ms"]
            merged_ae = audio_utils.pad_and_merge(spans, 0, dur)
            native = src_dir / "_ae_native.wav"
            beeped_w = src_dir / "_ae_beeped.wav"
            audio_utils.decode_to_wav(media, native)
            audio_utils.mix_beeps(native, beeped_w, merged_ae,
                                  common.CONFIG["beep_freq_hz"],
                                  common.CONFIG["beep_gain_db"])
            audio_utils.encode_audio(beeped_w,
                                     src_dir / common.beeped_name(fid))
            native.unlink(missing_ok=True)
            beeped_w.unlink(missing_ok=True)
            rendered["rendered_merged"] = [
                {"start_ms": s0, "end_ms": e0, "channel": c0}
                for s0, e0, c0 in merged_ae]
            common.save_json(src_dir / "rendered_intervals.json", rendered)
            print(f"  [auto-extend] {len(auto_ext)} digit-boundary "
                  "extension(s) applied -> re-QC + targeted edge listen")
            st2, _ = qc_report.qc_file(src_dir, fid,
                                       src_dir / "transcript.json",
                                       src_dir / "intervals.json",
                                       common.CONFIG["beep_freq_hz"])
            res["qc"] = st2
            listen_spans = [(max(0, min(ed["w_start"], ed["iv_start"]) - 400),
                             max(ed["w_end"], ed["iv_end"]) + 400)
                            for ed in edges]
            ev = edge_verify.verify_and_fix(
                src_dir, fid, media, model,
                common.CONFIG["beep_freq_hz"],
                common.CONFIG["beep_gain_db"],
                only_overlapping=listen_spans)

    # empirical beep-edge verification (opt-in via --edge-verify; slow because
    # it re-transcribes the audio around every beep edge). Audible fragments
    # of a beeped value extend the beep — defends against engine clock skew at
    # value onsets. Disabled by default.
    if edge_verify_on:
        print("  [edge-verify] checking beep edges by ear ...")
        ev = edge_verify.verify_and_fix(src_dir, fid, media, model,
                                        common.CONFIG["beep_freq_hz"],
                                        common.CONFIG["beep_gain_db"])
        if ev.get("extensions"):
            res2 = qc_report.qc_file(src_dir, fid, src_dir / "transcript.json",
                                     src_dir / "intervals.json",
                                     common.CONFIG["beep_freq_hz"])
            res["qc"] = res2[0]

    audit_summary = None
    if audit:
        print("  [audit] independent judge leak audit ...")
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "judge_leak_audit.py"),
             "--engines", engine_used, "--files", fid, "--force",
             "--out", str(src_dir / "audit")],
            capture_output=True, text=True)
        rep = src_dir / "audit" / "leak_audit_report.json"
        if rep.exists():
            j = json.loads(rep.read_text())
            s = j["engines"][engine_used]["summary"]
            audit_summary = {"leak_candidates": s.get("leak_candidates"),
                             "findings": s.get("total_findings"),
                             "rejected_by_verifier":
                                 s.get("findings_rejected_by_verifier", 0)}
            print(f"  [audit] leak candidates: {audit_summary['leak_candidates']}")
            if audit_summary.get("leak_candidates"):
                # confirm-on-second-audit (owner decision 2026-09-10): the
                # audit judge is unstable on borderline classes (observed 0 ->
                # 15 on identical transcripts). A candidate only gates when it
                # appears in TWO independent reads (same quote, line +/-3);
                # unconfirmed ones are recorded, never shipped silently.
                print("  [audit] confirming with a second independent read ...")
                subprocess.run(
                    [sys.executable, str(ROOT / "tools" / "judge_leak_audit.py"),
                     "--engines", engine_used, "--files", fid, "--force",
                     "--out", str(src_dir / "audit_confirm")],
                    capture_output=True, text=True)
                pf1 = src_dir / "audit" / f"{fid}__{engine_used}.json"
                pf2 = src_dir / "audit_confirm" / f"{fid}__{engine_used}.json"
                if pf1.exists() and pf2.exists():
                    def _k(f):
                        return " ".join(str(f.get("quoted_text", ""))
                                        .lower().split())
                    f1 = [f for f in json.loads(pf1.read_text())
                          .get("findings", []) if not f.get("profile_dependent")]
                    f2 = json.loads(pf2.read_text()).get("findings", [])
                    confirmed = [f for f in f1 if any(
                        _k(f) == _k(g)
                        and abs(f.get("line", 0) - g.get("line", 0)) <= 3
                        for g in f2)]
                    audit_summary["leak_candidates_first_read"] = \
                        audit_summary["leak_candidates"]
                    audit_summary["leak_candidates"] = len(confirmed)
                    audit_summary["unconfirmed_candidates"] = [
                        {"quote": str(f.get("quoted_text", ""))[:60],
                         "line": f.get("line")}
                        for f in f1 if f not in confirmed]
                    print(f"  [audit] confirmed on second read: "
                          f"{len(confirmed)} of "
                          f"{audit_summary['leak_candidates_first_read']}")
        else:
            audit_summary = {"error": (r.stdout + r.stderr)[-400:]}

    # ---- gate: QC must PASS; with --audit, zero leak candidates too ----
    quarantined_reason = None
    if res["qc"] != "PASS":
        quarantined_reason = f"QC {res['qc']}"
    elif ev is not None and not ev.get("clean", True):
        quarantined_reason = "edge verification unresolved (audible fragments at beep edges)"
    elif audit_summary and audit_summary.get("leak_candidates", 0):
        quarantined_reason = (f"judge audit found "
                              f"{audit_summary['leak_candidates']} leak candidate(s)")
    base = (ROOT / "quarantine") if quarantined_reason else outdir
    dst = base / fid
    # collision: same fid, different source -> suffix with source sha
    mf = dst / "manifest.json"
    if mf.exists():
        prev = json.loads(mf.read_text()).get("source_sha256")
        if prev and prev != sha(media):
            dst = base / f"{fid}-{sha(media)[:8]}"
    dst.mkdir(parents=True, exist_ok=True)
    beeped_name = common.beeped_name(fid)
    shutil.copy2(src_dir / beeped_name, dst / beeped_name)
    shutil.copy2(src_dir / f"{fid}_redacted.txt", dst / f"{fid}_redacted.txt")
    shutil.copy2(src_dir / "qc_report.md", dst / "qc_report.md")

    import policy
    policy_file = policy.POLICY_PROMPT_FILE
    keyterms = ROOT / "prompts" / "keyterms_india_v1.txt"
    manifest = {
        "workflow": "elevenlabs+gemini hybrid (decision gate 2026-09-01)",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_file": str(media), "source_sha256": sha(media),
        "profile": profile,
        "transcription": {"engine": engine_used
                          + (" (FALLBACK: elevenlabs unavailable)"
                             if fallback_note else ""),
                          "model": ("scribe_v2" if engine_used == "elevenlabs"
                                    else model),
                          "fallback_reason": fallback_note,
                          "keyterms_file": keyterms.name,
                          "keyterms_sha256": sha(keyterms) if keyterms.exists() else None},
        "detection": {"model": model, "policy_prompt": policy_file,
                      "policy_sha256": sha(ROOT / "prompts" / policy_file),
                      "taxonomy_version": "0.2", "input": "canonicalized words"},
        "assurance_mode": assurance,
        "assurance_extra_beeps": [
            {"start_ms": iv["start_ms"], "end_ms": iv["end_ms"],
             "categories": iv["categories"]} for iv in extra_beeps],
        "qc_status": res["qc"],
        "edge_verification": ev,
        "quarantined_reason": quarantined_reason,
        "video_note": ("video track not re-attached (audio-only delivery; "
                       "video remux requires ffmpeg)" if is_video else None),
        "redaction_count": len(common.load_json(
            src_dir / "intervals.json")["intervals"]),
        "judge_audit": audit_summary,
        "auto_extensions": auto_ext,
        "languages": res.get("languages"),
        "deliverables": [beeped_name, f"{fid}_redacted.txt",
                         "qc_report.md"],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  ensure_ascii=False))
    if quarantined_reason:
        print(f"  ** QUARANTINED ** -> {dst}  reason: {quarantined_reason}")
    if is_video:
        print("  note: video input — audio-only delivery (remux needs ffmpeg)")
    if manifest["redaction_count"] == 0 and not quarantined_reason:
        print("  note: 0 redactions — no PII detected (QC-verified)")
    print(f"  {'quarantined' if quarantined_reason else 'delivered'} -> {dst}  (qc={res['qc']}"
          + (f", audit leaks={audit_summary['leak_candidates']}"
             if audit_summary and "leak_candidates" in audit_summary else "")
          + ")")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("media", nargs="+")
    ap.add_argument("--profile", default=common.CONFIG["profile"],
                    choices=["strict", "medical_consult"])
    ap.add_argument("--outdir", default=str(ROOT / "deliveries"))
    ap.add_argument("--mono", action="store_true",
                    help="treat stereo input as a mono mix (disable "
                         "split-channel mode; for crosstalk-heavy stereo "
                         "where channels are not one speaker each)")
    ap.add_argument("--assurance", action="store_true",
                    help="also gemini-transcribe and union detections into "
                         "the audio beeps (high-risk content)")
    ap.add_argument("--audit", dest="audit", action="store_true", default=True,
                    help="run the independent judge leak audit (the default)")
    ap.add_argument("--no-audit", dest="audit", action="store_false",
                    help="skip the judge leak audit (not recommended for "
                         "deliveries)")
    ap.add_argument("--no-auto-extend", dest="auto_extend",
                    action="store_false",
                    help="hold files on digit-boundary QC flags instead of "
                         "auto-extending the beep over the digit word "
                         "(auto-extend + targeted edge listen is the default)")
    ap.add_argument("--edge-verify", dest="edge_verify", action="store_true",
                    default=False,
                    help="run empirical beep-edge verification (OFF by "
                         "default; slow — re-transcribes audio around every "
                         "beep edge)")
    args = ap.parse_args()

    if args.mono:
        common.CONFIG["channel_split"] = "never"
    model = common.resolve_model()
    import policy as _policy
    print(f"workflow: EL scribe_v2 (+keyterms) -> Gemini {model} "
          f"({_policy.POLICY_PROMPT_FILE}) | profile {args.profile}")
    outdir = Path(args.outdir)
    for m in args.media:
        p = Path(m).resolve()
        if not p.exists():
            print(f"SKIP {m}: not found")
            continue
        print(f"\n=== {p.name} ===")
        try:
            deliver(p, args.profile, outdir, args.assurance, args.audit, model,
                    args.edge_verify, args.auto_extend)
        except Exception as e:
            print(f"  ERROR {p.name}: {str(e)[:200]} — continuing with next file")


if __name__ == "__main__":
    main()
