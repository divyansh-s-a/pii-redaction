#!/usr/bin/env python3
"""Video arm — redact an mp4: its sound through the audio arm, its picture
here, then put them back together.

    python3 multimodal/video_pipeline/run_video.py AD.mp4 [AD2.mp4 ...]
        [--mapping m.yaml] [--replace beep|pseudonym] [--tts say|elevenlabs]
        [--profile-file p.yaml] [--outdir DIR] [--quarantine-dir DIR]
        [--sample-fps 2] [--detectors gemini,gpt] [--no-model-verify]
        [-- <audio-arm flags, e.g. --sweep --assurance>]

1. audio   the audio track is extracted and delivered by tools/redact.py
           (same flags: beep or invented-name mode, audits, listen check)
2. shots   cuts + per-shot motion (static shots: end cards, packshots)
3. detect  frames sampled at --sample-fps plus every shot's first/last frame;
           Apple Vision OCR/text regions/faces + mapping matcher + vision
           model(s) decide what to hide (frame_detect)
4. track   regions carried to every frame of their shot (track)
5. render  every region blurred; pseudonym mode lays an invented text plate
           over mapped text/logos on STATIC shots only (render_video)
6. mux     redacted picture + redacted sound
7. verify  OCR / face sharpness / model re-check on frames the detector did
           not sample, plus a broken control that must fail (verify_video)
8. script  shot-by-shot script from the redacted video + transcript, scanned
           for real names (ad_script)

Deliverables (outdir/<id>/): <id>_redacted.mp4, <id>_script.md,
<id>_redacted.txt (audio transcript), qc_report.md, manifest.json — any
problem sends the bundle to the quarantine dir instead. Work files (frames,
detections, the audio arm's caches) stay under the stage-cache root.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / "pipeline"))
sys.path.insert(0, str(PROJ / "tools"))
import ad_script
import frame_detect
import media_io
import render_video
import shots as shots_mod
import track
import verify_video


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sample_indices(shots, fps, rate):
    step = max(1, int(round(fps / rate)))
    idx = set()
    for s in shots:
        idx.update(range(s["f0"], s["f1"] + 1, step))
        idx.update((s["f0"], s["f1"]))
    return sorted(idx)


def detectors(spec):
    import common
    import judge_client
    out = []
    for d in [x.strip() for x in spec.split(",") if x.strip()]:
        if d == "gemini":
            out.append((common.CONFIG["model"], True))
        elif d == "gpt":
            out.append((judge_client.DOC_DETECTOR, False))
        else:
            out.append((d, d.startswith("gemini")))
    return out


def audio_track(src, fid, work, args, model, em):
    """Deliver the audio track through the audio arm; returns
    (manifest, beeped_audio_path, redacted_txt_path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tools_redact", PROJ / "tools" / "redact.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    wav = work / f"{fid}.wav"
    if not wav.exists():
        media_io.extract_audio(src, wav)
    extra = args.audio_args or []
    flags = {"assurance": "--assurance" in extra,
             "sweep": "--sweep" in extra}
    import common
    if flags["sweep"]:
        common.CONFIG["detect_sweep"] = True
    out_id = em.pseudonymise(fid) if em is not None else fid
    man = tr.deliver(wav, args.profile, work / "audio_deliveries",
                     flags["assurance"], not args.no_audit, model,
                     auto_extend=True, listen_check=None,
                     quarantine_dir=work / "audio_quarantine",
                     display_name=f"{out_id} (audio track)")
    for base in (work / "audio_deliveries", work / "audio_quarantine"):
        d = base / out_id
        if (d / "manifest.json").exists():
            b = d / common.beeped_name(out_id)
            return man, b, d / f"{out_id}_redacted.txt"
    raise RuntimeError("audio arm produced no bundle")


def run(src: Path, args, model, em):
    import common
    import run_pipeline
    info = media_io.probe(src)
    fid = run_pipeline.file_id_for(src)
    out_id = em.pseudonymise(fid) if em is not None else fid
    work = common.OUTPUT_ROOT / "video" / fid
    work.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {src.name} -> {out_id} ===  {info['width']}x{info['height']}"
          f" {info['fps']:.2f} fps, {info['duration_ms'] / 1000:.1f}s, "
          f"{info['frames']} frames, audio={info['has_audio']}")
    problems, notes = [], []

    # 1 audio
    a_man, a_audio, a_txt = (None, None, None)
    if info["has_audio"]:
        a_man, a_audio, a_txt = audio_track(src, fid, work, args, model, em)
        if a_man.get("quarantined_reason"):
            problems.append("audio track: " + a_man["quarantined_reason"])
        notes.append(f"audio track: qc {a_man.get('qc_status')}, "
                     f"{a_man.get('redaction_count')} redaction(s)")
    # 2 shots
    shots = shots_mod.analyze(media_io.frames(src), info["fps"])
    # 3 detect on samples
    samp = sample_indices(shots, info["fps"], args.sample_fps)
    fdir = work / "frames"
    fdir.mkdir(exist_ok=True)
    paths = {}
    want = set(samp)
    for k, fr in media_io.frames(src):
        if k in want:
            paths[k] = media_io.save_frame(fr, fdir / f"f_{k:06d}.jpg")
    order = sorted(paths)
    dets = detectors(args.detectors)
    # detections are cached per (samples, detectors, mapping): a re-render
    # after a render/tracking change must not pay for detection again
    sig = hashlib.sha256(json.dumps(
        [order, [m for m, _n in dets], em.sha256 if em else None,
         (HERE.parent / "prompts" / "video_frame_detection_v1.txt"
          ).read_text()]).encode()).hexdigest()
    cache = work / "detections.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    if cached.get("signature") == sig and not args.force_detect:
        by_sample = {int(k): v for k, v in cached["regions"].items()}
        print("  frame detection: cached")
    else:
        common.set_usage_context(f"video_detect:{fid}")
        try:
            reads, regions = frame_detect.detect([paths[k] for k in order],
                                                 em, dets)
        finally:
            common.set_usage_context("")
        by_sample = dict(zip(order, regions))
    errors = [k for k, rs in by_sample.items()
              if any(r.get("kind") == "error" for r in rs)]
    if errors:
        problems.append(f"a detector failed on {len(errors)} sampled "
                        "frame(s) — those frames are unchecked")
    cache.write_text(json.dumps(
        {"signature": sig,
         "regions": {str(k): v for k, v in by_sample.items()}}, indent=1,
        ensure_ascii=False))
    n_reg = sum(len([r for r in rs if r.get("box")])
                for rs in by_sample.values())
    notes.append(f"{len(order)} sampled frames, {n_reg} region(s) found")
    # 4 track (per shot, streaming)
    per_frame = {}
    cur, small = 0, []
    for k, fr in media_io.frames(src):
        small.append(track.small_gray(fr))
        s = shots[cur]
        if k == s["f1"]:
            local = {kk - s["f0"]: [r for r in by_sample[kk] if r.get("box")]
                     for kk in by_sample if s["f0"] <= kk <= s["f1"]}
            res = track.track_shot(small, local, info["width"],
                                   info["height"])
            for t, regs in res.items():
                per_frame[s["f0"] + t] = regs
            small = []
            cur = min(cur + 1, len(shots) - 1)
    held = sum(r["held"] for rs in per_frame.values() for r in rs)
    # a shot is static only if its picture AND every region in it stay put
    for s in shots:
        s["static"] = bool(s["static"]
                           and render_video.regions_still(per_frame, s))
    notes.insert(1 if a_man else 0, f"{len(shots)} shot(s), "
                 f"{sum(s['static'] for s in shots)} static (picture and "
                 "every region still)")
    # 5 render
    plates = {}
    if args.replace == "pseudonym":
        for s in shots:
            if s["static"]:
                plates[s["shot"]] = render_video.static_plates(per_frame, s,
                                                               em)
    vid = work / f"{fid}_video.mp4"
    wr = media_io.Writer(vid, info["fps"], info["width"], info["height"])
    cache = {}
    for k, fr in media_io.frames(src):
        for r in per_frame.get(k, []):
            render_video.blur(fr, r["px"])
        s = shots_mod.shot_of(shots, k)
        for p in plates.get(s["shot"], []):
            render_video.plate(fr, p["px"], p["text"], cache)
        wr.write(fr)
    wr.close()
    n_pl = sum(len(v) for v in plates.values())
    notes.append(f"render: {sum(len(v) for v in per_frame.values())} region-"
                 f"frames blurred ({held} held by the tracker), {n_pl} "
                 "invented-name plate(s) on static shots")
    # 6 mux
    red = work / f"{out_id}_redacted.mp4"
    if a_audio is not None:
        media_io.mux(vid, a_audio, red)
    else:
        shutil.copy2(vid, red)
    # 7 verify
    vmodels = [] if args.no_model_verify else dets[:1]
    common.set_usage_context(f"video_verify:{fid}")
    try:
        vprob, vrep = verify_video.verify(red, src, info, per_frame, em,
                                          work / "verify", vmodels, order)
    finally:
        common.set_usage_context("")
    problems += vprob
    ctrl = vrep["control"]
    notes.append(f"verify: {vrep['ocr_frames']} OCR/face frame(s), "
                 f"{vrep['model_frames']} model frame(s), "
                 f"{len(vrep['findings'])} finding(s); control "
                 + ("caught" if ctrl.get("verifier_ok") else
                    ("NOT caught" if ctrl.get("ran") else
                     "not run: " + ctrl.get("reason", ""))))
    # 8 script
    common.set_usage_context(f"video_script:{fid}")
    try:
        md, rows = ad_script.make(red, shots,
                                  a_txt.read_text() if a_txt else "",
                                  work / "script", dets[0][0], dets[0][1],
                                  out_id)
    finally:
        common.set_usage_context("")
    leaks = ad_script.scan(md, em, None if args.no_audit else
                           __import__("judge_client").DEFAULT_JUDGE)
    if leaks:
        problems.append(f"script names {len(leaks)} real entit(y/ies): "
                        + "; ".join(leaks[:3]))
    # bundle
    status = "PASS" if not problems else "FLAGGED"
    base = Path(args.quarantine_dir) if problems else Path(args.outdir)
    dst = base / out_id
    other = (Path(args.outdir) if problems else Path(args.quarantine_dir)) \
        / out_id
    if (other / "manifest.json").exists():
        shutil.rmtree(other)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(red, dst / f"{out_id}_redacted.mp4")
    (dst / f"{out_id}_script.md").write_text(md)
    if a_txt:
        shutil.copy2(a_txt, dst / f"{out_id}_redacted.txt")
    qc = [f"# QC report — {out_id}", "", f"**Status: {status}**", ""]
    qc += [f"- {n}" for n in notes]
    if problems:
        qc += ["", "## Problems", ""] + [f"- {p}" for p in problems]
    (dst / "qc_report.md").write_text("\n".join(qc) + "\n")
    manifest = {
        "workflow": "video arm: audio arm + frame redaction",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_file": em.pseudonymise(src.name) if em is not None
        else src.name,
        "source_sha256": sha(src), "video": info, "qc_status": status,
        "quarantined_reason": "; ".join(problems[:3]) if problems else None,
        "replace_mode": args.replace,
        "mapping": em.summary() if em is not None else None,
        "shots": [{k: s[k] for k in ("shot", "start_ms", "end_ms", "static",
                                     "motion")} for s in shots],
        "detection": {"sample_fps": args.sample_fps,
                      "sampled_frames": len(order),
                      "detectors": [m for m, _n in dets],
                      "regions": n_reg,
                      "prompt": "video_frame_detection_v1.txt"},
        "render": {"blur": "pixelate+gaussian", "plates": n_pl,
                   "held_region_frames": held},
        "verification": {k: vrep[k] for k in ("ocr_frames", "model_frames")}
        | {"findings": len(vrep["findings"]), "control": ctrl},
        "script": {"shots": len(rows), "real_name_hits": len(leaks)},
        "audio_track": ({k: a_man.get(k) for k in (
            "qc_status", "quarantined_reason", "redaction_count",
            "replacement", "mapped_audit", "judge_audit", "profile_file")}
            if a_man else None),
        "deliverables": [f"{out_id}_redacted.mp4", f"{out_id}_script.md"]
        + ([f"{out_id}_redacted.txt"] if a_txt else []) + ["qc_report.md"],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                  ensure_ascii=False,
                                                  default=str))
    (work / "per_frame_regions.json").write_text(json.dumps(
        {str(k): [r["px"] + [r.get("kind"), r.get("entity")] for r in rs]
         for k, rs in per_frame.items()}))
    (work / "per_frame_summary.json").write_text(json.dumps(
        {"frames_with_regions": len(per_frame), "held": held,
         "plates": plates}, indent=1, default=str))
    print(f"  {'QUARANTINED' if problems else 'delivered'} -> {dst} "
          f"(qc={status})")
    for p in problems[:6]:
        print("   - " + p)
    return manifest


def main():
    argv = sys.argv[1:]
    audio_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, audio_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("media", nargs="+")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--replace", default="beep",
                    choices=["beep", "pseudonym"])
    ap.add_argument("--tts", default=None, choices=["elevenlabs", "say"])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--profile-file", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quarantine-dir", default=None)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--detectors", default="gemini,gpt")
    ap.add_argument("--no-model-verify", action="store_true")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--force-detect", action="store_true",
                    help="redo frame detection even when cached")
    args = ap.parse_args(argv)
    args.audio_args = audio_args
    if args.mapping:
        os.environ["PII_MAPPING_FILE"] = str(Path(args.mapping).resolve())
    if args.replace == "pseudonym" and not args.mapping:
        ap.error("--replace pseudonym needs --mapping")
    os.environ["PII_REPLACE_MODE"] = args.replace
    if args.tts:
        os.environ["PII_TTS_ENGINE"] = args.tts
    import common
    if args.profile_file:
        os.environ["PII_PROFILE_FILE"] = str(Path(args.profile_file).resolve())
        import profiles
        args.profile = profiles.active()["name"]
    args.profile = args.profile or common.CONFIG["profile"]
    args.outdir = args.outdir or str(common.ROOT / "deliveries")
    args.quarantine_dir = args.quarantine_dir or str(common.ROOT /
                                                     "quarantine")
    import entity_map
    em = entity_map.active()
    model = common.resolve_model()
    for m in args.media:
        p = Path(m).resolve()
        if not p.exists():
            print(f"SKIP {m}: not found")
            continue
        try:
            run(p, args, model, em)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ERROR {p.name}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
