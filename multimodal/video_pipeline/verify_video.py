#!/usr/bin/env python3
"""Re-read the REDACTED video and report anything still identifiable.

Detected is not removed: every claim is checked on the output, on frames the
detector did NOT sample (an offset grid), by three independent means:
  1. OCR (Apple Vision) + the mapping matcher: a readable real name, or a
     token containing one (a URL), or a readable phone number;
  2. faces (Apple Vision) on the output, judged by SHARPNESS against the same
     region of the original frame — a blurred face may still be found as a
     face, but it no longer carries detail;
  3. a vision model re-check (prompts/video_leak_check_v1.txt) on a sparser
     offset grid, with the invented names as an allowlist.
And a deliberately broken CONTROL: a cosmetic redaction (a translucent box
over each detected region of a few original frames) must FAIL checks 1-2.
If it passes, the verifier is broken and the run is flagged.
"""
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frame_detect
import media_io

PROMPT = (HERE.parent / "prompts" / "video_leak_check_v1.txt").read_text()
OCR_FPS = 3.0
MODEL_FPS = 1.0
SHARP_RATIO = 0.35       # output face detail / original face detail
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"frames": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"frame": {"type": "integer"},
                       "findings": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False,
                           "properties": {
                               "kind": {"type": "string", "enum": [
                                   "entity", "logo", "face", "phone_url"]},
                               "entity_id": {"type": "string"},
                               "description": {"type": "string"}},
                           "required": ["kind", "entity_id",
                                        "description"]}}},
        "required": ["frame", "findings"]}}},
    "required": ["frames"],
}


def _detail(gray):
    import cv2
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _grab(path, indices):
    want = set(indices)
    got = {}
    for k, fr in media_io.frames(path):
        if k in want:
            got[k] = fr
        if len(got) == len(want):
            break
    return got


def ocr_face_check(paths, frames_out, frames_src, em):
    """Checks 1-2 on saved output frames. Returns findings."""
    import cv2
    items = sorted(paths.items())              # {frame index: image path}
    reads = frame_detect.read_frames([p for _k, p in items])
    out = []
    inv = em._invented_rx if em is not None else None
    for (k, _p), rd in zip(items, reads):
        text = "\n".join(ln["text"] for ln in rd["lines"])
        if em is not None:
            for h in em.scan_text(text):
                out.append({"frame": k, "kind": "entity",
                            "detail": f"readable '{h['quoted_text']}' "
                                      f"({h['entity']}, {h['match']})"})
            for ln in rd["lines"]:
                for w in ln["text"].split():
                    if inv is not None and inv.fullmatch(w.strip(".,")):
                        continue
                    kk = frame_detect._key(w)
                    for e in em.entries:
                        if any(len(ek) >= 5 and ek in kk and len(kk) > len(ek)
                               for ek in e["keys"]):
                            out.append({"frame": k, "kind": "entity",
                                        "detail": f"token containing a real "
                                                  f"name ({e['id']})"})
        if sum(c.isdigit() for c in text) >= 7 and frame_detect.PHONE_RX.search(
                text):
            out.append({"frame": k, "kind": "phone_url",
                        "detail": "readable phone-like number"})
        fo, fs = frames_out.get(k), frames_src.get(k)
        for f in rd["faces"]:
            if fo is None or fs is None:
                continue
            h, w = fo.shape[:2]
            x0, y0, x1, y1 = [int(v) for v in (f["box"][0] * w, f["box"][1] * h,
                                               f["box"][2] * w, f["box"][3] * h)]
            go = cv2.cvtColor(fo[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            gs = cv2.cvtColor(fs[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            d_src = _detail(gs)
            if d_src > 20 and _detail(go) > SHARP_RATIO * d_src:
                out.append({"frame": k, "kind": "face",
                            "detail": f"face with detail at {f['box']}"})
    return out


def model_check(paths, em, model, native):
    import judge_client
    block = em.prompt_block() if em is not None else "ENTITIES: (none)"
    allow = "; ".join(em.invented_terms()) if em is not None else ""
    out = []
    items = list(paths.items())
    for b in range(0, len(items), 6):
        chunk = items[b: b + 6]
        prompt = (PROMPT + "\n\n" + block + "\n\nINVENTED NAMES (never "
                  "findings): " + allow + "\n\nThe images follow (frame 0 "
                  "first).")
        try:
            raw = judge_client.judge_call(prompt, SCHEMA, model=model,
                                          images=[str(p) for _k, p in chunk],
                                          native_gemini=native,
                                          log_engine="video_verify",
                                          max_tokens=4000)
        except Exception as exc:
            out.append({"frame": chunk[0][0], "kind": "unchecked",
                        "detail": f"model check failed: {str(exc)[:100]}"})
            continue
        for fr in raw.get("frames", []):
            i = fr.get("frame")
            if not isinstance(i, int) or not (0 <= i < len(chunk)):
                continue
            for f in fr.get("findings", []):
                d = str(f.get("description", ""))
                if allow and re.search(re.escape(d.strip()), allow, re.I):
                    continue
                out.append({"frame": chunk[i][0], "kind": f.get("kind"),
                            "detail": "model: " + d[:120],
                            "entity": f.get("entity_id")})
    return out


def control(frames_src, regions, work, em):
    """Cosmetic redaction of a few original frames must FAIL checks 1-2."""
    import cv2
    picked = [k for k in sorted(regions)
              if any(r.get("px") and r.get("kind") in ("brand_text", "face",
                                                       "url_handle", "phone")
                     and r.get("source", "").startswith(("ocr", "vision"))
                     for r in regions[k])][:4]
    if not picked:
        return {"ran": False, "reason": "no OCR- or Vision-found region to "
                                        "test the verifier on"}
    paths, outs = {}, {}
    for k in picked:
        fr = frames_src[k].copy()
        for r in regions[k]:
            if not r.get("px"):
                continue
            x0, y0, x1, y1 = r["px"]
            over = fr[y0:y1, x0:x1].astype(np.float32)
            fr[y0:y1, x0:x1] = (0.65 * over + 0.35 * 0).astype(np.uint8)
        outs[k] = fr
        paths[k] = media_io.save_frame(fr, work / f"control_{k}.jpg")
    found = ocr_face_check(paths, outs, frames_src, em)
    return {"ran": True, "frames": picked, "findings": len(found),
            "verifier_ok": len(found) > 0}


def verify(out_video, src_video, info, regions_by_frame, em, work,
           models=(), detect_samples=()):
    """Returns (problems, report)."""
    fps, n = info["fps"], info["frames"]
    work.mkdir(parents=True, exist_ok=True)
    det = set(detect_samples)
    step = max(1, int(round(fps / OCR_FPS)))
    ocr_idx = [k for k in range(step // 2, n, step) if k not in det]
    mstep = max(1, int(round(fps / MODEL_FPS)))
    model_idx = [k for k in range(mstep // 2, n, mstep) if k not in det]
    allidx = sorted(set(ocr_idx) | set(model_idx))
    fo = _grab(out_video, allidx)
    fs = _grab(src_video, sorted(set(ocr_idx) | set(regions_by_frame)))
    p_ocr = {k: media_io.save_frame(fo[k], work / f"v_{k}.jpg")
             for k in ocr_idx if k in fo}
    findings = ocr_face_check(p_ocr, fo, fs, em)
    p_model = {k: (p_ocr.get(k) or media_io.save_frame(fo[k],
                                                       work / f"v_{k}.jpg"))
               for k in model_idx if k in fo}
    for model, native in models:
        findings += model_check(p_model, em, model, native)
    ctrl = control(fs, {k: regions_by_frame[k] for k in regions_by_frame
                        if k in fs}, work, em)
    problems = []
    for f in findings:
        t = f["frame"] / fps
        problems.append(f"{f['kind']} at {t:.2f}s (frame {f['frame']}): "
                        f"{f['detail']}")
    if ctrl.get("ran") and not ctrl.get("verifier_ok"):
        problems.append("CONTROL PASSED: a cosmetic redaction was not caught "
                        "— the verifier is broken")
    report = {"ocr_frames": len(p_ocr), "model_frames": len(p_model)
              if models else 0, "findings": findings, "control": ctrl}
    return problems, report
