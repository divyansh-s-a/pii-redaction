#!/usr/bin/env python3
"""Score a video-arm run on the synthetic ad (tools/synth_video.py) against
its per-frame ground truth.

  blur recall      share of (element, frame) pairs whose GT box is >= 90%
                   covered by blurred regions, by element kind
  blur precision   share of blurred pixels that fall inside some GT box
                   (the rest is over-blur)
  static shots     shot-level static/moving calls vs ground truth
  plates           invented-name plates placed on static shots for mapped
                   text, and none on moving shots
  plus the run's own verification findings, control result and script scan
  (from manifest.json), and the audio track's outcome.

  python3 tools/score_video.py --gt synth_ad.gt.json --work WORK_DIR
      --bundle BUNDLE_DIR [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text())
    work, bundle = Path(a.work), Path(a.bundle)
    regs = json.loads((work / "per_frame_regions.json").read_text())
    summ = json.loads((work / "per_frame_summary.json").read_text())
    man = json.loads((bundle / "manifest.json").read_text())
    W, H = gt["width"], gt["height"]
    by_kind, inside, total_blur = {}, 0, 0
    for k, fr in gt["frames"].items():
        mask = np.zeros((H, W), bool)
        for r in regs.get(k, []):
            x0, y0, x1, y1 = [int(v) for v in r[:4]]
            mask[y0:y1, x0:x1] = True
        gmask = np.zeros((H, W), bool)
        for el in fr["elements"]:
            x0, y0, x1, y1 = [max(0, int(v)) for v in el["box"]]
            gmask[y0:y1, x0:x1] = True
            area = max(1, (x1 - x0) * (y1 - y0))
            cov = mask[y0:y1, x0:x1].sum() / area
            kk = f"shot{fr['shot']}:{el['kind']}"
            b = by_kind.setdefault(kk, [0, 0])
            b[0] += cov >= 0.9
            b[1] += 1
        total_blur += mask.sum()
        inside += (mask & gmask).sum()
    recall = {k: round(v[0] / v[1], 3) for k, v in sorted(by_kind.items())}
    allc = sum(v[0] for v in by_kind.values())
    alln = sum(v[1] for v in by_kind.values())
    shots_run = {s["shot"]: s["static"] for s in man["shots"]}
    shots_gt = {s["shot"]: s["static"] for s in gt["shots"]}
    plates = summ.get("plates", {})
    plate_shots = sorted(int(k) for k, v in plates.items() if v)
    res = {
        "blur_recall_pooled": round(allc / alln, 3) if alln else None,
        "blur_recall_by_shot_kind": recall,
        "blur_precision": round(inside / total_blur, 3) if total_blur
        else None,
        "static_calls": {"run": shots_run, "truth": shots_gt,
                         "correct": sum(shots_run.get(k) == v
                                        for k, v in shots_gt.items()),
                         "of": len(shots_gt)},
        "plates_on_shots": plate_shots,
        "plates_on_moving_shots": [k for k in plate_shots
                                   if not shots_gt.get(k)],
        "qc_status": man["qc_status"],
        "quarantined_reason": man.get("quarantined_reason"),
        "verification": man["verification"],
        "script": man["script"],
        "audio_track": man.get("audio_track"),
        "detection": man["detection"],
    }
    txt = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "score_video.json").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
