#!/usr/bin/env python3
"""Byte-identity check: the default beep mode of THIS checkout must produce
exactly the deliverables another ref (default: main) produces.

Runs tools/offline_harness.py (full delivery workflow, network stages
replaced by the synthetic clips' answer keys) once in a temporary git
worktree of the ref and once here, then compares sha256 of every delivered
file (beeped audio, redacted transcript, QC report), the manifest without its
timestamp, and the work artifacts intervals.json and rendered_intervals.json.

  python3 tools/check_beep_identity.py --media DIR [--ref main] [--out DIR]
Audio that differs is decoded and compared sample by sample: macOS's AAC
decoder (afconvert) is not bit-reproducible between invocations — the same
.m4a decodes with a few +/-1 LSB differences from one run to the next, in
either checkout — so an audio file counts as the SAME AUDIO when at most
0.1% of samples differ, by at most 1 LSB. Anything larger is a difference.
Exit 1 on any difference.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_tree(tree: Path, media: Path, out: Path, work_dir: Path, extra=()):
    env = dict(os.environ)
    for k in ("PII_MAPPING_FILE", "PII_PROFILE_FILE", "PII_REPLACE_MODE",
              "PII_TTS_ENGINE"):
        env.pop(k, None)
    env["PII_WORK_DIR"] = str(work_dir)       # ignored by a tree without it
    r = subprocess.run([sys.executable, str(ROOT / "tools" /
                                            "offline_harness.py"),
                        "--tree", str(tree), "--media", str(media),
                        "--out", str(out), *extra],
                       capture_output=True, text=True, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith("HARNESS_RESULT ")]
    if r.returncode or not line:
        sys.exit(f"harness failed in {tree}:\n{r.stdout[-2000:]}\n"
                 f"{r.stderr[-3000:]}")
    return json.loads(line[-1].split(" ", 1)[1])


def pcm_compare(a: Path, b: Path):
    """(n_differing_samples, n_samples, max_abs_lsb) of two audio files."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import array
    import audio_utils
    with tempfile.TemporaryDirectory() as td:
        wa, wb = Path(td) / "a.wav", Path(td) / "b.wav"
        audio_utils.decode_to_wav(a, wa)
        audio_utils.decode_to_wav(b, wb)
        pa, _, _ = audio_utils.read_wav(wa)
        pb, _, _ = audio_utils.read_wav(wb)
    if len(pa) != len(pb):
        return None, max(len(pa), len(pb)), None
    nd, mx = 0, 0
    for x, y in zip(pa, pb):
        if x != y:
            nd += 1
            mx = max(mx, abs(x - y))
    return nd, len(pa), mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--ref", default="main")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    media = Path(a.media).resolve()
    out = Path(a.out or tempfile.mkdtemp(prefix="beep_identity_")).resolve()
    wt = out / f"worktree_{a.ref.replace('/', '_')}"
    if not wt.exists():
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach",
                        str(wt), a.ref], check=True, capture_output=True)
    # three variants: plain; a detector that misses a phone number's last
    # digit (QC digit-edge -> auto-extension RE-RENDER); and --assurance
    variants = {"plain": [], "phone_tail_miss": ["--phone-tail-miss"],
                "assurance": ["--assurance", "--phone-tail-miss"]}
    ref, here = {}, {}
    try:
        for vn, extra in variants.items():
            r1 = run_tree(wt, media, out / vn / "ref", out / vn / "ref_work",
                          extra)
            r2 = run_tree(ROOT, media, out / vn / "branch",
                          out / vn / "branch_work", extra)
            ref.update({f"{vn}:{c}": v for c, v in r1.items()})
            here.update({f"{vn}:{c}": v for c, v in r2.items()})
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove",
                        "--force", str(wt)], capture_output=True)
    diffs, total, noise = [], 0, []
    for clip in ref:
        for name, h in ref[clip]["files"].items():
            total += 1
            h2 = here.get(clip, {}).get("files", {}).get(name)
            ok = h == h2
            tag = "SAME" if ok else "DIFF"
            if not ok and name.endswith((".flac", ".m4a", ".wav")):
                nd, n, mx = pcm_compare(Path(ref[clip]["bundle"]) / name,
                                        Path(here[clip]["bundle"]) / name)
                if nd is not None and nd <= 0.001 * n and mx <= 1:
                    ok = True
                    noise.append(f"{clip}/{name}: {nd}/{n} samples, "
                                 f"max {mx} LSB")
                    tag = "SAME AUDIO"
            if not ok:
                diffs.append(f"{clip}/{name}")
            print(f"  {tag}  {clip}/{name}  "
                  f"{(h or '-')[:12]} {(h2 or '-')[:12]}")
        print(f"  {clip}: verdict ref={ref[clip]['verdict']} "
              f"here={here[clip]['verdict']}, qc ref={ref[clip]['qc']} "
              f"here={here[clip]['qc']}")
    print(f"\n{total - len(diffs) - len(noise)}/{total} artifacts "
          f"byte-identical to {a.ref}; {len(noise)} audio file(s) the same "
          f"audio up to AAC-decoder noise"
          + (f" ({'; '.join(noise)})" if noise else "")
          + (f"; DIFFERENT: {diffs}" if diffs else ""))
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
