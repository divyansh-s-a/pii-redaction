#!/usr/bin/env python3
"""Quarantine review actions: extend a beep edge, and release a reviewed file.

Used by the local web app (tools/webapp.py) and directly from the CLI:

    python3 tools/release.py extend  <file_id> <beep_start_ms> <beep_end_ms> \\
                                     start|end [--ms 400]
    python3 tools/release.py release <file_id> --note "reviewer decision text"
    python3 tools/release.py keep    <file_id> --note "why it stays held"

`extend` re-renders the quarantined file's beeped audio from the ORIGINAL
source with the chosen edge grown by --ms, updates rendered_intervals.json and
the quarantine copy, and records the action in the manifest. `release` moves
the folder to deliveries/ with the human decision recorded — the manifest
keeps the full gate history, so provenance shows exactly what a human
overrode and why.
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import audio_utils   # noqa: E402
import common        # noqa: E402


def _qdir(fid):
    d = ROOT / "quarantine" / fid
    if not (d / "manifest.json").exists():
        raise SystemExit(f"no quarantined file '{fid}'")
    return d


def _manifest(d):
    return json.loads((d / "manifest.json").read_text())


def _save_manifest(d, m):
    (d / "manifest.json").write_text(json.dumps(m, indent=2,
                                                ensure_ascii=False))


def _out_dir(m, fid):
    """The pipeline output dir this delivery was rendered from."""
    eng = "gemini" if "FALLBACK" in (m["transcription"]["engine"] or "") else \
        m["transcription"]["engine"].split()[0]
    sfx = "" if eng == "gemini" else "__elevenlabs"
    return ROOT / "output" / (fid + sfx)


def _log_review(m, entry):
    m.setdefault("human_review", {"actions": []})
    entry["at"] = datetime.datetime.now().isoformat(timespec="seconds")
    m["human_review"]["actions"].append(entry)


def extend(fid, beep_s, beep_e, side, ms=400):
    """Grow one beep edge by ms and re-render the quarantined audio."""
    q = _qdir(fid)
    m = _manifest(q)
    out = _out_dir(m, fid)
    ri = common.load_json(out / "rendered_intervals.json")
    merged = [(x["start_ms"], x["end_ms"], x.get("channel"))
              for x in ri["rendered_merged"]]
    hit = None
    for i, (s, e, c) in enumerate(merged):
        if abs(s - beep_s) <= 60 and abs(e - beep_e) <= 60:
            hit = i
            break
    if hit is None:
        raise SystemExit(f"no rendered beep near [{beep_s},{beep_e}]")
    s, e, c = merged[hit]
    merged[hit] = ((max(0, s - ms), e, c) if side == "start"
                   else (s, e + ms, c))
    original = Path(m["source_file"])
    nat, bp = out / "_rev_native.wav", out / "_rev_beeped.wav"
    audio_utils.decode_to_wav(original, nat)
    audio_utils.mix_beeps(nat, bp, merged, common.CONFIG["beep_freq_hz"],
                          common.CONFIG["beep_gain_db"])
    beeped_name = common.beeped_name(fid)
    audio_utils.encode_audio(bp, out / beeped_name)
    nat.unlink(missing_ok=True)
    bp.unlink(missing_ok=True)
    ri["rendered_merged"] = [{"start_ms": a, "end_ms": b, "channel": ch}
                             for a, b, ch in merged]
    ri["human_extended"] = True
    common.save_json(out / "rendered_intervals.json", ri)
    shutil.copy2(out / beeped_name, q / beeped_name)
    _log_review(m, {"action": "extend", "beep": [beep_s, beep_e],
                    "side": side, "ms": ms,
                    "new_beep": list(merged[hit][:2])})
    _save_manifest(q, m)
    print(f"extended beep {hit} ({side} +{ms}ms) -> {merged[hit][:2]}; "
          "audio re-rendered")
    return merged[hit]


def release(fid, note):
    """Move a reviewed quarantine folder to deliveries/ with the decision."""
    q = _qdir(fid)
    m = _manifest(q)
    _log_review(m, {"action": "release", "note": note,
                    "overrode": m.get("quarantined_reason")})
    m["quarantined_reason"] = None
    _save_manifest(q, m)
    dst = ROOT / "deliveries" / fid
    if dst.exists():
        raise SystemExit(f"deliveries/{fid} already exists")
    shutil.move(str(q), str(dst))
    print(f"released -> {dst} (human decision recorded in manifest)")
    return dst


def keep(fid, note):
    q = _qdir(fid)
    m = _manifest(q)
    _log_review(m, {"action": "keep", "note": note})
    _save_manifest(q, m)
    print("kept in quarantine; decision recorded")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extend")
    e.add_argument("fid")
    e.add_argument("beep_start_ms", type=int)
    e.add_argument("beep_end_ms", type=int)
    e.add_argument("side", choices=["start", "end"])
    e.add_argument("--ms", type=int, default=400)
    r = sub.add_parser("release")
    r.add_argument("fid")
    r.add_argument("--note", required=True)
    k = sub.add_parser("keep")
    k.add_argument("fid")
    k.add_argument("--note", required=True)
    a = ap.parse_args()
    if a.cmd == "extend":
        extend(a.fid, a.beep_start_ms, a.beep_end_ms, a.side, a.ms)
    elif a.cmd == "release":
        release(a.fid, a.note)
    else:
        keep(a.fid, a.note)


if __name__ == "__main__":
    main()
