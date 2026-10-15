#!/usr/bin/env python3
"""Scene-by-scene script of a redacted ad.

Built ONLY from redacted material — one frame per shot taken from the
redacted video, and the redacted transcript of the audio track — so the
script cannot know a name the deliverables do not show. It is then scanned
like a transcript (mapping matcher + a confirmed judge read): a real name in
the script would mean the redaction itself leaked.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import media_io

PROMPT = (HERE.parent / "prompts" / "video_script_v1.txt").read_text()
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"shots": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"shot": {"type": "integer"},
                       "visual": {"type": "string"},
                       "on_screen_text": {"type": "string"},
                       "camera": {"type": "string"}},
        "required": ["shot", "visual", "on_screen_text", "camera"]}}},
    "required": ["shots"],
}
TS = re.compile(r"^\[(\d+):(\d+):(\d+\.\d+) - (\d+):(\d+):(\d+\.\d+)\] (.*?) - (.*)$")


def _ms(h, m, s):
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def transcript_lines(text):
    out = []
    for line in (text or "").splitlines():
        m = TS.match(line.strip())
        if m:
            out.append({"start_ms": _ms(*m.group(1, 2, 3)),
                        "end_ms": _ms(*m.group(4, 5, 6)),
                        "speaker": m.group(7), "text": m.group(8)})
    return out


def _fmt(ms):
    return f"{ms // 60000:02d}:{ms % 60000 / 1000:04.1f}"


def make(red_video, shots, transcript_text, work, model, native, title):
    import judge_client
    work.mkdir(parents=True, exist_ok=True)
    mids = {s["shot"]: (s["f0"] + s["f1"]) // 2 for s in shots}
    want = set(mids.values())
    frames = {}
    for k, fr in media_io.frames(red_video):
        if k in want:
            frames[k] = fr
        if len(frames) == len(want):
            break
    desc = {}
    order = [s for s in shots if mids[s["shot"]] in frames]
    for b in range(0, len(order), 6):
        chunk = order[b: b + 6]
        paths = [media_io.save_frame(frames[mids[s["shot"]]],
                                     work / f"shot_{s['shot']}.jpg")
                 for s in chunk]
        listing = "\n".join(f"image {i}: shot {s['shot']} "
                            f"({_fmt(s['start_ms'])}-{_fmt(s['end_ms'])}, "
                            f"{'static' if s['static'] else 'moving'})"
                            for i, s in enumerate(chunk))
        raw = judge_client.judge_call(PROMPT + "\n\n" + listing, SCHEMA,
                                      model=model, images=[str(p) for p in
                                                           paths],
                                      native_gemini=native,
                                      log_engine="video_script",
                                      max_tokens=4000)
        for d in raw.get("shots", []):
            desc[d.get("shot")] = d
    lines = transcript_lines(transcript_text)
    md = [f"# Ad script — {title}", "",
          "Built from the redacted video and its redacted transcript only. "
          "Brand names shown are the invented ones; blurred areas are "
          "described as blurred.", ""]
    rows = []
    for s in shots:
        d = desc.get(s["shot"], {})
        vo = [f"{l['speaker']}: {l['text']}" for l in lines
              if l["start_ms"] < s["end_ms"] and s["start_ms"] < l["end_ms"]]
        md += [f"## Shot {s['shot']} · {_fmt(s['start_ms'])}–"
               f"{_fmt(s['end_ms'])} · {'static' if s['static'] else 'moving'}",
               "", f"- **Visual:** {d.get('visual', '(not described)')}",
               f"- **Camera:** {d.get('camera', '')}",
               f"- **On screen:** {d.get('on_screen_text') or '—'}",
               "- **Sound:** " + (" / ".join(vo) if vo else "(no speech)"),
               ""]
        rows.append({"shot": s["shot"], "start_ms": s["start_ms"],
                     "end_ms": s["end_ms"], "static": s["static"],
                     "visual": d.get("visual"), "camera": d.get("camera"),
                     "on_screen_text": d.get("on_screen_text"), "sound": vo})
    return "\n".join(md) + "\n", rows


def scan(md, em, model=None):
    """Real names readable in the script: matcher hits + confirmed judge."""
    if em is None:
        return []
    found = [f"line {h['line']}: '{h['quoted_text']}' ({h['entity']})"
             for h in em.scan_text(md)]
    if model:
        sys.path.insert(0, str(HERE.parent.parent / "tools"))
        import mapped_audit
        first = mapped_audit.judge_read(em, md, model)
        if first:
            second = mapped_audit.judge_read(em, md, model)
            for f in first:
                if any(mapped_audit._same(f, g) for g in second):
                    found.append(f"line {f['line']}: '{f['quoted_text']}' "
                                 f"(judge)")
    return found
