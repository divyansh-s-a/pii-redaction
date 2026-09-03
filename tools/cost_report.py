#!/usr/bin/env python3
"""End-to-end cost per configuration, from MEASURED usage logs.

Reads output/api_usage.jsonl (optionally from --since-offset, the byte offset
recorded when a run started) and computes, per configuration and per
audio-hour: transcription $, detection $, and the shared safety-gate overhead
(edge-verify snippets + judge audit + pverify). Sources, in order of trust:
  MEASURED   OpenRouter rows carry cost_usd (usage accounting); vendor audio
             rows carry measured seconds billed at the published per-hour rate
  DERIVED    Gemini rows carry token counts; billed at the price constants
             below (the same constants reproduce report v1's measured $0.58
             4-minute sample)
  ESTIMATE   OpenRouter rows WITHOUT cost_usd (pre-accounting cache entries)
             fall back to token prices marked as estimates

Usage:
  python3 tools/cost_report.py [--since-offset N] [--out output/cost]
  python3 tools/cost_report.py --self-test
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------- prices
EL_PER_HR = 0.22            # published scribe_v2 rate (keyterms ~ +$0.05/hr)
WHISPER_PER_HR = 0.36       # measured via OpenRouter per-call pricing
GEMINI_PRO_IN = 2.00        # $/1M prompt tokens   (reproduces v1's $0.58/4min)
GEMINI_PRO_OUT = 12.00      # $/1M output+thinking tokens
GEMINI_FLASH_IN = 0.30      # $/1M — edge-verify snippet ASR (estimate)
GEMINI_FLASH_OUT = 2.50
OPENROUTER_FALLBACK = {     # $/1M (in, out) when cost_usd missing — ESTIMATE
    "openai/gpt-5.6-sol": (1.25, 10.00),
    "openai/gpt-5.6-luna": (0.60, 4.80),
}

CONFIGS = {
    "elevenlabs": "ElevenLabs + Gemini (production)",
    "gemini": "Gemini end-to-end",
    "openai": "Whisper + Gemini",
    "elevenlabs__gpt": "ElevenLabs + GPT detector",
    "openai__gpt": "Whisper + GPT detector",
}


def gemini_cost(row):
    model = row.get("model") or ""
    pin, pout = ((GEMINI_FLASH_IN, GEMINI_FLASH_OUT) if "flash" in model
                 else (GEMINI_PRO_IN, GEMINI_PRO_OUT))
    out_tok = (row.get("output_tokens") or 0) + (row.get("thinking_tokens") or 0)
    return ((row.get("prompt_tokens") or 0) * pin
            + out_tok * pout) / 1e6


def openrouter_cost(row):
    """(usd, measured?) for a judge/detector row."""
    c = row.get("cost_usd")
    if isinstance(c, (int, float)) and c >= 0:
        return float(c), True
    pin, pout = OPENROUTER_FALLBACK.get(row.get("model") or "", (1.0, 5.0))
    usd = ((row.get("prompt_tokens") or 0) * pin
           + (row.get("output_tokens") or 0) * pout) / 1e6
    return usd, False


def _fid_from_file(path_str):
    stem = Path(str(path_str)).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")


def parse_rows(log_path, since_offset=0):
    raw = Path(log_path).read_bytes()[since_offset:]
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _ctx(row):
    """(kind, fid, config) from a context tag like 'detect:<fid>__openai__gpt'."""
    tag = row.get("context") or ""
    if ":" not in tag:
        return None, None, None
    kind, rest = tag.split(":", 1)
    rest = rest.split(":")[0]          # pverify:<fid>:<config> keeps 2nd part
    parts = tag.split(":")
    if kind == "pverify" and len(parts) >= 3:
        return kind, parts[1], parts[2]
    m = re.match(r"(.+?)((?:__[a-z]+)*)$", rest)
    fid, sfx = m.group(1), m.group(2)
    config = sfx.lstrip("_").replace("__", "__") if sfx else "gemini"
    if sfx:
        config = sfx[2:]
    return kind, fid, config


def collect(rows):
    """-> per-config/per-fid dollar buckets + gate overhead + audio seconds."""
    agg = {
        "transcribe": {},   # (config, fid) -> usd
        "detect": {},       # (config, fid) -> usd
        "gates": {},        # (kind, fid) -> usd   kind in edge_verify/audit/pverify
        "audio_seconds": {},  # (engine, fid) -> seconds
        "estimated_rows": 0,
        "measured_rows": 0,
    }

    def add(d, key, usd):
        d[key] = d.get(key, 0.0) + usd

    for r in rows:
        eng = r.get("engine")
        if eng in ("elevenlabs", "elevenlabs_entity", "openai_via_openrouter"):
            f = r.get("file") or ""
            if "_probe" in f or not r.get("audio_seconds"):
                continue
            add(agg["audio_seconds"], (eng, _fid_from_file(f)),
                float(r["audio_seconds"]))
            continue
        kind, fid, config = _ctx(r)
        if eng == "gemini":
            usd = gemini_cost(r)
            if kind == "transcribe":
                add(agg["transcribe"], (config, fid), usd)
            elif kind == "detect":
                add(agg["detect"], (config, fid), usd)
            elif kind == "edge_verify":
                add(agg["gates"], ("edge_verify", fid), usd)
            # untagged gemini rows (assurance etc.) are reported in the json
        elif eng in ("judge", "detector_gpt"):
            usd, measured = openrouter_cost(r)
            agg["measured_rows" if measured else "estimated_rows"] += 1
            if eng == "detector_gpt" and kind == "detect":
                add(agg["detect"], (config, fid), usd)
            elif kind == "audit":
                add(agg["gates"], ("audit", fid), usd)
            elif kind == "pverify":
                add(agg["gates"], ("pverify", fid), usd)
            else:
                add(agg["gates"], ("judge_other", fid or "-"), usd)
    return agg


def durations_hours(pred_root):
    """fid -> audio hours, from any config dir's prepare.json."""
    out = {}
    for d in sorted(Path(pred_root).iterdir()):
        p = d / "prepare.json"
        if not d.is_dir() or not p.exists():
            continue
        doc = json.loads(p.read_text())
        fid = doc.get("file_id") or d.name
        out.setdefault(fid, (doc.get("duration_ms") or 0) / 3600000.0)
    return out


def build(agg, hours):
    total_hours = sum(hours.values()) or None
    per_config = {}
    for cfg, label in CONFIGS.items():
        tr = sum(v for (c, f), v in agg["transcribe"].items() if c == cfg)
        det = sum(v for (c, f), v in agg["detect"].items() if c == cfg)
        # vendor-audio transcription billed by seconds
        if cfg.startswith("elevenlabs"):
            secs = sum(v for (e, f), v in agg["audio_seconds"].items()
                       if e == "elevenlabs")
            tr += secs / 3600.0 * EL_PER_HR
        if cfg.startswith("openai"):
            secs = sum(v for (e, f), v in agg["audio_seconds"].items()
                       if e == "openai_via_openrouter")
            tr += secs / 3600.0 * WHISPER_PER_HR
        per_config[cfg] = {"label": label, "transcription_usd": tr,
                           "detection_usd": det, "total_usd": tr + det}
    gates = {}
    for (kind, fid), v in agg["gates"].items():
        gates[kind] = gates.get(kind, 0.0) + v
    gates_total = sum(v for k, v in gates.items() if k != "judge_other")
    res = {"per_config": per_config, "gates_usd": gates,
           "gates_total_usd": gates_total, "audio_hours": total_hours,
           "hours_by_file": hours,
           "estimated_rows": agg["estimated_rows"],
           "measured_rows": agg["measured_rows"]}
    if total_hours:
        for cfg, c in per_config.items():
            c["per_audio_hour_usd"] = c["total_usd"] / total_hours
        res["gates_per_audio_hour_usd"] = gates_total / total_hours
    return res


def build_md(res):
    L = ["# End-to-end cost per configuration",
         "",
         "Generated %s by `tools/cost_report.py` | corpus: %.2f audio-hours "
         "| OpenRouter rows measured/estimated: %d/%d"
         % (datetime.date.today().isoformat(), res.get("audio_hours") or 0,
            res["measured_rows"], res["estimated_rows"]),
         "",
         "| config | transcription $ | detection $ | total $ | $/audio-hour |",
         "|---|---|---|---|---|"]
    for cfg, c in res["per_config"].items():
        L.append("| %s | %.2f | %.2f | %.2f | %.2f |" % (
            c["label"], c["transcription_usd"], c["detection_usd"],
            c["total_usd"], c.get("per_audio_hour_usd", 0.0)))
    L += ["",
          "Safety-gate overhead (production deliveries; applies to whichever "
          "config ships): $%.2f total, $%.2f per audio-hour"
          % (res["gates_total_usd"], res.get("gates_per_audio_hour_usd", 0.0)),
          "",
          "| gate | $ |", "|---|---|"]
    for k, v in sorted(res["gates_usd"].items()):
        L.append("| %s | %.2f |" % (k, v))
    L += ["",
          "Prices: ElevenLabs $%.2f/hr (published rate x measured seconds), "
          "Whisper $%.2f/hr (measured), Gemini Pro $%.2f/$%.2f per 1M "
          "in/out+thinking (derived; reproduces the v1 measured sample), "
          "Flash $%.2f/$%.2f per 1M (estimate), OpenRouter judge/detector "
          "rows measured via usage accounting where available."
          % (EL_PER_HR, WHISPER_PER_HR, GEMINI_PRO_IN, GEMINI_PRO_OUT,
             GEMINI_FLASH_IN, GEMINI_FLASH_OUT)]
    return "\n".join(L) + "\n"


def run(args):
    rows = parse_rows(ROOT / "output" / "api_usage.jsonl", args.since_offset)
    agg = collect(rows)
    hours = durations_hours(args.pred_root)
    res = build(agg, hours)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost_report.json").write_text(
        json.dumps(res, indent=2, default=str))
    md = build_md(res)
    (out / "cost_report.md").write_text(md)
    print(md)
    return 0


def self_test():
    rows = [
        # EL audio: 600 s
        {"engine": "elevenlabs", "file": "x/abc.mp3", "audio_seconds": 600},
        # whisper audio: 600 s
        {"engine": "openai_via_openrouter", "file": "x/abc.mp3",
         "audio_seconds": 600},
        # gemini transcription for the gemini config: 10k in, 40k out+think
        {"engine": "gemini", "model": "gemini-3.1-pro-preview",
         "context": "transcribe:abc", "prompt_tokens": 10000,
         "output_tokens": 30000, "thinking_tokens": 10000},
        # gemini detection on the EL transcript
        {"engine": "gemini", "model": "gemini-3.1-pro-preview",
         "context": "detect:abc__elevenlabs", "prompt_tokens": 5000,
         "output_tokens": 1000, "thinking_tokens": 0},
        # gemini detection on gemini transcript
        {"engine": "gemini", "model": "gemini-3.1-pro-preview",
         "context": "detect:abc", "prompt_tokens": 5000,
         "output_tokens": 1000, "thinking_tokens": 0},
        # gpt detector on whisper transcript, measured cost
        {"engine": "detector_gpt", "model": "openai/gpt-5.6-luna",
         "context": "detect:abc__openai__gpt", "prompt_tokens": 4000,
         "output_tokens": 1000, "cost_usd": 0.0123},
        # edge-verify flash snippets
        {"engine": "gemini", "model": "gemini-3.5-flash",
         "context": "edge_verify:abc", "prompt_tokens": 3000,
         "output_tokens": 500, "thinking_tokens": 250},
        # audit, measured
        {"engine": "judge", "model": "openai/gpt-5.6-sol",
         "context": "audit:abc", "prompt_tokens": 2000, "output_tokens": 500,
         "cost_usd": 0.02},
        # audit verifier, no cost -> estimate
        {"engine": "judge", "model": "openai/gpt-5.6-luna",
         "context": "audit:abc", "prompt_tokens": 1000, "output_tokens": 200},
        # probe row must be ignored
        {"engine": "elevenlabs", "file": "output/_probe.wav",
         "audio_seconds": 2},
    ]
    agg = collect(rows)
    res = build(agg, {"abc": 1.0})
    ok = True

    def case(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print("%s  %s%s" % ("PASS" if cond else "FAIL", name,
                            "  [%s]" % detail if detail and not cond else ""))

    el = res["per_config"]["elevenlabs"]
    case("EL config: audio 600s -> $0.0367 + detect",
         abs(el["transcription_usd"] - 600/3600*EL_PER_HR) < 1e-9
         and abs(el["detection_usd"] - (5000*2 + 1000*12)/1e6) < 1e-9,
         str(el))
    gm = res["per_config"]["gemini"]
    case("gemini config: transcription tokens + own detect",
         abs(gm["transcription_usd"] - (10000*2 + 40000*12)/1e6) < 1e-9
         and abs(gm["detection_usd"] - (5000*2 + 1000*12)/1e6) < 1e-9, str(gm))
    wg = res["per_config"]["openai__gpt"]
    case("whisper+gpt: measured detector cost used",
         abs(wg["detection_usd"] - 0.0123) < 1e-9
         and abs(wg["transcription_usd"] - 600/3600*WHISPER_PER_HR) < 1e-9,
         str(wg))
    case("gates: edge_verify flash-priced + audit measured + estimate fallback",
         abs(res["gates_usd"]["edge_verify"]
             - (3000*GEMINI_FLASH_IN + 750*GEMINI_FLASH_OUT)/1e6) < 1e-9
         and abs(res["gates_usd"]["audit"]
                 - (0.02 + (1000*0.60 + 200*4.80)/1e6)) < 1e-9,
         str(res["gates_usd"]))
    case("measured/estimated row counts", res["measured_rows"] == 2
         and res["estimated_rows"] == 1)
    case("per-audio-hour populated",
         abs(res["per_config"]["elevenlabs"]["per_audio_hour_usd"]
             - el["total_usd"]) < 1e-9)
    md = build_md(res)
    case("markdown renders", "$/audio-hour" in md and "Safety-gate" in md)
    print("\nself-test:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since-offset", type=int, default=0,
                    help="byte offset into api_usage.jsonl where the run "
                         "started (0 = whole log)")
    ap.add_argument("--pred-root", default=str(ROOT / "output"))
    ap.add_argument("--out", default=str(ROOT / "output" / "cost"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    sys.exit(self_test() if args.self_test else run(args))


if __name__ == "__main__":
    main()
