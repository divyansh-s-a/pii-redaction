#!/usr/bin/env python3
"""Phase A — independent LLM-judge leak audit of redacted transcripts.

For every engine x file that produced a <file_id>_redacted.txt (gemini at
output/<fid>/, elevenlabs at output/<fid>__elevenlabs/, openai at
output/<fid>__openai/), the judge (a different model family from the pipeline,
via pipeline/judge_client.py) reads the NUMBERED redacted transcript with the
full detection policy (prompts/judge_leak_audit_prompt_v1.txt) and reports PII
values that SURVIVED redaction.

Findings are validated before they count (hallucination guard): a finding is
dropped unless its quoted_text appears — after whitespace normalization — in
the cited line +/- 1 line of the actual transcript; categories outside
policy.CATEGORIES are dropped, and a finding with no valid category left is
dropped. Dropped findings are counted and reported.

Validated findings then pass an adversarial VERIFICATION PASS (default ON,
--no-verify to skip): judge_client.verify_findings sends ONE batched call per
file to the verifier model (default openai/gpt-5.6-luna, --verify-model to
override) asking, per finding, for the specific INCLUDE rule it satisfies —
otherwise the finding is REJECTED as over-reach (rule 11 calibration
exclusions, amount-less/product-name/generic-process/no-person-link cases).
Verification is fail-open per finding: a malformed or missing verdict keeps
the finding. Both the pre- and post-verification lists are cached; the report
shows findings_rejected_by_verifier per engine.

profile_dependent findings (e.g. HEALTH_MEDICAL_INFO under medical_consult)
are reported but counted separately — the remaining findings are the true
LEAK CANDIDATES. Leak candidates on the GEMINI engine's own transcripts also
mean our stage-C detector missed them (detection ran on that same transcript),
so the report calls those out as "detector misses (review queue)".

Per engine-file results are cached at <out>/<fid>__<engine>.json (--force to
redo; cache entries carry a "schema" field — entries without it predate the
verification pass and are stale for --verify runs). Report:
<out>/leak_audit_report.md + leak_audit_report.json.

Usage:
  python3 tools/judge_leak_audit.py [--engines gemini,elevenlabs,openai]
      [--files <fid>,...] [--judge-model openai/gpt-5.6-sol]
      [--no-verify] [--verify-model openai/gpt-5.6-luna]
      [--out output/judge/leak_audit] [--force]
  python3 tools/judge_leak_audit.py --self-test   # monkeypatched judge, no API
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import judge_client
import policy

ENGINE_SUFFIX = {"gemini": "", "elevenlabs": "__elevenlabs",
                 "openai": "__openai"}
QUOTE_TRUNC = 60
PROMPT_PATH = ROOT / "prompts" / "judge_leak_audit_prompt_v1.txt"
# Cache-entry schema. Bumped to 2 when the adversarial verification pass was
# added; entries without the "schema" field predate it and are stale for
# --verify runs (they carry no verification data).
CACHE_SCHEMA = 2

# Strict structured-output schema (OpenAI strict mode: additionalProperties
# false everywhere, every property required; unused fields come back as
# false / empty string).
FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "line": {"type": "integer",
                 "description": "1-based transcript line the value survives on"},
        "quoted_text": {"type": "string",
                        "description": "exact surviving text, quoted verbatim"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "profile_dependent": {"type": "boolean",
                              "description": "true only for findings that a "
                              "lenient profile (e.g. medical_consult) may "
                              "legitimately leave audible"},
        "reason": {"type": "string"},
    },
    "required": ["line", "quoted_text", "categories", "profile_dependent",
                 "reason"],
}
AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {"type": "array", "items": FINDING_SCHEMA},
        "transcript_language_ok": {
            "type": "boolean",
            "description": "true when the transcript text is coherent enough "
                           "to audit (not garbled beyond judgement)"},
        "notes": {"type": "string"},
    },
    "required": ["findings", "transcript_language_ok", "notes"],
}


# ------------------------------------------------------------ prompt building

def numbered_lines(text):
    """1-based numbered transcript lines, one per source line."""
    lines = text.splitlines()
    return lines, "\n".join("%d: %s" % (i + 1, l) for i, l in enumerate(lines))


def build_prompt(text):
    base = PROMPT_PATH.read_text()
    lines, numbered = numbered_lines(text)
    prompt = (base + "\n\n=== REDACTED TRANSCRIPT (numbered lines) ===\n"
              + numbered)
    return lines, prompt


# ----------------------------------------------------- finding validation

def _norm_ws(s):
    return " ".join(str(s or "").split())


def validate_findings(raw_findings, lines):
    """Hallucination guard + category filter.

    A finding survives only if its quoted_text (whitespace-normalized) occurs
    in the cited line +/- 1 line of the real transcript, and at least one of
    its categories is in policy.CATEGORIES. Returns (kept, stats)."""
    stats = {"raw": len(raw_findings), "kept": 0, "dropped_hallucinated": 0,
             "dropped_no_valid_category": 0, "categories_removed": 0}
    norm = [_norm_ws(l) for l in lines]
    n = len(norm)
    kept = []
    for f in raw_findings:
        try:
            ln = int(f.get("line", 0))
        except (TypeError, ValueError):
            ln = 0
        quote = _norm_ws(f.get("quoted_text", ""))
        window = " ".join(norm[max(0, ln - 2): min(n, ln + 1)])  # ln-1..ln+1
        if not quote or ln < 1 or ln > n or quote not in window:
            stats["dropped_hallucinated"] += 1
            continue
        raw_cats = list(f.get("categories", []))
        cats = [c for c in raw_cats if c in policy.CATEGORIES]
        stats["categories_removed"] += len(raw_cats) - len(cats)
        if not cats:
            stats["dropped_no_valid_category"] += 1
            continue
        kept.append({"line": ln,
                     "quoted_text": str(f.get("quoted_text", "")),
                     "categories": cats,
                     "profile_dependent": bool(f.get("profile_dependent",
                                                     False)),
                     "reason": str(f.get("reason", ""))})
    stats["kept"] = len(kept)
    return kept, stats


# ------------------------------------------------------------- one audit

def _cache_fresh(cached, verify):
    """--verify runs need a schema-2 cache entry written WITH verification;
    anything older (incl. pre-schema entries) is stale for them. --no-verify
    runs accept any cache entry."""
    if not verify:
        return True
    return (cached.get("schema") == CACHE_SCHEMA
            and bool((cached.get("verification") or {}).get("enabled")))


def _cache_view(cached, verify):
    """A --no-verify run reading a verified cache entry must see the
    PRE-verification findings (verification only removes; without --verify
    nothing may be removed)."""
    if verify or not (cached.get("verification") or {}).get("enabled"):
        return cached
    out = dict(cached)
    out["findings"] = list(cached.get("findings_pre_verify",
                                      cached.get("findings", [])))
    out["verification"] = {"enabled": False, "model": None, "checked": 0,
                           "rejected": 0, "rejected_findings": []}
    return out


def audit_one(file_id, engine, txt_path, cache_path, model, force=False,
              verify=False, verify_model=judge_client.DEFAULT_VERIFIER):
    """Audit one redacted transcript; returns (result_dict, from_cache).

    With verify=True the validated findings additionally pass the adversarial
    verification pass (judge_client.verify_findings — ONE batched call per
    file) before anything downstream sees them; rejected findings are kept in
    the cache under verification.rejected_findings."""
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text())
        if _cache_fresh(cached, verify):
            return _cache_view(cached, verify), True
    text = Path(txt_path).read_text()
    lines, prompt = build_prompt(text)
    import common
    common.set_usage_context("audit:%s" % file_id)
    # module-attribute lookup at call time so tests can monkeypatch it
    raw = judge_client.judge_call(prompt, AUDIT_SCHEMA, model=model)
    findings, stats = validate_findings(raw.get("findings", []), lines)
    findings_pre_verify = list(findings)
    verification = {"enabled": bool(verify),
                    "model": verify_model if verify else None,
                    "checked": 0, "rejected": 0, "rejected_findings": []}
    if verify and findings:
        # one line of surrounding context per finding: its cited line
        ctx = [lines[f["line"] - 1] for f in findings]
        keep = judge_client.verify_findings(
            findings, policy.DETECTION_POLICY, model=verify_model,
            context_texts=ctx)
        kept, rejected = [], []
        for f, k in zip(findings, keep):
            (kept if k else rejected).append(f)
        verification["checked"] = len(findings)
        verification["rejected"] = len(rejected)
        verification["rejected_findings"] = rejected
        findings = kept
    result = {
        "schema": CACHE_SCHEMA,
        "file_id": file_id, "engine": engine, "judge_model": model,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "transcript_path": str(txt_path), "n_lines": len(lines),
        "transcript_language_ok": bool(raw.get("transcript_language_ok",
                                               False)),
        "notes": str(raw.get("notes", "")),
        "validation": stats,
        "verification": verification,
        "findings": findings,
        "findings_pre_verify": findings_pre_verify,
        "raw_findings": raw.get("findings", []),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result, False


# ------------------------------------------------------------- aggregation

def _leak_candidates(findings):
    """A finding counts as a leak candidate unless its categories are
    annotate_only under the file's OWN render profile (code-side; the judge's
    profile_dependent assertion is audit metadata only)."""
    return [f for f in findings
            if not f.get("kept_by_profile", f.get("profile_dependent", False))]


def render_profile_for(pred_root, dirname, override=None):
    """The profile this output was rendered under, read from its own
    rendered_intervals.json (never assumed). --profile overrides for
    counterfactual scoring."""
    if override:
        return override
    import json as _json
    ri = Path(pred_root) / dirname / "rendered_intervals.json"
    if ri.exists():
        prof = _json.loads(ri.read_text()).get("profile")
        if prof:
            return prof
    import common
    return common.CONFIG.get("profile", "strict")


def classify_by_profile(findings, profile):
    """Code-side classification: a finding is kept-by-profile when EVERY one of
    its categories is annotate_only under the file's render profile. The
    judge's own profile_dependent assertion is retained for audit only."""
    import policy
    for f in findings:
        f["kept_by_profile"] = (
            policy.action_for(f["categories"], False, profile) == "annotate_only")
    return findings


def summarize_engine(results):
    """results: list of per-file result dicts for one engine."""
    per_cat = {}
    total = leak = prof = 0
    dropped_h = dropped_c = rejected_v = 0
    for r in results:
        rejected_v += (r.get("verification") or {}).get("rejected", 0)
        for f in r["findings"]:
            total += 1
            if f.get("kept_by_profile", f.get("profile_dependent", False)):
                prof += 1
            else:
                leak += 1
            for c in f["categories"]:
                b = per_cat.setdefault(c, {"findings": 0,
                                           "leak_candidates": 0})
                b["findings"] += 1
                if not f.get("kept_by_profile", f.get("profile_dependent", False)):
                    b["leak_candidates"] += 1
        v = r.get("validation", {})
        dropped_h += v.get("dropped_hallucinated", 0)
        dropped_c += v.get("dropped_no_valid_category", 0)
    return {"files_audited": len(results), "total_findings": total,
            "leak_candidates": leak, "profile_dependent": prof,
            "dropped_hallucinated": dropped_h,
            "dropped_no_valid_category": dropped_c,
            "findings_rejected_by_verifier": rejected_v,
            "per_category": per_cat}


# ------------------------------------------------------------- report

def _q(text, n=QUOTE_TRUNC):
    t = _norm_ws(text).replace("|", "\\|")
    return t[:n]


def build_report(cfg, per_engine, missing):
    """per_engine: ordered {engine: [result dicts]}; missing: {engine: [fid]}."""
    summaries = {e: summarize_engine(rs) for e, rs in per_engine.items()}
    verify_line = ("verifier `%s` (adversarial over-reach filter ON)"
                   % cfg.get("verify_model")
                   if cfg.get("verify") else "verification OFF")
    L = ["# Judge leak audit — redacted transcripts (Phase A)",
         "",
         "Generated %s by `tools/judge_leak_audit.py` | judge model `%s` | "
         "%s | out `%s`" % (cfg["generated_at"], cfg["judge_model"],
                            verify_line, cfg["out_dir"]),
         "",
         "The judge reads each engine's REDACTED transcript with the full "
         "detection policy and reports PII values that survived redaction. "
         "Findings failing the hallucination guard (quote not found in the "
         "cited line +/-1) or citing unknown categories were dropped and are "
         "counted below. With verification ON, validated findings must also "
         "survive an adversarial verification pass (each must satisfy a "
         "specific INCLUDE rule of the policy or be rejected as over-reach); "
         "rejections are counted per engine. `profile_dependent` findings "
         "(may legitimately stay audible under a lenient profile, e.g. "
         "HEALTH_MEDICAL_INFO under medical_consult) are reported "
         "separately; the rest are the true **leak candidates**.",
         "",
         "## Aggregate — engine comparison", "",
         "| engine | files audited | validated findings | leak candidates | "
         "profile-dependent | dropped (hallucinated) | dropped (bad category) "
         "| rejected by verifier |",
         "|---|---|---|---|---|---|---|---|"]
    for e in per_engine:
        s = summaries[e]
        L.append("| %s | %d | %d | **%d** | %d | %d | %d | %d |" % (
            e, s["files_audited"], s["total_findings"], s["leak_candidates"],
            s["profile_dependent"], s["dropped_hallucinated"],
            s["dropped_no_valid_category"],
            s.get("findings_rejected_by_verifier", 0)))
    L.append("")

    for e, results in per_engine.items():
        s = summaries[e]
        L += ["## Engine `%s`" % e, "",
              "- files audited: %d%s" % (
                  s["files_audited"],
                  (" (missing redacted transcript: %s)" % ", ".join(missing[e])
                   if missing.get(e) else "")),
              "- validated findings: %d | leak candidates (excl. "
              "profile-dependent): %d | profile-dependent: %d"
              % (s["total_findings"], s["leak_candidates"],
                 s["profile_dependent"]),
              "- hallucination guard: %d finding(s) dropped (quote not in "
              "cited line +/-1); %d dropped for no valid category"
              % (s["dropped_hallucinated"], s["dropped_no_valid_category"]),
              "- findings_rejected_by_verifier: %d (adversarial over-reach "
              "filter)" % s.get("findings_rejected_by_verifier", 0),
              ""]
        if s["per_category"]:
            L += ["### Findings per category", "",
                  "| category | findings | leak candidates |", "|---|---|---|"]
            for c in sorted(s["per_category"]):
                b = s["per_category"][c]
                L.append("| %s | %d | %d |" % (c, b["findings"],
                                               b["leak_candidates"]))
            L.append("")
        L += ["### Per file", "",
              "| file_id | lines | findings | leak cand. | dropped | "
              "lang ok | judge notes |", "|---|---|---|---|---|---|---|"]
        for r in results:
            v = r.get("validation", {})
            L.append("| %s | %d | %d | %d | %d | %s | %s |" % (
                r["file_id"], r["n_lines"], len(r["findings"]),
                len(_leak_candidates(r["findings"])),
                v.get("dropped_hallucinated", 0)
                + v.get("dropped_no_valid_category", 0),
                "yes" if r.get("transcript_language_ok") else "NO",
                _q(r.get("notes", ""), 80)))
        L.append("")
        rows = [(r["file_id"], f) for r in results for f in r["findings"]]
        L += ["### Findings (validated)", ""]
        if rows:
            L += ["| file_id | line | categories | profile-dep | quote | "
                  "reason |", "|---|---|---|---|---|---|"]
            for fid, f in rows:
                L.append("| %s | %d | %s | %s | %s | %s |" % (
                    fid, f["line"], ",".join(f["categories"]),
                    "yes" if f["profile_dependent"] else "no",
                    _q(f["quoted_text"]), _q(f["reason"], 80)))
        else:
            L.append("None.")
        L.append("")

    # Judge disagreement: gemini leak candidates == stage-C detector misses
    L += ["## Judge disagreement — detector misses (review queue)", "",
          "Stage-C detection ran on the GEMINI engine's own transcript, so a "
          "leak candidate the judge can still read on a gemini redacted "
          "transcript is PII our detector missed (or under-covered). These "
          "findings are queued for human review:", ""]
    misses = []
    for r in per_engine.get("gemini", []):
        for f in _leak_candidates(r["findings"]):
            misses.append({"file_id": r["file_id"], "line": f["line"],
                           "categories": f["categories"],
                           "quoted_text": f["quoted_text"],
                           "reason": f["reason"]})
    if misses:
        L += ["| file_id | line | categories | quote | reason |",
              "|---|---|---|---|---|"]
        for m in misses:
            L.append("| %s | %d | %s | %s | %s |" % (
                m["file_id"], m["line"], ",".join(m["categories"]),
                _q(m["quoted_text"]), _q(m["reason"], 80)))
    else:
        L.append("None — no non-profile-dependent leak candidates on gemini "
                 "transcripts.")
    L.append("")

    report_json = {
        "generated_at": cfg["generated_at"], "judge_model": cfg["judge_model"],
        "out_dir": cfg["out_dir"],
        "engines": {e: {"summary": summaries[e],
                        "missing_redacted": missing.get(e, []),
                        "files": [{k: r.get(k) for k in
                                   ("file_id", "n_lines", "judge_model",
                                    "transcript_language_ok", "notes",
                                    "validation", "verification", "findings")}
                                  for r in rs]}
                    for e, rs in per_engine.items()},
        "detector_misses_review_queue": misses,
    }
    return "\n".join(L) + "\n", report_json


# ------------------------------------------------------------- run

def discover_fids(pred_root):
    fids = set()
    for d in sorted(pred_root.iterdir()):
        if not d.is_dir():
            continue
        base = d.name.split("__")[0]
        if (d / (base + "_redacted.txt")).exists():
            fids.add(base)
    return sorted(fids)


def run_audit(args):
    pred_root = Path(args.pred_root)
    if not pred_root.is_absolute():
        pred_root = ROOT / pred_root
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    bad = [e for e in engines if e not in ENGINE_SUFFIX]
    if bad:
        print("unknown engine(s): %s (choose from %s)"
              % (", ".join(bad), ", ".join(sorted(ENGINE_SUFFIX))))
        return 2
    if args.files:
        fids = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        fids = discover_fids(pred_root)
    if not fids:
        print("no files with a redacted transcript under %s" % pred_root)
        return 0

    per_engine = {}
    missing = {}
    for engine in engines:
        results = []
        miss = []
        for fid in fids:
            txt = pred_root / (fid + ENGINE_SUFFIX[engine]) / (
                fid + "_redacted.txt")
            if not txt.exists():
                miss.append(fid)
                print("  [%s] %s: redacted transcript missing — skipped"
                      % (engine, fid))
                continue
            cache = out_dir / ("%s__%s.json" % (fid, engine))
            res, cached = audit_one(fid, engine, txt, cache,
                                    args.judge_model, force=args.force,
                                    verify=args.verify,
                                    verify_model=args.verify_model)
            profile = render_profile_for(pred_root, fid + ENGINE_SUFFIX[engine],
                                         getattr(args, "profile", None))
            classify_by_profile(res["findings"], profile)
            res["render_profile"] = profile
            v = res.get("validation", {})
            print("  [%s] %s: %d finding(s) kept (%d leak candidates), "
                  "%d dropped, %d rejected by verifier%s" % (
                      engine, fid, len(res["findings"]),
                      len(_leak_candidates(res["findings"])),
                      v.get("dropped_hallucinated", 0)
                      + v.get("dropped_no_valid_category", 0),
                      (res.get("verification") or {}).get("rejected", 0),
                      "  [cached]" if cached else ""))
            results.append(res)
        per_engine[engine] = results
        missing[engine] = miss

    cfg = {"generated_at": datetime.date.today().isoformat(),
           "judge_model": args.judge_model, "out_dir": str(out_dir),
           "verify": bool(args.verify),
           "verify_model": args.verify_model if args.verify else None}
    md, js = build_report(cfg, per_engine, missing)
    md_path = out_dir / "leak_audit_report.md"
    js_path = out_dir / "leak_audit_report.json"
    md_path.write_text(md)
    js_path.write_text(json.dumps(js, indent=2, ensure_ascii=False))
    total_leak = sum(summarize_engine(rs)["leak_candidates"]
                     for rs in per_engine.values())
    print("audited %d engine(s) x %d file id(s) | %d leak candidate(s) total"
          % (len(engines), len(fids), total_leak))
    print("report -> %s\njson   -> %s" % (md_path, js_path))
    return 0


# ------------------------------------------------------------- self-test

def _schema_strict(node):
    """Every object node: additionalProperties False + all properties required."""
    if not isinstance(node, dict):
        return True
    if node.get("type") == "object":
        if node.get("additionalProperties") is not False:
            return False
        if sorted(node.get("required", [])) != sorted(node.get("properties",
                                                               {})):
            return False
    return all(_schema_strict(v) for v in node.values()
               if isinstance(v, dict))


def self_test():
    import tempfile
    cases = []

    def case(name, ok, detail=""):
        cases.append((name, ok))
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  [%s]" % detail) if (detail and not ok) else ""))

    lines = [
        "[00:00:01.00 - 00:00:02.00] S1 - my number is nine eight seven six"
        " five four three two one zero",
        "[00:00:03.00 - 00:00:04.00] S2 - I was diagnosed with diabetes last"
        " year",
        "[00:00:05.00 - 00:00:06.00] S1 - okay thank you bye",
    ]
    text = "\n".join(lines) + "\n"

    fake_response = {
        "findings": [
            # 1. planted surviving phone — must be kept
            {"line": 1, "quoted_text":
             "nine eight seven six five four three two one zero",
             "categories": ["PHONE"], "profile_dependent": False,
             "reason": "10 spoken digits survived redaction"},
            # 2. hallucinated quote — must be dropped
            {"line": 3, "quoted_text": "this text never appears anywhere",
             "categories": ["PHONE"], "profile_dependent": False,
             "reason": "made up"},
            # 3. profile-dependent health info — kept, counted separately
            {"line": 2, "quoted_text": "diagnosed with diabetes",
             "categories": ["HEALTH_MEDICAL_INFO"], "profile_dependent": True,
             "reason": "condition audible; medical_consult may allow"},
            # 4. unknown category only — must be dropped
            {"line": 3, "quoted_text": "okay thank you",
             "categories": ["BOGUS_CATEGORY"], "profile_dependent": False,
             "reason": "not a real category"},
            # 5. off-by-one line citation — window +/-1 keeps it
            {"line": 2, "quoted_text": "my number is nine eight",
             "categories": ["PHONE"], "profile_dependent": False,
             "reason": "cited neighbouring line"},
            # 6. whitespace-mangled quote — normalization keeps it
            {"line": 1, "quoted_text": "nine  eight   seven\tsix",
             "categories": ["PHONE"], "profile_dependent": False,
             "reason": "extra whitespace in quote"},
            # 7. out-of-range line — dropped
            {"line": 99, "quoted_text": "okay thank you",
             "categories": ["PHONE"], "profile_dependent": False,
             "reason": "bad line number"},
        ],
        "transcript_language_ok": True,
        "notes": "synthetic",
    }

    calls = {"n": 0}

    def fake_judge(prompt, schema, model=judge_client.DEFAULT_JUDGE,
                   max_tokens=16000):
        calls["n"] += 1
        assert schema is AUDIT_SCHEMA
        assert "=== REDACTED TRANSCRIPT (numbered lines) ===" in prompt
        assert "1: [00:00:01.00" in prompt and "3: [00:00:05.00" in prompt
        return json.loads(json.dumps(fake_response))

    real = judge_client.judge_call
    judge_client.judge_call = fake_judge
    try:
        # 0. schema strictness
        case("schema strict (additionalProperties/required everywhere)",
             _schema_strict(AUDIT_SCHEMA))

        kept, stats = validate_findings(fake_response["findings"], lines)
        case("planted phone kept + validated",
             any(f["categories"] == ["PHONE"] and f["line"] == 1
                 and "nine eight seven" in f["quoted_text"] for f in kept),
             str(kept))
        case("hallucinated + out-of-range dropped",
             stats["dropped_hallucinated"] == 2, str(stats))
        case("unknown category dropped",
             stats["dropped_no_valid_category"] == 1
             and stats["categories_removed"] == 1, str(stats))
        case("kept count", stats["kept"] == 4 and len(kept) == 4, str(stats))
        case("off-by-one line window kept",
             any(f["line"] == 2 and f["quoted_text"].startswith("my number")
                 for f in kept), str(kept))
        case("whitespace-normalized quote kept",
             any("nine  eight" in f["quoted_text"] for f in kept), str(kept))
        leaks = _leak_candidates(kept)
        case("profile_dependent HEALTH_MEDICAL_INFO counted separately",
             len(leaks) == 3
             and all("HEALTH_MEDICAL_INFO" not in f["categories"]
                     for f in leaks)
             and sum(1 for f in kept if f["profile_dependent"]) == 1,
             str(kept))

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            txt = tdp / "fid1_redacted.txt"
            txt.write_text(text)
            cache = tdp / "fid1__gemini.json"
            r1, c1 = audit_one("fid1", "gemini", txt, cache, "fake-model")
            r2, c2 = audit_one("fid1", "gemini", txt, cache, "fake-model")
            case("cache: first call live, second cached",
                 calls["n"] == 1 and not c1 and c2
                 and r2["findings"] == r1["findings"],
                 "calls=%d c1=%s c2=%s" % (calls["n"], c1, c2))
            r3, c3 = audit_one("fid1", "gemini", txt, cache, "fake-model",
                               force=True)
            case("--force redoes the call",
                 calls["n"] == 2 and not c3, "calls=%d" % calls["n"])

            per_engine = {"gemini": [r1], "elevenlabs": []}
            md, js = build_report(
                {"generated_at": "test", "judge_model": "fake-model",
                 "out_dir": td},
                per_engine, {"gemini": [], "elevenlabs": ["fid1"]})
            s = js["engines"]["gemini"]["summary"]
            case("engine summary counts",
                 s["total_findings"] == 4 and s["leak_candidates"] == 3
                 and s["profile_dependent"] == 1
                 and s["dropped_hallucinated"] == 2
                 and s["dropped_no_valid_category"] == 1, str(s))
            case("per-category buckets",
                 s["per_category"]["PHONE"]["leak_candidates"] == 3
                 and s["per_category"]["HEALTH_MEDICAL_INFO"]
                 ["leak_candidates"] == 0, str(s["per_category"]))
            case("detector-misses review queue from gemini leak candidates",
                 len(js["detector_misses_review_queue"]) == 3
                 and "detector misses (review queue)" in md
                 and "nine eight seven six five four three two one zero" in md,
                 md[:200])
            case("quotes truncated to %d chars" % QUOTE_TRUNC,
                 all(len(_q(f["quoted_text"])) <= QUOTE_TRUNC
                     for f in r1["findings"])
                 and len(_q("x" * 200)) == QUOTE_TRUNC, "")
            case("missing transcript noted in report",
                 "missing redacted transcript: fid1" in md, "")
    finally:
        judge_client.judge_call = real

    # ---------------- adversarial verification pass (over-reach filter)
    import re
    vlines = [
        "[00:00:01.00 - 00:00:02.00] S1 - I already paid the hospital bill",
        "[00:00:03.00 - 00:00:04.00] S2 - my salary is thirty thousand"
        " rupees",
        "[00:00:05.00 - 00:00:06.00] S1 - call me on nine eight seven six"
        " five four three two one zero",
    ]
    vtext = "\n".join(vlines) + "\n"
    v_findings = [
        # planted over-reach: amount-less FINANCIAL_INFO -> verifier REJECTS
        {"line": 1, "quoted_text": "paid the hospital bill",
         "categories": ["FINANCIAL_INFO"], "profile_dependent": False,
         "reason": "payment happened but no value (over-reach)"},
        # valued FINANCIAL_INFO -> verifier KEEPS
        {"line": 2, "quoted_text": "salary is thirty thousand rupees",
         "categories": ["FINANCIAL_INFO"], "profile_dependent": False,
         "reason": "salary with an amount"},
        {"line": 3, "quoted_text":
         "nine eight seven six five four three two one zero",
         "categories": ["PHONE"], "profile_dependent": False,
         "reason": "complete phone"},
    ]
    v_audit = {"findings": json.loads(json.dumps(v_findings)),
               "transcript_language_ok": True, "notes": "synthetic"}
    vcalls = {"audit": 0, "verify": 0, "verify_prompts": []}

    def fake_judge_v(prompt, schema, model="fake", max_tokens=16000):
        if schema is judge_client.VERIFY_SCHEMA:
            vcalls["verify"] += 1
            vcalls["verify_prompts"].append(prompt)
            verdicts = []
            for m in re.finditer(r'(?m)^(\d+)\. text: "([^"]*)"', prompt):
                keep = "paid the hospital bill" not in m.group(2)
                verdicts.append({
                    "index": int(m.group(1)), "keep": keep,
                    "rule": ("include rule satisfied" if keep else
                             "REJECT: rule 11 amount-less payment claim")})
            return {"verdicts": verdicts}
        vcalls["audit"] += 1
        assert schema is AUDIT_SCHEMA
        return json.loads(json.dumps(v_audit))

    judge_client.judge_call = fake_judge_v
    try:
        case("verify schema strict (additionalProperties/required everywhere)",
             _schema_strict(judge_client.VERIFY_SCHEMA))

        keep = judge_client.verify_findings(
            v_findings, policy.DETECTION_POLICY, model="fake",
            context_texts=[vlines[f["line"] - 1] for f in v_findings])
        case("verify: amount-less FINANCIAL_INFO rejected, valued one kept",
             keep == [False, True, True], str(keep))
        p = vcalls["verify_prompts"][-1]
        case("verify prompt: ONE batched numbered list + policy + context",
             vcalls["verify"] == 1
             and '1. text: "paid the hospital bill"' in p
             and '3. text: "nine eight seven' in p
             and "categories: FINANCIAL_INFO" in p
             and "Calibration" in p              # policy rule 11 included
             and "must be REJECTED" in p
             and "my salary is thirty thousand" in p, p[:400])
        case("verify: empty findings -> [] and no call",
             judge_client.verify_findings([], "policy", model="fake") == []
             and vcalls["verify"] == 1)

        def fake_failopen(prompt, schema, model="fake", max_tokens=16000):
            return {"verdicts": [
                {"index": 2, "keep": False, "rule": "REJECT: x"},
                {"index": 99, "keep": False, "rule": "out of range"},
                {"index": "bogus", "keep": False, "rule": "bad index"},
                "not-a-dict"]}
        judge_client.judge_call = fake_failopen
        keep = judge_client.verify_findings(v_findings, "policy", model="f")
        case("fail-open: missing index defaults keep; malformed ignored",
             keep == [True, False, True], str(keep))
        judge_client.judge_call = fake_judge_v

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            txt = tdp / "fid2_redacted.txt"
            txt.write_text(vtext)
            cache = tdp / "fid2__gemini.json"
            r1, c1 = audit_one("fid2", "gemini", txt, cache, "fake-model",
                               verify=True, verify_model="fake-verifier")
            ver = r1["verification"]
            case("audit_one --verify: planted over-reach removed, both "
                 "lists stored",
                 not c1 and len(r1["findings"]) == 2
                 and len(r1["findings_pre_verify"]) == 3
                 and ver["enabled"] and ver["model"] == "fake-verifier"
                 and ver["checked"] == 3 and ver["rejected"] == 1
                 and ver["rejected_findings"][0]["categories"]
                 == ["FINANCIAL_INFO"]
                 and all("paid the hospital" not in f["quoted_text"]
                         for f in r1["findings"]), str(ver))
            case("batching: one verify call per file",
                 vcalls["verify"] == 2 and vcalls["audit"] == 1, str(vcalls))
            case("cache entry carries schema field",
                 json.loads(cache.read_text()).get("schema") == CACHE_SCHEMA)
            r2, c2 = audit_one("fid2", "gemini", txt, cache, "fake-model",
                               verify=True, verify_model="fake-verifier")
            case("verified cache reused by --verify run (no new calls)",
                 c2 and vcalls["audit"] == 1 and vcalls["verify"] == 2
                 and r2["findings"] == r1["findings"])
            r3, c3 = audit_one("fid2", "gemini", txt, cache, "fake-model",
                               verify=False)
            case("--no-verify view of verified cache = pre-verify findings",
                 c3 and len(r3["findings"]) == 3
                 and not r3["verification"]["enabled"]
                 and vcalls["audit"] == 1, str(len(r3["findings"])))
            # simulate a pre-schema (old) cache entry
            old = json.loads(cache.read_text())
            old.pop("schema", None)
            old.pop("verification", None)
            old.pop("findings_pre_verify", None)
            cache.write_text(json.dumps(old))
            r4, c4 = audit_one("fid2", "gemini", txt, cache, "fake-model",
                               verify=False)
            case("pre-schema cache still fresh for --no-verify runs",
                 c4 and vcalls["audit"] == 1)
            r5, c5 = audit_one("fid2", "gemini", txt, cache, "fake-model",
                               verify=True, verify_model="fake-verifier")
            case("cache staleness on schema bump: --verify redoes old entry",
                 not c5 and vcalls["audit"] == 2 and vcalls["verify"] == 3
                 and len(r5["findings"]) == 2)

            classify_by_profile(r5["findings"], "strict")
            md2, js2 = build_report(
                {"generated_at": "test", "judge_model": "fake-model",
                 "out_dir": td, "verify": True,
                 "verify_model": "fake-verifier"},
                {"gemini": [r5]}, {"gemini": []})
            s2 = js2["engines"]["gemini"]["summary"]
            case("report: findings_rejected_by_verifier per engine",
                 s2["findings_rejected_by_verifier"] == 1
                 and "rejected by verifier" in md2
                 and "verifier `fake-verifier`" in md2, md2[:300])
    finally:
        judge_client.judge_call = real

    failed = [n for n, ok in cases if not ok]
    print("\nself-test: %d/%d passed" % (len(cases) - len(failed),
                                         len(cases)))
    return 1 if failed else 0


# ------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engines", default="gemini,elevenlabs,openai",
                    help="comma-separated engines (gemini,elevenlabs,openai)")
    ap.add_argument("--files", default=None,
                    help="comma-separated file ids (default: every id with a "
                         "redacted transcript under --pred-root)")
    ap.add_argument("--judge-model", default=judge_client.DEFAULT_JUDGE,
                    help="OpenRouter judge model (default %s)"
                         % judge_client.DEFAULT_JUDGE)
    ap.add_argument("--verify", dest="verify", action="store_true",
                    default=True,
                    help="adversarial verification pass over validated "
                         "findings (default ON)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the verification pass")
    ap.add_argument("--verify-model", default=judge_client.DEFAULT_VERIFIER,
                    help="OpenRouter verifier model (default %s)"
                         % judge_client.DEFAULT_VERIFIER)
    ap.add_argument("--out", default="output/judge/leak_audit",
                    help="cache + report directory")
    ap.add_argument("--pred-root", default="output",
                    help="directory holding <fid><engine-suffix>/ dirs")
    ap.add_argument("--profile", default=None,
                    help="override the render profile for counterfactual "
                         "scoring (default: read per file from its "
                         "rendered_intervals.json)")
    ap.add_argument("--force", action="store_true",
                    help="redo cached engine-file audits")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline self-tests (judge_call monkeypatched)")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    sys.exit(run_audit(args))


if __name__ == "__main__":
    main()
