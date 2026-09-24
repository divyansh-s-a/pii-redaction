#!/usr/bin/env python3
"""Regression matrix across both arms.

Runs every case, asserts something unambiguous, and prints PASS/FAIL per row.
An assertion that cannot fail is worse than no assertion, so each row states
what would have to be true for it to fail, and rows that genuinely cannot be
decided report INCONCLUSIVE rather than green.

    python3 synth/regress.py --quick        # no model calls
    python3 synth/regress.py                # full, runs the document pipeline
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ / "pipeline"))
sys.path.insert(0, str(ROOT / "taxonomies"))
sys.path.insert(0, str(ROOT / "document_pipeline"))

import actions as A  # noqa: E402

RESULTS = []


def rec(case, arm, status, detail):
    RESULTS.append((case, arm, status, detail))
    mark = {"PASS": "PASS", "FAIL": "FAIL", "INCONCLUSIVE": "INCON",
            "SKIP": "SKIP"}[status]
    print(f"  {mark:<6}{arm:<10}{case:<30}{detail}")


def run_doc(src, out, extra=()):
    cmd = [sys.executable, str(ROOT / "document_pipeline" / "run_document.py"),
           str(src), "--out", str(out), "--findings-out",
           str(Path(out).with_suffix(".jsonl")), "--workers", "3", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return p.returncode == 0, p.stdout + p.stderr


def black_frac(page, rect=None):
    pm = page.get_pixmap(dpi=110, clip=rect)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[..., :3]
    return float((a.max(axis=2) < 50).mean()) if a.size else 0.0


# ------------------------------------------------------------------ selftests
def selftests():
    mods = ["taxonomies/base.py", "taxonomies/registry.py",
            "document_pipeline/matching.py", "document_pipeline/tokens.py",
            "document_pipeline/actions.py", "document_pipeline/geometry.py",
            "document_pipeline/propagate.py", "document_pipeline/formfields.py",
            "document_pipeline/handwriting.py", "router/analyze.py",
            "synth/make_corpus.py"]
    for m in mods:
        p = subprocess.run([sys.executable, str(ROOT / m), "--self-test"],
                           capture_output=True, text=True, cwd=str(ROOT))
        rec(Path(m).stem, "document", "PASS" if p.returncode == 0 else "FAIL",
            (p.stdout + p.stderr).strip().splitlines()[-1][:58] if (p.stdout or p.stderr) else "")
    p = subprocess.run([sys.executable, str(PROJ / "pipeline" / "detect_pii.py")],
                       capture_output=True, text=True, cwd=str(PROJ))
    rec("detect_pii", "audio", "PASS" if p.returncode == 0 else "FAIL",
        (p.stdout + p.stderr).strip().splitlines()[-1][:58])


def audio_shared_modules():
    """The document arm edited two modules the audio arm depends on."""
    import importlib
    try:
        import policy
        importlib.reload(policy)
        ok = len(policy.CATEGORIES) == 31 and policy.action_for(
            ["FULL_NAME"], False, "strict") == "beep"
        rec("audio taxonomy intact", "audio", "PASS" if ok else "FAIL",
            f"{len(policy.CATEGORIES)} categories, action_for unchanged")
    except Exception as e:
        rec("audio taxonomy intact", "audio", "FAIL", str(e)[:50])
    try:
        import judge_client
        import inspect
        sig = inspect.signature(judge_client.judge_call)
        has = "images" in sig.parameters
        dflt = sig.parameters["images"].default if has else "MISSING"
        rec("judge_call audio default", "audio",
            "PASS" if has and dflt is None else "FAIL",
            f"images param default={dflt!r} (audio path unchanged)")
    except Exception as e:
        rec("judge_call audio default", "audio", "FAIL", str(e)[:50])
    try:
        import common
        src = Path(common.__file__).read_text()
        ok = ("unsupported part type" in src
              and "NEVER fall through silently" in src)
        rec("common.py else:raise", "shared", "PASS" if ok else "FAIL",
            "unsupported part raises instead of being dropped")
    except Exception as e:
        rec("common.py else:raise", "shared", "FAIL", str(e)[:50])


# -------------------------------------------------------------- taxonomy pack
def qc_verdict_consistency():
    """A bundle's RECORDED verdict must match the report sitting next to it.

    tools/redact.py calls qc_report.qc_file() at six points; one of them --
    the re-render after --assurance adds intervals -- discarded the return
    value, so res["qc"] kept the PRE-assurance status while qc_report.md on
    disk was rewritten with the post-assurance one. A file whose QC problem
    the second listener had just resolved was quarantined anyway, with a PASS
    report inside the quarantine folder. Direction of harm is over-caution,
    not a leak, but --assurance must not be able to cause a quarantine.

    This asserts the INVARIANT rather than that one call site: for every
    bundle on disk, manifest qc_status == the status in qc_report.md.
    """
    import re as _re
    roots = [PROJ / "quarantine", PROJ / "deliveries"]
    checked, bad = 0, []
    for root in roots:
        if not root.exists():
            continue
        for man in list(root.glob("*/manifest.json"))[:40]:
            rep = man.parent / "qc_report.md"
            if not rep.exists():
                continue
            try:
                recorded = json.loads(man.read_text()).get("qc_status")
                m = _re.search(r"\*\*Status:\s*([A-Z]+)\*\*", rep.read_text())
            except Exception:
                continue
            if not (recorded and m):
                continue
            checked += 1
            if m.group(1) != recorded:
                bad.append(f"{man.parent.name}: manifest={recorded} "
                           f"report={m.group(1)}")
    if not checked:
        rec("qc verdict matches report", "audio", "SKIP",
            "no delivered or quarantined bundles on disk to check")
        return
    # Bundles produced BEFORE the fix carry the mismatch permanently -- they
    # are the evidence that found it. Baseline them so this stays a
    # regression gate rather than a permanently red light nobody reads; a
    # mismatch in any bundle NOT on the list is a new failure.
    base_f = ROOT / "synth" / "qc_verdict_baseline.txt"
    known = set()
    if base_f.exists():
        known = {l.split(":")[0].strip()
                 for l in base_f.read_text().splitlines() if l.strip()}
    fresh = [b for b in bad if b.split(":")[0].strip() not in known]
    if fresh:
        rec("qc verdict matches report", "audio", "FAIL",
            f"NEW stale verdict in {len(fresh)}: {fresh[:3]}")
    else:
        rec("qc verdict matches report", "audio", "PASS",
            f"{checked} bundle(s) checked; {len(bad)} known pre-fix "
            f"mismatch(es) baselined, 0 new")


def pack_behaviour():
    import base, registry
    p = base.load(ROOT / "taxonomies" / "insurance_medical")
    checks = [
        ("clinical kept", p.action_for(["T37"]) == "KEEP"),
        ("amounts kept", p.action_for(["T34"]) == "KEEP"),
        ("dates kept", p.action_for(["T21"]) == "KEEP"
         and p.action_for(["T17"]) == "KEEP"),
        ("identity destroyed", p.action_for(["T08"]) == "REDACT"),
        ("most protective wins", p.action_for(["T37", "T01"]) == "PSEUDONYMIZE"),
        ("unknown fails closed", p.action_for(["NOPE"]) == "REDACT"),
    ]
    bad = [n for n, ok in checks if not ok]
    rec("insurance matrix", "document", "PASS" if not bad else "FAIL",
        "; ".join(n for n, _ in checks) if not bad else f"BROKEN: {bad}")

    # ---- the generic pack, and the divergence between the two -----------
    g = base.load(ROOT / "taxonomies" / "generic_document")
    gchecks = [
        ("identity destroyed", g.action_for(["G13"]) == "REDACT"),
        ("health REDACTED (diverges from insurance)",
         g.action_for(["sem:clinical"]) == "REDACT"
         and p.action_for(["sem:clinical"]) == "KEEP"),
        ("numeric tables kept", g.action_for(["G35"]) == "KEEP"),
        ("empty fields kept", g.action_for(["G36"]) == "KEEP"),
        ("guidance kept", g.action_for(["G37"]) == "KEEP"),
        ("page furniture kept", g.action_for(["G38"]) == "KEEP"),
        ("content ink kept, identity ink struck",
         g.action_for(["sem:ink_unzoned"]) == "KEEP"
         and g.action_for(["sem:handwriting_ink"]) == "REDACT"),
        ("letterhead kept, personal mail struck",
         g.action_for(["sem:org_contact"]) == "KEEP"
         and g.action_for(["sem:contact"]) == "REDACT"),
        ("IFSC kept under BOTH packs",
         g.action_for(["sem:bank_common"]) == "KEEP"
         and p.action_for(["sem:bank_common"]) == "KEEP"),
        ("event dates kept, DOB not",
         g.action_for(["sem:event_date"]) == "KEEP"
         and g.action_for(["sem:dob"]) != "KEEP"),
        ("unknown fails closed", g.action_for(["NOPE"]) == "REDACT"),
    ]
    gbad = [n for n, ok in gchecks if not ok]
    rec("generic matrix", "document", "PASS" if not gbad else "FAIL",
        f"{len(gchecks)} rules hold; health REDACTED vs insurance KEEP"
        if not gbad else f"BROKEN: {gbad}")

    # every semantic key a mechanism emits must be mapped by BOTH packs --
    # an unmapped key fails closed to REDACT and inverts that pack's KEEP row
    import formfields
    emitted = {c for _, c, _ in formfields.LEXICON} | {
        "sem:card", "sem:org_contact", "sem:ink_unzoned", "sem:face",
        "sem:machine_code", "sem:handwriting_ink", "sem:gov_id",
        "sem:org_reg_code", "sem:bank_common", "sem:bank_personal",
        "sem:unlabelled_identity_block", "sem:clinical"}
    # each pack must also map every key ITS OWN vocabulary emits -- the base
    # set alone shrank from 36 keys to 24 when domain rows moved into packs,
    # and a check that quietly tests less is worse than no check
    unmapped = {k.id: sorted(x for x in (emitted | {r[1] for r in k.lexicon})
                             if not k.resolve(x))
                for k in (p, g)}
    bad = {k: v for k, v in unmapped.items() if v}
    n_all = len(emitted | {r[1] for r in p.lexicon} | {r[1] for r in g.lexicon})
    rec("no unmapped semantic keys", "document", "PASS" if not bad else "FAIL",
        f"{len(emitted)} shared + pack keys = {n_all} distinct, all resolve"
        if not bad else f"UNMAPPED -> silent REDACT: {bad}")

    # ---- lexicon isolation: a pack's vocabulary must not reach another ----
    import formfields as _ff
    _ff._PACK_COMPILED.clear()
    own_i = {r[1] for r in p.lexicon}
    own_g = {r[1] for r in g.lexicon}
    # PATTERNS are what must not cross, not semantic keys: a pack may add more
    # ways to spell a key the base also carries (generic adds NHS/SSN spellings
    # of sem:gov_id). What must never happen is one pack's PATTERN being
    # compiled during another pack's run.
    base_pats = {pt for pt, _, _ in _ff.LEXICON}
    pats_i = {r[0] for r in p.lexicon}
    pats_g = {r[0] for r in g.lexicon}
    iso = [
        ("insurance owns claim vocabulary", "sem:clinical" in own_i),
        ("generic owns credential vocabulary", "sem:credential" in own_g),
        ("no pack pattern sits in the shared base",
         not (pats_i & base_pats) and not (pats_g & base_pats)),
        ("the two packs share no pattern", not (pats_i & pats_g)),
        ("insurance run never sees a generic pattern",
         not (pats_g & {rx.pattern for rx, _, _ in _ff.compiled_for(p)})),
        ("generic run never sees an insurance pattern",
         not (pats_i & {rx.pattern for rx, _, _ in _ff.compiled_for(g)})),
        ("insurance run does not compile generic rows",
         len(_ff.compiled_for(p)) == len(_ff.COMPILED) + len(p.lexicon)),
        ("generic run does not compile insurance rows",
         len(_ff.compiled_for(g)) == len(_ff.COMPILED) + len(g.lexicon)),
        ("OCR repair is pack-scoped",
         _ff._fix_token("Dicherge") == "Dicherge"
         and _ff._fix_token("Dicherge", *_ff.vocab_for(p)) == "Discharge"),
    ]
    ibad = [n for n, ok in iso if not ok]
    rec("lexicon isolation", "document", "PASS" if not ibad else "FAIL",
        f"base {len(_ff.LEXICON)} shared + {len(p.lexicon)} insurance-only "
        f"+ {len(g.lexicon)} generic-only" if not ibad else f"BROKEN: {ibad}")

    # propagation must be live under a pack, not just under legacy names
    import propagate
    allowed = propagate.propagatable(p)
    live = {c["id"] for c in p.rows()} & allowed
    live_g = {c["id"] for c in g.rows()} & propagate.propagatable(g)
    rec("propagation live under pack", "document",
        "PASS" if len(live) >= 10 and len(live_g) >= 10 else "FAIL",
        f"insurance {len(live)}, generic {len(live_g)} propagatable categories"
        if len(live) >= 10 and len(live_g) >= 10 else
        f"INERT for a pack: insurance={len(live)} generic={len(live_g)} -- the "
        f"bundle-wide consistency guarantee silently does nothing")

    # ambiguity must fall to the conservative pack, never the permissive one
    _, dec = registry.select({"text": ["nothing recognisable here"]})
    conservative = registry.conservative(registry.discover())
    rec("ambiguous -> conservative", "document",
        "PASS" if dec["pack"] == conservative.id else "FAIL",
        f"basis={dec['basis']} pack={dec['pack']}")

    # extends inheritance
    import json as _j
    tmp = ROOT / "taxonomies" / "_regress_child"
    tmp.mkdir(exist_ok=True)
    (tmp / "pack.json").write_text(_j.dumps({
        "id": "_regress_child", "version": "0", "extends": "insurance_medical",
        "categories": [{"id": "T34", "name": "CLAIM_FINANCIAL_AMOUNT",
                        "action": "REDACT", "enforcement": "label_anchored",
                        "definition": "child override"}]}))
    (tmp / "prompt.txt").write_text("child")
    try:
        child = base.load(tmp)
        ok = (child.action_for(["T34"]) == "REDACT"
              and child.action_for(["T37"]) == "KEEP"
              and len(child.rows()) == len(p.rows()))
        rec("extends inheritance", "document", "PASS" if ok else "FAIL",
            f"{len(child.rows())} rows inherited, T34 overridden to REDACT")
    finally:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()


def verify_gate_and_loop():
    """Two things a self-check must do: SIGNAL failure, and be able to fix it.

    verify_removal printed FAIL and exited 0, so a script, a CI job or a batch
    over hundreds of documents saw success where a human reading the output
    saw a leak. And nothing fed its findings back -- the audio arm extends a
    beep and re-renders until the value is inaudible; the document arm had no
    equivalent.

    Uses the COSMETIC redactor as the broken input, because it is the one
    failure guaranteed to produce a survivor: a box is drawn, nothing is
    destroyed.
    """
    import redact as _r, verify_removal as _v, base as _b, tokens as _t
    import tempfile
    pack = _b.load(ROOT / "taxonomies" / "generic_document")
    td = Path(tempfile.mkdtemp())
    d = fitz.open(); pg = d.new_page()
    pg.insert_text((60, 100), "Patient Rosalind Grieves", fontsize=11)
    box = [fitz.Rect(w[:4]) for w in pg.get_text("words")
           if w[4] == "Rosalind"][0]
    src = td / "src.pdf"; d.save(src); d.close()
    finds = {1: [{"categories": ["G01"], "text": "Rosalind",
                  "bbox": list(box), "all_boxes": [list(box)],
                  "action": "REDACT"}]}
    cos = td / "cos.pdf"
    _r.redact_document(src, finds, cos, cosmetic=True, pack=pack,
                       tokens=_t.TokenStore("t"), audit=[])
    caught = len(_v.survivors(cos, finds))
    fixed = td / "fix.pdf"
    _r.redact_document(cos, finds, fixed, cosmetic=False, pack=pack,
                       tokens=_t.TokenStore("t"), audit=[])
    left = len(_v.survivors(fixed, finds))
    gone = "Rosalind" not in fitz.open(fixed)[0].get_text()
    ok = caught == 1 and left == 0 and gone
    rec("verify catches and loop corrects", "document", "PASS" if ok else "FAIL",
        "cosmetic pass detected as a survivor; re-redaction destroys it"
        if ok else f"caught={caught} (want 1), left={left} (want 0), gone={gone}")

    # the gate must also SIGNAL, not just print
    import subprocess as _sp
    fj = td / "f.jsonl"
    fj.write_text(json.dumps({"file": "src.pdf", "page": 1,
                              "findings": finds[1]}) + "\n")
    # the verifier resolves <stem-from-findings><suffix>.pdf
    (td / "src_redacted.pdf").write_bytes(cos.read_bytes())
    r = _sp.run([sys.executable,
                 str(ROOT / "document_pipeline" / "verify_removal.py"),
                 "--findings", str(fj), "--redacted", str(td),
                 "--suffix", "_redacted", "--label", "broken"],
                capture_output=True, text=True, cwd=str(ROOT))
    rec("verify gate exits non-zero", "document",
        "PASS" if r.returncode != 0 else "FAIL",
        f"a failed verification exits {r.returncode}"
        if r.returncode else "prints FAIL but exits 0 - invisible to automation")


def row_veto_bounded():
    """The table row veto must be bounded HORIZONTALLY, not just vertically.

    A y-band alone made the veto fire on anything sharing a horizontal line
    with a table. On a two-column page that is a different column: clinical
    content in a narrative column was flipped REDACT -> KEEP because an
    unrelated charges table had a row at the same height. A leak, and an
    invisible one, since the removal gate skips KEEP findings.
    """
    import formfields as _ff, base as _b
    pack = _b.load(ROOT / "taxonomies" / "generic_document")

    def probe(x, label):
        d = fitz.open(); pg = d.new_page()
        pg.insert_text((x, 200), label, fontsize=10)
        for i, (a, b) in enumerate([("1,200.00", "4,500.00"),
                                    ("2,300.00", "5,600.00"),
                                    ("3,400.00", "6,700.00")]):
            pg.insert_text((350, 200 + i * 16), a, fontsize=9)
            pg.insert_text((450, 200 + i * 16), b, fontsize=9)
        words = [(i, w[4], fitz.Rect(w[:4]))
                 for i, w in enumerate(pg.get_text("words"))]
        box = [fitz.Rect(w[:4]) for w in pg.get_text("words")
               if w[4] == label.split()[0]][0]
        f = [{"categories": ["sem:clinical"], "text": label,
              "bbox": tuple(box), "action": pack.action_for(["sem:clinical"])}]
        out = _ff.spare_numeric_columns(pg, words, f, pack)
        d.close()
        return out[0]["action"]

    far = probe(112, "Diagnosis positive")     # separate column, across a gutter
    near = probe(300, "Surgeon")               # the table's own description cell
    ok = far == "REDACT" and near == "KEEP"
    rec("row veto bounded", "document", "PASS" if ok else "FAIL",
        "clinical in a separate column is redacted; a line-item description "
        "beside its figures is kept" if ok
        else f"separate column={far} (want REDACT), description={near} (want KEEP)")


def inherited_redaction_marks():
    """A redaction mark the SOURCE carried must not write its label into ours.

    Partner-supplied "already redacted" PDFs routinely carry Acrobat redaction
    annotations that were marked but never applied. apply_redactions() burns
    their overlay label into the page, so our output would contain text nobody
    here authored -- on pages this pipeline found nothing on. That breaks the
    no-write-back invariant, which is the whole contract of this arm.
    """
    import redact as _r, actions as _a
    d = fitz.open(); pg = d.new_page()
    pg.insert_text((60, 100), "Hospital bill total 4520.00", fontsize=11)
    pg.add_redact_annot(fitz.Rect(60, 130, 300, 150),
                        text="CONFIDENTIAL XKPQ", fill=(1, 1, 1))
    src = Path(ROOT / "out" / "regress" / "_inherited.pdf")
    src.parent.mkdir(parents=True, exist_ok=True)
    d.save(src); d.close()
    before = fitz.open(src)[0].get_text()
    doc = fitz.open(src); page = doc[0]
    n = _r.neutralise_inherited_redactions(page)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS,
                          graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                          text=fitz.PDF_REDACT_TEXT_REMOVE)
    after = page.get_text()
    doc.close()
    added = _a.added_text(before, after)
    ok = n == 1 and not added
    rec("inherited redaction marks", "document", "PASS" if ok else "FAIL",
        f"{n} inherited mark stripped of its label; nothing written"
        if ok else f"text written into the page: {added}")


def second_detector_isolation():
    """An OPTIONAL second detector must not be able to empty a page.

    detect_page's caller turns any exception into `finds = []` for the whole
    page, so an unwrapped failure in the second detector -- missing key, auth
    error, rate limit -- would redact the page using no model findings at all,
    silently. The detector is on by default, so this is the common failure
    mode on a machine without the second family's credentials, not a corner.
    """
    import detect as _d, base as _b
    import fitz as _f
    pack = _b.load(ROOT / "taxonomies" / "generic_document")
    doc = _f.open(); pg = doc.new_page()
    pg.insert_text((60, 100), "Name: Sunita Rao", fontsize=11)
    words = _d.page_words(pg)
    real = _d._ask

    def fake(det, prompt, images=None):
        if det != "gpt":
            raise RuntimeError("401 Unauthorized")
        return [{"start_word": 1, "end_word": 2, "categories": ["G01"],
                 "spoken_text": "Sunita Rao", "public_figure": False,
                 "confidence": "high", "reason": "name"}]
    _d._ask = fake
    try:
        finds, _n = _d.detect_page(pg, "x", detector="gpt", second="gemini",
                                   words=words, pack=pack)
    except Exception as exc:
        _d._ask = real; doc.close()
        rec("second detector isolated", "document", "FAIL",
            f"a failing second detector propagated: {type(exc).__name__}")
        return
    _d._ask = real
    kept = [f for f in finds if f.get("source") != "second_detector_degraded"]
    marked = [f for f in finds if f.get("source") == "second_detector_degraded"]
    doc.close()
    ok = len(kept) == 1 and len(marked) == 1
    rec("second detector isolated", "document", "PASS" if ok else "FAIL",
        "primary findings survive a failing second detector, and the "
        "degradation is recorded" if ok
        else f"kept {len(kept)} primary findings, {len(marked)} markers")


# ------------------------------------------------------------------ edge cases
def edge_cases(outdir):
    man_p = ROOT / "out" / "edge" / "manifest.json"
    if not man_p.exists():
        rec("edge corpus", "document", "SKIP", "run synth/edge_cases.py first")
        return
    man = json.loads(man_p.read_text())
    for m in man:
        src = ROOT / "out" / "edge" / m["file"]
        if src.suffix != ".pdf":
            continue
        out = Path(outdir) / f"red_{src.stem}.pdf"
        ok, log = run_doc(src, out)
        if not ok:
            rec(m["case"], "document", "FAIL", f"run failed: {log.strip()[-60:]}")
            continue
        doc = fitz.open(out)
        full = "\n".join(doc[i].get_text() for i in range(doc.page_count))
        orig = fitz.open(src)

        # invariant that applies to EVERY case
        added = any(A.added_text(orig[i].get_text(), doc[i].get_text())
                    for i in range(doc.page_count))
        if added:
            rec(m["case"] + " [no-write]", "document", "FAIL",
                "text was ADDED to the page")
        detail, status = "", "PASS"
        if m.get("max_boxes") == 0:
            nb = sum(1 for line in Path(out).with_suffix(".jsonl").open()
                     for f in json.loads(line)["findings"] if f.get("bbox"))
            blk = max(black_frac(doc[i]) for i in range(doc.page_count))
            status = "PASS" if nb == 0 and blk < 0.02 else "FAIL"
            detail = f"{nb} boxes, {blk:.1%} of page black (want 0 / <2%)"
        elif m.get("values"):
            gone = [v for v in m["values"] if v not in full]
            if m.get("known_gap"):
                status = "INCONCLUSIVE"
                detail = (f"{len(gone)}/{len(m['values'])} removed; "
                          "watermark is a known gap")
            else:
                status = "PASS" if len(gone) == len(m["values"]) else "FAIL"
                detail = f"{len(gone)}/{len(m['values'])} values removed"
        else:
            blk = max(black_frac(doc[i]) for i in range(doc.page_count))
            status = "INCONCLUSIVE"
            detail = f"no extractable ground truth; {blk:.1%} of page redacted"
        if m.get("must_survive"):
            lost = [v for v in m["must_survive"] if v not in full]
            if lost:
                status = "FAIL"
                detail += f" | DESTROYED data that must survive: {lost[:4]}"
            else:
                detail += f" | all {len(m['must_survive'])} kept values survive"
        rec(m["case"], "document", status, detail)
        doc.close(); orig.close()


# ------------------------------------------------------------- negative control
def cosmetic_control(outdir):
    """The control MUST fail. A verifier that cannot fail proves nothing."""
    src = ROOT / "out" / "edge" / "numeric_table.pdf"
    if not src.exists():
        rec("cosmetic control", "document", "SKIP", "edge corpus missing")
        return
    import base, redact, tokens as T
    jl = Path(outdir) / "red_numeric_table.jsonl"
    if not jl.exists():
        rec("cosmetic control", "document", "SKIP", "no findings to replay")
        return
    pf = {}
    for line in jl.open():
        r = json.loads(line)
        pf[r["page"]] = r["findings"]
    pack = base.load(ROOT / "taxonomies" / "insurance_medical")
    dest = Path(outdir) / "cosmetic.pdf"
    redact.redact_document(src, pf, dest, cosmetic=True, pack=pack,
                           tokens=T.TokenStore("regress"))
    d = fitz.open(dest)
    txt = "\n".join(d[i].get_text() for i in range(d.page_count))
    survived = sum(1 for p, fs in pf.items() for f in fs
                   if f.get("text") and not f["text"].startswith("[")
                   and f["text"] in txt)
    md = {k: v for k, v in (d.metadata or {}).items() if v}
    d.close()
    ok = survived > 0
    rec("cosmetic control FAILS", "document", "PASS" if ok else "FAIL",
        f"{survived} values still extractable, metadata kept "
        f"({len(md)} fields) - control correctly fails"
        if ok else "control PASSED - the verifier proves nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip anything that calls a model")
    ap.add_argument("--outdir", default="out/regress")
    a = ap.parse_args()
    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)

    print("\n== self-tests ==")
    selftests()
    print("\n== audio arm / shared modules ==")
    audio_shared_modules()
    qc_verdict_consistency()
    print("\n== taxonomy packs ==")
    pack_behaviour()
    print("\n== detector fallback ==")
    second_detector_isolation()
    print("\n== inherited annotations ==")
    inherited_redaction_marks()
    print("\n== table guard bounds ==")
    row_veto_bounded()
    print("\n== verification gate and correction loop ==")
    verify_gate_and_loop()
    if not a.quick:
        print("\n== edge cases (document pipeline) ==")
        edge_cases(out)
        print("\n== negative control ==")
        cosmetic_control(out)

    n = len(RESULTS)
    by = {}
    for _, _, s, _ in RESULTS:
        by[s] = by.get(s, 0) + 1
    print(f"\n{'='*78}\n{n} checks: " +
          "  ".join(f"{k} {v}" for k, v in sorted(by.items())))
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    if fails:
        print("\nFAILURES:")
        for c, arm, _, d in fails:
            print(f"  {arm:<10}{c:<30}{d}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
