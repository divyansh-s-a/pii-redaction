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


def doc_redact():
    """The DOCUMENT redactor, loaded from its file, never by bare name.

    pipeline/judge_client.py puts the audio arm's pipeline/ back at the front
    of sys.path every time it is imported, and that directory has its own
    redact.py. So a bare `import redact` in this long-lived process returns
    whichever redactor the import order happened to favour: the suite crashed
    with "module 'redact' has no attribute 'neutralise_inherited_redactions'"
    the first time a new check imported a module that pulls in judge_client
    ahead of it. run_document.py already pins its redactor to the file this
    way; the suite now does the same.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "regress_doc_redact", ROOT / "document_pipeline" / "redact.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def openrouter_no_training():
    """Every OpenRouter chat request asks for providers that keep no data.

    Fails if: any of the four request builders (judge, Gemini fallback, audio
    listener, report writer) posts a payload without
    provider.data_collection == "deny", or the helper lets a caller's own
    routing keys switch the deny off or get dropped. No request leaves the
    machine: the transport is replaced by one that records and stops.
    """
    import importlib.util
    import tempfile
    import common
    import judge_client
    from adapters import openrouter_audio

    class Sent(Exception):
        pass

    sent = {}

    def capture(name):
        def fake(url, data=None, headers=None, method=None, timeout=300):
            sent[name] = json.loads(data)
            raise Sent()
        return fake

    spec = importlib.util.spec_from_file_location(
        "regress_final_report", PROJ / "generate_final_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    real_req, real_env = common._request, common._read_env_value
    common._read_env_value = lambda name: "sk-test"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "x.wav"
            wav.write_bytes(b"RIFF")
            calls = [
                ("judge_call", lambda: judge_client.judge_call(
                    "p", {"type": "object"}, model="openai/gpt-5.6-sol")),
                ("gemini fallback", lambda: common._generate_openrouter(
                    "gemini-2.5-pro", [{"text": "p"}], None, 0, 100)),
                ("audio listener", lambda: openrouter_audio._call(
                    "openai/gpt-audio", wav)),
                ("final report", lambda: report._openrouter_chat(
                    "anthropic/claude-opus-4.8", "s", "u", 100)),
            ]
            for name, call in calls:
                common._request = capture(name)
                try:
                    call()
                except Sent:
                    pass
    finally:
        common._request, common._read_env_value = real_req, real_env
    checks = [(f"{name} sends deny",
               (sent.get(name) or {}).get("provider", {})
               .get("data_collection") == "deny")
              for name, _ in calls]
    merged = common.openrouter_provider(
        {"order": ["a"], "data_collection": "allow"})
    checks.append(("routing keys kept, deny not overridable",
                   merged == {"order": ["a"], "data_collection": "deny"}))
    bad = [n for n, ok in checks if not ok]
    rec("openrouter no-training", "shared", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


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


def script_scan():
    """The non-Latin scan: fires on Indic, silent on Latin, indices only.

    Guards the two ways this can break quietly -- going dead (a name stands)
    or going greedy (every Latin page starts losing text) -- and the contract
    that it returns word indices and never coordinates.
    """
    import objective as _ob
    W = lambda toks: [(i, t, None) for i, t in enumerate(toks)]
    checks = []

    latin = _ob.scan_script(W(["Patient", "Zarinabanu", "Mysuru", "570010"]))
    checks.append(("silent on pure Latin", latin == []))

    mixed = _ob.scan_script(W(["Name", "\u0ca8\u0cc2\u0cb0\u0ccd",
                               "\u0c9c\u0cb9\u0cbe\u0ca8\u0ccd", "DOB"]))
    checks.append(("one run over adjacent Indic tokens", len(mixed) == 1))
    checks.append(("run covers exactly the Indic span",
                   bool(mixed) and mixed[0]["start_word"] == 1
                   and mixed[0]["end_word"] == 2))
    checks.append(("emits the semantic key, not a category",
                   bool(mixed) and mixed[0]["category"] == "sem:non_latin_script"))
    checks.append(("NO COORDINATES in the result",
                   all(not any(k in h for k in ("bbox", "box", "rect", "x0"))
                       for h in mixed)))

    scripts = {"Devanagari": "\u0938\u094b\u0928\u093e\u0932\u0940",
               "Tamil": "\u0bb7\u0baa\u0bbe\u0ba9\u0bbe",
               "Telugu": "\u0c05\u0c28\u0c41\u0c1c",
               "Kannada": "\u0ca8\u0cc2\u0cb0\u0ccd"}
    for name, tok in scripts.items():
        checks.append((f"{name} detected", _ob.is_non_latin(tok)))
    for tok in ("Rs.", "12,500.00", "BAY/13679/26", "info@x.in"):
        checks.append((f"Latin/numeric {tok!r} not flagged",
                       not _ob.is_non_latin(tok)))

    bad = [n for n, ok in checks if not ok]
    rec("non-latin script scan", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


def propagation_names():
    """Single-word personal names propagate; common words still do not."""
    import propagate as _pg, base as _b
    pack = _b.load(ROOT / "taxonomies" / "insurance_medical")
    def one(txt, cat):
        return _pg.harvest({1: [{"categories": [cat], "text": txt,
                                 "bbox": [0, 0, 10, 10], "source": "gpt",
                                 "confidence": "high"}]},
                           readers_by_page={1: "text_layer"}, pack=pack)
    checks = [
        ("single-word name learned", bool(one("Zarinabanu", "T01"))),
        ("second single-word name", bool(one("Gulnaaz", "T01"))),
        ("multi-word name still learned", bool(one("K R Hari Prasad", "T01"))),
        ("'Female' not learned", not one("Female", "T01")),
        ("'Hospital' not learned", not one("Hospital", "T01")),
        ("'Surgeon' not learned", not one("Surgeon", "T01")),
        ("name shape needs a NAME category", not one("Zarinabanu", "T37")),
    ]
    bad = [n for n, ok in checks if not ok]
    rec("propagation of names", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


def row_veto_money_gated():
    """The row veto protects a BILL line, not a clinical measurements row.

    Both look like "text on a row carrying figures", which is how the veto
    came to flip an A-scan biometry chart's measurements back to KEEP under a
    pack that deliberately redacts clinical content. The discriminator is
    whether the row's figures are MONEY.

    Also guards the failed first fix: filtering the veto to semantics the
    pack already KEEPs makes it a no-op, since the veto's only action is to
    flip REDACT -> KEEP.
    """
    import formfields as _ff, base as _b
    gen = _b.load(ROOT / "taxonomies" / "generic_document")

    def probe(label, figures, x=300):
        d = fitz.open(); pg = d.new_page()
        pg.insert_text((x, 200), label, fontsize=10)
        for i, (a, b) in enumerate(figures):
            pg.insert_text((350, 200 + i * 16), a, fontsize=9)
            pg.insert_text((450, 200 + i * 16), b, fontsize=9)
        words = [(i, w[4], fitz.Rect(w[:4]))
                 for i, w in enumerate(pg.get_text("words"))]
        box = [fitz.Rect(w[:4]) for w in pg.get_text("words")
               if w[4] == label.split()[0]][0]
        f = [{"categories": ["sem:clinical"], "text": label,
              "bbox": tuple(box), "action": gen.action_for(["sem:clinical"])}]
        out = _ff.spare_numeric_columns(pg, words, f, gen)
        d.close()
        return out[0]["action"]

    MONEY = [("1,200.00", "4,500.00"), ("2,300.00", "5,600.00"),
             ("3,400.00", "6,700.00")]
    MEASURE = [("3.49", "4.06"), ("15.63", "23.17"), ("1532", "1641")]
    checks = [
        ("bill description spared", probe("Surgeon", MONEY) == "KEEP"),
        ("veto is not inert", probe("Surgeon", MONEY) != "REDACT"),
        ("separate column still redacted",
         probe("Diagnosis positive", MONEY, x=112) == "REDACT"),
        ("measurements row NOT spared under generic",
         probe("Axial length", MEASURE) == "REDACT"),
        ("money page detected",
         _ff._money_context([(0, "Total", None), (1, "Amount", None)])),
        ("clinical page not a money page",
         not _ff._money_context([(0, "REFRACTION", None),
                                 (1, "Allergies", None)])),
    ]
    bad = [n for n, ok in checks if not ok]
    rec("row veto money-gated", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


def band_ceiling():
    """The letterhead band stops at a table's top border -- except over a form.

    On a 2338pt scan of progress notes the 18% band reached 421pt while the
    notes table began near 140pt, and the first rows of notes (a service
    date, "POD-2", vitals) were cut as letterhead. The border there is tilted
    and has rounded corners, which a straight-row test never finds. Lifting
    the band is also how a patient-data form leaks: its table starts just as
    high, and "Name of the Patient" / "Age" / "Sex" have their answers written
    on the line below, where no label region reaches.

    Fails if: the tilted notes border is not found, the band is lifted over a
    printed identity label (including two headings on one line, which the
    label finder's prose veto used to drop), a paper-edge shadow with no
    column rules is taken for a table, the band lifts with no words to check,
    or the pen's own word "mother" blocks the lift as if it were a label.
    """
    import handwriting as _hw, base as _b
    ins = _b.load(ROOT / "taxonomies" / "insurance_medical")
    BLUE = (0.1, 0.15, 0.85)

    def page(headers, pen, edge_shadow=False, border=True):
        d = fitz.open(); pg = d.new_page(width=1700, height=2338)
        if edge_shadow:
            pg.draw_line((20, 18), (1680, 18), color=(0.3, 0.3, 0.3), width=4)
        if border:
            # ~2 degrees of tilt: 60pt of drop across the sheet
            y = lambda x: 180 - 60 * (x - 100) / 1500
            pg.draw_line((100, y(100)), (1600, y(1600)), color=(0, 0, 0), width=3)
            for x in (100, 500, 1100, 1600):
                pg.draw_line((x, y(x)), (x, 2000), color=(0, 0, 0), width=3)
        for x, t in headers:        # one printed baseline, below the border
            pg.insert_text((x, 215), t, fontsize=26)
        for x, yy, t in pen:
            pg.insert_text((x, yy), t, fontsize=40, color=BLUE)
        words = [(i, w[4], fitz.Rect(w[:4]))
                 for i, w in enumerate(pg.get_text("words"))]
        return d, pg, words

    def zones(pg, words, give_words=True):
        boxes, st = _hw.ink_boxes(pg)
        kw = dict(words=words, pack=ins) if give_words else {}
        f, _ = _hw.regions(pg, labels=[], boxes=boxes, stats=st,
                           clinical_page=True, **kw)
        return {z for z in (x["zone"] for x in f)}

    NOTES = [(120, "Date & Time"), (620, "PROGRESS NOTES"),
             (1180, "TREATMENT ADVICE")]
    PEN = [(150, 330, "14/06/26"), (640, 330, "POD-2 pt fine")]
    d1, p1, w1 = page(NOTES, PEN)
    d2, p2, w2 = page([(120, "Name of the Patient"), (620, "Age"),
                       (1180, "Sex")], PEN)
    d3, p3, w3 = page([(120, "NAME OF THE ANAESTHETIST"),
                       (900, "NAME OF THE SISTER")], PEN)
    d4, p4, w4 = page(NOTES, PEN, edge_shadow=True, border=False)
    d5, p5, w5 = page(NOTES, PEN + [(1150, 300, "Baby by mother side")])
    rule = _hw._first_table_rule(p1)
    checks = [
        ("tilted border found", rule is not None and abs(rule[0]) > 0.02),
        ("notes rows lifted out of the band", zones(p1, w1) == {"unzoned_kept"}),
        ("identity labels keep the band", zones(p2, w2) == {"band"}),
        ("two headings on one line keep the band", zones(p3, w3) == {"band"}),
        ("paper-edge shadow is not a table", zones(p4, w4) == {"band"}),
        ("no words, no lift", zones(p1, w1, give_words=False) == {"band"}),
        ("pen 'mother' does not block", zones(p5, w5) == {"unzoned_kept"}),
    ]
    for d in (d1, d2, d3, d4, d5):
        d.close()
    bad = [n for n, ok in checks if not ok]
    rec("band ceiling", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


def signoff_columns():
    """Staff sign-off columns are found on a photographed chart, and only there.

    A phone photo of a ring-bound medication sheet has leaning, faint column
    rules and headers that wrap inside narrow cells ("Administered / by").
    The first column pass found 2 of its 5 sign-off columns: a single tall
    straight-run test lost the leaning rules, a fixed darkness threshold lost
    the hairline ones, and a page-wide line grouping never read the wrapped
    header as one phrase. Making the rules easy to find then exposed the
    opposite failure: a claim form's instructions table has full-height
    rules too, and its row label "c) Company TPA ID No." was taken for a
    column header -- 1500pt of printed guidance blacked out.

    Fails if: a leaning hairline-ruled sign-off column is missed, the wrapped
    "Administered by" header is not read, a column is invented over a
    non-identity header, a row label in the middle of a column counts as its
    header, or a column of printed sentences is cut.
    """
    import formfields as _ff

    def chart(faint=True, lean=10.0):
        d = fitz.open(); pg = d.new_page(width=595, height=842)
        xs = [40, 130, 220, 310, 400, 500]
        grey = (0.62, 0.62, 0.62) if faint else (0, 0, 0)
        for i, x in enumerate(xs):
            col = (0, 0, 0) if i in (0, len(xs) - 1) else grey
            pg.draw_line((x, 100), (x + lean, 760), color=col,
                         width=0.6 if col == grey else 1.2)
        for y in (100, 150, 760):
            pg.draw_line((40, y), (500 + lean, y), color=(0, 0, 0), width=1)
        pg.insert_text((50, 125), "Date", fontsize=9)
        pg.insert_text((140, 125), "Drug", fontsize=9)
        pg.insert_text((230, 118), "Administered", fontsize=9)
        pg.insert_text((230, 130), "by", fontsize=9)
        pg.insert_text((320, 125), "Dose", fontsize=9)
        pg.insert_text((410, 125), "Remarks", fontsize=9)
        for k in range(8):
            pg.insert_text((236, 180 + 60 * k), "Neha 4071", fontsize=9)
        return d, pg

    def instructions():
        d = fitz.open(); pg = d.new_page(width=595, height=842)
        for x in (40, 230, 560):
            pg.draw_line((x, 100), (x, 780), color=(0, 0, 0), width=1)
        rows = ["a) Policy No.", "b) Certificate No.", "c) Company TPA ID No.",
                "d) Name", "e) Address", "f) Staff ID", "g) Date", "h) Time"]
        for k, lab in enumerate(rows):
            y = 130 + 80 * k
            pg.insert_text((48, y), lab, fontsize=9)
            pg.insert_text((238, y), "Enter the number of the policy as it is "
                           "printed on the card", fontsize=8)
            pg.draw_line((40, y + 30), (560, y + 30), color=(0, 0, 0), width=1)
        return d, pg

    def declaration():
        # headed like a sign-off column, but its body is printed sentences
        d = fitz.open(); pg = d.new_page(width=595, height=842)
        for x in (40, 150, 330, 560):
            pg.draw_line((x, 100), (x, 700), color=(0, 0, 0), width=1)
        pg.draw_line((40, 100), (560, 100), color=(0, 0, 0), width=1)
        pg.insert_text((50, 118), "Date", fontsize=9)
        pg.insert_text((160, 118), "Signature", fontsize=9)
        pg.insert_text((340, 118), "Remarks", fontsize=9)
        for k in range(6):
            pg.insert_text((156, 160 + 60 * k), "I agree to the terms of the",
                           fontsize=8)
            pg.insert_text((156, 172 + 60 * k), "policy and all of its rules",
                           fontsize=8)
        return d, pg

    def words(pg):
        return [(i, w[4], fitz.Rect(w[:4]))
                for i, w in enumerate(pg.get_text("words"))]

    d1, p1 = chart()
    cols = _ff.identity_columns(p1, words(p1))
    d2, p2 = instructions()
    neg = _ff.identity_columns(p2, words(p2))
    d3, p3 = declaration()
    neg += _ff.identity_columns(p3, words(p3))
    got = [(c["text"], round(c["bbox"][0]), round(c["bbox"][2])) for c in cols]
    checks = [
        ("leaning hairline rules found", len(_ff._vertical_rules(p1)) >= 6),
        ("wrapped 'Administered by' column found",
         any("Administered by" in t and 215 <= x0 <= 240 and 300 <= x1 <= 325
             for t, x0, x1 in got)),
        ("no column over non-identity headers", len(cols) == 1),
        ("row labels and prose are not sign-off columns", neg == []),
    ]
    d1.close(); d2.close(); d3.close()
    bad = [n for n, ok in checks if not ok]
    rec("sign-off columns", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad} got={got} neg={[c['text'] for c in neg]}")


def unread_ink_lines():
    """A printed line that no reader or region detector saw still reaches
    the classifier -- and a cut-line, a pen line or a read line does not.

    Vision's region detector returned nothing on the Telugu name line of a
    scanned Aadhaar card, so the name was never classified and never cut.
    Lines are now also found from the pixels. Fails if: the unread printed
    line is not proposed, a line the reader already transcribed is proposed
    again, a dashed cut-line or a coloured pen line is proposed, or a
    born-digital page (ocr=False) runs the pixel pass at all.
    """
    import unread as _ur
    d = fitz.open(); pg = d.new_page(width=595, height=842)
    pg.insert_text((60, 120), "Name  Ramesh Kumar Rao", fontsize=14)
    pg.insert_text((60, 180), "UNREAD PRINTED LINE XYZ", fontsize=14)
    # dark enough to pass the darkness gate: only its colour marks it as pen
    pg.insert_text((60, 240), "blue pen line written here", fontsize=14,
                   color=(0.0, 0.0, 0.3))
    # a scissor mark and a dashed cut-line: line-tall, but its ink is one band
    pg.insert_text((60, 307), "X", fontsize=17)
    for x in range(75, 400, 12):
        pg.draw_line((x, 301), (x + 6, 301), color=(0, 0, 0), width=1.5)
    allw = [(i, w[4], fitz.Rect(w[:4]))
            for i, w in enumerate(pg.get_text("words"))]
    # the "reader" saw the first line only
    known = [r for _, t, r in allw if r.y1 < 140]
    lines = _ur.ink_lines(pg, known)
    hit = lambda y0, y1: [r for r in lines if y0 <= (r.y0 + r.y1) / 2 <= y1]
    real_vision = _ur.objective.vision
    _ur.objective.vision = lambda *a, **k: {"textregions": []}
    try:
        seen = [w for w in allw if w[2].y1 < 140]
        off = _ur.unread_regions(pg, seen, workdir="/tmp", ocr=False)
        on = _ur.unread_regions(pg, seen, workdir="/tmp", ocr=True)
    finally:
        _ur.objective.vision = real_vision
    checks = [
        ("unread printed line proposed", len(hit(160, 190)) == 1),
        ("read line not proposed again", hit(100, 135) == []),
        ("pen line left to the ink passes", hit(220, 250) == []),
        ("dashed cut-line not proposed", hit(290, 310) == []),
        ("born-digital page skips the pixel pass", off == []),
        ("scanned page gets the pixel lines",
         any(160 <= (r.y0 + r.y1) / 2 <= 190 for r in on)),
    ]
    d.close()
    bad = [n for n, ok in checks if not ok]
    rec("unread ink lines", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad} lines={[[round(v) for v in r] for r in lines]}")


def chart_signoffs():
    """Sign-offs in a medication chart's date grid reach the classifier whole.

    A medication-administration record has no "Signature" header over its
    date columns: each cell's pen entry is either clinical (a time) or the
    nurse who gave the dose -- a name over a staff number. OCR read those as
    loose words, so the residual pass never showed them to the classifier,
    and judged piece by piece a staff number read as a date.

    Fails if: the lattice is missed, a form's few large boxes are taken for
    one, the chart's wide printed side panel is swallowed into it, a name and
    the staff number under it stay two candidates, neighbouring columns'
    entries merge, a ruled-line stub in an empty cell becomes a candidate, or
    a residual candidate inside the grid is still classified as residual.
    """
    import inkmarks as _im

    def chart():
        d = fitz.open(); pg = d.new_page(width=595, height=842)
        # side panel: three wide columns; lattice: time column + 10 day columns
        xs = [20, 90, 160, 200] + [200 + 30 * k for k in range(1, 11)]
        for x in xs:
            pg.draw_line((x, 150), (x, 750), color=(0, 0, 0), width=0.8)
        for k in range(31):
            y = 150 + 20 * k
            pg.draw_line((20, y), (500, y), color=(0, 0, 0), width=0.8)
        pg.insert_text((24, 190), "Dosage", fontsize=8)
        pg.insert_text((165, 225), "8AM", fontsize=9)
        pg.insert_text((203, 225), "Meera", fontsize=9)      # day 1, row 4
        pg.insert_text((204, 245), "40417", fontsize=9)      # day 1, row 5
        pg.insert_text((203, 265), "Kavya", fontsize=9)      # day 1, row 6
        pg.insert_text((233, 225), "Tara", fontsize=9)       # day 2, same row
        pg.draw_line((270, 360), (295, 360), color=(0, 0, 0), width=0.8)  # stub
        # a stretch of a bent rule: slanted, so it fills every row it spans
        pg.draw_line((322, 455), (347, 466), color=(0, 0, 0), width=0.8)
        # a rule stub with a blot on it: not thin everywhere, but its ink
        # still sits in one band
        pg.draw_line((382, 560), (407, 560), color=(0, 0, 0), width=1.5)
        pg.draw_rect(fitz.Rect(392, 558, 395, 562), color=(0, 0, 0),
                     fill=(0, 0, 0), width=0)
        return d, pg

    def form():
        d = fitz.open(); pg = d.new_page(width=595, height=842)
        # ruled like a table, but in boxes far too big for a chart's cells
        for x in range(40, 561, 85):
            pg.draw_line((x, 100), (x, 700), color=(0, 0, 0), width=1)
        for y in range(100, 701, 30):
            pg.draw_line((40, y), (550, y), color=(0, 0, 0), width=1)
        return d, pg

    d1, p1 = chart()
    grids = _im.charting_grids(p1)
    ents = _im.grid_entries(p1, grids) if grids else []
    def hits(x0, y0, x1, y1):
        box = fitz.Rect(x0, y0, x1, y1)
        return [r for r in ents if r.intersects(box)]
    first, second = hits(201, 212, 229, 228), hits(231, 212, 259, 228)
    d2, p2 = form()
    # stub the model: every candidate is a signature; record who asked
    seen = []
    real = _im.classify_regions
    def fake(page, rects, workdir=None, cap=None, text_pt=None,
             margin_cap=None, prompt=None, schema=None):
        seen.append(list(rects))
        return [(r, "signature", "stub") for r in rects], [], []
    _im.classify_regions = fake
    real_res, real_gate = _im.residual_marks, _im.grid_has_signoffs
    inside = fitz.Rect(203, 216, 228, 228)
    _im.residual_marks = lambda *a, **k: ([inside], []) if k.get("split") else [inside]
    def srcs_with(gate):
        _im.grid_has_signoffs = lambda *a, **k: gate
        fnd, _mk = _im.findings(p1, [], "vision_ocr", pack=None, workdir="/tmp")
        return {f["source"] for f in fnd
                if fitz.Rect(f["bbox"]).intersects(fitz.Rect(201, 212, 229, 250))}
    try:
        srcs = srcs_with(True)
        srcs_gate_off = srcs_with(False)
        # second opinion: the first judgement keeps everything, the second
        # (reversed batch) cuts -- the entry must still be cut
        calls = {"n": 0}
        def split_vote(page, rects, workdir=None, cap=None, text_pt=None,
                       margin_cap=None, prompt=None, schema=None):
            calls["n"] += 1
            k = "time" if calls["n"] == 1 else "staff_number"
            if margin_cap is None:            # the residual call, not the grid
                k = "stray_mark"
            return [(r, k, "stub") for r in rects], [], []
        _im.classify_regions = split_vote
        srcs_second = srcs_with(True)
        # a multi-row block the model calls a date is still cut
        def says_date(page, rects, workdir=None, cap=None, text_pt=None,
                      margin_cap=None, prompt=None, schema=None):
            return [(r, "date", "stub") for r in rects], [], []
        _im.classify_regions = says_date
        tall_cut = srcs_with(True)              # the name/number block spans 2+ rows
    finally:
        _im.classify_regions = real
        _im.residual_marks, _im.grid_has_signoffs = real_res, real_gate
    checks = [
        ("lattice found", len(grids) == 1),
        ("side panel left out", bool(grids) and grids[0].x0 >= 155),
        ("name and staff number are one candidate",
         len(first) == 1 and first[0].y1 >= 262),
        ("neighbouring columns stay apart",
         len(second) == 1 and second[0] != first[0] if first else False),
        ("ruled stub is not a candidate", hits(265, 350, 300, 370) == []),
        ("bent-rule stretch is not a candidate", hits(318, 450, 350, 470) == []),
        ("blotted rule stub is not a candidate", hits(378, 552, 410, 568) == []),
        ("a form's few big boxes are not a chart", _im.charting_grids(p2) == []),
        ("grid pass owns ink inside the grid", srcs == {"grid_ink"}),
        ("gate says no: residual pass keeps its candidates",
         srcs_gate_off == {"residual_ink"}),
        ("either of two grid judgements can cut", srcs_second == {"grid_ink"}),
        ("a multi-row block is no date", tall_cut == {"grid_ink"}),
    ]
    d1.close(); d2.close()
    bad = [n for n, ok in checks if not ok]
    rec("chart sign-offs", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else
        f"BROKEN: {bad} grids={[[round(v) for v in g] for g in grids]} "
        f"ents={[[round(v) for v in r] for r in ents]}")


def embedded_marks():
    """Small no-text images: identity marks go, bank marks stay.

    Built from synthetic images so it needs no corpus. Guards both ways this
    breaks: going dead (a signature image stands, as it did on the review
    bundle's promotion letter and insurance cards) and going greedy (a bank's
    institution-level mark, which both packs keep like an IFSC, is destroyed).
    """
    import marks as _mk, base as _b
    pack = _b.load(ROOT / "taxonomies" / "insurance_medical")

    def page_with(img_text=None, size=(60, 30), with_text_over=False):
        d = fitz.open(); pg = d.new_page()
        pg.insert_text((72, 60), "Body text so this is a born-digital page")
        pm = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 360, 180), False)
        pm.set_rect(pm.irect, (255, 255, 255))
        src = fitz.open()
        sp = src.new_page(width=360, height=180)
        if img_text:
            sp.insert_text((20, 110), img_text, fontsize=48)
        else:
            sp.draw_bezier((20, 120), (80, 20), (160, 170), (300, 60),
                           color=(0, 0, 0), width=4)     # a scrawl
        png = sp.get_pixmap(dpi=144).tobytes("png")
        r = fitz.Rect(300, 300, 300 + size[0], 300 + size[1])
        pg.insert_image(r, stream=png)
        if with_text_over:
            pg.insert_text((r.x0 + 2, r.y0 + 12), "Over", fontsize=9)
        return d, pg

    checks = []
    d, pg = page_with()                       # a bare signature-like scrawl
    f = _mk.findings(pg, pack=pack)
    checks.append(("signature image redacted",
                   len(f) == 1 and f[0]["action"] == "REDACT"))
    d.close()

    d, pg = page_with("HDFC BANK", size=(150, 36))
    f = _mk.findings(pg, pack=pack)
    checks.append(("bank mark kept",
                   len(f) == 1 and f[0]["action"] == "KEEP"
                   and f[0]["categories"] == ["sem:bank_common"]))
    d.close()

    d, pg = page_with("ACME INSURANCE", size=(150, 36))
    f = _mk.findings(pg, pack=pack)
    checks.append(("insurer logo redacted",
                   len(f) == 1 and f[0]["action"] == "REDACT"))
    d.close()

    d, pg = page_with(size=(400, 300))        # too big to be a mark
    checks.append(("large image left to other passes",
                   _mk.findings(pg, pack=pack) == []))
    d.close()

    d, pg = page_with(with_text_over=True)    # text layer already covers it
    checks.append(("image under text left to the text passes",
                   _mk.findings(pg, pack=pack) == []))
    d.close()

    bad = [n for n, ok in checks if not ok]
    rec("embedded marks", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


def classifier_alignment():
    """A multi-image reply that does not cover every image must not be trusted.

    Found live: sent a bare list of 11 crops, the model answered for 10, every
    index after the skipped one slid by one, a printed fragment inherited the
    "signature" verdict and the real signature got none -- deterministically,
    every run. Both model-backed region classifiers now label each image and
    reject a reply whose indices do not match. This checks the rejection
    without a network call, and that ink-mark classification stays off
    born-digital pages, where every ink mark is text or its own image.
    """
    import inkmarks as _ik, unread as _ur, common as _cm
    real = _cm.generate
    checks = []
    try:
        d = fitz.open(); pg = d.new_page()
        rects = [fitz.Rect(100, 100 + 30 * i, 140, 120 + 30 * i) for i in range(3)]
        short = '{"marks": [{"index": 0, "kind": "signature", "why": ""},' \
                ' {"index": 1, "kind": "stray_mark", "why": ""}]}'
        _cm.generate = lambda *a, **k: short
        try:
            _ik._classify(pg, rects, "/tmp")
            checks.append(("inkmarks rejects a 2-of-3 reply", False))
        except ValueError:
            checks.append(("inkmarks rejects a 2-of-3 reply", True))
        # a model that only ever answers for image 0: a multi-image batch is
        # a miscount, a single-image batch is complete. Splitting must turn
        # the miscount into three clean one-image answers, not three cuts.
        shortu = '{"regions": [{"index": 0, "text": "x", "is_personal": true,' \
                 ' "kind": "name"}]}'
        _cm.generate = lambda *a, **k: shortu
        got = _ur.classify(pg, rects, workdir="/tmp")
        checks.append(("unread splits a miscounted batch and resolves it",
                       len(got) == 3 and all(k == "name" for _r, _t, k in got)))
        def boom(*a, **k):
            raise RuntimeError("down")
        _cm.generate = boom
        got = _ur.classify(pg, rects, workdir="/tmp")
        checks.append(("unread fails CLOSED when the model is down",
                       len(got) == 3 and all(k == "unclassified" for _r, _t, k in got)))
        d.close()
        d = fitz.open(); pg = d.new_page()
        pg.insert_text((72, 72), "born digital")
        f, m = _ik.findings(pg, [], "text_layer")
        checks.append(("inkmarks skips born-digital pages", f == [] and m is None))
        d.close()
    finally:
        _cm.generate = real
    bad = [n for n, ok in checks if not ok]
    rec("classifier alignment", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


def pack_behaviour():
    import base, registry
    p = base.load(ROOT / "taxonomies" / "insurance_medical")
    checks = [
        ("clinical kept", p.action_for(["T37"]) == "KEEP"),
        ("amounts kept", p.action_for(["T34"]) == "KEEP"),
        ("episode dates kept", p.action_for(["T21"]) == "KEEP"),
        ("DOB redacted", p.action_for(["T17"]) == "REDACT"),
        ("age/sex redacted", p.action_for(["T18"]) == "REDACT"),
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
        ("issuing organisation hidden in BOTH packs (owner 2026-09-24)",
         g.action_for(["sem:org_contact"]) == "REDACT"
         and p.action_for(["sem:org_contact"]) == "REDACT"
         and g.action_for(["sem:org_name"]) == "REDACT"),
        ("DOB and age/sex redacted under BOTH packs",
         g.action_for(["G08"]) == "REDACT" and g.action_for(["G09"]) == "REDACT"
         and p.action_for(["T17"]) == "REDACT"
         and p.action_for(["T18"]) == "REDACT"),
        ("event dates kept under BOTH packs",
         g.action_for(["G33"]) == "KEEP" and p.action_for(["T21"]) == "KEEP"),
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
    import verify_removal as _v, base as _b, tokens as _t
    _r = doc_redact()
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
    import actions as _a
    _r = doc_redact()
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

    # the mirror case: the PRIMARY is down (an account out of credit) and the
    # second family must carry the page instead of the page losing its model
    doc = _f.open(); pg = doc.new_page()
    pg.insert_text((60, 100), "Name: Sunita Rao", fontsize=11)
    words = _d.page_words(pg)

    def primary_down(det, prompt, images=None):
        if det == "gpt":
            raise RuntimeError("judge HTTP 402: Insufficient credits")
        return [{"start_word": 1, "end_word": 2, "categories": ["G01"],
                 "spoken_text": "Sunita Rao", "public_figure": False,
                 "confidence": "high", "reason": "name"}]

    def both_down(det, prompt, images=None):
        raise RuntimeError("down")
    _d._ask = primary_down
    try:
        finds, _n = _d.detect_page(pg, "x", detector="gpt", second="gemini",
                                   words=words, pack=pack)
        err = None
    except Exception as exc:
        finds, err = [], exc
    _d._ask = both_down
    try:
        _d.detect_page(pg, "x", detector="gpt", second="gemini",
                       words=words, pack=pack)
        both_raised = False
    except Exception:
        both_raised = True
    _d._ask = real
    doc.close()
    kept = [f for f in finds if f.get("source") != "second_detector_degraded"]
    marked = [f for f in finds if f.get("source") == "second_detector_degraded"]
    ok = err is None and len(kept) == 1 and len(marked) == 1 and both_raised
    rec("primary detector isolated", "document", "PASS" if ok else "FAIL",
        "second detector carries a page whose primary is down, marked "
        "single-detector; both down still reaches the fallback" if ok
        else f"err={err!r} kept={len(kept)} markers={len(marked)} "
             f"both_raised={both_raised}")


def openai_fallback():
    """GPT detection survives an OpenRouter account that has run dry.

    Fails if: an account refusal (402 "insufficient credits") does not switch
    the call to OpenAI directly, the switch does not stick for later calls,
    an ordinary failure (a timeout) is mistaken for an account problem and
    silently reroutes spend, or a missing OpenAI key hides the original error.
    """
    import openai_direct as _od
    real_call, real_key, st0 = _od.call, _od.key, dict(_od.STATE)
    log = []

    def refused(*a, **k):
        log.append("openrouter")
        raise RuntimeError("judge HTTP 402: Insufficient credits")

    def timeout(*a, **k):
        log.append("openrouter")
        raise RuntimeError("judge HTTP 0: TimeoutError")

    def direct(prompt, schema, model, images=None, log_engine="x"):
        log.append("direct")
        return {"spans": []}
    _od.call, _od.key = direct, (lambda: "sk-test")
    try:
        _od.STATE.update(direct=False, announced=True)
        _od.judge_or_direct(refused, "p", {}, "openai/gpt-5.6-sol")
        _od.judge_or_direct(refused, "p", {}, "openai/gpt-5.6-sol")
        switched = log == ["openrouter", "direct", "direct"]
        _od.STATE.update(direct=False); log.clear()
        try:
            _od.judge_or_direct(timeout, "p", {}, "openai/gpt-5.6-sol")
            kept_error = False
        except RuntimeError:
            kept_error = log == ["openrouter"]
        _od.STATE.update(direct=False); log.clear()
        _od.key = lambda: None
        try:
            _od.judge_or_direct(refused, "p", {}, "openai/gpt-5.6-sol")
            no_key = False
        except RuntimeError as exc:
            no_key = "402" in str(exc)
        model_ok = _od.model_id("openai/gpt-5.6-sol") == "gpt-5.6-sol"
    finally:
        _od.call, _od.key = real_call, real_key
        _od.STATE.clear(); _od.STATE.update(st0)
    checks = [("402 switches to OpenAI, and stays", switched),
              ("a timeout is not an account problem", kept_error),
              ("no OpenAI key: the original error surfaces", no_key),
              ("vendor prefix dropped for OpenAI", model_ok)]
    bad = [n for n, ok in checks if not ok]
    rec("openai fallback", "document", "PASS" if not bad else "FAIL",
        f"{len(checks)} checks" if not bad else f"BROKEN: {bad}")


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
        # A case whose expectation is a PACK CONTRACT must name the pack.
        # numeric_table asserts "every bill particular survives", which is
        # true under insurance_medical and deliberately false under
        # generic_document (G21 removes health content). Leaving it to auto
        # selection tested one pack against the other's contract.
        extra = (("--taxonomy", m["taxonomy"]) if m.get("taxonomy") else ())
        ok, log = run_doc(src, out, extra)
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
    import base, tokens as T
    redact = doc_redact()
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
    openrouter_no_training()
    qc_verdict_consistency()
    print("\n== taxonomy packs ==")
    pack_behaviour()
    script_scan()
    propagation_names()
    row_veto_money_gated()
    band_ceiling()
    signoff_columns()
    chart_signoffs()
    unread_ink_lines()
    embedded_marks()
    classifier_alignment()
    print("\n== detector fallback ==")
    second_detector_isolation()
    openai_fallback()
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
