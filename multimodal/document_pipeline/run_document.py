#!/usr/bin/env python3
"""Redact an ENTIRE document, every page, end to end.

    python3 document_pipeline/run_document.py INPUT.pdf --out OUT.pdf

Per page: pick a reader (text layer or vision OCR) -> detect PII (GPT) ->
project findings to exact boxes -> add the objective visual detections that no
text reader can see (QR codes, faces) -> redact the whole document in one pass
-> scrub object-level PII -> verify the result by re-reading it.

The visual pass is not optional decoration. An Aadhaar Secure QR decodes to the
complete identity - name, guardian, address, DOB - so a page whose printed
number is blacked out while its QR survives has not been redacted at all.
Faces are included for the same reason: a portrait identifies a person whether
or not their name is readable.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(PKG.parent / "taxonomies"))
sys.path.insert(0, str(PKG))

import importlib.util

import common
import policy
import detect as detector
import objective
import readers
import formfields
import handwriting
import propagate
import tokens as tokenstore
import registry as taxonomies
import fitz

HERE = Path(__file__).resolve().parent


def _load(name):
    """Import by file path.

    `pipeline/redact.py` (the audio renderer) shares a module name with this
    package's `redact.py`, and importing a shared audio module re-inserts
    pipeline/ at the front of sys.path -- so a plain `import redact` here
    silently picks up the wrong one. Pin it to the file.
    """
    spec = importlib.util.spec_from_file_location(
        f"mmpdf_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


redactor = _load("redact")

VISUAL_DPI = 300          # measured floor for dense QR decoding


def visual_findings(doc, pno, workdir):
    """QR codes and faces as redaction boxes in PDF points."""
    page = doc[pno - 1]
    W, H = page.rect.width, page.rect.height
    out = []
    try:
        got = objective.page_qr_and_faces(doc, pno - 1, dpi=VISUAL_DPI,
                                          workdir=workdir)
    except Exception:
        return out
    for q in got.get("qr", []):
        b = q.get("box")
        if q.get("native_xref") is not None:
            # native XObject: use its real placement rect(s) on the page
            try:
                for r in page.get_image_rects(q["native_xref"]):
                    out.append({"categories": ["sem:machine_code"],
                                "text": "[QR code]",
                                "bbox": [r.x0, r.y0, r.x1, r.y1],
                                "source": "objective_qr",
                                "aadhaar": bool(q.get("aadhaar"))})
            except Exception:
                pass
        elif b:
            out.append({"categories": ["sem:machine_code"],
                        "text": "[QR code]",
                        "bbox": [b["x0"] * W, b["y0"] * H,
                                 b["x1"] * W, b["y1"] * H],
                        "source": "objective_qr",
                        "aadhaar": bool(q.get("aadhaar"))})
    # Deterministic ID-photo fallback. Apple Vision misses the Aadhaar
    # portrait: it is placed at ~51x63pt, grayscale and heavily compressed, and
    # the detector wants more face than that. But on a page whose Secure QR
    # DECODED as an Aadhaar identity payload we already know what this page is,
    # and a portrait-shaped image object on an Aadhaar card is the cardholder.
    # Gating on the decode is what keeps this from blacking out every logo in
    # the corpus -- it fires only on confirmed identity documents.
    if any(q.get("aadhaar") for q in got.get("qr", [])):
        for im in page.get_images(full=True):
            xref, w, h = im[0], im[2], im[3]
            if not (100 <= w <= 600 and 0.6 <= w / h <= 0.9):
                continue
            for r in page.get_image_rects(xref):
                out.append({"categories": ["sem:face"],
                            "text": "[ID photograph]",
                            "bbox": [r.x0, r.y0, r.x1, r.y1],
                            "source": "objective_idphoto"})

    # QR by SHAPE, unioned with Vision's decode. The two fail differently:
    # Vision decodes a crisp symbol but returned 0 on the photocopied Aadhaar
    # card at p24, where the finder-pattern scan finds it; and the scan misses
    # a small symbol that Vision reads. Union, never either alone.
    try:
        have = [fitz.Rect(f["bbox"]) for f in out]
        for z in objective.locate_qr_by_pattern(page):
            if not any(abs(z.x0 - h.x0) < 6 and abs(z.y0 - h.y0) < 6
                       for h in have):
                out.append({"categories": ["sem:machine_code"], "text": "[QR code]",
                            "bbox": [z.x0, z.y0, z.x1, z.y1],
                            "source": "objective_qr_shape"})
    except Exception:
        pass

    # the drawer's signature on a cancelled cheque
    try:
        for z in objective.cheque_signature_zones(page):
            out.append({"categories": ["sem:signature"], "text": "[cheque signature]",
                        "bbox": [z.x0, z.y0, z.x1, z.y1],
                        "source": "objective_cheque_sig"})
    except Exception:
        pass

    for f in got.get("faces", []):
        b = f["box"]
        # pad: Vision's face box is tight to the features, and an ID portrait
        # is identifiable well beyond them (hair, ears, shoulders)
        w, h = (b["x1"] - b["x0"]) * W, (b["y1"] - b["y0"]) * H
        out.append({"categories": ["sem:face"], "text": "[face]",
                    "source": "objective_face",
                    "bbox": [b["x0"] * W - 0.25 * w, b["y0"] * H - 0.35 * h,
                             b["x1"] * W + 0.25 * w, b["y1"] * H + 0.35 * h]})
    return out


def _verify_and_maybe_fix(src, args, per_page, pack, toks, audit):
    """Re-read the finished document; optionally re-redact what survived.

    The document counterpart of the audio arm's edge_verify.verify_and_fix,
    which extends a beep and re-renders until the value is no longer audible.
    Here the equivalent is: find the values still readable inside their own
    box, hand them back as findings, redact again, look again.

    Bounded by --max-rounds, because an unbounded self-correcting loop is a
    worse failure than the miss it is chasing. A round that fixes nothing
    stops immediately -- if re-redacting the same box did not remove the
    value, doing it a third time will not either, and the honest outcome is
    to report it.

    Safe by construction in a way the audio loop is not: a further pass can
    only ADD boxes, never remove one, so the worst case is over-redaction --
    which the column and row guards already bound -- not a fresh leak.
    """
    verifier = _load("verify_removal")
    out = Path(args.out)
    rounds = args.max_rounds if args.remediate else 1
    left = []
    for rnd in range(1, rounds + 1):
        left = verifier.survivors(out, per_page)
        if not left:
            print(f"  verify: round {rnd}: nothing survived inside its box")
            return 0
        print(f"  verify: round {rnd}: {len(left)} value(s) still readable "
              f"inside their own box"
              + (f" -> re-redacting" if args.remediate and rnd < rounds else ""))
        for v in left[:5]:
            print(f"     p{v['page']} {v['text'][:40]!r}")
        if not args.remediate or rnd == rounds:
            break
        # hand the survivors back as findings and redact the OUTPUT again
        extra = {}
        for v in left:
            extra.setdefault(v["page"], []).append({
                "categories": v["categories"] or ["sem:unresolved"],
                "text": v["text"], "bbox": v["bbox"],
                "all_boxes": [v["bbox"]],
                "action": "REDACT",            # a survivor is not a judgement
                "source": f"remediation_round_{rnd}",
                "locator": "verify_survivor"})
        before = sum(len(x) for x in extra.values())
        tmp = out.with_suffix(f".r{rnd}.pdf")
        redactor.redact_document(out, extra, tmp, cosmetic=False, pack=pack,
                                 tokens=toks, audit=audit)
        tmp.replace(out)
        for pno, finds in extra.items():
            per_page.setdefault(pno, []).extend(finds)
        print(f"  verify: round {rnd}: re-redacted {before} box(es)")
    if left:
        print(f"  VERIFY FAILED: {len(left)} value(s) still present after "
              f"{rounds} round(s). The redaction mechanism did not remove "
              f"them; this is not a detection gap.")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--findings-out", default=None)
    ap.add_argument("--detector", default="gpt")
    # OPT-IN, like the audio arm's --assurance. It measurably reduces both
    # the leak rate and the run-to-run variance on scanned pages (see
    # results/document_redaction_report.md), but it doubles detection spend
    # and requires credentials for a second model family, so the operator
    # chooses it rather than inheriting it.
    ap.add_argument("--second-detector", default=None,
                    help="union a second detector family's findings with the "
                         "primary's (e.g. gemini). Cuts leak rate and "
                         "run-to-run variance on scans; ~2x detection cost")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--no-labels", action="store_true",
                    help="skip label-anchored form-field regions")
    ap.add_argument("--no-propagate", action="store_true",
                    help="skip bundle-wide value propagation")
    ap.add_argument("--no-handwriting", action="store_true",
                    help="skip chromatic-ink (pen) handwriting regions")
    ap.add_argument("--profile", default="strict")
    ap.add_argument("--taxonomy", default=None,
                    help="force a taxonomy pack id; default is to select one "
                         "from the document's own signals")
    ap.add_argument("--token-secret", default="",
                    help="keys the pseudonym hash; same secret + same bundle "
                         "reproduces the same tokens")
    ap.add_argument("--workdir", default="/private/tmp/claude-501/"
                    "-Users-shivangi-Documents-Claude-Audio-PII-Project/"
                    "87ddc989-26e9-49df-9724-61c42bfe09fe/scratchpad/rundoc")
    ap.add_argument("--pages", default=None, help="e.g. 1-10 (default: all)")
    # Opt-in, like the audio arm's --remediate-audit. Verification costs a
    # second read of the finished document; the loop costs a re-render per
    # round. Neither changes what is detected, only whether the pipeline
    # checks its own work and acts on the answer.
    ap.add_argument("--verify", action="store_true",
                    help="re-read the finished PDF and report any value that "
                         "survived; exits non-zero if the mechanism failed")
    ap.add_argument("--remediate", action="store_true",
                    help="as --verify, then RE-REDACT whatever survived and "
                         "check again, up to --max-rounds (implies --verify)")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="bound on --remediate; a loop that cannot terminate "
                         "is worse than a miss (default 3)")
    ap.add_argument("--workers", type=int, default=6,
                    help="pages detected concurrently (detection is the "
                         "bottleneck: ~7-40s/page depending on PII density)")
    args = ap.parse_args()

    common.CONFIG["profile"] = args.profile
    src = Path(args.src)
    doc = fitz.open(src)
    lo, hi = 1, doc.page_count
    if args.pages:
        a, _, b = args.pages.partition("-")
        lo, hi = int(a), int(b or a)
    # Which taxonomy applies. Cheap and deterministic: sample text from the
    # document itself and score each pack's declared signals against it.
    sample = []
    for i in range(min(6, doc.page_count)):
        t = doc[i].get_text().strip()
        if t:
            sample.append(t[:4000])
    if not sample:                       # an all-scan bundle has no text layer
        try:
            r0 = readers.read_page(doc, lo, workdir=args.workdir)
            sample.append(" ".join(t for _, t, _ in r0["words"])[:4000])
        except Exception:
            pass
    pack, decision = taxonomies.select({"text": sample, "file": src.name},
                                       override=args.taxonomy)
    print(f"taxonomy: {pack.id} v{pack.version}  ({decision['basis']}"
          f"{', AMBIGUOUS' if decision['ambiguous'] else ''})")
    print(f"  scores: {decision['scores']}")
    if decision["ambiguous"]:
        print("  -> fell back to the conservative pack; pass --taxonomy to override")
    toks = tokenstore.TokenStore(src.stem, secret=args.token_secret,
                                 path=Path(args.out).with_suffix(".tokens.json"))

    if str(args.second_detector).lower() in ("none", "off", ""):
        args.second_detector = None
    model = common.resolve_model()
    per_page, stats = {}, Counter()
    t0 = time.time()

    # One page at a time is dominated by the detector's round trip, and pages
    # are independent, so fan them out. PyMuPDF Documents are NOT safe for
    # concurrent use, so each worker thread opens its own handle to the file.
    local = threading.local()
    lock = threading.Lock()
    words_by_page = {}
    readers_by_page = {}
    hw_by_page = {}

    def handle(pno):
        if not hasattr(local, "doc"):
            local.doc = fitz.open(src)
        d = local.doc
        try:
            rd = readers.read_page(d, pno, workdir=args.workdir)
            finds, _ = detector.detect_page(
                d[pno - 1], model, detector=args.detector,
                second=args.second_detector, words=rd["words"], pack=pack)
            finds = detector.locate_unboxed(d[pno - 1], finds, rd["words"])
            finds = formfields.spare_numeric_columns(
                d[pno - 1], rd["words"], finds, pack)
            reader_used = rd["reader"]
        except Exception as exc:
            with lock:
                print(f"  p{pno:<4} DETECTION FAILED: {str(exc)[:70]}",
                      flush=True)
            finds, reader_used = [], "error"
        vis = [] if args.no_visual else visual_findings(d, pno, args.workdir)
        for v in vis:
            v["action"] = pack.action_for(v["categories"])
        # ink first, so the label pass can tell a filled field from a blank one
        raw_ink, raw_stats = [], {"blue_share": 0.0}
        if not args.no_handwriting and reader_used != "error":
            try:
                raw_ink, raw_stats = handwriting.ink_boxes(d[pno - 1])
            except Exception:
                raw_ink, raw_stats = [], {"blue_share": 0.0}
        labs = []
        if not args.no_labels and reader_used != "error":
            try:
                labs = formfields.value_regions(d[pno - 1], rd["words"],
                                                ink=raw_ink, pack=pack)
                # Identity blocks that NO label claims -- a letter's recipient
                # address, which carries no colon and no keyword. Runs after
                # value_regions and skips anything already inside a label's
                # region, so the two never contest the same box; on a fully
                # labelled form (every insurance page) it finds nothing, which
                # is why it costs nothing to leave on.
                blocks = formfields.unlabelled_identity_blocks(
                    d[pno - 1], rd["words"], labels=labs, pack=pack)
                labs.extend(blocks)
                for L in labs:
                    L["action"] = pack.action_for(L["categories"])
            except Exception as exc:
                # never swallow this silently: an exception here empties the
                # label fallback for the whole page, and the page still looks
                # plausible because the ink pass keeps drawing
                with lock:
                    print(f"  p{pno} LABEL PASS FAILED: {type(exc).__name__}: "
                          f"{str(exc)[:90]}", flush=True)
                labs = []
        hw, hwst, shrunk = [], {"blue_share": 0.0}, 0
        if not args.no_handwriting:
            try:
                hw, hwst = handwriting.regions(d[pno - 1], labels=labs,
                                               boxes=raw_ink, stats=raw_stats)
                for H in hw:
                    H["action"] = pack.action_for(H["categories"])
                # ink first: it inherited its categories from `labs` above,
                # so dropping the label fallback here loses no classification
                labs, shrunk, basis = handwriting.resolve_labels_vs_ink(
                    labs, hw, hwst["blue_share"])
            except Exception:
                hw = []
        hw_by_page[pno] = hw          # added after the repeat filter
        boxed = [f for f in finds if f.get("bbox")] + vis + labs
        words_by_page[pno] = rd["words"] if reader_used != "error" else []
        readers_by_page[pno] = reader_used
        with lock:
            stats["pages"] += 1
            stats["findings"] += len(finds)
            stats["boxed"] += len(boxed)
            stats["qr"] += sum(1 for v in vis if v["source"] == "objective_qr")
            stats["faces"] += sum(1 for v in vis if v["source"] in
                                  ("objective_face", "objective_idphoto"))
            stats["labels"] += len(labs)
            flags = "QR" if any(v["source"] == "objective_qr" for v in vis) else ""
            flags += ("+PHOTO" if any(v["source"] in ("objective_face",
                                                      "objective_idphoto")
                                      for v in vis) else "")
            flags += (f"+{len(labs)}L" if labs else "")
            flags += (f"+{len(hw)}HW" if hw else "")
            flags += (f"(ink>{shrunk}L)" if shrunk else "")
            print(f"  p{pno:<4}{reader_used:<11}{len(boxed):>3} boxes {flags}"
                  f"   [{stats['pages']}/{hi - lo + 1}]", flush=True)
        return pno, boxed

    common.set_usage_context(f"rundoc:{src.stem}")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pno, boxed in ex.map(handle, range(lo, hi + 1)):
                per_page[pno] = boxed
    finally:
        common.set_usage_context("")
    doc.close()

    if not args.no_handwriting:
        rep = handwriting.drop_repeated(hw_by_page)
        for pno, regs in hw_by_page.items():
            per_page[pno].extend(regs)
        stats["handwriting"] = sum(len(v) for v in hw_by_page.values())
        print(f"\n  handwriting: {stats['handwriting']} pen-ink regions kept, "
              f"{rep} repeated printed elements dropped")

    if not args.no_propagate:
        values = propagate.harvest(per_page, readers_by_page=readers_by_page,
                                   pack=pack)
        pdoc = fitz.open(src)

        def act(cats):
            return pack.action_for(cats)

        added = 0
        for pno in sorted(per_page):
            w = words_by_page.get(pno) or []
            if not w:
                continue
            extra = propagate.apply_to_page(pdoc[pno - 1], w, values,
                                            per_page[pno], act)
            per_page[pno].extend(extra)
            added += len(extra)
        pdoc.close()
        stats["propagated"] = added
        print(f"\n  propagation: {len(values)} bundle values -> {added} "
              f"additional boxes on pages that had missed them")
        for k, v in sorted(values.items(),
                           key=lambda kv: -len(kv[1]["pages"]))[:8]:
            print(f"     {v['categories'][0]:<20}{v['text'][:34]!r:<38}"
                  f"first seen p{v['pages'][0]}")

    if args.findings_out:
        with Path(args.findings_out).open("w") as fh:
            for pno, finds in per_page.items():
                fh.write(json.dumps({"file": src.name, "page": pno,
                                     "findings": finds},
                                    ensure_ascii=False) + "\n")
    audit = []
    st = redactor.redact_document(src, per_page, args.out, cosmetic=False,
                                  pack=pack, tokens=toks, audit=audit)
    Path(args.out).with_suffix(".audit.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in audit))
    dt = time.time() - t0
    print(f"\n{stats['pages']} pages | {stats['findings']} findings | "
          f"{stats['boxed']} boxes drawn | {stats['qr']} QR | "
          f"{stats['faces']} faces | {stats['labels']} label-regions | "
          f"{stats.get('propagated', 0)} propagated | "
          f"{stats.get('handwriting', 0)} pen-ink | {dt/60:.1f} min")
    print(f"  redaction: pages={st['pages_touched']} boxes={st['boxes']}")
    print(f"  by action: {st['by_action']}")
    if st["degraded"]:
        print(f"  DEGRADED (action could not be applied as asked): {st['degraded']}")
    print(f"  tokens minted: {len(toks)} -> "
          f"{Path(args.out).with_suffix('.tokens.json').name} "
          f"(RE-IDENTIFICATION KEY: do not ship with the PDF)")
    rc = 0
    if args.verify or args.remediate:
        rc = _verify_and_maybe_fix(src, args, per_page, pack, toks, audit)
    print(f"  -> {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
