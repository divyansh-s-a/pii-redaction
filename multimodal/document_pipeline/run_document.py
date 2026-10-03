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
import unread
import carriers
import marks as markpass   # not `marks`: handle() has a local of that name
import inkmarks
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
DETECT_RETRIES = 2      # a timed-out page is retried, not discarded


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
    # ON BY DEFAULT (owner, 2026-09-24). A single detector over a scanned
    # page can return NOTHING -- no error, no log line, a clean verify -- and
    # the page ships unredacted. Measured on a real 29-page claim: 2 of 29
    # pages got zero model findings with one detector and 0 of 29 with the
    # union, and 5 of 10 PII probes readable became 1. Two model families do
    # not fail on the same page. `--second-detector none` disables it; it
    # roughly doubles detection cost.
    ap.add_argument("--second-detector", default="gemini",
                    help="union a second model family's findings with the "
                         "primary's (default: gemini). `none` disables. "
                         "Covers pages a single detector silently misses")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--no-labels", action="store_true",
                    help="skip label-anchored form-field regions")
    ap.add_argument("--no-propagate", action="store_true",
                    help="skip bundle-wide value propagation")
    ap.add_argument("--no-inkmarks", action="store_true",
                    help="skip classifying residual ink on SCANNED pages. It "
                         "is the only pass that sees a black-ink signature "
                         "baked into a scan; it costs one model request per "
                         "scanned page that carries unexplained ink.")
    ap.add_argument("--no-marks", action="store_true",
                    help="skip the embedded-mark pass: small images on a "
                         "born-digital page that carry no text -- signatures "
                         "and logos. Without it a signature placed as its own "
                         "image, or an insurer/TPA logo, is readable in the "
                         "output because no text mechanism can see it.")
    ap.add_argument("--no-carriers", action="store_true",
                    help="skip the stamp/signature carrier pass. That pass "
                         "destroys a whole rubber-stamp or signature "
                         "impression EXCEPT a date found inside it; without "
                         "it a box lands on the stamp's centre and leaves "
                         "the provider name readable around the ring.")
    ap.add_argument("--no-unread", action="store_true",
                    help="skip the unread-text-region pass. That pass is the "
                         "ONLY thing that sees Indic script on a SCANNED "
                         "page: Apple Vision supports no Indian script, so "
                         "the state-language half of a bilingual ID is "
                         "otherwise never tokenised at all.")
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
    # No token store: PSEUDONYMIZE destroys exactly as REDACT does and writes
    # nothing back, so minting one only produced a re-identification key that
    # nothing consumed. Kept as an in-memory object so the call signatures
    # below are unchanged; it is never saved.
    toks = tokenstore.TokenStore(src.stem, secret=args.token_secret, path=None)

    if str(args.second_detector).lower() in ("none", "off", ""):
        args.second_detector = None
    model = common.resolve_model()
    per_page, stats = {}, Counter()
    markers = {}          # pno -> [marker findings]
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
            if rd["reader"] == "ocr_empty":
                # Not a page with nothing on it: the OCR subprocess failed
                # after its own retries. detect_page would return before
                # calling a model and before its regex pass, so this page
                # would be redacted on faces and QR alone. Say so.
                with lock:
                    print(f"  p{pno:<4} OCR RETURNED NO WORDS -- "
                          f"{rd.get('ocr_error') or 'unknown'}; this page has "
                          f"NO text coverage", flush=True)
            finds, _ = detector.detect_page(
                d[pno - 1], model, detector=args.detector,
                second=args.second_detector, words=rd["words"], pack=pack)
            finds = detector.locate_unboxed(d[pno - 1], finds, rd["words"])
            finds = formfields.spare_numeric_columns(
                d[pno - 1], rd["words"], finds, pack)
            reader_used = rd["reader"]
        except Exception as exc:
            # A transient failure here costs the WHOLE page: finds=[] and
            # reader_used="error" also disable the label and ink passes, so
            # the page ships with visual findings only. That is too much to
            # lose to one timed-out request, and timeouts are exactly what
            # happens when the API is busy, so retry before giving up.
            with lock:
                print(f"  p{pno:<4} detection error ({str(exc)[:60]}) "
                      f"-- retrying", flush=True)
            finds, reader_used = None, "error"
            for attempt in range(DETECT_RETRIES):
                time.sleep(4 * (attempt + 1))
                try:
                    finds, _ = detector.detect_page(
                        d[pno - 1], model, detector=args.detector,
                        second=args.second_detector, words=rd["words"],
                        pack=pack)
                    finds = detector.locate_unboxed(d[pno - 1], finds,
                                                    rd["words"])
                    finds = formfields.spare_numeric_columns(
                        d[pno - 1], rd["words"], finds, pack)
                    reader_used = rd["reader"]
                    break
                except Exception as exc2:
                    exc = exc2
            if finds is None:
                # THE MODEL FAILED; THE PAGE WAS STILL READ. Losing the model's
                # spans is unavoidable, but nothing else here needs a model:
                # the ID patterns, the script scan, the field-label pass, the
                # handwriting, unread-text and ink-mark passes. Marking the
                # whole page "error" switched every one of them off -- a
                # claim form shipped with its handwritten PAN and account
                # number readable after the detector's API account ran out of
                # credit, though the label pass would have cut both. Only a
                # page that could not be READ now loses everything.
                if "rd" in locals() and rd.get("words"):
                    try:
                        finds = detector.deterministic_findings(
                            d[pno - 1], rd["words"], pack)
                        finds = detector.locate_unboxed(d[pno - 1], finds,
                                                        rd["words"])
                        finds = formfields.spare_numeric_columns(
                            d[pno - 1], rd["words"], finds, pack)
                    except Exception:
                        finds = []
                    reader_used = rd["reader"]
                    with lock:
                        print(f"  p{pno:<4} MODEL DETECTION FAILED after "
                              f"{DETECT_RETRIES} retries: {str(exc)[:60]} -- "
                              f"NO MODEL COVERAGE; fixed-rule checks still ran",
                              flush=True)
                else:
                    with lock:
                        print(f"  p{pno:<4} DETECTION FAILED after "
                              f"{DETECT_RETRIES} retries: {str(exc)[:60]} -- "
                              f"this page has NO text coverage", flush=True)
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
                # Staff sign-off columns of a ruled table (medication charts,
                # nursing logs): the header says every cell below is a name,
                # a signature or a staff code, which no per-cell reading can.
                labs.extend(formfields.identity_columns(
                    d[pno - 1], rd["words"], pack=pack))
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
        # The labelled fields as the label pass zoned them, before handwriting
        # resolution rewrites `labs`. unread.py defers to these: text inside a
        # field the pack KEEPS (a bank name, an IFSC) must not be cut by a
        # classifier that saw one cell with no label beside it.
        zones = [dict(L) for L in labs]
        if not args.no_handwriting:
            try:
                hw, hwst = handwriting.regions(d[pno - 1], labels=labs,
                                               boxes=raw_ink, stats=raw_stats,
                                               words=rd["words"], pack=pack)
                for H in hw:
                    H["action"] = pack.action_for(H["categories"])
                # ink first: it inherited its categories from `labs` above,
                # so dropping the label fallback here loses no classification
                labs, shrunk, basis = handwriting.resolve_labels_vs_ink(
                    labs, hw, hwst["blue_share"])
            except Exception:
                hw = []
        hw_by_page[pno] = hw          # added after the repeat filter
        # Text NO reader transcribed -- on this corpus, the state-language
        # half of a bilingual ID page, which Apple Vision cannot read in any
        # Indian script. Geometry from Vision's region detector, the keep/cut
        # decision from the model; it costs one request only on pages that
        # actually carry unread text.
        # Rubber stamps and ink signatures, destroyed whole except a date
        # inside them. Pigment-based, so it needs no text and no model.
        car = []
        if not args.no_carriers:
            try:
                car = carriers.findings(d[pno - 1], pack=pack,
                                        workdir=args.workdir)
            except Exception as exc:
                with lock:
                    print(f"  p{pno} CARRIER PASS FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                car = []
        # Signatures and logos placed as their own small images. Bank marks
        # come back as sem:bank_common and the pack keeps them, as it keeps
        # an IFSC; everything else is identity and goes.
        mk = []
        if not args.no_marks:
            try:
                mk = markpass.findings(d[pno - 1], pack=pack,
                                    workdir=args.workdir)
            except Exception as exc:
                with lock:
                    print(f"  p{pno} MARK PASS FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                mk = []
        unr = []
        if not args.no_unread and reader_used != "error":
            try:
                unr = unread.findings(d[pno - 1], rd["words"], pack=pack,
                                      workdir=args.workdir, zones=zones,
                                      ocr=reader_used != "text_layer")
            except Exception as exc:
                with lock:
                    print(f"  p{pno} UNREAD-REGION PASS FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                unr = []
        # Last, so it can skip whatever the passes above already cut: black
        # ink on a scan that no reader explained, kept or cut by the model.
        ink, ink_marker = [], None
        if not args.no_inkmarks and reader_used != "error":
            already = [f["bbox"] for f in vis + car + mk + unr if f.get("bbox")]
            try:
                ink, ink_marker = inkmarks.findings(
                    d[pno - 1], rd["words"], reader_used, pack=pack,
                    covered=already, workdir=args.workdir)
            except Exception as exc:
                with lock:
                    print(f"  p{pno} INK-MARK PASS FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
                ink = []
            if ink_marker is not None:
                with lock:
                    print(f"  p{pno} INK MARKS DEGRADED: "
                          f"{ink_marker['reason'][:90]}", flush=True)
        boxed = ([f for f in finds if f.get("bbox")] + vis + labs + unr + car
                 + mk + ink)
        # Page markers have no bbox by design -- they describe the page, not a
        # region -- so the line above drops them. Keep them separately or the
        # run has no record that a page went unexamined.
        marks = [f for f in finds
                 if f.get("locator") == "page_marker"]
        if ink_marker is not None:
            marks.append(ink_marker)
        if reader_used == "ocr_empty":
            marks.append({
                "categories": [], "text": "", "bbox": None, "action": "KEEP",
                "source": "ocr_empty", "locator": "page_marker",
                "reason": str(rd.get("ocr_error") or "")[:300]})
        words_by_page[pno] = rd["words"] if reader_used != "error" else []
        readers_by_page[pno] = reader_used
        with lock:
            if marks:
                markers[pno] = marks
            stats["pages"] += 1
            stats["findings"] += len(finds)
            stats["boxed"] += len(boxed)
            stats["qr"] += sum(1 for v in vis if v["source"] == "objective_qr")
            stats["faces"] += sum(1 for v in vis if v["source"] in
                                  ("objective_face", "objective_idphoto"))
            stats["labels"] += len(labs)
            stats["unread"] += len(unr)
            stats["carriers"] += len(car)
            stats["marks"] += sum(1 for m in mk if m.get("action") != "KEEP")
            stats["inkmarks"] += len(ink)
            flags = "QR" if any(v["source"] == "objective_qr" for v in vis) else ""
            flags += ("+PHOTO" if any(v["source"] in ("objective_face",
                                                      "objective_idphoto")
                                      for v in vis) else "")
            flags += (f"+{len(labs)}L" if labs else "")
            flags += (f"+{len(unr)}U" if unr else "")
            flags += (f"+{len(car)}C" if car else "")
            flags += (f"+{len(mk)}M" if mk else "")
            flags += (f"+{len(ink)}I" if ink else "")
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
                                     "findings": finds + markers.get(pno, [])},
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
          f"{stats.get('handwriting', 0)} pen-ink | "
          f"{stats.get('unread', 0)} unread-text | "
          f"{stats.get('carriers', 0)} carriers | "
          f"{stats.get('marks', 0)} marks | "
          f"{stats.get('inkmarks', 0)} ink-marks | {dt/60:.1f} min")
    print(f"  redaction: pages={st['pages_touched']} boxes={st['boxes']}")
    print(f"  by action: {st['by_action']}")
    # A page examined only by the visual pass is the leak mode the owner
    # found on the ID pages, and it is invisible in every other number here:
    # the page contributes boxes (faces, QR) and looks like it was processed.
    blind = sorted(pno for pno, fs in markers.items()
                   if any(f.get("source") == "no_model_coverage" for f in fs))
    if blind:
        print(f"  NO MODEL COVERAGE on {len(blind)} page(s): {blind} "
              f"-- text on these pages was never examined by a detector")
    solo = sorted(pno for pno, fs in markers.items()
                  if any(f.get("source") == "second_detector_degraded"
                         for f in fs))
    if solo:
        print(f"  SINGLE-DETECTOR on {len(solo)} page(s): {solo} "
              f"-- union coverage was not applied there")
    noocr = sorted(pno for pno, fs in markers.items()
                   if any(f.get("source") == "ocr_empty" for f in fs))
    if noocr:
        print(f"  OCR FAILED on {len(noocr)} page(s): {noocr} "
              f"-- redacted on visual findings only")
    if st["degraded"]:
        print(f"  DEGRADED (action could not be applied as asked): {st['degraded']}")

    rc = 0
    if args.verify or args.remediate:
        rc = _verify_and_maybe_fix(src, args, per_page, pack, toks, audit)
    print(f"  -> {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
