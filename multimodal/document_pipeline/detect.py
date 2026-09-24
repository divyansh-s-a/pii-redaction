#!/usr/bin/env python3
"""P6/P7 — detect PII on a document page and localise it to exact boxes.

    python3 document_pipeline/detect.py --sample out/sample_born.json \
            --out out/findings_born.jsonl

The architectural rule inherited from the audio pipeline: THE MODEL NEVER
EMITS COORDINATES. It returns inclusive word-index ranges into a deterministic
token stream, and geometry comes from the stream. On a text-layer page that
makes localisation exact by construction rather than approximate - which is
why this path is built first: any localisation error here is a harness bug,
not a model limitation, and that makes it a calibration instrument for the
scoring code before the hard pages need it.

Reuses verbatim: policy.CATEGORIES / DETECTION_SCHEMA / the v4 policy prompt,
common.generate() with its AI-Studio->OpenRouter fallback and usage logging.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent      # Audio_PII_Project
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import judge_client
import policy
import objective
import geometry
import matching
import readers
import fitz

# Owner decisions 2026-09-17:
#   GPT detects, Gemini judges (reverse of the audio pipeline).
#   Gemini is available as an OPTIONAL SECOND DETECTOR - the document analogue
#   of the audio second listener, which cut must-redact leak rate 35.9% -> 11%
#   there by supplying content the primary reader never produced.
DETECTORS = {
    "gpt":    {"family": "openai", "model": judge_client.DOC_DETECTOR},
    "gemini": {"family": "google", "model": None},      # None -> resolve_model()
}

SUPPLEMENT = (Path(__file__).resolve().parent.parent / "prompts"
              / "pdf_detection_supplement_v1.txt")
CHUNK = 1200          # words per request; a dense page is ~700


def page_words(page):
    """-> [(index, text, fitz.Rect)] in extraction (reading) order."""
    out = []
    for i, w in enumerate(page.get_text("words")):
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
        out.append((i, txt, fitz.Rect(x0, y0, x1, y1)))
    return out


def words_as_prompt(words):
    return "\n".join(f"{i}\t{t}" for i, t, _ in words)


def _ask(detector, prompt, images=None):
    """One detection request, routed to the chosen family. Returns spans."""
    spec = DETECTORS[detector]
    if spec["family"] == "openai":
        got = judge_client.judge_call(
            prompt, _GPT_SPAN_SCHEMA, model=spec["model"],
            log_engine="detector_gpt", images=images)
        return got.get("spans", [])
    parts = [{"text": prompt}]
    for im in (images or []):
        parts.append({"image_path": str(im), "mime": "image/png"})
    raw = common.generate(spec["model"] or common.resolve_model(), parts,
                          policy.DETECTION_SCHEMA, thinking_budget=2048)
    return json.loads(raw).get("spans", [])


# OpenAI strict-mode mirror of policy.DETECTION_SCHEMA
_GPT_SPAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"spans": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "start_word": {"type": "integer"},
            "end_word": {"type": "integer"},
            "categories": {"type": "array", "items": {"type": "string"}},
            "spoken_text": {"type": "string"},
            "public_figure": {"type": "boolean"},
            "confidence": {"type": "string",
                           "enum": ["high", "medium", "low"]},
            "reason": {"type": "string"}},
        "required": ["start_word", "end_word", "categories", "spoken_text",
                     "public_figure", "confidence", "reason"]}}},
    "required": ["spans"]}



def detect_page(page, model, detector="gpt", second=None, words=None,
                pack=None):
    """-> list of findings with exact boxes. Empty list is a valid result.

    `second` adds a UNION detector: anything the second family finds that the
    primary missed is added, never subtracted. More redaction, never less -
    the same safe direction the audio union mode used.
    """
    # words may be supplied by readers.read_page() (vision OCR on a scan);
    # falling back to the native layer keeps the function usable standalone
    if words is None:
        words = page_words(page)
    if not words:
        return [], 0
    findings = []
    # A pack brings its OWN prompt and its own action matrix. Falling back to
    # the audio policy prompt is only for standalone/legacy use; a production
    # run always passes a pack, because the prompt and the matrix have to
    # describe the same taxonomy or the detector reports categories the
    # redactor cannot act on.
    if pack is not None:
        head = pack.prompt
    else:
        head = policy.DETECTION_POLICY + "\n\n" + SUPPLEMENT.read_text()
    degraded = []
    for base in range(0, len(words), CHUNK):
        chunk = words[base:base + CHUNK]
        prompt = head + "\n\nPAGE WORDS:\n" + words_as_prompt(chunk)
        spans = [(sp, detector) for sp in _ask(detector, prompt)]
        if second:
            covered = {(int(x["start_word"]), int(x["end_word"]))
                       for x, _ in spans
                       if str(x.get("start_word", "")).lstrip("-").isdigit()}
            # THE SECOND OPINION MUST NOT BE ABLE TO DESTROY THE FIRST.
            # Unwrapped, a failure here -- no key for the second family, an
            # auth error, a rate limit, a malformed response -- propagates out
            # of detect_page, and run_document's page-level handler turns that
            # into `finds = []` for the WHOLE page. A page would then be
            # redacted using no model findings at all, silently, because an
            # OPTIONAL extra detector was unavailable. Since this detector is
            # on by default, that would be the common failure, not a corner.
            try:
                extra = _ask(second, prompt)
            except Exception as exc:
                extra = []
                degraded.append(f"{second}: {type(exc).__name__}: "
                                f"{str(exc)[:90]}")
            for sp in extra:
                try:
                    key = (int(sp["start_word"]), int(sp["end_word"]))
                except (KeyError, TypeError, ValueError):
                    continue
                # union-only: keep it if no primary span already overlaps it
                if not any(a <= key[1] and key[0] <= b for a, b in covered):
                    spans.append((sp, f"{second}_union_only"))
        for s, src in spans:
            try:
                a, b = int(s["start_word"]), int(s["end_word"])
            except (KeyError, TypeError, ValueError):
                continue
            lo, hi = chunk[0][0], chunk[-1][0]
            if not (lo <= a <= b <= hi):
                continue                     # out-of-range index: drop
            raw_cats = s.get("categories", []) or []
            if pack is not None:
                cats = [c for c in raw_cats if c in pack.categories]
                if raw_cats and not cats:
                    # the detector named categories this pack does not define.
                    # Keep the span and let the matrix fail closed rather than
                    # dropping it, which would be a silent leak.
                    cats = raw_cats
            else:
                cats = [c for c in raw_cats if c in policy.CATEGORIES]
            if not cats:
                continue
            sel = [w for w in words if a <= w[0] <= b]
            if not sel:
                continue
            # One box per contiguous run, NOT one union over the whole span:
            # a span whose tokens sit on different lines or opposite sides of
            # the page unions into a rectangle that swallows everything
            # between them.
            rb = geometry.run_boxes(sel)
            keep = [r for r in rb
                    if not geometry.too_large([r.x0, r.y0, r.x1, r.y1],
                                              page.rect)]
            text = " ".join(t for _, t, _ in sel)
            if not keep:
                findings.append({
                    "categories": cats, "text": text, "bbox": None,
                    "locator": "oversize_rejected", "source": src,
                    "n_words": len(sel),
                    "reason": "every candidate box exceeded the page-area gate",
                })
                continue
            box = keep[0]
            findings.append({
                "start_word": a, "end_word": b,
                "categories": cats,
                "text": text,
                "bbox": [round(v, 2) for v in (box.x0, box.y0, box.x1, box.y1)],
                "all_boxes": [[round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]
                              for r in keep],
                "n_boxes": len(keep),
                "n_words": len(sel),
                "confidence": s.get("confidence", "medium"),
                "public_figure": bool(s.get("public_figure", False)),
                "action": (pack.action_for(cats) if pack is not None
                           else policy.action_for(
                               cats, bool(s.get("public_figure", False)),
                               common.CONFIG.get("profile", "strict"))),
                "source": src,
                "locator": "word_index",
                "reason": str(s.get("reason", ""))[:160],
            })
    # deterministic objective pass over the same text: catches what the model
    # skipped and marks which values are checksum-PROVEN (evidence tier T0)
    full = " ".join(t for _, t, _ in words)
    for hit in objective.scan_text(full):
        findings.append({
            "categories": [hit["category"]], "text": hit["value"],
            "bbox": None, "locator": "search_needed",
            "checksum_ok": hit["checksum_ok"], "kind": hit["kind"],
            "source": "objective_regex",
            "action": (pack.action_for([hit["category"]]) if pack is not None
                       else policy.action_for(
                           [hit["category"]], False,
                           common.CONFIG.get("profile", "strict"))),
        })
    if degraded:
        # Loud on purpose. The justification for running two detectors is that
        # one alone is not repeatable on a degraded scan; if the second never
        # ran, the page carries single-detector exposure and nothing in the
        # output would otherwise say so.
        print(f"  SECOND DETECTOR UNAVAILABLE on this page -- falling back to "
              f"{detector} alone; union coverage NOT applied. "
              f"{degraded[0]}", flush=True)
        findings.append({
            "categories": [], "text": "", "bbox": None, "action": "KEEP",
            "source": "second_detector_degraded", "locator": "page_marker",
            "reason": "; ".join(degraded)[:300],
        })
    return findings, len(words)



def token_runs_matching(words, target):
    """Every CONTIGUOUS run of tokens that reads as `target`.

    Delegates to matching.find_runs so detection, propagation and the token
    store all agree on what "the same value" means -- they used to disagree,
    and a value could be redacted on one page and missed on the next purely
    because two modules normalised it differently.
    """
    return matching.find_runs(words, target)


def locate_unboxed(page, findings, words=None):
    """Give the objective-regex findings a box via search_for().

    search_for returns EVERY occurrence, which is what redaction wants: a value
    printed three times on one page needs three boxes, not one.
    """
    for f in findings:
        if f.get("bbox") or f.get("locator") != "search_needed":
            continue
        try:
            rects = page.search_for(f["text"], quads=False)
        except Exception:
            rects = []
        if not rects and words:
            # Scanned page: search_for has no text layer, so match the reader's
            # own tokens. The test must be "does this RUN OF TOKENS spell the
            # target", never "is this token a substring of the target" -- the
            # latter matches 'in', 'o', 'care' and 'com' anywhere on the page
            # and unions them into one page-sized rectangle.
            rects = token_runs_matching(words, f["text"])
        rects = [r for r in rects
                 if not geometry.too_large([r[0], r[1], r[2], r[3]],
                                           page.rect)]
        if rects:
            f["bbox"] = [round(v, 2) for v in rects[0]]
            f["all_boxes"] = [[round(v, 2) for v in r] for r in rects]
            f["n_boxes"] = len(rects)
            f["locator"] = "search_for"
        else:
            f["locator"] = "unlocatable"
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", default="out/sample_born.json")
    ap.add_argument("--out", default="out/findings.jsonl")
    ap.add_argument("--corpus", default=str(readers.corpus_dir()))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detector", default="gpt", choices=list(DETECTORS),
                    help="primary detector family (owner default: gpt)")
    ap.add_argument("--reader", default="auto",
                    choices=["auto", "text_layer", "vision_ocr"],
                    help="auto picks the text layer only for born-digital "
                         "pages; scans and invisible-OCR pages use vision OCR")
    ap.add_argument("--workdir", default="/private/tmp/claude-501/"
                    "-Users-shivangi-Documents-Claude-Audio-PII-Project/"
                    "87ddc989-26e9-49df-9724-61c42bfe09fe/scratchpad/det")
    ap.add_argument("--second-detector", default=None, choices=list(DETECTORS),
                    help="optional union detector, e.g. gemini - adds what "
                         "the primary missed, never removes")
    args = ap.parse_args()

    doc_sample = json.loads(Path(args.sample).read_text())
    pages = doc_sample["pages"][:args.limit or None]
    model = common.resolve_model()
    corpus = Path(args.corpus)
    opened, n_find, n_words = {}, 0, 0

    with Path(args.out).open("w") as fh:
        for i, p in enumerate(pages, 1):
            d = opened.get(p["file"]) or opened.setdefault(
                p["file"], fitz.open(corpus / p["file"]))
            page = d[p["page"] - 1]
            common.set_usage_context(f"pdfdetect:{p['file']}:p{p['page']}")
            try:
                rd = readers.read_page(d, p["page"], reader=args.reader,
                                       workdir=args.workdir)
                finds, nw = detect_page(page, model,
                                        detector=args.detector,
                                        second=args.second_detector,
                                        words=rd["words"])
                for f in finds:
                    f["reader"] = rd["reader"]
                finds = locate_unboxed(page, finds, rd["words"])
            except Exception as exc:
                print(f"  !! {p['file'][:24]} p{p['page']}: {str(exc)[:90]}")
                finds, nw = [], 0
            finally:
                common.set_usage_context("")
            n_find += len(finds)
            n_words += nw
            fh.write(json.dumps({"file": p["file"], "page": p["page"],
                                 "weight": p.get("weight"), "words": nw,
                                 "findings": finds}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(pages)}] {p['file'][:26]:<28}p{p['page']:<5}"
                  f"{nw:>5}w -> {len(finds):>3} findings", flush=True)
    for d in opened.values():
        d.close()
    print(f"\n{len(pages)} pages, {n_words} words, {n_find} findings")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
