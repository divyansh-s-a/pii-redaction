#!/usr/bin/env python3
"""What to hide on a sampled frame.

Geometry comes from deterministic readers wherever it can (the repository
rule: a model's box is a hint, never truth):
  * Apple Vision OCR — Latin text with per-token boxes; the mapping matcher
    runs over these tokens directly (no model needed for a readable brand);
  * Apple Vision text regions — WHERE unreadable (Indic) text sits;
  * Apple Vision faces.
A vision model then decides what each token/region IS (by id) and adds boxes
only for what no reader locates: logos without text, packaging, mascots,
faces Vision missed. Two model families can run (Gemini primary, GPT second
detector, union) — one can silently return nothing on a frame.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "document_pipeline"))
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
PROMPT = (HERE.parent / "prompts" / "video_frame_detection_v1.txt").read_text()
KINDS = ["logo", "brand_text", "packaging", "mascot", "signage", "url_handle",
         "phone", "face", "other_brand"]
PHONE_RX = re.compile(r"(?:\+?\d[\d\s-]{6,}\d)")
URL_RX = re.compile(r"(?:www\.|https?://|\.com\b|\.in\b|@\w{3,}|#\w{3,})", re.I)
BATCH = 4

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"frames": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "frame": {"type": "integer"},
            "items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": KINDS},
                    "entity_id": {"type": "string"},
                    "ocr_ids": {"type": "array", "items": {"type": "integer"}},
                    "region_ids": {"type": "array",
                                   "items": {"type": "integer"}},
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                    "label": {"type": "string"}},
                "required": ["kind", "entity_id", "ocr_ids", "region_ids",
                             "box_2d", "label"]}}},
        "required": ["frame", "items"]}}},
    "required": ["frames"],
}


def vision_batch(paths, mode):
    """Apple Vision over many images in one helper process."""
    import vision_exec
    if not paths:
        return []
    r = subprocess.run(vision_exec.argv(["--batch", mode] + [str(p) for p in
                                                              paths]),
                       capture_output=True, text=True, timeout=1800)
    try:
        out = json.loads(r.stdout)["batch"]
    except (ValueError, KeyError):
        raise RuntimeError(f"vision helper failed ({mode}): "
                           f"{(r.stderr or r.stdout)[:300]}")
    if len(out) != len(paths):
        raise RuntimeError("vision helper returned a short batch")
    return out


def _b(d):
    return [float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"])]


def read_frames(paths):
    """-> per frame {'tokens': [...], 'lines': [...], 'regions': [...],
    'faces': [...]} with normalised [x0, y0, x1, y1] boxes (top-left
    origin). Token ids are per frame."""
    texts = vision_batch(paths, "text")
    regs = vision_batch(paths, "textregions")
    faces = vision_batch(paths, "faces")
    out = []
    for t, r, f in zip(texts, regs, faces):
        toks, lines = [], []
        for li, line in enumerate(t.get("text", [])):
            ids = []
            for tk in line.get("tokens", []):
                ids.append(len(toks))
                toks.append({"id": len(toks), "text": tk["text"],
                             "box": _b(tk["box"]), "line": li})
            lines.append({"text": line["text"], "token_ids": ids,
                          "box": _b(line["box"])})
        regions = [{"id": k, "box": [rr["x0"], rr["y0"], rr["x1"], rr["y1"]]}
                   for k, rr in enumerate(r.get("textregions") or [])]
        # regions the OCR already read are not "unread"
        unread = [rg for rg in regions
                  if not any(_iou(rg["box"], ln["box"]) > 0.3 for ln in lines)]
        out.append({"tokens": toks, "lines": lines, "regions": unread,
                    "faces": [{"box": _b(x["box"]),
                               "confidence": x.get("confidence")}
                              for x in f.get("faces", [])]})
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def deterministic(read, em):
    """Regions the readers alone justify: mapped text (matcher over OCR
    tokens, plus tokens that CONTAIN a mapped name, e.g. a URL), phone
    numbers, URLs/handles, and every Vision face."""
    out = []
    toks = read["tokens"]
    for ln in read["lines"]:
        ids = ln["token_ids"]
        words = [toks[i]["text"] for i in ids]
        if em is not None:
            for i0, i1, e, how in em.match_tokens(words):
                out.append({"box": _union_box([toks[ids[j]]["box"]
                                               for j in range(i0, i1 + 1)]),
                            "kind": "brand_text", "entity": e["id"],
                            "source": "ocr+mapping", "match": how})
        for j, w in enumerate(words):
            hit = None
            if em is not None:
                k = _key(w)
                for e in em.entries:
                    if any(len(ek) >= 5 and ek in k for ek in e["keys"]):
                        hit = e
                        break
            if hit is not None:
                out.append({"box": toks[ids[j]]["box"], "kind": "url_handle",
                            "entity": hit["id"], "source": "ocr+mapping",
                            "match": "contains"})
            elif URL_RX.search(w):
                out.append({"box": toks[ids[j]]["box"], "kind": "url_handle",
                            "entity": None, "source": "ocr"})
        if PHONE_RX.search(ln["text"]) and sum(c.isdigit()
                                               for c in ln["text"]) >= 7:
            dig = [toks[i]["box"] for i in ids
                   if any(c.isdigit() for c in toks[i]["text"])]
            if dig:
                out.append({"box": _union_box(dig), "kind": "phone",
                            "entity": None, "source": "ocr"})
    for f in read["faces"]:
        out.append({"box": f["box"], "kind": "face", "entity": None,
                    "source": "vision_faces"})
    return out


def _key(w):
    import entity_map
    return entity_map.key(w)


def _frame_block(k, read):
    toks = " | ".join(f"t{t['id']}: {t['text']}" for t in read["tokens"])
    regs = " ".join(f"r{r['id']}" for r in read["regions"])
    return (f"FRAME {k}: OCR tokens: {toks or '(none)'}; unread text "
            f"regions: {regs or '(none)'}")


def model_pass(paths, reads, em, model, native_gemini):
    """Ask a vision model about a batch of frames; returns per-frame region
    lists (boxes normalised, from token/region ids or the model's box_2d)."""
    import judge_client
    block = em.prompt_block() if em is not None else "ENTITIES: (none)"
    prompt = (PROMPT + "\n\n" + block + "\n\n"
              + "\n".join(_frame_block(k, r) for k, r in enumerate(reads))
              + "\n\nThe images follow in the same order (frame 0 first).")
    raw = judge_client.judge_call(prompt, SCHEMA, model=model,
                                  images=[str(p) for p in paths],
                                  native_gemini=native_gemini,
                                  log_engine="video_detect", max_tokens=8000)
    res = [[] for _ in paths]
    for fr in raw.get("frames", []):
        k = fr.get("frame")
        if not isinstance(k, int) or not (0 <= k < len(paths)):
            continue
        read = reads[k]
        for it in fr.get("items", []):
            boxes = [read["tokens"][i]["box"] for i in it.get("ocr_ids", [])
                     if 0 <= i < len(read["tokens"])]
            boxes += [read["regions"][i]["box"]
                      for i in it.get("region_ids", [])
                      if 0 <= i < len(read["regions"])]
            src = "model_ids"
            b2 = it.get("box_2d") or []
            if not boxes and len(b2) == 4:
                y0, x0, y1, x1 = [max(0, min(1000, v)) / 1000 for v in b2]
                if x1 > x0 and y1 > y0:
                    boxes = [[x0, y0, x1, y1]]
                    src = "model_box"
            if not boxes:
                continue
            e = em.entry_for_prompt_id(it.get("entity_id")) if em else None
            res[k].append({"box": _union_box(boxes), "kind": it.get("kind"),
                           "entity": e["id"] if e else None,
                           "source": f"{src}:{model}",
                           "label": str(it.get("label", ""))[:60]})
    return res


def detect(paths, em, models):
    """paths: sampled frame images. models: [(model, native_gemini), ...].
    Returns (reads, regions_per_frame)."""
    reads = read_frames(paths)
    regions = [deterministic(r, em) for r in reads]
    for model, native in models:
        for b in range(0, len(paths), BATCH):
            try:
                got = model_pass(paths[b: b + BATCH], reads[b: b + BATCH],
                                 em, model, native)
            except Exception as exc:          # one family failing must not
                print(f"    !! {model} frames {b}-{b + BATCH - 1}: "
                      f"{str(exc)[:120]}")    # silently pass a frame
                for k in range(b, min(b + BATCH, len(paths))):
                    regions[k].append({"box": None, "kind": "error",
                                       "source": model, "entity": None})
                continue
            for k, rs in enumerate(got):
                regions[b + k] += rs
    return reads, regions
