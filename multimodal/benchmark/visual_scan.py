#!/usr/bin/env python3
"""Objective visual signals (QR + faces) over a page sample -> out/visual.jsonl

Deliberately samples two groups so the rates mean something:
  candidates  pages whose census row suggests an ID card (square image >=300px,
              or a small photo-aspect image) - where QR/faces should be found
  control     random image-only pages - to expose false positives
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "document_pipeline"))
import objective
import fitz

D = readers.corpus_dir()
OUT = Path("out"); OUT.mkdir(exist_ok=True)
WORK = Path("/private/tmp/claude-501/-Users-shivangi-Documents-Claude-Audio-"
            "PII-Project/87ddc989-26e9-49df-9724-61c42bfe09fe/scratchpad/vis")

N_CONTROL = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def candidates(rows):
    """Pages that LOOK like they carry an ID card, from the free census."""
    out = []
    for r in rows:
        fp = r.get("fingerprint", "")
        sizes = []
        for tok in fp.split("|")[-1].split(","):
            if "x" in tok:
                try:
                    w, h = (int(x) for x in tok.split("x"))
                    sizes.append((w, h))
                except ValueError:
                    pass
        square = any(w == h and w >= 300 for w, h in sizes)
        photo = any(60 <= w <= 400 and 0.6 <= (w / h if h else 0) <= 0.95
                    for w, h in sizes)
        if square or photo:
            out.append((r, "square_image" if square else "photo_aspect"))
    return out


def main():
    rows = [json.loads(l) for l in open(OUT / "census.jsonl")]
    cand = candidates(rows)
    rng = random.Random(20260917)
    imgonly = [r for r in rows if r["kind"] == "image_only"]
    control = [(r, "control") for r in rng.sample(imgonly,
                                                  min(N_CONTROL, len(imgonly)))]
    todo = cand + control
    print(f"candidate pages: {len(cand)} | control pages: {len(control)}")

    results = []
    docs = {}
    for r, why in todo:
        doc = docs.get(r["file"]) or docs.setdefault(r["file"],
                                                     fitz.open(D / r["file"]))
        try:
            got = objective.page_qr_and_faces(doc, r["page"] - 1, workdir=WORK)
        except Exception as exc:
            got = {"error": str(exc)[:120], "qr": [], "faces": []}
        row = {"file": r["file"], "page": r["page"], "group": why,
               "kind": r["kind"], "n_qr": len(got.get("qr", [])),
               "n_faces": len(got.get("faces", [])),
               "qr_source": got.get("source"),
               "qr_decoded": sum(1 for q in got.get("qr", [])
                                 if q.get("payload")),
               "aadhaar_qr": sum(1 for q in got.get("qr", [])
                                 if q.get("aadhaar")),
               "latin_text_regions": got.get("latin_text_regions"),
               "error": got.get("error")}
        results.append(row)
        flag = ("QR" if row["n_qr"] else "") + ("F" if row["n_faces"] else "")
        print(f"  {r['file'][:26]:<28}p{r['page']:<4}{why:<14}{flag}", flush=True)
    for d in docs.values():
        d.close()
    with (OUT / "visual.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    cands = [r for r in results if r["group"] != "control"]
    ctrl = [r for r in results if r["group"] == "control"]
    print(f"\n=== candidates (n={len(cands)}) ===")
    print(f"  pages with a QR: {sum(1 for r in cands if r['n_qr'])}"
          f" | decoded: {sum(r['qr_decoded'] for r in cands)}"
          f" | Aadhaar-parsed: {sum(r['aadhaar_qr'] for r in cands)}")
    print(f"  pages with a face: {sum(1 for r in cands if r['n_faces'])}"
          f" | total faces: {sum(r['n_faces'] for r in cands)}")
    print(f"=== control (n={len(ctrl)}) ===")
    print(f"  pages with a QR: {sum(1 for r in ctrl if r['n_qr'])}"
          f" | pages with a face: {sum(1 for r in ctrl if r['n_faces'])}")
    print(f"\n  -> {OUT/'visual.jsonl'}")


if __name__ == "__main__":
    main()
