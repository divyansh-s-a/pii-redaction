#!/usr/bin/env python3
"""Visual leftover pass for the insurance route (--visual-verify; owner's request, 2026-10-22).

The route's own --verify / --remediate re-read the TEXT of the output and catch a value still readable inside its own box. What they
cannot see is what a vision reader sees: a signature, stamp, logo, seal or ID photo only partly covered, handwriting that runs past
the edge of a box, a name in another script, a printed name the detector never read. This pass looks at every page of the finished
output as a person would, and blacks out what is still readable, page by page, without any list of pages or files:

  FIND    a vision model sees the whole redacted page on a lettered grid (columns A-J, rows 1-14) and lists every item the pack's
          rules say must be removed that is still readable, with the grid cells it touches
  FINE    per item, a zoomed crop on a fine grid (columns a-p); the model lists every fine cell the item touches
  COVER   each connected run of chosen cells becomes ONE box, the cells' union tightened to the ink inside (the crop's own paper
          level decides what is ink, so faint stamps and show-through count), so a stroke crossing several cells is covered whole
  VERIFY  the model sees the proposed boxes on the crop and vetoes any that would also destroy something the rules keep (clinical
          text, amounts, episode dates, printed labels) or that miss the item; a vetoed run falls back to per-cell ink boxes,
          verified the same way
  CHECK   FIND again on the fixed page; at most --visual-rounds fix rounds, then one last look that only reports

New boxes are true redactions (the pixels and any text under them are destroyed), placed in the page's unrotated space, so they land
correctly on a rotated page. A further pass can only ADD boxes. What is still reported after the last round is listed in the report,
not hidden. Calls go to OpenRouter with data collection off and stop at --visual-spend-cap dollars.

    python3 visual_verify.py --self-test          (no model calls)
"""
import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import common  # noqa: E402  (pipeline/common.py: the OpenRouter key, the no-training provider object, the usage log)

COLS, ROWS, FCOLS, LONG = "ABCDEFGHIJ", 14, "abcdefghijklmnop", 1560
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
FZ = threading.Lock()                       # PyMuPDF is not thread-safe
SPEND = {"usd": 0.0, "calls": 0}
LOCK = threading.Lock()
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()

INSURANCE_RULES = """INSURANCE REDACTION RULES (Indian health-insurance claim bundles; the
redacted copy goes to human annotators, so over-redaction is as bad as a leak).
MUST BE REMOVED if still readable:
- names of people: patient, relatives, nominee, proposer, any third person,
  doctors, nurses and other clinicians, signatories (printed or handwritten);
- numbers tied to a person or this case: Aadhaar/VID, PAN, voter ID, passport,
  driving licence, policy/member/card/claim numbers, patient IDs (MRN, UHID,
  IP/OP no.), bill/invoice/receipt/visit/report reference numbers, employee
  codes, doctor registration numbers, implant serials, the person's own bank
  account/card/cheque numbers and UPI IDs;
- the person's address (house, street, locality, village, PIN code), personal
  phone numbers and emails;
- date of birth, age, sex/gender, relationship;
- faces and photos of people, signatures, thumbprints, initials used as a
  sign-off; QR codes and barcodes;
- identity of the hospital, clinic, lab, pharmacy, insurer and TPA: their
  names (also inside a longer legal string), logos, letterhead, stamps and
  seals, web domain, email, phone numbers, postal address, GSTIN/CIN/licence/
  registration numbers; the insured person's employer name;
- on a personal bank statement: transaction narration and reference numbers.
Any script (English, Hindi, Assamese, Bengali, Kannada, Telugu, Tamil, ...)
and any orientation (sideways or upside-down text counts).
MUST BE KEPT (never report): clinical content (diagnoses, symptoms, tests and
values, units, ranges, medicines, doses, procedures, clinical notes); money
amounts, quantities, bill line items, service descriptions; episode dates
(admission, discharge, visit, sample, report, bill dates); the BANK's own
details (bank name, branch name and branch address, IFSC, MICR, the bank's
helpline, registered office); printed labels, headings, boilerplate, page
numbers, ward/bed/department, form designations; government issuer text and
sites ("Government of India", UIDAI, Income Tax Department, uidai.gov.in);
the person's city or state on its own (not the PIN or street).
Black boxes are existing redactions - fine. A fragment too partial to identify
anyone (1-2 stray letters at a box edge) is not a leak. A faint background
watermark lying under body text is not reported (cutting it would destroy the
page); a logo or name standing in clear space is."""


def rules_for(pack):
    """the insurance pack's rules as tested on claim files; any other pack's own categories, removed vs kept"""
    if pack is None or getattr(pack, "id", "") == "insurance_medical":
        return INSURANCE_RULES
    raw = getattr(pack, "_raw", {}) or {}
    rm = [f"- {c['name'].replace('_', ' ').lower()}: {c.get('definition', '')[:240]}" for c in raw.get("categories", [])
          if c.get("action") != "KEEP"]
    keep = [f"- {c['name'].replace('_', ' ').lower()}" for c in raw.get("categories", []) if c.get("action") == "KEEP"]
    return ("REDACTION RULES (the redacted copy goes to human readers, so over-redaction is as bad as a leak).\nMUST BE REMOVED if "
            "still readable:\n" + "\n".join(rm) + "\nMUST BE KEPT (never report):\n" + "\n".join(keep) +
            "\nBlack boxes are existing redactions - fine. A fragment too partial to identify anyone is not a leak.")


FIND = """

The image is one REDACTED page with a faint grid: columns A-J (letters on top),
rows 1-14 (numbers on the left). List every item that breaks the rules above
and is still readable. For each give the grid cells it touches (e.g. ["C4","D4"]).
Answer ONLY with JSON:
{"leaks": [{"what": "<short description, quote the text if readable>",
 "kind": "name|id|address|phone|email|dob_age_sex|face|signature|stamp|code|org_name_or_logo|other",
 "cells": ["C4"]}]}
An empty list means the page is clean."""
FINE = """

The image is a zoomed part of a redacted page with a fine grid: columns a-{lastcol} (letters on top), rows 1-{rows}
(numbers on the left). The leak to remove: {what} ({kind}).
List EVERY fine cell the leak touches, including faint strokes, loops and partial letters next to existing black
boxes, so that blacking out those cells removes it completely. Never list a cell whose only content is something the
rules say to keep. If the leak is not in this image, return an empty list. Answer ONLY with JSON: {{"cells": ["c4", "d4"]}}"""
VERIFY = """

The image is a zoomed part of a redacted page. Each numbered red box is about
to be blacked out to remove this leak: {what} ({kind}).
Look at what is under each box. List the numbers of boxes that would ALSO
destroy something the rules say to keep (clinical text or values, medicines,
amounts, episode dates, printed labels or headings, body text under a faint
watermark) or that do not touch the leak at all. Those boxes will be dropped.
Answer ONLY with JSON: {{"drop": [3, 5]}} (empty list if every box is right)."""


def _key():
    try:
        import common
        k = common.openrouter_key()
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY")


def _log_usage(row):
    try:
        import common
        log = common.OUTPUT_ROOT / "api_usage.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with LOCK, open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


def _b64(im):
    buf = io.BytesIO(); im.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()


class Asker:
    def __init__(self, model, cap):
        self.model, self.cap, self.key = model, float(cap), _key()
        if not self.key:
            raise SystemExit("--visual-verify needs OPENROUTER_API_KEY")

    def __call__(self, prompt, img, max_tokens=3000, default=None):
        """-> parsed JSON. Claim pages never train a model: provider data collection is off on every call."""
        if SPEND["usd"] > self.cap:
            raise RuntimeError(f"visual-verify spend cap ${self.cap:.0f} reached")
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "usage": {"include": True},
                "provider": common.openrouter_provider(),
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(img)}},
                    {"type": "text", "text": prompt}]}]}
        err, bad = None, 0
        for attempt in range(6):
            req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.loads(r.read())
            except Exception as e:
                err = e; time.sleep(8 * (attempt + 1)); continue
            cost = float((resp.get("usage") or {}).get("cost") or 0)
            with LOCK:
                SPEND["usd"] += cost; SPEND["calls"] += 1
            _log_usage({"engine": "visual_verify", "model": self.model, "usd": cost, "ts": time.time()})
            txt = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            try:
                return json.loads(m.group(0))
            except Exception as e:
                err = e; bad += 1
                if bad >= 3:
                    break
        if default is not None:
            return default
        raise RuntimeError(f"visual-verify call failed: {err}")


def grid_image(page):
    z = LONG / max(page.rect.width, page.rect.height)
    pm = page.get_pixmap(matrix=fitz.Matrix(z, z))
    pg = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    m = 26
    im = Image.new("RGB", (pm.width + m, pm.height + m), "white"); im.paste(pg, (m, m))
    dr = ImageDraw.Draw(im, "RGBA")
    cw, rh = pm.width / len(COLS), pm.height / ROWS
    for i, c in enumerate(COLS):
        x = m + i * cw; dr.line([(x, m), (x, m + pm.height)], fill=(0, 140, 255, 90), width=1)
        dr.text((x + cw / 2 - 5, 4), c, fill=(0, 90, 220), font=FONT)
    for j in range(ROWS):
        y = m + j * rh; dr.line([(m, y), (m + pm.width, y)], fill=(0, 140, 255, 90), width=1)
        dr.text((3, y + rh / 2 - 8), str(j + 1), fill=(0, 90, 220), font=FONT)
    return im


def cells_rect(page, cells, margin=1):
    """grid cells -> the displayed-page rectangle they span, one cell of margin"""
    W, H = page.rect.width, page.rect.height
    cw, rh = W / len(COLS), H / ROWS
    xs, ys = [], []
    for c in cells:
        mm = re.match(r"^\s*([A-J])\s*(\d{1,2})\s*$", str(c).upper())
        if mm and 1 <= int(mm.group(2)) <= ROWS:
            xs.append(COLS.index(mm.group(1))); ys.append(int(mm.group(2)) - 1)
    if not xs:
        return None
    return fitz.Rect(max(0, min(xs) - margin) * cw, max(0, min(ys) - margin) * rh,
                     min(len(COLS), max(xs) + 1 + margin) * cw, min(ROWS, max(ys) + 1 + margin) * rh)


def fine_image(page, clip):
    z = min(4.0, 1400 / max(clip.width, clip.height))
    pm = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip)
    rows = max(4, min(24, round(len(FCOLS) * clip.height / clip.width)))
    m = 24
    im = Image.new("RGB", (pm.width + m, pm.height + m), "white")
    im.paste(Image.frombytes("RGB", (pm.width, pm.height), pm.samples), (m, m))
    dr = ImageDraw.Draw(im, "RGBA")
    cw, rh = pm.width / len(FCOLS), pm.height / rows
    for i, c in enumerate(FCOLS):
        x = m + i * cw; dr.line([(x, m), (x, m + pm.height)], fill=(0, 140, 255, 110), width=1)
        dr.text((x + cw / 2 - 4, 3), c, fill=(0, 90, 220), font=FONT)
    for j in range(rows):
        y = m + j * rh; dr.line([(m, y), (m + pm.width, y)], fill=(0, 140, 255, 110), width=1)
        dr.text((2, y + rh / 2 - 8), str(j + 1), fill=(0, 90, 220), font=FONT)
    return im, rows


def pick_image(page, clip, boxes):
    z = min(4.0, 1400 / max(clip.width, clip.height))
    pm = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip)
    im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    dr = ImageDraw.Draw(im, "RGBA")
    for i, r in enumerate(boxes, 1):
        x0, y0, x1, y1 = [(r.x0 - clip.x0) * z, (r.y0 - clip.y0) * z, (r.x1 - clip.x0) * z, (r.y1 - clip.y0) * z]
        dr.rectangle([x0, y0, x1, y1], outline=(230, 0, 0, 255), width=2)
        tw = 9 * len(str(i)) + 4; tx, ty = max(0, x0), max(0, y0 - 17)
        dr.rectangle([tx, ty, tx + tw, ty + 17], fill=(255, 255, 255, 215))
        dr.text((tx + 2, ty), str(i), fill=(0, 0, 230), font=FONT)
    return im


def _ink(page, clip):
    dpi = 150; sc = 72.0 / dpi
    pm = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
    g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
    solid = ndi.binary_dilation(ndi.binary_opening(g < 20, np.ones((12, 12))), np.ones((3, 3)))   # existing black boxes
    paper = np.percentile(g[~solid], 90) if (~solid).any() else 255
    return (g < max(190, paper - 22)) & ~solid, sc                     # darker than this crop's own paper


def _grid(cells, rows):
    grid = np.zeros((rows, len(FCOLS)), bool)
    for c in cells:
        mm = re.match(r"^\s*([a-p])\s*(\d{1,2})\s*$", str(c).lower())
        if mm and 1 <= int(mm.group(2)) <= rows:
            grid[int(mm.group(2)) - 1, FCOLS.index(mm.group(1))] = True
    return grid


def runs_cut(page, clip, rows, cells):
    """chosen fine cells -> one box per connected run of cells, tightened to the ink inside it (displayed-page space)"""
    grid = _grid(cells, rows)
    if not grid.any():
        return []
    ink, sc = _ink(page, clip)
    cw, rh = clip.width / len(FCOLS), clip.height / rows
    lab, _ = ndi.label(grid, np.ones((3, 3)))
    out = []
    for sl in ndi.find_objects(lab):
        x0, x1 = int(sl[1].start * cw / sc), int(sl[1].stop * cw / sc) + 1
        y0, y1 = int(sl[0].start * rh / sc), int(sl[0].stop * rh / sc) + 1
        sub = ink[y0:y1, x0:x1]
        if sub.sum() < 4:
            continue
        ys, xs = np.nonzero(sub)
        out.append(fitz.Rect(clip.x0 + (x0 + xs.min()) * sc - 1.5, clip.y0 + (y0 + ys.min()) * sc - 1.5,
                             clip.x0 + (x0 + xs.max() + 1) * sc + 1.5, clip.y0 + (y0 + ys.max() + 1) * sc + 1.5) & clip)
    return out


def cells_cut(page, clip, rows, cells):
    """the fallback: one ink-tight box per chosen cell"""
    grid = _grid(cells, rows)
    ink, sc = _ink(page, clip)
    cw, rh = clip.width / len(FCOLS), clip.height / rows
    out = []
    for j, i in zip(*np.nonzero(grid)):
        x0, x1 = int(i * cw / sc), int((i + 1) * cw / sc) + 1
        y0, y1 = int(j * rh / sc), int((j + 1) * rh / sc) + 1
        cell = ink[y0:y1, x0:x1]
        if cell.sum() < 4:
            continue
        ys, xs = np.nonzero(cell)
        out.append(fitz.Rect(clip.x0 + (x0 + xs.min()) * sc, clip.y0 + (y0 + ys.min()) * sc,
                             clip.x0 + (x0 + xs.max() + 1) * sc, clip.y0 + (y0 + ys.max() + 1) * sc))
    return out


def apply_cuts(page, cuts):
    """boxes are in displayed (rotated) space; redaction annotations take the unrotated page's space"""
    for r in cuts:
        page.add_redact_annot(((fitz.Rect(r) + (-1.5, -1.5, 1.5, 1.5)) & page.rect) * page.derotation_matrix, fill=(0, 0, 0))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS, text=fitz.PDF_REDACT_TEXT_REMOVE)


def _verify(ask, rules, page, clip, chosen, leak, ent, key):
    if not chosen:
        return []
    with FZ:
        im = pick_image(page, clip, chosen)
    vr = ask(rules + VERIFY.format(what=leak.get("what", ""), kind=leak.get("kind", "")), im, max_tokens=1500,
             default={"drop": list(range(1, len(chosen) + 1))})          # no answer: drop them all (never cut blind)
    drop = {int(i) for i in (vr.get("drop") or []) if str(i).isdigit()}
    ent[key] = len([i for i in drop if 1 <= i <= len(chosen)])
    return [b for i, b in enumerate(chosen, 1) if i not in drop]


def fix_page(ask, rules, page, rounds):
    rep = {"page": page.number + 1, "rounds": []}
    for rnd in range(1, rounds + 2):
        with FZ:
            gim = grid_image(page)
        leaks = ask(rules + FIND, gim, default={"leaks": []}).get("leaks") or []
        rd = {"round": rnd, "leaks": [], "boxes": 0}; rep["rounds"].append(rd)
        if not leaks or rnd == rounds + 1:                                  # the last round only looks
            rd["leaks"] = [{k: l.get(k) for k in ("what", "kind", "cells")} for l in leaks]
            break
        cut = []
        for l in leaks:
            ent = {k: l.get(k) for k in ("what", "kind", "cells")}
            clip = cells_rect(page, l.get("cells") or [])
            if clip is None:
                rd["leaks"].append(ent); continue
            with FZ:
                fim, rows = fine_image(page, clip)
            cells = ask(rules + FINE.format(lastcol=FCOLS[-1], rows=rows, what=l.get("what", ""), kind=l.get("kind", "")), fim,
                        max_tokens=1500, default={"cells": []}).get("cells") or []
            with FZ:
                big = runs_cut(page, clip, rows, cells)
            chosen = _verify(ask, rules, page, clip, big, l, ent, "runs_dropped")
            if len(chosen) < len(big):                                       # a run hit kept content: per-cell boxes instead
                with FZ:
                    small = cells_cut(page, clip, rows, cells)
                small = [b for b in small if not any(abs(b & k) > 0.5 * abs(b) for k in chosen)]
                chosen += _verify(ask, rules, page, clip, small, l, ent, "cells_dropped")
            ent["boxes"] = len(chosen); cut += chosen; rd["leaks"].append(ent)
        rd["boxes"] = len(cut)
        if not cut:
            break
        rd["cuts"] = [[round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)] for r in cut]
        with FZ:
            apply_cuts(page, cut)
    rep["boxes"] = sum(r["boxes"] for r in rep["rounds"])
    rep["still_seen"] = [l for l in rep["rounds"][-1]["leaks"] if not l.get("boxes")] if rep["rounds"] else []
    return rep


def run(pdf, pack=None, report=None, model="anthropic/claude-opus-5.5", rounds=2, workers=8, cap=100.0, pages=None):
    """the pass over every page of a finished output, in place -> stats"""
    ask = Asker(model, cap); rules = rules_for(pack)
    doc = fitz.open(pdf)
    todo = [p for p in range(1, doc.page_count + 1) if not pages or p in pages]
    reps, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        fut = {ex.submit(fix_page, ask, rules, doc[p - 1], rounds): p for p in todo}
        for f in as_completed(fut):
            p = fut[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"page": p, "error": str(e)[:300], "rounds": [], "boxes": 0, "still_seen": []}
            reps.append(r)
    tmp = Path(str(pdf) + ".vv.pdf")
    doc.save(tmp, garbage=4, deflate=True); doc.close(); tmp.replace(pdf)
    reps.sort(key=lambda r: r["page"])
    if report:
        Path(report).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reps))
    st = {"pages": len(reps), "pages_cut": sum(1 for r in reps if r["boxes"]), "boxes": sum(r["boxes"] for r in reps),
          "pages_still_seen": [r["page"] for r in reps if r["still_seen"]], "errors": [r["page"] for r in reps if r.get("error")],
          "usd": round(SPEND["usd"], 2), "calls": SPEND["calls"], "minutes": round((time.time() - t0) / 60, 1)}
    return st


def self_test():
    """grid geometry, connected runs and the rotated-page placement, with a scripted model (no calls)"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "a.pdf"
        d = fitz.open(); p = d.new_page()
        p.insert_text((80, 400), "Signed: Meera Iyer", fontsize=14)
        p.insert_text((80, 600), "Haemoglobin 11.2 g/dL", fontsize=12)
        d.save(src)
        doc = fitz.open(src); page = doc[0]
        clip = cells_rect(page, ["B8", "C8", "D8"])
        assert clip.contains(page.search_for("Meera Iyer")[0]), "the grid cells span the leak"
        fim, rows = fine_image(page, clip)
        cw, rh = clip.width / len(FCOLS), clip.height / rows
        hit = page.search_for("Meera Iyer")[0]
        cells = [f"{FCOLS[i]}{j + 1}" for i in range(len(FCOLS)) for j in range(rows)
                 if fitz.Rect(clip.x0 + i * cw, clip.y0 + j * rh, clip.x0 + (i + 1) * cw, clip.y0 + (j + 1) * rh).intersects(hit)]
        boxes = runs_cut(page, clip, rows, cells)
        assert len(boxes) == 1 and boxes[0].intersects(hit), "connected cells make one box over the ink"
        apply_cuts(page, boxes)
        assert "Meera" not in page.get_text() and "Haemoglobin" in page.get_text(), "the leak is gone, kept text stays"
        q = fitz.open(); r = q.new_page(); r.set_rotation(90)                 # rotated page: the box lands on the text
        r.insert_text((100, 100), "Arjun Rao", fontsize=14)
        hit = [w for w in r.get_text("words") if w[4] == "Arjun"][0]
        disp = fitz.Rect(hit[:4]) * r.rotation_matrix
        apply_cuts(r, [disp])
        assert "Arjun" not in r.get_text(), "a cut in displayed space removes the text on a rotated page"
        print("visual_verify self-test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
