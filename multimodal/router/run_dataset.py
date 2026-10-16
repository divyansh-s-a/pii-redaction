#!/usr/bin/env python3
"""One run over a whole dataset — redact or replace, every modality.

    python3 multimodal/router/run_dataset.py INPUT --mode redact|replace \
        --taxonomy market_research --out DELIVERY --work WORK
        [--stop-after detect|mapping] [--mapping FILE] [--context TEXT]
        [--skip-media] [--no-sweep] [--no-assurance] [--workers 24]

INPUT is a folder (or one file). DELIVERY receives only redacted files, in a
mirror of the input tree with folder and file names cleaned. WORK holds the
ledgers, the mapping, the code book, caches and reports — they contain real
names and values: never share WORK, never put it in a repository.

Stage 0  inventory   every file gets a route by its type: audio, video,
                     office (docx/pptx/xlsx/xlsm; doc/rtf converted to docx),
                     text and survey (csv, txt, qdf, qtf, stpn, qpsmran, asc),
                     pdf, image, zip (members routed, re-zipped). Anything
                     unreadable is HELD and listed, never copied through.
Stage 1  detect      the taxonomy pack decides what counts. Documents: model
                     pass over each distinct text piece once (cached across
                     the dataset), patterns (email, phone), survey column
                     rules; images classified. Audio/video: the audio arm's
                     detection with the pack's audio profile, plus entity
                     discovery over the transcript (and on-screen text).
                     File and folder names are detected like text. Every
                     hit goes into one ledger per file.
Stage 2  mapping     Replace mode only: ONE mapping for the whole dataset,
                     built from every ledger before anything is written —
                     spellings grouped into entities, invented names for
                     people and brands (prompts/dataset_naming_v1.txt),
                     checked by entity_map. --stop-after mapping pauses here;
                     a later run with --mapping FILE renders without
                     detecting again.
Stage 3  render      the dataset net (every removable form found in ANY file
                     is applied to EVERY file) + each file's own spans.
                     Redact: black boxes / beeps / blur. Replace: invented
                     names (spoken in audio), codes for linking ids
                     (R0001, INT001), contact data, addresses and other ids
                     removed.
Stage 4  verify      every output is read back (every XML part, the PDF text
                     layer, OCR of images and scans) and must hold no
                     removable form from the ledgers, no email or phone, and
                     must open; a broken control per route (the original)
                     must fail the same check. A failing file is HELD.
Stage 5  report      WORK/report.json and WORK/report.md.
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
for p in (ROOT / "pipeline", PKG / "document_pipeline", PKG / "taxonomies",
          ROOT / "tools"):
    sys.path.insert(0, str(p))

AUDIO_EXT = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "opus", "wma"}
VIDEO_EXT = {"mp4", "mov", "m4v", "avi", "mkv", "webm"}
OOXML_EXT = {"docx", "docm", "pptx", "pptm", "xlsx", "xlsm"}
CONVERT_EXT = {"doc", "rtf"}
TEXT_EXT = {"csv", "tsv", "txt", "asc", "qdf", "qtf", "stpn", "qpsmran",
            "md", "json", "xml", "log"}
IMAGE_EXT = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

PERSON_CATS = {"FULL_NAME", "NAME_IN_FREE_TEXT"}
ENTITY_CATS = {"BRAND": "BRAND", "PRODUCT_OR_MODEL": "PRODUCT",
               "DEALER_OR_OUTLET": "DEALER",
               "RESEARCH_OR_AD_AGENCY": "AGENCY",
               "BRAND_IDENTIFYING_DETAIL": "SLOGAN",
               "ISSUING_ORGANISATION": "ORGANISATION"}
TAGS = {"FULL_NAME": "NAME", "NAME_IN_FREE_TEXT": "NAME", "BRAND": "BRAND",
        "PRODUCT_OR_MODEL": "PRODUCT", "DEALER_OR_OUTLET": "DEALER",
        "RESEARCH_OR_AD_AGENCY": "AGENCY", "ISSUING_ORGANISATION": "ORG",
        "BRAND_IDENTIFYING_DETAIL": "BRAND", "CONTACT_CHANNEL": "CONTACT",
        "ADDRESS": "ADDRESS", "ASSIGNED_ID": "ID", "RECORD_LOCATOR_ID": "ID"}
NAMING = (ROOT / "prompts" / "dataset_naming_v1.txt").read_text()

LOCK = threading.Lock()
HONORIFICS = {"shri", "shree", "smt", "kumari", "miss", "mister", "madam",
              "sahab", "saheb", "ji", "doctor", "prof", "professor", "team",
              "respondent", "moderator", "interviewer", "client", "manager",
              # account and software words that sit in author fields
              "microsoft", "office", "user", "users", "admin", "administrator",
              "windows", "owner", "guest", "google", "apple", "default"}
CALL_TIMEOUT = 360         # seconds per model call before it is retried
MAX_PIECES = 120           # text pieces per detection call


def route_of(path: Path):
    e = path.suffix.lower().lstrip(".")
    if e in AUDIO_EXT:
        return "audio"
    if e in VIDEO_EXT:
        return "video"
    if e in OOXML_EXT or e in CONVERT_EXT:
        return "office"
    if e in TEXT_EXT:
        return "text"
    if e == "pdf":
        return "pdf"
    if e in IMAGE_EXT:
        return "image"
    if e == "zip":
        return "zip"
    return "held"


def fid_of(rel):
    return hashlib.sha1(rel.encode()).hexdigest()[:16]


def jdump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)


def jload(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


# =============================================================== stage 0

def inventory(src: Path, work: Path):
    items = []
    if src.is_file():
        files, base = [src], src.parent
    else:
        base = src
        files = sorted(f for f in src.rglob("*") if f.is_file() and not any(
            part.startswith(".") for part in f.relative_to(src).parts))
    for f in files:
        rel = f.relative_to(base).as_posix()
        study = rel.split("/")[0] if "/" in rel else ""
        it = {"id": fid_of(rel), "rel": rel, "path": str(f),
              "route": route_of(f), "study": study}
        items.append(it)
        if it["route"] == "zip":
            dst = work / "expanded" / it["id"]
            try:
                with zipfile.ZipFile(f) as z:
                    z.extractall(dst)
                    names = [n for n in z.namelist() if not n.endswith("/")]
            except zipfile.BadZipFile:
                it["route"] = "held"
                it["held_reason"] = "unreadable zip"
                continue
            it["members"] = []
            for n in names:
                mp = dst / n
                mrel = f"{rel}!/{n}"
                sub = {"id": fid_of(mrel), "rel": mrel, "path": str(mp),
                       "route": route_of(mp), "study": study,
                       "container": it["id"], "member": n}
                it["members"].append(sub["id"])
                items.append(sub)
    for it in items:
        if it["route"] == "held" and "held_reason" not in it:
            it["held_reason"] = "no reader for this file type"
    return items


# =============================================================== detection

FRAME = """

THE MATERIAL BELOW is text from ONE file of a market-research dataset
({label}), as numbered words; an empty line separates its pieces (paragraphs,
table or sheet cells, slide text, notes, comments, chart labels, properties,
links, data-file lines, or file and folder names). Return each span as
s = its first word number, e = its last word number (inclusive) and c = its
categories (the category NAMES above). Nothing else. Never return a span that
crosses a piece boundary. A value repeated in many pieces is returned every
time it occurs.

WORDS:
"""


# lean answer format: spans only (a reason per span doubled the output of
# label-heavy files and the latency with it)
LEAN_SCHEMA = {"type": "OBJECT", "properties": {"spans": {
    "type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "s": {"type": "INTEGER"}, "e": {"type": "INTEGER"},
        "c": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["s", "e", "c"]}}}, "required": ["spans"]}


class TextDetector:
    """The model pass over text pieces. A piece's text is detected ONCE
    per dataset (cache by content); pieces are batched in file order so the
    model sees their neighbours."""

    def __init__(self, pack, model, cache_file: Path, workers=12,
                 chunk_words=1200):
        self.pack, self.model, self.workers = pack, model, workers
        self.chunk = chunk_words
        self.names = set(pack.categories)
        self.cache_file = cache_file
        self.cache = {}
        if cache_file.exists():
            for ln in cache_file.read_text().splitlines():
                try:
                    h, res = json.loads(ln)
                    self.cache[h] = res
                except ValueError:
                    pass
        self.failed = set()
        self.calls = 0
        self.thinking = None

    @staticmethod
    def key(text):
        return hashlib.sha1(text.encode("utf-8", "surrogateescape")
                            ).hexdigest()

    @staticmethod
    def wanted(seg):
        t = seg["text"]
        return len(t.strip()) >= 2 and any(ch.isalpha() for ch in t)

    def run(self, jobs):
        """jobs: [(key, segs, label)] -> {key: (spans, incomplete)}"""
        pending, batches = set(), []
        for _key, segs, label in jobs:
            cur, n = [], 0
            for seg in segs:
                if not self.wanted(seg):
                    continue
                h = self.key(seg["text"])
                if h in self.cache or h in pending:
                    continue
                pending.add(h)
                w = len(seg["text"].split())
                # bounded by words AND pieces: hundreds of one-word labels in
                # one call make an answer too long to finish in time
                if cur and (n + w > self.chunk or len(cur) >= MAX_PIECES):
                    batches.append((label, cur))
                    cur, n = [], 0
                cur.append((h, seg["text"]))
                n += w
            if cur:
                batches.append((label, cur))
        if batches:
            print(f"  model pass: {len(batches)} batch(es), "
                  f"{len(pending)} distinct piece(s)", flush=True)
            done = [0]

            def one(b):
                self._batch(b)
                with LOCK:
                    done[0] += 1
                    if done[0] % 20 == 0:
                        print(f"    {done[0]}/{len(batches)} batches",
                              flush=True)
            with cf.ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(one, batches))
        out = {}
        for key, segs, _label in jobs:
            spans, incomplete = [], False
            for si, seg in enumerate(segs):
                if not self.wanted(seg):
                    continue
                res = self.cache.get(self.key(seg["text"]))
                if res is None:
                    incomplete = True
                    continue
                for a, b, cats in res:
                    spans.append({"seg": si, "a": a, "b": b,
                                  "text": seg["text"][a:b],
                                  "categories": cats, "source": "model"})
            out[key] = (spans, incomplete)
        return out

    def _batch(self, batch):
        import common
        import policy
        label, items = batch
        words, lines = [], []
        for k, (_h, text) in enumerate(items):
            if lines:
                lines.append("")
            for m in re.finditer(r"\S+", text):
                words.append((k, m.start(), m.end()))
                lines.append(f"{len(words) - 1}\t{m.group(0)}")
        prompt = self.pack.prompt + FRAME.format(label=label) + \
            "\n".join(lines)
        raw = None
        for attempt in range(4):
            # a call that hangs is abandoned after CALL_TIMEOUT and retried
            # (the HTTP layer's own timeout is 20 minutes)
            ex = cf.ThreadPoolExecutor(1)
            fut = ex.submit(common.generate, self.model, [{"text": prompt}],
                            LEAN_SCHEMA, 0.0, 65536, self.thinking)
            try:
                raw = json.loads(fut.result(timeout=CALL_TIMEOUT))
                ex.shutdown(wait=False)
                break
            except Exception as exc:          # noqa: BLE001
                ex.shutdown(wait=False)
                err = exc if str(exc) else TimeoutError("call timed out")
                time.sleep(3 * (attempt + 1))
        with LOCK:
            self.calls += 1
        if raw is None:
            print(f"    !! batch failed ({str(err)[:100]})", flush=True)
            with LOCK:
                self.failed |= {h for h, _t in items}
            return
        res = {k: [] for k in range(len(items))}
        for s in raw.get("spans", []):
            a, b = s.get("s"), s.get("e")
            if not isinstance(a, int) or not isinstance(b, int) or \
                    a < 0 or b >= len(words) or b < a:
                continue
            if words[a][0] != words[b][0]:
                continue
            cats = [c for c in s.get("c", []) if c in self.names]
            if cats:
                res[words[a][0]].append([words[a][1], words[b][2], cats])
        with LOCK:
            with open(self.cache_file, "a") as f:
                for k, (h, _t) in enumerate(items):
                    self.cache[h] = res[k]
                    f.write(json.dumps([h, res[k]], ensure_ascii=False) + "\n")


# =============================================================== readers

class Ctx:
    def __init__(self, a):
        import base
        self.a = a
        self.mode = a.mode
        self.src = Path(a.input).resolve()
        self.out = Path(a.out).resolve()
        self.work = Path(a.work).resolve()
        self.pack = base.load(PKG / "taxonomies" / a.taxonomy)
        self.model = a.model
        self.img_model = a.image_model
        self.img_cache_file = self.work / "image_cache.json"
        self.img_cache = jload(self.img_cache_file, {}) or {}
        self.detector = TextDetector(self.pack, self.model,
                                     self.work / "text_cache.jsonl",
                                     a.workers)
        self.detector.thinking = a.thinking or None
        self.em = None
        self.codes = {}
        self.keep_forms = set()
        self.only_ids = None

    def classify(self, data, ext):
        import office
        return office.classify_image(data, ext, self.img_cache,
                                     self.img_model)

    def save_images(self):
        with LOCK:
            jdump(self.img_cache_file, self.img_cache)


def doc_blob(ctx, it):
    """Bytes the office route edits (a .doc/.rtf is converted to .docx)."""
    import office
    p = Path(it["path"])
    if p.suffix.lower().lstrip(".") in CONVERT_EXT:
        conv = ctx.work / "converted" / (it["id"] + ".docx")
        if not conv.exists():
            conv.parent.mkdir(parents=True, exist_ok=True)
            tmp = office.convert_to_docx(p, conv.parent)
            tmp.replace(conv)
        return conv.read_bytes(), "docx"
    return p.read_bytes(), p.suffix.lower().lstrip(".")


def read_item(ctx, it):
    """-> list of (subkey, segs, rule_spans, extra) for detection. An office
    file adds one entry per embedded office file (subkey = member path)."""
    import office
    r = it["route"]
    if r == "office":
        blob, _ext = doc_blob(ctx, it)
        out = []

        def walk(b, sub):
            parts, segs, media, embedded, rules = office.read_ooxml(b)
            out.append((sub, segs, rules, {"media": media}))
            with zipfile.ZipFile(__import__("io").BytesIO(b)) as z:
                for n in embedded:
                    walk(z.read(n), (sub + "::" if sub else "") + n)
        walk(blob, "")
        return out
    if r == "text":
        ext = Path(it["path"]).suffix.lower().lstrip(".")
        _t, _e, _b, segs, rules = office.read_text(
            Path(it["path"]).read_bytes(), ext)
        return [("", segs, rules, {})]
    if r == "pdf":
        import pdfimg
        segs, info = pdfimg.read_pdf(it["path"])
        return [("", segs, [], {"info": info})]
    if r == "image":
        import pdfimg
        segs, info = pdfimg.read_image(it["path"])
        return [("", segs, [], {"info": info})]
    return []


def classify_media(ctx, blob, media):
    """Pre-classify an office file's images (parallel, cached)."""
    import io
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in media:
            ext = n.rsplit(".", 1)[-1].lower()
            if n.startswith("docProps/thumbnail") or ext in ("wmf", "emf"):
                continue
            ctx.classify(z.read(n), ext)


# =============================================================== stage 1

def stage_detect(ctx, items):
    import office
    ledir = ctx.work / "ledger"
    docs = [] if ctx.a.media_only else [
        it for it in items if it["route"] in ("office", "text", "pdf",
                                              "image")
        and (ctx.only_ids is None or it["id"] in ctx.only_ids)]
    jobs, meta = [], {}

    def prep(it):
        lf = ledir / f"{it['id']}.json"
        if lf.exists() and not ctx.a.redetect:
            return it, None
        try:
            return it, read_item(ctx, it)
        except Exception as exc:              # noqa: BLE001
            return it, exc
    print(f"\n[stage 1] reading {len(docs)} document(s)", flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        read = list(ex.map(prep, docs))
    for it, res in read:
        if res is None:
            continue
        if isinstance(res, Exception):
            it["route"], it["held_reason"] = "held", \
                f"could not read: {str(res)[:120]}"
            print(f"  !! held {it['rel']}: {str(res)[:100]}")
            continue
        meta[it["id"]] = res
        for sub, segs, _rules, _x in res:
            jobs.append((f"{it['id']}|{sub}", segs,
                         f"file: {Path(it['rel']).name}"
                         + (f", embedded {sub}" if sub else "")))
    # file and folder names, detected once as one job
    comps = sorted({c for it in items for c in _name_parts(it["rel"])})
    name_segs = [{"part": "names", "kind": "name", "pieces": [],
                  "text": _name_text(c)} for c in comps]
    jobs.append(("__names__", name_segs, "file and folder names"))
    res = ctx.detector.run(jobs)
    # images inside office files (parallel; cached)
    img_jobs = []
    for it, r in read:
        if isinstance(r, list) and it["route"] == "office":
            blob, _e = doc_blob(ctx, it)
            img_jobs.append((blob, r[0][3].get("media", [])))
    if img_jobs:
        print(f"  classifying images of {len(img_jobs)} office file(s)",
              flush=True)
        with cf.ThreadPoolExecutor(ctx.a.workers) as ex:
            list(ex.map(lambda j: classify_media(ctx, *j), img_jobs))
        ctx.save_images()
    for iid, entries in meta.items():
        led = {"id": iid, "subs": {}}
        for sub, segs, rules, extra in entries:
            spans, incomplete = res.get(f"{iid}|{sub}", ([], False))
            spans = office._dedupe(spans + rules + office.deterministic(segs))
            rec = {"n_segs": len(segs), "spans": spans,
                   "incomplete": incomplete}
            if "info" in extra:
                rec["segs"] = segs
                rec["info"] = extra["info"]
            led["subs"][sub] = rec
        jdump(ledir / f"{iid}.json", led)
    name_spans, inc = res["__names__"]
    jdump(ledir / "__names__.json",
          {"segs": [s["text"] for s in name_segs], "spans": name_spans,
           "incomplete": inc})
    # media
    media = [it for it in items if it["route"] in ("audio", "video")]
    if media and not ctx.a.skip_media:
        todo = [it for it in media if ctx.a.redetect
                or not (ledir / f"{it['id']}.json").exists()]

        def det(it):
            try:
                jdump(ledir / f"{it['id']}.json", media_detect(ctx, it))
            except Exception as exc:          # noqa: BLE001
                it["route"], it["held_reason"] = "held", \
                    f"media detection failed: {str(exc)[:160]}"
        # each file is its own process (the audio / video arm): parallel
        with cf.ThreadPoolExecutor(ctx.a.media_workers) as ex:
            list(ex.map(det, todo))
    elif media:
        for it in media:
            it["route"], it["held_reason"] = "held", "skipped (--skip-media)"
    print(f"  model calls this run: {ctx.detector.calls}; failed pieces: "
          f"{len(ctx.detector.failed)}", flush=True)


def _name_text(comp):
    """A path component as detection reads it: '_' and '.' between words
    are breaks (same length, so offsets map back to the real name)."""
    return re.sub(r"[_.]", " ", comp)


def _name_parts(rel):
    parts = rel.replace("!/", "/").split("/")
    return [Path(p).stem if i == len(parts) - 1 else p
            for i, p in enumerate(parts)]


# =============================================================== media

def audio_profile(ctx):
    p = PKG / "taxonomies" / ctx.pack.id / "audio_profile.yaml"
    return p if p.exists() else None


def media_env(ctx, mapping=None, replace="beep"):
    env = dict(os.environ)
    env["PII_WORK_DIR"] = str(ctx.work / "media_cache")
    prof = audio_profile(ctx)
    if prof:
        env["PII_PROFILE_FILE"] = str(prof)
    if mapping:
        env["PII_MAPPING_FILE"] = str(mapping)
    else:
        env.pop("PII_MAPPING_FILE", None)
    env["PII_REPLACE_MODE"] = replace
    env["PYTHONUNBUFFERED"] = "1"
    return env


def media_cmd(ctx, it, outdir, mapping=None, replace="beep", audit=True,
              sweep=None):
    tool = (ROOT / "tools" / "redact.py" if it["route"] == "audio"
            else PKG / "video_pipeline" / "run_video.py")
    cmd = [sys.executable, str(tool), it["path"], "--outdir", str(outdir),
           "--quarantine-dir", str(outdir.parent / (outdir.name + "_held"))]
    prof = audio_profile(ctx)
    if prof:
        # the audio arm takes the profile's name from the file itself
        cmd += ["--profile-file", str(prof)]
        if it["route"] == "video":
            import profiles
            cmd += ["--profile", profiles.load(prof)["name"]]
    if mapping:
        cmd += ["--mapping", str(mapping)]
    if replace == "pseudonym":
        cmd += ["--replace", "pseudonym"]
    if not audit:
        cmd += ["--no-audit"]
    # dataset defaults, as run_all.py: category sweep and the second
    # listener (Gemini) unioned with the primary — the strongest recall
    # configuration; --no-sweep / --no-assurance turn them off
    sweep = ctx.a.sweep if sweep is None else sweep
    arm = (["--sweep"] if sweep else []) + \
        (["--assurance"] if ctx.a.assurance else [])
    if arm:
        cmd += arm if it["route"] == "audio" else ["--"] + arm
    return cmd


def media_detect(ctx, it):
    """Audio arm detection (beep render, no audit) -> intervals; entity
    discovery over the transcript (and on-screen text) for brands."""
    import build_mapping
    import common
    out = ctx.work / "media_stage1"
    log = ctx.work / "logs" / f"stage1_{it['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [media] detecting {it['rel']}", flush=True)
    with open(log, "w") as lf:
        # stage 1 finds who and what is named (one detection pass); the
        # render detects again with the mapping and the full sweep
        r = subprocess.run(media_cmd(ctx, it, out, audit=False, sweep=False),
                           env=media_env(ctx), stdout=lf, stderr=lf,
                           cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"audio arm exited {r.returncode} (see {log})")
    cache = ctx.work / "media_cache"
    import run_pipeline
    path = Path(it["path"])
    if it["route"] == "video":
        fid = run_pipeline.file_id_for(path)
        wav = cache / "video" / fid / f"{fid}.wav"
        afid = run_pipeline.file_id_for(wav)
    else:
        afid = run_pipeline.file_id_for(path)
    d = cache / f"{afid}__elevenlabs"
    if not (d / "intervals.json").exists():
        d = cache / afid
    ivs = jload(d / "intervals.json", {}).get("intervals", [])
    texts = [build_mapping.transcript_text(d / "transcript.json")]
    if (d / "assurance" / "transcript.json").exists():
        texts.append(build_mapping.transcript_text(
            d / "assurance" / "transcript.json"))
    if it["route"] == "video":
        try:
            work = cache / "video" / run_pipeline.file_id_for(path)
            scr = build_mapping.onscreen_text(path, work)
            if scr:
                texts.append("ON-SCREEN TEXT:\n" + scr)
        except Exception:                     # noqa: BLE001
            pass
    found = build_mapping.discover(ctx.model, texts, ctx.a.context)
    corpus = "\n".join(texts).lower()
    ents = []
    for e in found:
        forms = [f for f in (e.get("forms") or []) + [e.get("name", "")]
                 if f and f.strip() and f.strip().lower() in corpus]
        if forms:
            ents.append({"name": e.get("name"), "type": e.get("type"),
                         "forms": sorted(set(forms)),
                         "mentions": int(e.get("mentions") or 1)})
    people = [{"text": iv.get("spoken_text", ""), "categories":
               iv["categories"]} for iv in ivs
              if "FULL_NAME" in iv["categories"]]
    return {"id": it["id"], "media": True, "audio_dir": str(d),
            "people": people, "entities": ents}


def audio_dir(ctx, it):
    """The audio arm's work folder for a media item (video: its track)."""
    import run_pipeline
    cache = ctx.work / "media_cache"
    path = Path(it["path"])
    if it["route"] == "video":
        fid = run_pipeline.file_id_for(path)
        afid = run_pipeline.file_id_for(cache / "video" / fid / f"{fid}.wav")
        d = cache / "video" / fid / "audio_deliveries"
        cand = [cache / f"{afid}__elevenlabs", cache / afid]
    else:
        afid = run_pipeline.file_id_for(path)
        cand = [cache / f"{afid}__elevenlabs", cache / afid]
    return next((c for c in cand if (c / "intervals.json").exists()), cand[0])


def listen_remap(ctx, it, mapping):
    """A file held ONLY because the listen check heard a real name where an
    invented one was spoken (an invented name that sounds like a real one):
    for this file those invented names are beeped instead of spoken."""
    import yaml
    d = audio_dir(ctx, it)
    ma = jload(d / "mapped_audit" / "mapped_audit.json", {}) or {}
    listen = ma.get("listen") or {}
    hits = listen.get("hits") or []
    if not hits or ma.get("transcript_hits") or \
            (ma.get("judge") or {}).get("confirmed"):
        return None
    ivs = (jload(d / "intervals.json", {}) or {}).get("intervals", [])
    ids = {iv["mapping_id"] for iv in ivs
           if iv.get("render") == "voice" and iv.get("mapping_id")
           and any(not (iv["end_ms"] <= h["start_ms"]
                        or h["end_ms"] <= iv["start_ms"]) for h in hits)}
    if not ids:
        return None
    m = yaml.safe_load(Path(mapping).read_text()) or {}
    changed = 0
    for sec, items in m.items():
        if isinstance(items, list):
            for i, e in enumerate(items):
                if isinstance(e, dict) and f"{sec}:{i}" in ids:
                    e["render"] = "beep"
                    changed += 1
    if not changed:
        return None
    out = ctx.work / "media_remap" / f"{it['id']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False,
                                  width=1000))
    return out


def media_render(ctx, it, mapping, _retry=True):
    out = ctx.work / "media_out" / it["id"]
    log = ctx.work / "logs" / (f"render_{it['id']}.log" if _retry
                               else f"render_{it['id']}_remap.log")
    replace = "pseudonym" if ctx.mode == "replace" else "beep"
    with open(log, "w") as lf:
        r = subprocess.run(media_cmd(ctx, it, out, mapping, replace),
                           env=media_env(ctx, mapping, replace), stdout=lf,
                           stderr=lf, cwd=str(ROOT))
    held = out.parent / (out.name + "_held")
    if r.returncode != 0:
        return None, f"media render exited {r.returncode} (see {log})"
    bundles = [d for d in out.iterdir() if d.is_dir()] if out.exists() else []
    if not bundles:
        hb = [d for d in held.iterdir()] if held.exists() else []
        why = ""
        for d in hb:
            m = jload(d / "manifest.json", {}) or {}
            why = m.get("quarantined_reason") or why
        if _retry and ctx.mode == "replace" and "mapped-entity audit" in why:
            remap = listen_remap(ctx, it, mapping)
            if remap is not None:
                # automated fix: render again with those names beeped
                shutil.rmtree(out, ignore_errors=True)
                shutil.rmtree(held, ignore_errors=True)
                bundle, why2 = media_render(ctx, it, remap, _retry=False)
                if bundle is not None:
                    return bundle, None
                why = why2 or why
        return None, f"held by the media arm: {why or 'see ' + str(log)}"
    return bundles[0], None


# =============================================================== stage 2

def collect_entities(ctx, items):
    """Entities from every ledger: forms, type, counts, contexts."""
    import entity_map
    ledir = ctx.work / "ledger"
    forms = {}            # normalised form -> record

    def add(text, etype, ctxt="", study=""):
        t = re.sub(r"\s+", " ", str(text)).strip(" \t.,;:!?\"'()[]{}<>-–—")
        if len(t) < 2 or not any(ch.isalpha() for ch in t) or len(t) > 80:
            return
        k = t.lower()
        r = forms.setdefault(k, {"form": t, "type": etype, "n": 0,
                                 "ctx": []})
        r["n"] += 1
        if etype == "person" and r["type"] != "person":
            r["type"] = "person"
        if ctxt and len(r["ctx"]) < 2:
            r["ctx"].append(ctxt[:140])
        if study:
            r.setdefault("studies", set()).add(study)
    for it in items:
        led = jload(ledir / f"{it['id']}.json")
        if not led:
            continue
        if led.get("media"):
            for p in led["people"]:
                add(p["text"], "person")
            for e in led["entities"]:
                et = (e.get("type") or "brand").lower()
                if et in ("staff", "endorser"):
                    et = "person"
                for f in e["forms"]:
                    add(f, et)
            continue
        for sub, rec in led["subs"].items():
            for s in rec["spans"]:
                cats = set(s["categories"])
                if cats & PERSON_CATS:
                    add(s["text"], "person", study=it["study"])
                else:
                    for c in cats:
                        if c in ENTITY_CATS:
                            add(s["text"], ENTITY_CATS[c].lower(),
                                study=it["study"])
                            break
    nl = jload(ledir / "__names__.json", {})
    for s in nl.get("spans", []):
        cats = set(s["categories"])
        if cats & PERSON_CATS:
            add(s["text"], "person")
        else:
            for c in cats:
                if c in ENTITY_CATS:
                    add(s["text"], ENTITY_CATS[c].lower())
    # group spellings: same phonetic key -> one entity
    groups = {}
    for k, r in forms.items():
        g = entity_map.key(r["form"]) or k
        if r["type"] == "person":
            g = "P:" + g
        groups.setdefault(g, []).append(r)
    ents = []
    for g, rs in groups.items():
        rs.sort(key=lambda r: -r["n"])
        ents.append({"name": rs[0]["form"], "type": rs[0]["type"],
                     "forms": sorted({r["form"] for r in rs}),
                     "mentions": sum(r["n"] for r in rs),
                     "ctx": [c for r in rs for c in r["ctx"]][:2],
                     "studies": sorted({st for r in rs
                                        for st in r.get("studies", ())})[:4]})
    ents.sort(key=lambda e: (e["type"] != "person", e["name"].lower()))
    return ents


NAME_SCHEMA = {
    "type": "OBJECT",
    "properties": {"decisions": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"id": {"type": "STRING"}, "core": {"type": "STRING"},
                       "invented": {"type": "STRING"},
                       "category": {"type": "STRING"},
                       "keep": {"type": "BOOLEAN"}},
        "required": ["id", "invented", "category", "keep"]}}},
    "required": ["decisions"]}


def name_batch(ctx, batch, avoid=()):
    import common
    listing = "\n".join(
        f"E{k}\t{e['type']}\t{e['name']}\tforms: "
        + "; ".join(e["forms"][:6])
        + (f"\tcontext: {' | '.join(e['ctx'])}" if e["ctx"] else "")
        + (f"\tstudy: {'; '.join(e.get('studies') or [])}"
           if e.get("studies") else "")
        for k, e in enumerate(batch, 1))
    extra = ("\n\nDo NOT use these invented names (already taken): "
             + "; ".join(sorted(avoid)[:400])) if avoid else ""
    for attempt in range(3):
        try:
            raw = json.loads(common.generate(ctx.model, [{"text": (
                NAMING + "\n\nSTUDY CONTEXT: " + (ctx.a.context or "(none)")
                + "\n\nENTITIES (id, type, name, forms, context):\n" + listing
                + extra)}], NAME_SCHEMA))
            break
        except Exception:                     # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError("naming call failed")
    out = {}
    for d in raw.get("decisions", []):
        m = re.fullmatch(r"E(\d+)", str(d.get("id", "")).strip())
        if m and 1 <= int(m.group(1)) <= len(batch):
            out[int(m.group(1)) - 1] = d
    return out


def stage_mapping(ctx, items):
    import build_mapping
    import entity_map
    ents = collect_entities(ctx, items)
    print(f"\n[stage 2] mapping: {len(ents)} entities "
          f"({sum(e['type'] == 'person' for e in ents)} people)", flush=True)
    # naming answers are cached by entity, so a rebuilt mapping (new rules
    # below, a resumed run) never pays for or re-rolls a name
    cache_f = ctx.work / "naming_cache.json"
    cache = jload(cache_f, {}) or {}

    def ekey(e):
        return e["type"] + "|" + "|".join(sorted(f.lower() for f in e["forms"]))
    if not cache and (ctx.work / "mapping_entities.json").exists():
        # a run before the cache existed: its decisions seed it
        for r in jload(ctx.work / "mapping_entities.json", []) or []:
            if r.get("decision"):
                cache[ekey(r)] = r["decision"]
    decisions = [cache.get(ekey(e)) for e in ents]
    todo = [i for i, d in enumerate(decisions) if d is None]
    batches = [todo[i:i + 120] for i in range(0, len(todo), 120)]
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda b: name_batch(ctx, [ents[i] for i in b]),
                          batches))
    for b, r in zip(batches, res):
        for k, d in r.items():
            decisions[b[k]] = d
            cache[ekey(ents[b[k]])] = d
    jdump(cache_f, cache)
    canonical_names(ents, decisions)
    # two different entities must not share an invented name
    seen, dup = {}, []
    for i, d in enumerate(decisions):
        if not d or d.get("keep"):
            continue
        inv = (d.get("invented") or "").strip().lower()
        if inv in seen and ents[seen[inv]]["name"].lower() != \
                ents[i]["name"].lower():
            dup.append(i)
        else:
            seen[inv] = i
    if dup:
        fresh = name_batch(ctx, [ents[i] for i in dup], avoid=set(seen))
        for k, i in enumerate(dup):
            if k in fresh:
                decisions[i] = fresh[k]
    fix_collisions(ctx, ents, decisions)
    canonical_names(ents, decisions)          # again: after any re-invention
    split = {}
    for e, d in zip(ents, decisions):
        if d and not d.get("keep") and e["type"] != "person" and \
                len((d.get("core") or e["name"]).split()) == 1:
            split.setdefault((d.get("core") or e["name"]).strip().lower(),
                             set()).add(d.get("invented"))
    multi = {k: v for k, v in split.items() if len(v) > 1}
    print(f"  one name per core: {len(split) - len(multi)}/{len(split)}"
          + (f" (still split: {list(multi)[:5]})" if multi else ""),
          flush=True)
    mapping = build_entries(ents, decisions)
    path = ctx.work / "mapping.yaml"
    validate_and_repair(mapping, path, "dataset mapping (run_dataset.py)")
    jdump(ctx.work / "mapping_entities.json",
          [{"name": e["name"], "type": e["type"], "forms": e["forms"],
            "decision": d} for e, d in zip(ents, decisions)])
    return path


def canonical_names(ents, decisions):
    """One invented name per real name across the dataset. Entities named in
    different batches can share a core ("Zorvex" inside "Zorvex Mobility"
    and "Zorvex Edge"): the core takes ONE invented name (that of its most
    mentioned entity), and a compound core is composed from the canonical
    names of the cores inside it, so a model line reads the same everywhere."""
    canon = {}
    order = sorted(range(len(ents)), key=lambda i: -ents[i]["mentions"])
    for i in order:
        d = decisions[i]
        if not d or d.get("keep") or ents[i]["type"] == "person" or \
                not (d.get("invented") or "").strip():
            continue
        core = (d.get("core") or ents[i]["name"]).strip()
        canon.setdefault(core.lower(), d["invented"].strip())
    cores = sorted(canon, key=len, reverse=True)
    for i, d in enumerate(decisions):
        if not d or d.get("keep") or ents[i]["type"] == "person" or \
                not (d.get("invented") or "").strip():
            continue
        core = (d.get("core") or ents[i]["name"]).strip()
        if core.lower() in canon and len(core.split()) == 1:
            inv = canon[core.lower()]
            d["invented"] = inv.lower() if core.islower() else \
                inv.upper() if core.isupper() and len(core) > 1 else inv
            continue
        # compose: every smaller canonical core inside this one
        out, low, hit = core, core.lower(), False
        for c in cores:
            if c == low or len(c) < 3:
                continue
            m = re.search(rf"(?<!\w){re.escape(c)}(?!\w)", low)
            if m:
                hit = True
                inv = canon[c]
                inv = inv.upper() if core[m.start():m.end()].isupper() else inv
                out = out[:m.start()] + inv + out[m.end():]
                low = low[:m.start()] + "\0" * len(inv) + low[m.end():]
        if hit:
            d["invented"] = out
        elif core.lower() in canon:
            d["invented"] = canon[core.lower()]


def fix_collisions(ctx, ents, decisions, extra_real=()):
    """An invented name must not contain any token of a REAL name in the
    dataset (with hundreds of respondents, common first names collide): a
    colliding person token is swapped for a pool name that occurs nowhere
    among the real names (the same swap everywhere, so a first name alone
    stays consistent with the full name); a colliding brand is re-invented
    once, then made distinct mechanically."""
    import entity_map
    pool = json.loads((ROOT / "prompts" / "name_pool_india_v1.json"
                       ).read_text())
    real_tok, real_key = set(), set()
    for e in list(ents) + list(extra_real):
        for f in list(e["forms"]) + [e["name"]]:
            for t in re.findall(r"[^\W\d_]+", str(f)):
                if len(t) >= 2:
                    real_tok.add(t.lower())
                    real_key.add(entity_map.key(t))

    def bad(t):
        return t.lower() in real_tok or entity_map.key(t) in real_key
    lists = {k: [n for n in pool[k] if not bad(n)]
             for k in ("female", "male", "surname")}
    sets = {k: {n.lower() for n in pool[k]} for k in ("female", "male")}
    cache, count = {}, {"female": 0, "male": 0, "surname": 0}

    def pick(kind):
        lst = lists[kind] or pool[kind]
        n = lst[count[kind] % len(lst)]
        count[kind] += 1
        return n
    brands_bad = []
    for i, (e, d) in enumerate(zip(ents, decisions)):
        if not d or d.get("keep") or not (d.get("invented") or "").strip():
            continue
        toks = d["invented"].split()
        if e["type"] != "person":
            if any(bad(re.sub(r"[^\w]", "", t)) for t in toks
                   if re.sub(r"[^\w]", "", t)):
                brands_bad.append(i)
            continue
        real_first = (re.findall(r"[^\W\d_]+", e["name"]) or [""])[0].lower()
        out = []
        for j, t in enumerate(toks):
            core = re.sub(r"[^\w]", "", t)
            if core and bad(core):
                k = core.lower()
                if k not in cache:
                    if j == 0:
                        kind = ("female" if k in sets["female"] or
                                real_first in sets["female"] else "male")
                    else:
                        kind = "surname"
                    cache[k] = pick(kind)
                new = cache[k]
                t = t.replace(core, new.upper() if core.isupper() else new)
            out.append(t)
        d["invented"] = " ".join(out)
    if brands_bad:
        avoid = sorted(real_tok)[:600]
        fresh = name_batch(ctx, [ents[i] for i in brands_bad], avoid=avoid)
        for k, i in enumerate(brands_bad):
            f = fresh.get(k)
            if f and f.get("invented") and not any(
                    bad(re.sub(r"[^\w]", "", t)) for t in f["invented"].split()
                    if re.sub(r"[^\w]", "", t)):
                decisions[i]["invented"] = f["invented"]
                if f.get("core"):
                    decisions[i]["core"] = f["core"]
            else:
                decisions[i]["invented"] = " ".join(
                    (t + "o") if bad(re.sub(r"[^\w]", "", t) or "_") else t
                    for t in decisions[i]["invented"].split())
    # two different people never share one invented full name
    seen = {}
    for i, (e, d) in enumerate(zip(ents, decisions)):
        if not d or d.get("keep") or e["type"] != "person":
            continue
        inv = (d.get("invented") or "").strip()
        k = inv.lower()
        if k in seen and ents[seen[k]]["name"].lower() != e["name"].lower():
            toks = inv.split()
            if len(toks) > 1:
                toks[-1] = pick("surname")
            else:
                toks.append(pick("surname"))
            d["invented"] = " ".join(toks)
        seen[d["invented"].lower()] = i


ERR_RX = re.compile(r"(\w+):(\d+): invented name '([^']+)' (matches real "
                    r"pattern|sounds the same as a real form of) (\w+):(\d+)")


def validate_and_repair(mapping, path, note, tries=8):
    """Write the mapping and load it through entity_map's checks; an
    invented name a real pattern matches (or that sounds like a real form)
    gets exactly its offending token swapped — a pool name for a person, a
    changed ending for a brand — until the mapping loads."""
    import build_mapping
    import entity_map
    pool = json.loads((ROOT / "prompts" / "name_pool_india_v1.json"
                       ).read_text())
    used = {t.lower() for sec in mapping.values() if isinstance(sec, list)
            for e in sec if isinstance(e, dict)
            for t in str(e.get("replace", "")).split()}
    pools = {k: [n for n in pool[k] if n.lower() not in used]
             for k in ("male", "female", "surname")}
    for attempt in range(tries):
        build_mapping.dump(mapping, path, note)
        try:
            return entity_map.load(path)
        except ValueError as exc:
            errs = ERR_RX.findall(str(exc))
            if not errs or attempt == tries - 1:
                raise
            print(f"  repairing {len(errs)} invented name(s) "
                  f"(round {attempt + 1})", flush=True)
            for sec, i, inv, how, rsec, ri in errs:
                e = mapping[sec][int(i)]
                if e.get("replace") != inv:
                    continue           # already changed this round
                real = mapping[rsec][int(ri)]
                toks = inv.split()
                bad_idx = set(range(len(toks)))
                if how.startswith("matches"):
                    rx = re.compile(str(real["pattern"]),
                                    0 if rsec == "abbreviations" else re.I)
                    m = rx.search(inv)
                    if m:
                        pos, bad_idx = 0, set()
                        for k, t in enumerate(toks):
                            a = inv.index(t, pos)
                            b = a + len(t)
                            pos = b
                            if a < m.end() and b > m.start():
                                bad_idx.add(k)
                for k in sorted(bad_idx):
                    t = toks[k]
                    if sec == "people":
                        kind = ("surname" if k > 0 else
                                "female" if t.lower() in {
                                    n.lower() for n in pool["female"]}
                                else "male")
                        lst = pools[kind] or pool[kind]
                        new = lst.pop(0) if pools[kind] else \
                            lst[(attempt * 7 + k) % len(lst)]
                        toks[k] = new.upper() if t.isupper() else new
                    else:
                        toks[k] = t + ("ro" if attempt % 2 == 0 else "vi")
                e["replace"] = " ".join(toks)


def build_entries(ents, decisions):
    import build_mapping
    m = {"brands": [], "abbreviations": [], "brand_facts": [], "agency": [],
         "people": [], "keep": []}
    for e, d in zip(ents, decisions):
        if not d or d.get("keep") or not (d.get("invented") or "").strip():
            m["keep"].append(f"{e['name']} ({e['type']}, kept)")
            continue
        cat = (d.get("category") or "BRAND").upper()
        core = (d.get("core") or e["name"]).strip()
        if sum(ch.isalpha() for ch in core) < 2:
            m["keep"].append(f"{e['name']} (core too short to match safely)")
            continue
        if e["type"] == "person":
            cat = "PERSON"
            core = e["name"]
        inv = d["invented"].strip()
        full = e["name"]
        if core != full and core.lower() in full.lower():
            # the invented name replaces the CORE only: generic words around
            # it ("Electronics Hub") stay in the text, so drop them here
            k = full.lower().index(core.lower())
            pre, post = full[:k].strip(), full[k + len(core):].strip()
            if post and inv.lower().endswith(post.lower()):
                inv = inv[:-len(post)].strip()
            if pre and inv.lower().startswith(pre.lower()):
                inv = inv[len(pre):].strip()
        inv = inv or d["invented"].strip()
        n = len(core.split())
        aliases = sorted(f for f in e["forms"] if f != core
                         and len(f.split()) <= max(n, 1))
        entry = {"pattern": build_mapping.core_regex(core), "replace": inv,
                 "aliases": aliases, "category": cat, "name": core}
        if cat == "PERSON":
            entry["render"] = "voice"
            m["people"].append(entry)
            continue
        if build_mapping.ordinary_word(core):
            entry["context_only"] = True
        sec = build_mapping.SECTION.get(cat, "brands")
        if len(core) <= 3 and core.isupper():
            sec = "abbreviations"
        m[sec].append(entry)
    return {k: build_mapping._merge_same(v) if k != "keep" else v
            for k, v in m.items() if v}


def grow_mapping(ctx, items):
    """Append invented names for ledger entities the given mapping does
    not cover; existing entries never change."""
    import build_mapping
    import yaml
    ents = collect_entities(ctx, items)
    # what the mapping deliberately kept ("Name (type, kept)" notes, and the
    # keep decisions of the run that built it) is not new
    m0 = __import__("yaml").safe_load(Path(ctx.a.mapping_path).read_text()) \
        or {}
    kept = {str(k).split(" (")[0].strip().lower()
            for k in m0.get("keep") or [] if isinstance(k, str)}
    kept |= ctx.keep_forms
    for f in ctx.work.glob("mapping_entities*.json"):
        for r in jload(f, []) or []:
            if (r.get("decision") or {}).get("keep"):
                kept |= {x.lower() for x in r.get("forms", [])}
    new = []
    for e in ents:
        if any(f.lower() in kept for f in e["forms"]):
            continue
        if any(ctx.em.match_tokens(f.split()) for f in e["forms"] if f):
            continue
        new.append(e)
    if not new:
        return ctx.a.mapping_path
    print(f"  mapping: {len(new)} new entit(ies) to name "
          f"({sum(e['type'] == 'person' for e in new)} people)", flush=True)
    taken = {e["replace"].lower() for e in ctx.em.entries if e["replace"]}
    decisions = [None] * len(new)
    batches = [list(range(i, min(i + 120, len(new))))
               for i in range(0, len(new), 120)]
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda b: name_batch(
            ctx, [new[i] for i in b], avoid=taken), batches))
    for b, r in zip(batches, res):
        for k, d in r.items():
            decisions[b[k]] = d
    real = [{"name": e.get("name") or "", "forms": e["forms"]}
            for e in ctx.em.entries]
    fix_collisions(ctx, new, decisions, extra_real=real)
    add = build_entries(new, decisions)
    m = yaml.safe_load(Path(ctx.a.mapping_path).read_text()) or {}
    for sec, its in add.items():
        if sec == "keep":
            m.setdefault("keep", []).extend(its)
        else:
            m.setdefault(sec, []).extend(its)
    out = ctx.work / "mapping.yaml"
    validate_and_repair(m, out, "dataset mapping, grown with names from "
                        "files the given mapping had not seen")
    jdump(ctx.work / "mapping_entities_grown.json",
          [{"name": e["name"], "type": e["type"], "forms": e["forms"],
            "decision": d} for e, d in zip(new, decisions)])
    return out


def extend_mapping(ctx, cands):
    """Names the audit found that the mapping lacks: named once, appended
    (existing invented names never change)."""
    import build_mapping
    import entity_map
    import yaml
    path = ctx.a.mapping_path
    m = yaml.safe_load(Path(path).read_text()) or {}
    have = {f.lower() for e in ctx.em.entries for f in e["forms"]}
    ents, seen = [], set()
    for c in cands:
        t = c["text"]
        if t.lower() in have or t.lower() in seen:
            continue
        seen.add(t.lower())
        ents.append({"name": t, "type": "person" if c["kind"] == "person"
                     else c["kind"], "forms": [t], "mentions": 1,
                     "ctx": [c.get("why") or ""]})
    if not ents:
        return path
    taken = {e["replace"].lower() for e in ctx.em.entries if e["replace"]}
    dec = name_batch(ctx, ents, avoid=taken)
    decs = [dec.get(k) for k in range(len(ents))]
    for d in decs:
        if d:
            d["keep"] = False      # the audit found it: it is replaced
    real = [{"name": e.get("name") or "", "forms": e["forms"]}
            for e in ctx.em.entries]
    fix_collisions(ctx, ents, decs, extra_real=real)
    add = build_entries(ents, decs)
    for sec, items in add.items():
        if sec == "keep":
            continue
        m.setdefault(sec, []).extend(items)
    new = ctx.work / "mapping.yaml"
    validate_and_repair(m, new, "dataset mapping (run_dataset.py), "
                        "extended by the leak audit")
    print(f"  mapping extended with {sum(len(v) for k, v in add.items() if k != 'keep')} "
          f"audit name(s)", flush=True)
    return new


# =============================================================== the net

class Net:
    """Every removable form found anywhere in the dataset, found again in
    any text (render: applied to every file; verify: must be gone)."""

    def __init__(self, forms):        # {form: categories}
        import build_mapping
        self.by_first = {}
        self.cats = {}
        for f, cats in forms.items():
            f = f.strip()
            toks = re.findall(r"\w+", f.lower())
            if not toks:
                continue
            digits = sum(ch.isdigit() for ch in f)
            letters = sum(ch.isalpha() for ch in f)
            if letters == 0 and digits < 6:
                continue
            if letters and len(f) < 3:
                continue
            if len(toks) == 1 and letters and build_mapping.ordinary_word(f):
                continue
            self.by_first.setdefault(toks[0], []).append(f)
            self.cats[f.lower()] = cats
        self.rx = {}

    def find_loose(self, text):
        """Forms of 5+ characters found anywhere, word boundaries ignored —
        for text a PDF or OCR glued together ("HierarchyZorvex")."""
        if not hasattr(self, "_loose"):
            fs = sorted({f for fl in self.by_first.values() for f in fl
                         if (len(f) >= 5 and any(c.isalpha() for c in f))
                         or (len(f) >= 8 and f.isdigit())},
                        key=len, reverse=True)
            self._loose = re.compile("|".join(
                re.escape(f).replace(r"\ ", r"\s*") for f in fs), re.I) \
                if fs else None
        if self._loose is None:
            return []
        return [(m.start(), m.end(), self.cats[m.group(0).lower()]
                 if m.group(0).lower() in self.cats else ["OTHER"])
                for m in self._loose.finditer(text)]

    def find(self, text):
        low_toks = set(re.findall(r"\w+", text.lower()))
        hits = []
        for t in low_toks & self.by_first.keys():
            for f in self.by_first[t]:
                rx = self.rx.get(f)
                if rx is None:
                    body = re.escape(f).replace(r"\ ", r"\s+")
                    rx = re.compile(rf"(?<!\w){body}(?!\w)", re.I)
                    self.rx[f] = rx
                for m in rx.finditer(text):
                    hits.append((m.start(), m.end(), self.cats[f.lower()]))
        return hits


NAMESPACE_WORDS = {"microsoft", "openxmlformats", "schemas", "purl", "w3",
                   "xmlsoap", "office", "word", "excel", "powerpoint"}


def build_net(ctx, items, extra_forms=None):
    forms = {f: c for f, c in (extra_forms or {}).items()
             if f.strip().lower() not in NAMESPACE_WORDS}
    ledir = ctx.work / "ledger"
    for it in items:
        led = jload(ledir / f"{it['id']}.json")
        if not led or led.get("media"):
            continue
        for rec in led["subs"].values():
            for s in rec["spans"]:
                if ctx.pack.action_for(s["categories"]) == "KEEP":
                    continue
                if s.get("source") == "column_rule" and \
                        set(s["categories"]) & {"ASSIGNED_ID"}:
                    continue      # codes are per value, never netted
                t = s["text"].strip()
                if t.lower() in ctx.keep_forms:
                    continue
                cats = set(s["categories"])
                if getattr(ctx, "em", None) is not None and \
                        cats & set(ENTITY_CATS) and not cats & PERSON_CATS:
                    # replace mode: a brand span is netted by the part the
                    # mapping replaces (its core); generic words around it
                    # ("service center") stay in the text by design
                    toks = t.split()
                    for i0, i1, _e, _how in ctx.em.match_tokens(toks):
                        forms.setdefault(" ".join(toks[i0:i1 + 1]),
                                         s["categories"])
                    continue
                forms.setdefault(t, s["categories"])
                if set(s["categories"]) & PERSON_CATS:
                    # a name seen once in full is caught alone too
                    for tok in re.findall(r"[^\W\d_]{4,}", t):
                        if tok.lower() not in HONORIFICS:
                            forms.setdefault(tok, s["categories"])
    nl = jload(ledir / "__names__.json", {})
    for s in nl.get("spans", []):
        if ctx.pack.action_for(s["categories"]) != "KEEP":
            forms.setdefault(s["text"].strip(), s["categories"])
    return Net(forms)


def net_spans(net, segs):
    out = []
    for si, seg in enumerate(segs):
        for a, b, cats in net.find(seg["text"]):
            out.append({"seg": si, "a": a, "b": b,
                        "text": seg["text"][a:b], "categories": cats,
                        "source": "dataset_net"})
    return out


# =============================================================== stage 3

class NameReplacer:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, span):
        act = self.ctx.pack.action_for(span["categories"])
        if act == "KEEP":
            return None
        if self.ctx.mode == "replace" and act == "PSEUDONYMIZE" and \
                self.ctx.em is not None:
            import office
            rep = office.Replacer("replace", self.ctx.pack, self.ctx.em
                                  ).lookup(span)
            if rep:
                return rep
        tag = next((TAGS[c] for c in span["categories"] if c in TAGS),
                   "REDACTED")
        return f"[{tag}]"


def clean_name(ctx, comp, name_led, net):
    """One path component with its detected spans replaced."""
    idx = {t: i for i, t in enumerate(name_led["segs"])}
    i = idx.get(_name_text(comp))
    spans = [s for s in name_led["spans"] if s["seg"] == i] if i is not None \
        else []
    seg = {"text": _name_text(comp)}
    spans = spans + [dict(s, seg=i) for s in net_spans(net, [seg])]
    if ctx.em is not None:
        import office
        spans += [dict(s, seg=i) for s in office.deterministic([dict(
            seg, kind="name")], ctx.em) if s["source"] == "mapping"]
    import office
    spans = office._dedupe([dict(s, seg=0) for s in spans])
    rep = NameReplacer(ctx)
    out = comp
    for s in sorted(spans, key=lambda s: -s["a"]):
        new = rep(s)
        if new is not None:
            out = out[:s["a"]] + new + out[s["b"]:]
    out = re.sub(r"[/\\:]", "_", out).strip() or "_"
    return out


def out_rel(ctx, it, name_led, net, cache={}):
    rel = it["rel"].split("!/")[0] if it.get("container") else it["rel"]
    parts = rel.split("/")
    new = []
    for i, p in enumerate(parts):
        if i == len(parts) - 1:
            stem, ext = Path(p).stem, Path(p).suffix
            if ext.lower() in (".doc", ".rtf") and it["route"] == "office":
                ext = ".docx"
            new.append(clean_name(ctx, stem, name_led, net) + ext)
        else:
            new.append(clean_name(ctx, p, name_led, net))
    return "/".join(new)


def render_doc(ctx, it, led, net):
    """-> (bytes, ext, report) for a document item."""
    import office
    import pdfimg
    r = it["route"]
    codes = ctx.codes.setdefault(it["study"], {})
    if r == "office":
        blob, ext = doc_blob(ctx, it)
        reports = {}

        def render(b, sub):
            parts, segs, media, embedded, rules = office.read_ooxml(b)
            rec = led["subs"].get(sub)
            if rec is None or rec["n_segs"] != len(segs):
                raise RuntimeError(f"ledger does not match {sub or 'file'}")
            spans = office._dedupe(rec["spans"] + net_spans(net, segs))
            spans = [s for s in spans if s["text"].strip().lower()
                     not in ctx.keep_forms or s.get("source") == "pattern"]

            def emb(n, data):
                return render(data, (sub + "::" if sub else "") + n)
            out, rep = office.process_ooxml(
                b, ctx.pack, ctx.model, ctx.mode, ctx.em, codes,
                ctx.img_cache, ctx.img_model, detections=spans,
                embedded_fn=emb)
            reports[sub or "file"] = rep
            return out
        out = render(blob, "")
        return out, ext, reports
    if r == "text":
        ext = Path(it["path"]).suffix.lower().lstrip(".")
        raw = Path(it["path"]).read_bytes()
        _t, _e, _b, segs, _r = office.read_text(raw, ext)
        rec = led["subs"][""]
        extra = []
        if ext in office.FIXED_WIDTH:
            # fixed-width records glue fields together ("10001" + a phone):
            # every known form is searched without word boundaries
            extra = [{"seg": si, "a": a, "b": b, "text": sg["text"][a:b],
                      "categories": c, "source": "dataset_net"}
                     for si, sg in enumerate(segs)
                     for a, b, c in net.find_loose(sg["text"])]
        spans = office._dedupe(rec["spans"] + net_spans(net, segs) + extra)
        out, rep = office.process_text(raw, ctx.pack, ctx.model, ctx.mode,
                                       ctx.em, codes, detections=spans,
                                       ext=ext)
        return out, ext, rep
    rec = led["subs"][""]
    segs, info = rec["segs"], rec["info"]
    loose = [{"seg": si, "a": a, "b": b, "text": sg["text"][a:b],
              "categories": c, "source": "dataset_net"}
             for si, sg in enumerate(segs)
             for a, b, c in net.find_loose(sg["text"])]
    spans = office._dedupe(rec["spans"] + net_spans(net, segs) + loose)
    if ctx.em is not None:
        spans = office._dedupe(spans + office.deterministic(segs, ctx.em))
    rp = office.Replacer(ctx.mode, ctx.pack, ctx.em, codes)
    if r == "pdf":
        out, rep = pdfimg.render_pdf(it["path"], segs, spans, rp, info,
                                     ctx.classify)
        return out, "pdf", rep
    out, rep = pdfimg.render_image(it["path"], segs, spans, rp, info,
                                   ctx.classify)
    return out, Path(it["path"]).suffix.lower().lstrip("."), rep


# =============================================================== stage 4

def output_text(blob, ext, route):
    import office
    import pdfimg
    if route == "pdf":
        return pdfimg.output_text_pdf(blob)
    if route == "image":
        return pdfimg.output_text_image(blob)
    return office.output_text(blob, ext)


def check_format(blob, ext, route, src_path=None):
    import io
    try:
        if ext in OOXML_EXT:
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                if z.testzip() is not None:
                    return "corrupt zip member"
                for n in z.namelist():
                    if n.endswith((".xml", ".rels")):
                        ET.fromstring(z.read(n))
        elif route == "pdf":
            import fitz
            fitz.open(stream=blob, filetype="pdf").close()
        elif route == "image":
            from PIL import Image
            Image.open(io.BytesIO(blob)).verify()
        elif ext == "asc" and src_path:
            import survey
            a = survey.decode(Path(src_path).read_bytes())[0].splitlines()
            b = survey.decode(blob)[0].splitlines()
            if [len(x) for x in a] != [len(x) for x in b]:
                return "fixed-width line lengths changed"
    except Exception as exc:                  # noqa: BLE001
        return f"does not open: {str(exc)[:80]}"
    return None


def repair(ctx, it, blob, ext, net, codes):
    import office
    import pdfimg
    if it["route"] == "office":
        _p, segs, _m, _e, _r = office.read_ooxml(blob)
        spans = office._dedupe(net_spans(net, segs))
        if not spans:
            return blob
        out, _rep = office.process_ooxml(
            blob, ctx.pack, ctx.model, ctx.mode, ctx.em, codes,
            ctx.img_cache, ctx.img_model, detections=spans,
            embedded_fn=lambda n, d: d)
        return out
    if it["route"] == "text":
        _t, _e, _b, segs, _r = office.read_text(blob, ext)
        spans = net_spans(net, segs)
        if ext in office.FIXED_WIDTH:
            spans += [{"seg": si, "a": a, "b": b, "text": sg["text"][a:b],
                       "categories": c, "source": "dataset_net"}
                      for si, sg in enumerate(segs)
                      for a, b, c in net.find_loose(sg["text"])]
        spans = office._dedupe(spans)
        if not spans:
            return blob
        out, _rep = office.process_text(blob, ctx.pack, ctx.model, ctx.mode,
                                        ctx.em, codes, detections=spans,
                                        ext=ext)
        return out
    if it["route"] == "pdf":
        return pdfimg.flatten_pages_with(blob, lambda t: bool(net.find(t)))
    return blob


def binary_member_text(blob):
    import io
    out = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            low = n.lower()
            if low.endswith((".xml", ".rels", ".png", ".jpg", ".jpeg",
                             ".gif", ".emf", ".wmf", ".tif", ".tiff", ".wdp",
                             ".jxr", ".bmp", ".svg")) or "/media/" in low \
                    or "printersettings" in low \
                    or low.endswith(tuple("." + e for e in OOXML_EXT)):
                continue
            data = z.read(n)
            out.append(data.decode("latin-1"))
            out.append(data.decode("utf-16-le", "ignore"))
    return "\n".join(out)


def invented_spans(ctx, text):
    """Where the mapping's invented names occur (replace mode): a real
    first name that is also part of an invented one is not a leftover."""
    if getattr(ctx, "em", None) is None:
        return []
    if not hasattr(ctx, "_inv_rx"):
        inv = sorted({e["replace"] for e in ctx.em.entries
                      if e["replace"] and len(e["replace"]) >= 3},
                     key=len, reverse=True)
        ctx._inv_rx = re.compile("|".join(
            rf"(?<!\w){re.escape(v)}(?!\w)" for v in inv), re.I) \
            if inv else None
    if ctx._inv_rx is None:
        return []
    return [(m.start(), m.end()) for m in ctx._inv_rx.finditer(text)]


def leftovers(ctx, text, net):
    import office
    found = []
    inv = invented_spans(ctx, text)
    for a, b, cats in net.find(text):
        if any(x <= a and b <= y for x, y in inv):
            continue
        found.append({"kind": "form", "categories": cats,
                      "len": b - a})
    for _m in office.EMAIL_RX.finditer(text):
        found.append({"kind": "email"})
    for _m in VERIFY_PHONE_RX.finditer(text):
        found.append({"kind": "phone"})
    return found


# a phone inside one piece of text (spaces or hyphens between its groups,
# never a line break: two neighbouring cells of figures are not a phone)
VERIFY_PHONE_RX = re.compile(
    r"(?<![\d.])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?![\d.])"
    r"|(?<![\d.])\+\d{1,3}[ -]?\d{3,4}[ -]?\d{3,4}[ -]?\d{3,4}")


# =============================================================== audit

AUDIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"leaks": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"text": {"type": "string"},
                       "kind": {"type": "string", "enum": [
                           "person", "brand", "product", "dealer", "agency",
                           "phone", "email", "address", "id", "other"]},
                       "why": {"type": "string"}},
        "required": ["text", "kind", "why"]}}},
    "required": ["leaks"]}
AUDIT_PROMPT = """You audit a document that was {what} before release. Find
anything that should have been removed and is still there:
- names of real people (respondents, staff, clients, interviewers, authors);
- phone numbers, email addresses, street addresses, ID numbers;
- names of brands, companies, products, models, dealers, banks or the
  research / advertising agency that a study is about or compares.
Allowed (NOT leaks): cities, states and countries; ages, gender, income or
SEC groups, occupations; dates; numbers in tables; generic product words;
platforms and marketplaces that only carry products (shopping, delivery,
payment apps); social networks and media channels (Facebook, Instagram,
YouTube, Twitter, WhatsApp, TV, newspapers); typefaces and fonts; software,
file formats and office or survey tools named in metadata or syntax; theme
and template names; government bodies; black boxes (█ or X runs); codes like
R0001, INT001, ID0001; tags like [NAME] or [BRAND]{allowed}.
Return each leak with its EXACT text as it appears below. Return an empty
list when nothing leaks. Do not report anything that is not in the text.

DOCUMENT ({label}):
"""


def audit_sample(ctx, items, results, k=60):
    """A stratified sample of delivered documents (by route and study)."""
    pool = [it for it in items if results.get(it["id"], {}).get("status")
            == "ok" and it["route"] in ("office", "text", "pdf", "image")]
    by = {}
    for it in pool:
        by.setdefault((it["route"], it["study"]), []).append(it)
    out, i = [], 0
    while len(out) < k and any(by.values()):
        for key in list(by):
            if by[key] and len(out) < k:
                out.append(by[key].pop(0))
        i += 1
    return out


def run_audit(ctx, items, results):
    """Second model family reads delivered documents; returns leak strings
    that really occur in the output (hallucinated ones are dropped)."""
    import judge_client
    import office
    sample = audit_sample(ctx, items, results, ctx.a.audit_files)
    allowed = ""
    invented = set()
    if ctx.em is not None:
        invented = {e["replace"].lower() for e in ctx.em.entries
                    if e["replace"] and len(e["replace"]) >= 3}
        inv = sorted({e["replace"] for e in ctx.em.entries if e["replace"]})
        allowed = ("; and these INVENTED names, which replaced the real "
                   "ones: " + ", ".join(inv[:300]))
    what = ("redacted (removed values are black boxes)" if ctx.mode ==
            "redact" else "pseudonymised (real names replaced by invented "
            "ones; contact data removed)")
    jobs = []
    for it in sample:
        res = results[it["id"]]
        blob = Path(res["staged"]).read_bytes()
        ext = Path(it["rel"]).suffix.lower().lstrip(".")
        if it["route"] == "office":
            _p, segs, _m, _e, _r = office.read_ooxml(
                blob if ext not in CONVERT_EXT else blob)
            text = "\n".join(sg["text"] for sg in segs)
        else:
            text = output_text(blob, ext, it["route"])
        text = text[:ctx.a.audit_chars]
        for c in range(0, len(text), 12000):
            jobs.append((it, text[c:c + 12000]))
    found = []

    def one(job):
        it, chunk = job
        prompt = AUDIT_PROMPT.format(what=what, allowed=allowed,
                                     label=Path(it["rel"]).suffix) + chunk
        try:
            r = judge_client.judge_call(prompt, AUDIT_SCHEMA,
                                        model=ctx.a.audit_model,
                                        log_engine="dataset_audit",
                                        max_tokens=4000)
        except Exception as exc:              # noqa: BLE001
            return it, chunk, None, str(exc)[:80]
        return it, chunk, r.get("leaks", []), None
    print(f"\n[stage 4b] leak audit: {len(sample)} file(s), {len(jobs)} "
          f"call(s) ({ctx.a.audit_model})", flush=True)
    errors = 0
    with cf.ThreadPoolExecutor(12) as ex:
        for it, chunk, leaks, err in ex.map(one, jobs):
            if leaks is None:
                errors += 1
                continue
            for lk in leaks:
                t = str(lk.get("text", "")).strip()
                if len(t) < 3 or t not in chunk or set(t) <= set("█X "):
                    continue
                if re.fullmatch(r"(R|INT|ID)\d{3,4}|\[[A-Z]+\]", t):
                    continue
                if invented and any(v in t.lower() for v in invented):
                    continue          # our own invented name
                if t.isdigit() and len(t) > 12:
                    continue          # a whole fixed-width record, not a value
                found.append({"file_id": it["id"], "text": t,
                              "kind": lk.get("kind"), "why": lk.get("why")})
    return {"files": len(sample), "calls": len(jobs), "errors": errors,
            "candidates": found}


KIND_CAT = {"person": "FULL_NAME", "brand": "BRAND",
            "product": "PRODUCT_OR_MODEL", "dealer": "DEALER_OR_OUTLET",
            "agency": "RESEARCH_OR_AD_AGENCY", "phone": "CONTACT_CHANNEL",
            "email": "CONTACT_CHANNEL", "address": "ADDRESS",
            "id": "ASSIGNED_ID", "other": "OTHER"}


# =============================================================== main flow

def render_item(ctx, net, name_led, staging, it):
    """Render, verify and (when needed) repair one document; -> (it, res)."""
    ledir = ctx.work / "ledger"
    led = jload(ledir / f"{it['id']}.json")
    if not led:
        return it, {"status": "held", "reason": "no ledger"}
    if any(r.get("incomplete") for r in led["subs"].values()):
        return it, {"status": "held",
                    "reason": "detection incomplete (model call failed)"}
    try:
        blob, ext, rep = render_doc(ctx, it, led, net)
    except Exception as exc:              # noqa: BLE001
        return it, {"status": "held", "reason":
                    f"render failed: {str(exc)[:160]}"}
    fmt = check_format(blob, ext, it["route"], it["path"])
    text = output_text(blob, ext, it["route"])
    left = leftovers(ctx, text, net)
    if any(x["kind"] == "form" for x in left) and \
            it["route"] in ("office", "text", "pdf"):
        # automated repair: the net applied once more to the OUTPUT
        # itself (a hidden copy the first pass did not edit, a PDF page
        # whose text sat in an object redaction cannot reach)
        try:
            blob = repair(ctx, it, blob, ext, net,
                          ctx.codes.setdefault(it["study"], {}))
            text = output_text(blob, ext, it["route"])
            left = leftovers(ctx, text, net)
            fmt = check_format(blob, ext, it["route"], it["path"])
        except Exception as exc:          # noqa: BLE001
            left.append({"kind": f"repair failed: {str(exc)[:60]}"})
    if ext == "asc":
        left += [{"kind": "form_in_fixed_width", "categories": c,
                  "len": b - a_} for a_, b, c in net.find_loose(text)]
    if it["route"] == "office":
        # binary members the reader cannot edit (OLE objects, VBA): a
        # real name inside one holds the file rather than shipping it
        # (names only: random bytes look like phone numbers)
        bt = binary_member_text(blob)
        left += [{"kind": "form_in_binary_member", "categories": c,
                  "len": b - a_}
                 for a_, b, c in net.find(bt)
                 if b - a_ >= 5 and any(ch.isalpha() for ch in bt[a_:b])]
    res = {"status": "held" if (fmt or left) else "ok",
           "reason": fmt or (f"{len(left)} leftover(s): "
                             + ", ".join(sorted({x['kind'] for x in left}))
                             if left else ""),
           "report": rep, "leftovers": left[:20]}
    rel = out_rel(ctx, it, name_led, net)
    if it.get("container"):
        res["member_out"] = "/".join(clean_name(ctx, p, name_led, net)
                                     if i < len(it["member"].split("/"))
                                     - 1 else
                                     clean_name(ctx, Path(p).stem,
                                                name_led, net)
                                     + Path(p).suffix
                                     for i, p in enumerate(
                                         it["member"].split("/")))
    dst = staging / it["id"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)
    res["staged"] = str(dst)
    res["out_rel"] = rel
    res["tokens"] = sorted(set(re.findall(r"\w+", text.lower())))[:200000]
    return it, res


_W = {}


def _init_worker(a_dict, extra, codes, keep):
    """A render process: its own context, mapping and net (built once)."""
    import argparse
    import entity_map
    a = argparse.Namespace(**a_dict)
    ctx = Ctx(a)
    if a.mapping_path and a.mode == "replace":
        ctx.em = entity_map.load(a.mapping_path)
    ctx.codes = codes
    ctx.keep_forms = set(keep)
    items = jload(ctx.work / "inventory.json", [])
    _W.update(ctx=ctx, net=build_net(ctx, items, extra),
              name_led=jload(ctx.work / "ledger" / "__names__.json",
                             {"segs": [], "spans": []}),
              staging=ctx.work / "staging")


def _render_worker(it):
    return render_item(_W["ctx"], _W["net"], _W["name_led"], _W["staging"],
                       it)


def precompute_codes(ctx, items):
    """The code book for every linking id in every ledger, in a fixed order,
    before rendering (parallel renderers then agree on every code)."""
    import office
    ledir = ctx.work / "ledger"
    for it in items:
        led = jload(ledir / f"{it['id']}.json")
        if not led or led.get("media"):
            continue
        rp = office.Replacer("replace", ctx.pack, None,
                             ctx.codes.setdefault(it["study"], {}))
        for sub in sorted(led["subs"]):
            for sp in sorted(led["subs"][sub]["spans"],
                             key=lambda x: (x["seg"], x["a"])):
                cats = set(sp["categories"])
                if cats & office.ID_CATS and not (
                        cats - office.ID_CATS - {"OTHER"}) and \
                        ctx.pack.action_for(sp["categories"]) == "PSEUDONYMIZE":
                    rp.code_for(sp["text"], sp.get("code_prefix"))


def broken_controls(ctx, items, results, net):
    """Per route, the ORIGINAL of one file with removals must fail the same
    check (proves the check can catch a leak)."""
    controls = {}
    ledir = ctx.work / "ledger"
    for it in items:
        r = it["route"]
        if r in controls or r not in ("office", "text", "pdf", "image") or \
                it["id"] not in results:
            continue
        led = jload(ledir / f"{it['id']}.json")
        if not led or not led_has_removals(ctx, led):
            continue
        try:
            if r == "office":
                orig, oext = doc_blob(ctx, it)
            else:
                orig = Path(it["path"]).read_bytes()
                oext = Path(it["path"]).suffix.lower().lstrip(".")
            caught = bool(leftovers(ctx, output_text(orig, oext, r), net))
            controls[r] = {"file_id": it["id"], "caught": caught}
        except Exception as exc:              # noqa: BLE001
            controls[r] = {"file_id": it["id"], "caught": None,
                           "error": str(exc)[:80]}
    return controls


def stage_render(ctx, items, extra_forms=None, only=None, prior=None):
    import entity_map
    ledir = ctx.work / "ledger"
    name_led = jload(ledir / "__names__.json", {"segs": [], "spans": []})
    net = build_net(ctx, items, extra_forms)
    staging = ctx.work / "staging"
    if only is None and staging.exists():
        shutil.rmtree(staging)
    results = dict(prior or {})
    docs = [] if ctx.a.media_only else [
        it for it in items if it["route"] in ("office", "text", "pdf",
                                              "image")
        and (only is None or it["id"] in only)]
    print(f"\n[stage 3] rendering {len(docs)} document(s) "
          f"({ctx.mode} mode, {ctx.a.processes} process(es))", flush=True)
    if docs and ctx.mode == "replace":
        precompute_codes(ctx, items)

    def show(it, res):
        results[it["id"]] = res
        flag = "ok  " if res["status"] == "ok" else "HELD"
        print(f"  {flag} {it['route']:6s} {Path(it['rel']).name[:60]}"
              + (f"  ({res['reason']})" if res["status"] != "ok" else ""),
              flush=True)
    if docs and ctx.a.processes > 1:
        a_dict = dict(vars(ctx.a))
        with cf.ProcessPoolExecutor(
                ctx.a.processes, initializer=_init_worker,
                initargs=(a_dict, extra_forms or {}, ctx.codes,
                          sorted(ctx.keep_forms))) as ex:
            futs = [ex.submit(_render_worker, it) for it in docs]
            for fut in cf.as_completed(futs):
                try:
                    it, res = fut.result()
                except Exception as exc:      # noqa: BLE001
                    print(f"  !! render process failed: {str(exc)[:120]}")
                    continue
                show(it, res)
    elif docs:
        with cf.ThreadPoolExecutor(ctx.a.workers) as ex:
            futs = [ex.submit(render_item, ctx, net, name_led, staging, it)
                    for it in docs]
            for fut in cf.as_completed(futs):
                show(*fut.result())
    for it in docs:
        if it["id"] not in results:
            results[it["id"]] = {"status": "held",
                                 "reason": "render process failed"}
    controls = broken_controls(ctx, items, results, net)
    ctx.save_images()
    for it in items:
        if it["route"] == "zip":
            results[it["id"]] = {"status": "zip",
                                 "out_rel": out_rel(ctx, it, name_led, net)}
    # media
    media = [it for it in items if it["route"] in ("audio", "video")]
    if media and not ctx.a.skip_media:
        mapping = ctx.a.mapping_path
        if ctx.mode == "redact":
            mapping = redact_media_list(ctx, items)
        def rend(it):
            if not (ledir / f"{it['id']}.json").exists():
                return it, None, "no media ledger"
            bundle, why = media_render(ctx, it, mapping)
            return it, bundle, why
        print(f"  rendering {len(media)} audio/video file(s), "
              f"{ctx.a.media_workers} at a time", flush=True)
        with cf.ThreadPoolExecutor(ctx.a.media_workers) as ex:
            futs = [ex.submit(rend, it) for it in media]
            for fut in cf.as_completed(futs):
                it, bundle, why = fut.result()
                if bundle is None:
                    results[it["id"]] = {"status": "held", "reason": why}
                else:
                    results[it["id"]] = {"status": "ok",
                                         "bundle": str(bundle),
                                         "out_rel": out_rel(ctx, it,
                                                            name_led, net)}
                print(f"  {'ok  ' if bundle else 'HELD'} {it['route']:6s} "
                      f"{Path(it['rel']).name[:60]}"
                      + (f"  ({why})" if why else ""), flush=True)
    return results, controls, net


def led_has_removals(ctx, led):
    return any(ctx.pack.action_for(s["categories"]) != "KEEP"
               for r in led["subs"].values() for s in r["spans"])


def redact_media_list(ctx, items):
    """Redact mode needs no invented names, but the audio arm finds brands
    through a list of spellings: the brand and person forms detected in the
    dataset, written as a beep-only list (never delivered)."""
    import build_mapping
    ents = collect_entities(ctx, items)
    m = {"brands": [], "people": []}
    for e in ents:
        core = e["name"]
        if e["type"] != "person" and build_mapping.ordinary_word(core):
            continue
        entry = {"pattern": build_mapping.core_regex(core),
                 "replace": "[redacted]", "aliases": e["forms"],
                 "render": "beep"}
        (m["people"] if e["type"] == "person" else m["brands"]).append(entry)
    path = ctx.work / "redact_beep_list.yaml"
    build_mapping.dump(m, path, "beep-only list for redact mode")
    return path


MEDIA_OUT_EXT = (".mp3", ".wav", ".mp4", ".m4a", ".mov", ".flac", ".aac",
                 ".ogg")


def scrub_text(ctx, text, net):
    """A media bundle's side file: every dataset form (and in replace mode
    every mapped name) replaced, line by line."""
    import office
    segs = office.text_segments(text)
    spans = net_spans(net, segs)
    if ctx.em is not None:
        spans += [s for s in office.deterministic(segs, ctx.em)
                  if s["source"] == "mapping"]
    spans = office._dedupe(spans)
    rp = NameReplacer(ctx)
    for sp in sorted(spans, key=lambda s: (s["seg"], -s["a"])):
        new = rp(sp)
        if new is None:
            continue
        a0 = segs[sp["seg"]]["pieces"][0][0]
        text = text[:a0 + sp["a"]] + new + text[a0 + sp["b"]:]
    return text


def deliver(ctx, items, results, net):
    out = ctx.out
    by_id = {it["id"]: it for it in items}
    delivered, held = [], []
    used = set()

    def unique(rel):
        base, k = rel, 2
        while rel.lower() in used:
            p = Path(base)
            rel = str(p.with_name(f"{p.stem} ({k}){p.suffix}"))
            k += 1
        used.add(rel.lower())
        return rel
    for it in items:
        res = results.get(it["id"])
        if ctx.a.media_only and it["route"] not in ("audio", "video") and \
                not it.get("held_reason", "").startswith("media"):
            continue
        if it["route"] == "held":
            held.append({"rel_id": it["id"], "route": it["route"],
                         "reason": it.get("held_reason")})
            continue
        if it.get("container") or res is None:
            continue
        if ctx.a.media_only and it["route"] not in ("audio", "video"):
            continue
        if it["route"] == "zip":
            members = [results.get(m) for m in it["members"]]
            bad = [by_id[m]["member"] for m, r in zip(it["members"], members)
                   if not r or r["status"] != "ok"]
            if bad:
                held.append({"rel_id": it["id"], "route": "zip",
                             "reason": f"{len(bad)} member(s) held"})
                continue
            rel = unique(res["out_rel"])
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
                for m in it["members"]:
                    r = results[m]
                    z.write(r["staged"], r["member_out"])
            delivered.append({"id": it["id"], "out": rel})
            continue
        if res["status"] != "ok":
            held.append({"rel_id": it["id"], "route": it["route"],
                         "reason": res["reason"]})
            continue
        rel = unique(res["out_rel"])
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if it["route"] in ("audio", "video"):
            # the arm's bundle: media, redacted transcript, script, QC,
            # manifest — side files are named by kind (the arm's own names
            # carry the work id) and their text goes through the net
            stem = Path(rel).stem
            for f in Path(res["bundle"]).iterdir():
                if not f.is_file():
                    continue
                low = f.name.lower()
                if f.suffix.lower() in MEDIA_OUT_EXT:
                    name = stem + f.suffix
                elif low.endswith("_redacted.txt"):
                    name = stem + "_transcript.txt"
                elif low.endswith("_script.md"):
                    name = stem + "_script.md"
                elif low.endswith("qc_report.md"):
                    name = stem + "_qc_report.md"
                elif low.endswith("manifest.json"):
                    name = stem + "_manifest.json"
                else:
                    name = stem + "_" + f.suffix.lstrip(".") + f.suffix
                if f.suffix.lower() in (".txt", ".md", ".json"):
                    (dst.parent / name).write_text(
                        scrub_text(ctx, f.read_text(errors="replace"), net))
                else:
                    shutil.copy2(f, dst.parent / name)
        else:
            shutil.copy2(res["staged"], dst)
        delivered.append({"id": it["id"], "out": rel})
    return delivered, held


def report(ctx, items, results, controls, delivered, held, t0, audit=None):
    by_route = {}
    for it in items:
        by_route.setdefault(it["route"], 0)
        by_route[it["route"]] += 1
    acts = {}
    for r in results.values():
        rep = r.get("report") or {}
        reps = rep.values() if isinstance(rep, dict) and all(
            isinstance(v, dict) for v in rep.values()) and rep else [rep]
        for x in reps:
            for k, v in (x.get("by_action") or {}).items():
                acts[k] = acts.get(k, 0) + v
    summary = {
        "when": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": ctx.mode, "taxonomy": ctx.pack.id, "model": ctx.model,
        "files": len([i for i in items if not i.get("container")]),
        "zip_members": len([i for i in items if i.get("container")]),
        "routes": by_route, "delivered": len(delivered), "held": len(held),
        "spans_by_action": acts, "broken_controls": controls,
        "model_calls": ctx.detector.calls,
        "minutes": round((time.time() - t0) / 60, 1),
        "mapping": str(ctx.a.mapping_path) if ctx.a.mapping_path else None,
        "audit": None if not audit else {
            "files": audit["files"], "candidates": len(audit["candidates"]),
            "after_remediation": (len(audit["after"]["candidates"])
                                  if "after" in audit else None)},
    }
    jdump(ctx.work / "report.json", {"summary": summary, "held": held,
                                     "delivered": delivered,
                                     "results": results})
    lines = [f"# Dataset run — {ctx.mode} mode ({ctx.pack.id})", "",
             f"- files: {summary['files']} (+{summary['zip_members']} zip "
             f"members); delivered {len(delivered)}, held {len(held)}",
             f"- routes: {by_route}",
             f"- spans by action: {acts}",
             f"- broken controls (original must fail the check): "
             f"{ {k: v['caught'] for k, v in controls.items()} }",
             f"- leak audit (second model family, sample): "
             f"{summary['audit']}",
             f"- model calls: {ctx.detector.calls}; minutes: "
             f"{summary['minutes']}", "", "## Held", ""]
    ids = {it["id"]: it for it in items}
    for h in held:
        lines.append(f"- `{h['rel_id']}` ({h['route']}): {h['reason']}")
    (ctx.work / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:8]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("--mode", required=True, choices=["redact", "replace"])
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stop-after", choices=["detect", "mapping"])
    ap.add_argument("--mapping", default=None,
                    help="render with this mapping (no new mapping built)")
    ap.add_argument("--context", default="")
    ap.add_argument("--skip-media", action="store_true")
    ap.add_argument("--only-held", action="store_true",
                    help="render again only the files the last run held "
                         "(their ledgers are re-read); other results kept")
    ap.add_argument("--media-only", action="store_true",
                    help="process only the audio and video files (documents "
                         "untouched: e.g. while their own run is going)")
    ap.add_argument("--media-workers", type=int, default=6,
                    help="audio / video files processed at a time")
    ap.add_argument("--no-sweep", dest="sweep", action="store_false",
                    help="audio / video: one free-form detection pass instead "
                         "of the category sweep (sweep is on by default)")
    ap.add_argument("--no-assurance", dest="assurance", action="store_false",
                    help="audio / video: no second listener (on by default: "
                         "Gemini also transcribes, its detections join the "
                         "beeps)")
    ap.add_argument("--redetect", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--processes", type=int, default=6,
                    help="documents rendered at a time, in separate "
                         "processes (1 = threads in this process)")
    ap.add_argument("--thinking", type=int, default=4096,
                    help="thinking-token budget per detection call "
                         "(0 = the model's default)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--image-model", default="google/gemini-3.8-flash")
    ap.add_argument("--audit-files", type=int, default=40,
                    help="documents the leak audit reads (0 = no audit)")
    ap.add_argument("--audit-chars", type=int, default=36000)
    ap.add_argument("--audit-model", default="openai/gpt-5.6-sol")
    a = ap.parse_args()
    src, out, work = (Path(a.input).resolve(), Path(a.out).resolve(),
                      Path(a.work).resolve())
    # usage logs and media caches follow the work folder (set before the
    # pipeline modules read it)
    os.environ["PII_WORK_DIR"] = str(work / "media_cache")
    import common
    a.model = a.model or common.CONFIG["model"]
    if ROOT in out.parents or ROOT in work.parents:
        sys.exit("refusing: outputs and work files hold data; keep them "
                 "outside this repository")
    if out == src or src in out.parents:
        sys.exit("refusing: the delivery folder must not be inside the input")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = Ctx(a)
    a.mapping_path = None
    items = inventory(src, work)
    jdump(work / "inventory.json", items)
    counts = {}
    for it in items:
        counts[it["route"]] = counts.get(it["route"], 0) + 1
    print(f"[stage 0] {len(items)} item(s): {counts}", flush=True)
    if a.only_held:
        # a second pass over the files the last run held (after a fix):
        # they are read and rendered again, every other result is kept
        prev = (jload(work / "report.json", {}) or {}).get("results", {})
        ctx.only_ids = {k for k, r in prev.items()
                        if r.get("status") == "held"}
        a.redetect = True
        print(f"  --only-held: {len(ctx.only_ids)} held file(s) from the "
              f"last run", flush=True)
    # stage 1 skips files whose ledger exists (unless --redetect, which
    # re-reads them; answers come from the text cache)
    stage_detect(ctx, items)
    jdump(work / "inventory.json", items)
    if a.stop_after == "detect":
        return
    if ctx.mode == "replace":
        import entity_map
        if a.mapping:
            a.mapping_path = Path(a.mapping).resolve()
            ctx.em = entity_map.load(a.mapping_path)
            # the mapping only grows: names found in files it has not seen
            # (e.g. spoken in the recordings) are invented and appended
            a.mapping_path = grow_mapping(ctx, items)
        else:
            a.mapping_path = stage_mapping(ctx, items)
            if a.stop_after == "mapping":
                print(f"  mapping -> {a.mapping_path} (paused; resume with "
                      f"--mapping)")
                return
        ctx.em = entity_map.load(a.mapping_path)
        ents = jload(work / "mapping_entities.json", []) or []
        ctx.keep_forms = {f.lower() for e in ents
                          if (e.get("decision") or {}).get("keep")
                          for f in e["forms"]}
        os.environ["PII_MAPPING_FILE"] = str(a.mapping_path)
    ctx.codes = jload(work / "codebook.json", {}) or {}
    ctx.codes = {k: {tuple(json.loads(kk)): v for kk, v in d.items()}
                 for k, d in ctx.codes.items()}
    prior = None
    if a.only_held:
        prior = (jload(work / "report.json", {}) or {}).get("results", {})
    results, controls, _net = stage_render(
        ctx, items, only=ctx.only_ids, prior=prior)
    audit = None
    if a.audit_files > 0:
        audit = run_audit(ctx, items, results)
        cands = audit["candidates"]
        print(f"  audit: {len(cands)} leak candidate(s) in {audit['files']} "
              f"file(s)", flush=True)
        # what the audit flags is replaced even if naming had kept it
        ctx.keep_forms -= {c["text"].lower() for c in cands}
        if cands and ctx.mode == "replace":
            # a missed real brand or person must get an invented name: the
            # audit's names join the mapping (one naming call) and the dataset
            # is rendered again
            extra_names = [c for c in cands if c["kind"] in (
                "person", "brand", "product", "dealer", "agency")]
            if extra_names:
                a.mapping_path = extend_mapping(ctx, extra_names)
                ctx.em = entity_map.load(a.mapping_path)
        extra = {}
        rounds = 0
        while cands and rounds < 2:
            rounds += 1
            new = {c["text"]: [KIND_CAT.get(c["kind"], "OTHER")]
                   for c in cands if c["text"] not in extra}
            if not new:
                break
            extra.update(new)
            jdump(work / "audit_forms.json", extra)
            print(f"  re-rendering with the audit's findings (round "
                  f"{rounds})", flush=True)
            # only files whose output contains a new finding are rendered
            # again; everything else keeps its verified result
            firsts = {t for f in new for t in re.findall(r"\w+", f.lower())[:1]}
            affected = {k for k, r in results.items()
                        if firsts & set(r.get("tokens") or ())}
            print(f"  {len(affected)} file(s) contain a finding", flush=True)
            results, controls, _net = stage_render(
                ctx, items, extra, only=affected, prior=results)
            audit["after"] = run_audit(ctx, items, results)
            cands = audit["after"]["candidates"]
            ctx.keep_forms -= {c["text"].lower() for c in cands}
            print(f"  audit after remediation: {len(cands)} candidate(s)",
                  flush=True)
            if cands and ctx.mode == "replace":
                more = [c for c in cands if c["kind"] in (
                    "person", "brand", "product", "dealer", "agency")]
                if more:
                    a.mapping_path = extend_mapping(ctx, more)
                    ctx.em = entity_map.load(a.mapping_path)
        jdump(work / "audit.json", audit)
    jdump(work / "codebook.json",
          {k: {json.dumps(list(kk)): v for kk, v in d.items()}
           for k, d in ctx.codes.items()})
    if out.exists() and a.mapping is None and not a.stop_after:
        pass
    delivered, held = deliver(ctx, items, results, _net)
    report(ctx, items, results, controls, delivered, held, t0, audit)


if __name__ == "__main__":
    main()
