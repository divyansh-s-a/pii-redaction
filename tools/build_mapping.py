#!/usr/bin/env python3
"""Build a study's name mapping automatically, for --replace pseudonym.

    python3 tools/build_mapping.py INPUT [INPUT ...] --out MAPPING.yaml
        [--context "what the study is about"] [--context-file BRIEF.txt]
        [--seed EARLIER.yaml] [--reconcile OTHER.yaml] [--no-assurance]
    python3 tools/build_mapping.py --reconcile-only GENERATED.yaml OTHER.yaml \
        --out MAPPING.yaml

1. inputs (files or folders, walked): audio and video are prepared and
   transcribed by both listeners (stages A-B, cached exactly where the
   redaction will look, so nothing is transcribed twice); a video adds the
   text on one frame per shot (Apple Vision OCR, local). Documents add their
   text — docx/pptx/xlsx/xlsm (text, comments, notes, sheet names,
   properties), pdf, doc/rtf (macOS textutil), csv/txt and survey-tabulation
   text files, images (OCR), zip members — and every file and folder NAME
   (they often carry the brand). Numbers-only tokens are dropped.
2. discovery: a model lists the brands, products and models, dealers,
   banks, platforms, agencies, slogans, endorsers and staff in the
   transcripts, with every spelling actually heard (Devanagari and
   mis-hearings included — spellings not found in the text are dropped).
3. naming: a second call decides replace / keep / beep by the project's
   rules (client, competitors, their products and dealers, the agency and
   identifying slogans are replaced; platforms and places kept; people
   beeped) and invents names for the replaced ones.
4. the result is checked by entity_map (an invented name that a real
   pattern matches, or that sounds the same as a real form, is refused and
   re-invented) and written in the mapping YAML format.
5. --seed EARLIER: names from an earlier mapping (another study of the same
   project) win for every brand it shares, and its brands and agency are
   carried over — one invented name per brand across studies; its
   study-specific facts and staff are not.
   --reconcile OTHER: the same for a mapping of the SAME material (e.g. the
   one a project uses for its documents): its names win and ALL its entries
   are added. Changing only invented names never re-detects: a re-run
   re-renders.

The mapping holds real names: it must live outside this repository.
"""
import argparse
import datetime
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import common        # noqa: E402
import detect_pii    # noqa: E402
import entity_map    # noqa: E402
import run_pipeline  # noqa: E402

DISCOVER = (ROOT / "prompts" / "entity_discovery_v1.txt").read_text()
NAMING = (ROOT / "prompts" / "entity_naming_v1.txt").read_text()
TYPES = ["brand", "product", "model", "dealer", "bank_financial", "platform",
         "agency", "slogan", "endorser", "staff", "organisation"]
VIDEO_EXT = {"mp4", "mov", "m4v", "avi", "mkv", "webm"}
AUDIO_EXT = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "opus", "wma"}
OFFICE_EXT = {"docx", "pptx", "xlsx", "xlsm", "docm", "pptm"}
TEXT_EXT = {"csv", "txt", "asc", "qdf", "qtf", "stpn", "qpsmran", "tsv",
            "json", "xml", "md"}
IMAGE_EXT = {"jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff"}
DOC_CAP = 60000               # characters of text kept per document
CHUNK_WORDS = 5000
DISC_SCHEMA = {
    "type": "OBJECT",
    "properties": {"entities": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "type": {"type": "STRING", "enum": TYPES},
            "forms": {"type": "ARRAY", "items": {"type": "STRING"}},
            "role": {"type": "STRING",
                     "enum": ["client", "competitor", "other"]},
            "mentions": {"type": "INTEGER"}},
        "required": ["name", "type", "forms", "role"]}}},
    "required": ["entities"],
}
NAME_SCHEMA = {
    "type": "OBJECT",
    "properties": {"decisions": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "decision": {"type": "STRING",
                         "enum": ["replace", "keep", "beep"]},
            "core": {"type": "STRING"},
            "invented": {"type": "STRING"},
            "category": {"type": "STRING"}},
        "required": ["id", "decision"]}}},
    "required": ["decisions"],
}


# ------------------------------------------------------------- 1. inputs

def media_audio(path: Path):
    """(audio path, on-screen text) for a media file. A video's track is
    extracted where run_video extracts it, so the caches are shared."""
    if path.suffix.lower().lstrip(".") not in VIDEO_EXT:
        return path, ""
    sys.path.insert(0, str(ROOT / "multimodal" / "video_pipeline"))
    import media_io
    fid = run_pipeline.file_id_for(path)
    work = common.OUTPUT_ROOT / "video" / fid
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{fid}.wav"
    if not wav.exists():
        media_io.extract_audio(path, wav)
    return wav, onscreen_text(path, work)


def onscreen_text(video: Path, work: Path):
    """Latin text on one frame per shot (Apple Vision, local, no cost)."""
    import frame_detect
    import media_io
    import shots as shots_mod
    info = media_io.probe(video)
    sh = shots_mod.analyze(media_io.frames(video), info["fps"])
    mids = {(s["f0"] + s["f1"]) // 2 for s in sh}
    paths = []
    for k, fr in media_io.frames(video):
        if k in mids:
            paths.append(media_io.save_frame(fr, work / f"ocr_{k:06d}.jpg"))
    try:
        reads = frame_detect.read_frames(paths)
    except RuntimeError as exc:
        print(f"  on-screen text unavailable: {str(exc)[:80]}")
        return ""
    lines = [ln["text"] for r in reads for ln in r["lines"]]
    return "\n".join(dict.fromkeys(lines))


def transcript_text(tr_path: Path):
    t = common.load_json(tr_path)
    return "\n".join(f"{u['speaker']}: " + " ".join(w["w"] for w in u["words"])
                     for u in t["utterances"])


PAIRS = []        # (primary transcript, second-listener transcript) paths


def transcripts(audio: Path, model: str, assurance: bool):
    """Both listeners' transcripts as text, from the shared caches."""
    fid, _sfx, out, tr, work, chw = run_pipeline.stage_ab(
        audio, model, set(), "elevenlabs", None)
    texts = [transcript_text(tr)]
    if assurance:
        adir = out / "assurance"
        adir.mkdir(exist_ok=True)
        atr = adir / "transcript.json"
        if not atr.exists():
            import gemini_transcribe
            print("  [assurance] gemini transcription ...")
            gemini_transcribe.transcribe_file(model, audio, work, atr,
                                              channel_wavs=chw)
        texts.append(transcript_text(atr))
        PAIRS.append((tr, atr))
    return texts


def _strip_xml(xml: str):
    xml = re.sub(r"</w:p>|</a:p>|<w:tab/>|<w:br/>|</si>|</c>", "\n", xml)
    return re.sub(r"<[^>]+>", " ", xml)


def office_text(blob_or_path):
    """Text of an OOXML file: body, comments, notes, headers, sheet names
    and shared strings, document properties."""
    import html
    import zipfile
    out = []
    with zipfile.ZipFile(blob_or_path) as z:
        for n in z.namelist():
            if not n.endswith(".xml"):
                continue
            if re.search(r"(document|comments|footnotes|endnotes|header\d*|"
                         r"footer\d*|slide\d+|notesSlide\d+|sharedStrings|"
                         r"sheet\d+|workbook|core|app|chart\d+)\.xml$", n):
                out.append(html.unescape(_strip_xml(
                    z.read(n).decode("utf-8", "ignore"))))
    return "\n".join(out)


def doc_text(path: Path):
    """Text of one document by type ('' when unsupported)."""
    ext = path.suffix.lower().lstrip(".")
    try:
        if ext in OFFICE_EXT:
            return office_text(path)
        if ext == "pdf":
            import fitz
            with fitz.open(path) as d:
                return "\n".join(pg.get_text() for pg in d)
        if ext in ("doc", "rtf"):
            import subprocess
            r = subprocess.run(["textutil", "-convert", "txt", "-stdout",
                                str(path)], capture_output=True, text=True)
            return r.stdout
        if ext in TEXT_EXT:
            raw = path.read_bytes()[:400000]
            return raw.decode("utf-8", "ignore") if raw.isascii() or \
                b"\xef\xbb\xbf" in raw[:3] else raw.decode("latin-1")
        if ext in IMAGE_EXT:
            import frame_detect
            r = frame_detect.vision_batch([path], "text")[0]
            return "\n".join(ln["text"] for ln in r.get("text", []))
        if ext == "zip":
            import io
            import zipfile
            out = []
            with zipfile.ZipFile(path) as z:
                for n in z.namelist():
                    e = n.rsplit(".", 1)[-1].lower()
                    out.append(n)
                    if e in OFFICE_EXT:
                        out.append(office_text(io.BytesIO(z.read(n))))
                    elif e in TEXT_EXT:
                        out.append(z.read(n)[:200000].decode("latin-1"))
            return "\n".join(out)
    except Exception as exc:
        print(f"  could not read {path.name}: {str(exc)[:80]}")
    return ""


def tidy(text: str):
    """Drop numbers-only tokens (tab books are mostly figures) and cap."""
    toks = [t for t in text.split() if not re.fullmatch(r"[\d.,%()\-+/:]+",
                                                        t)]
    return " ".join(toks)[:DOC_CAP]


def walk(inputs):
    """Expand folders; -> (media files, document files, path names)."""
    media, docs, names = [], [], []
    for i in inputs:
        p = Path(i).resolve()
        files = ([p] if p.is_file() else
                 sorted(f for f in p.rglob("*") if f.is_file()
                        and not f.name.startswith(".")))
        if p.is_dir():
            names += [str(d.relative_to(p.parent)) for d in p.rglob("*")
                      if d.is_dir()] + [p.name]
        for f in files:
            names.append(f.name)
            e = f.suffix.lower().lstrip(".")
            if e in AUDIO_EXT or e in VIDEO_EXT:
                media.append(f)
            else:
                docs.append(f)
    return media, docs, names


# ---------------------------------------------------------- 2. discovery

def chunks(text):
    words = text.split()
    for i in range(0, len(words), CHUNK_WORDS):
        yield " ".join(words[i:i + CHUNK_WORDS])


def discover(model, texts, context):
    found = []
    for t in texts:
        for ch in chunks(t):
            raw = json.loads(common.generate(model, [{"text": (
                DISCOVER + "\n\nSTUDY CONTEXT: " + (context or "(none)")
                + "\n\nTEXT:\n" + ch)}], DISC_SCHEMA))
            found += raw.get("entities", [])
    return found


def merge(found, corpus):
    """One group per entity: spellings unioned (only spellings that really
    occur in the material), mentions summed, the most specific role kept."""
    low = corpus.lower()
    groups = []
    for e in found:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        forms = [f.strip() for f in e.get("forms") or [] if f.strip()
                 and f.strip().lower() in low]
        keys = {entity_map.key(x) for x in [name] + forms
                if len(entity_map.key(x)) >= 3}
        nk = entity_map.key(name)
        for g in groups:
            if g["keys"] & keys or (len(nk) >= 4 and difflib.SequenceMatcher(
                    None, nk, entity_map.key(g["name"])).ratio() >= 0.9):
                g["forms"] |= set(forms)
                g["keys"] |= keys
                g["mentions"] += int(e.get("mentions") or 1)
                if e.get("role") in ("client", "competitor"):
                    g["role"] = (e["role"] if g["role"] == "other"
                                 else g["role"])
                break
        else:
            groups.append({"name": name, "type": e.get("type", "brand"),
                           "forms": set(forms), "keys": keys,
                           "role": e.get("role", "other"),
                           "mentions": int(e.get("mentions") or 1)})
    return groups


def link_listeners(groups):
    """Merge entities the two listeners heard at the SAME moment: where the
    primary heard "Doorways" and the second listener, aligned word for word,
    heard the brand, they are one brand and must get one invented name."""
    import align_listeners
    keyed = [{entity_map.key(f) for f in g["forms"] | {g["name"]}
              if len(entity_map.key(f)) >= 3} for g in groups]

    def which(words, i):
        for size in (1, 2, 3):
            k = "".join(entity_map.key(w["w"]) for w in words[i:i + size])
            for gi, ks in enumerate(keyed):
                if k in ks:
                    return gi
        return None
    parent = list(range(len(groups)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    pair_links = {}
    for tr, atr in PAIRS:
        P = detect_pii.flatten_words(common.load_json(tr))
        S = detect_pii.flatten_words(common.load_json(atr))
        amap = align_listeners.align(P, S)
        back = {}
        for j, (_s, _e, idx) in amap.items():
            for i in idx:
                back.setdefault(i, []).append(j)
        for i in range(len(P)):
            gp = which(P, i)
            if gp is None:
                continue
            for j in back.get(i, []):
                gs = which(S, j)
                if gs is not None and gs != gp:
                    pr = tuple(sorted((gp, gs)))
                    pair_links[pr] = pair_links.get(pr, 0) + 1
    # one misaligned word must not merge two brands: two shared moments, or
    # one side a rare spelling (a mis-hearing is typically heard once or twice)
    links = 0
    for (a, b), n in sorted(pair_links.items(), key=lambda kv: -kv[1]):
        if n < 2 and min(groups[a]["mentions"], groups[b]["mentions"]) > 2:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        keep, drop = ((ra, rb) if groups[ra]["mentions"]
                      >= groups[rb]["mentions"] else (rb, ra))
        parent[drop] = keep
        links += 1
    out = {}
    for gi, g in enumerate(groups):
        r = find(gi)
        if r not in out:
            out[r] = dict(groups[r], forms=set(groups[r]["forms"]),
                          keys=set(groups[r]["keys"]))
        if gi != r:
            out[r]["forms"] |= g["forms"] | {g["name"]}
            out[r]["keys"] |= g["keys"]
            out[r]["mentions"] += g["mentions"]
    if links:
        print(f"  {links} cross-listener link(s): {len(groups)} -> "
              f"{len(out)} entities")
    return list(out.values())


# ------------------------------------------------------------- 3. naming

def name_entities(model, groups, context, avoid=()):
    listing = "\n".join(
        f"N{k}\t{g['type']}\t{g['role']}\t{g['name']}\tforms: "
        + "; ".join(sorted(g["forms"])[:8]) + f"\tmentions: {g['mentions']}"
        for k, g in enumerate(groups, 1))
    extra = ("\n\nDo NOT use these names (taken or too close to a real "
             "one): " + "; ".join(avoid)) if avoid else ""
    raw = json.loads(common.generate(model, [{"text": (
        NAMING + "\n\nSTUDY CONTEXT: " + (context or "(none)")
        + "\n\nENTITIES (id, type, role, name, forms, mentions):\n" + listing
        + extra)}], NAME_SCHEMA))
    out = {}
    for d in raw.get("decisions", []):
        m = re.fullmatch(r"N(\d+)", str(d.get("id", "")).strip())
        if m and 1 <= int(m.group(1)) <= len(groups):
            out[int(m.group(1)) - 1] = d
    return out


def core_regex(core: str):
    toks = core.split()
    body = r"\s*".join(re.escape(t) for t in toks)
    if core.isascii():
        return r"\b" + body + r"\b"
    return body


SECTION = {"SLOGAN": "brand_facts", "AGENCY": "agency"}
_WORDS = None


def ordinary_word(core: str):
    """A one-word core that is also an ordinary word (the system word list,
    plural folded): matched only where the context shows the brand."""
    global _WORDS
    if len(core.split()) != 1:
        return False
    if _WORDS is None:
        try:
            _WORDS = {w.strip().lower() for w in
                      open("/usr/share/dict/words", errors="ignore")}
        except OSError:
            _WORDS = set()
    w = core.lower().strip(".,'’")
    return w in _WORDS or (w.endswith("s") and w[:-1] in _WORDS)


def build(groups, decisions):
    """Mapping dict in the project's YAML format."""
    m = {"brands": [], "abbreviations": [], "brand_facts": [], "agency": [],
         "people": [], "keep": []}
    for k, g in enumerate(groups):
        d = decisions.get(k) or {"decision": "keep"}
        kind = ("beep" if g["type"] in ("staff", "endorser")
                else d.get("decision", "keep"))
        if kind == "keep":
            m["keep"].append(f"{g['name']} ({g['type']}, kept)")
            continue
        if kind == "beep":
            m["people"].append({"pattern": core_regex(g["name"]),
                                "replace": ("colleague" if g["type"] == "staff"
                                            else "a well-known person"),
                                "aliases": sorted(g["forms"])})
            continue
        core = (d.get("core") or g["name"]).strip()
        inv = (d.get("invented") or "").strip()
        if not inv:
            m["keep"].append(f"{g['name']} (no invented name returned)")
            continue
        n = len(core.split())
        aliases = sorted(f for f in g["forms"] if len(f.split()) <= n)
        cat = (d.get("category") or "BRAND").upper()
        entry = {"pattern": core_regex(core), "replace": inv,
                 "aliases": aliases, "category": cat,
                 "name": core}
        if ordinary_word(core):
            entry["context_only"] = True
        sec = SECTION.get(cat, "brands")
        if len(core) <= 3 and core.isupper():
            sec = "abbreviations"
        m[sec].append(entry)
    return {k: _merge_same(v) if k != "keep" else v
            for k, v in m.items() if v}


def _merge_same(entries):
    """One entry per invented name: spellings found as separate entities
    (a mis-heard brand) that got the same invented name are combined."""
    out, by = [], {}
    for e in entries:
        k = (e.get("replace"), e.get("category"))
        if k in by and e.get("replace"):
            m = by[k]
            if e["pattern"] not in m["pattern"]:
                m["pattern"] = m["pattern"] + "|" + e["pattern"]
            m["aliases"] = sorted(set(m["aliases"]) | set(e["aliases"]))
            m["context_only"] = (m.get("context_only", False)
                                 and e.get("context_only", False)) or None
            if not m["context_only"]:
                m.pop("context_only", None)
        else:
            by[k] = dict(e, aliases=list(e.get("aliases") or []))
            out.append(by[k])
    return out


def dump(mapping, path: Path, note):
    import yaml
    head = ("# Name mapping — generated by tools/build_mapping.py on "
            f"{datetime.date.today().isoformat()}.\n# {note}\n"
            "# Holds REAL names: keep it out of any repository.\n")
    path.write_text(head + yaml.safe_dump(mapping, allow_unicode=True,
                                          sort_keys=False, width=1000))


def validated(model, groups, decisions, path, note, context, tries=2):
    """Write and load (entity_map refuses an invented name a real pattern
    matches, or one that sounds the same as a real form); re-invent the
    offending names, then make any survivor distinct mechanically."""
    for attempt in range(tries + 1):
        mapping = build(groups, decisions)
        dump(mapping, path, note)
        try:
            entity_map.load(path)
            return mapping
        except ValueError as exc:
            bad = re.findall(r"invented name '([^']+)'", str(exc))
            print(f"  re-inventing {len(bad)} name(s): mapping check failed")
            if attempt < tries:
                fresh = name_entities(model, groups, context, avoid=bad)
                for k, d in decisions.items():
                    if (d.get("invented") or "") in bad and k in fresh:
                        d["invented"] = fresh[k].get("invented") or d["invented"]
            else:
                for d in decisions.values():
                    if (d.get("invented") or "") in bad:
                        d["invented"] = d["invented"] + "o"
    mapping = build(groups, decisions)
    dump(mapping, path, note)
    entity_map.load(path)          # raises if still not valid
    return mapping


# ---------------------------------------------------------- 5. reconcile

def _forms(entry):
    return ([entry.get("name")] if entry.get("name") else []) \
        + entity_map.literals(str(entry.get("pattern", ""))) \
        + [str(a) for a in entry.get("aliases") or []]


def reconcile(gen: dict, other: dict, add_sections=None):
    """Adopt OTHER's invented names for entities it covers; add its entries
    that are missing here (only from add_sections, when given). Returns
    (mapping, report)."""
    o_entries = []
    for sec, items in other.items():
        if not isinstance(items, list):
            continue
        for i, it in enumerate(items):
            if isinstance(it, dict) and "pattern" in it:
                flags = 0 if sec == "abbreviations" else re.I
                o_entries.append({"sec": sec, "i": i, "item": it,
                                  "rx": re.compile(str(it["pattern"]), flags)})
    used, adopted = set(), 0
    for sec, items in gen.items():
        if not isinstance(items, list):
            continue
        for e in items:
            if not isinstance(e, dict):
                continue
            g_rx = re.compile(str(e["pattern"]),
                              0 if sec == "abbreviations" else re.I)
            for k, o in enumerate(o_entries):
                hit = any(o["rx"].search(f) for f in _forms(e) if f) or any(
                    g_rx.search(f) for f in _forms(o["item"]) if f)
                if not hit:
                    continue
                used.add(k)
                if o["item"].get("replace") and sec != "people":
                    if e.get("replace") != o["item"]["replace"]:
                        adopted += 1
                    e["replace"] = o["item"]["replace"]
                    e["reconciled_with"] = f"{o['sec']}:{o['i']}"
                break
    added = 0
    for k, o in enumerate(o_entries):
        if k in used or (add_sections is not None
                         and o["sec"] not in add_sections):
            continue
        item = dict(o["item"], from_reconcile=f"{o['sec']}:{o['i']}")
        gen.setdefault(o["sec"], []).append(item)
        added += 1
    keep = other.get("keep") or []
    if keep:
        gen.setdefault("keep", [])
        gen["keep"] += [k for k in keep if isinstance(k, str)
                        and k not in gen["keep"]]
    return gen, {"entries_matched": len(used), "names_adopted": adopted,
                 "entries_added_from_other": added}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*",
                    help="audio/video/document files or folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--context-file", default=None)
    ap.add_argument("--seed", default=None,
                    help="earlier mapping whose names win for shared brands "
                         "(its brands and agency are carried over)")
    ap.add_argument("--reconcile", default=None)
    ap.add_argument("--reconcile-only", nargs=2, default=None,
                    metavar=("GENERATED", "OTHER"))
    ap.add_argument("--no-assurance", action="store_true")
    a = ap.parse_args()
    import yaml
    out = Path(a.out).resolve()
    if ROOT in out.parents:
        sys.exit("refusing: a mapping holds real names; write it outside "
                 "this repository")
    if a.reconcile_only:
        gen = yaml.safe_load(Path(a.reconcile_only[0]).read_text()) or {}
        other = yaml.safe_load(Path(a.reconcile_only[1]).read_text()) or {}
        gen, rep = reconcile(gen, other)
        dump(gen, out, f"reconciled with {Path(a.reconcile_only[1]).name}")
        entity_map.load(out)
        print(json.dumps(rep))
        return
    if not a.inputs:
        ap.error("no inputs given")
    context = a.context
    if a.context_file:
        context += "\n" + Path(a.context_file).read_text()[:8000]
    model = common.resolve_model()
    media, docs, names = walk(a.inputs)
    texts, corpus = [], ""
    for p in media:
        print(f"\n=== {p.name} ===")
        audio, screen = media_audio(p)
        t = transcripts(audio, model, not a.no_assurance)
        if screen:
            t.append("ON-SCREEN TEXT:\n" + screen)
        texts += t
        corpus += "\n".join(t) + "\n"
    sys.path.insert(0, str(ROOT / "multimodal" / "document_pipeline"))
    for p in docs:
        t = tidy(doc_text(p))
        if t:
            texts.append(f"DOCUMENT {p.name}:\n{t}")
            corpus += t + "\n"
    if names:
        listing = "\n".join(dict.fromkeys(names))
        texts.append("FILE AND FOLDER NAMES:\n" + listing)
        corpus += listing + "\n"
    print(f"\n  inputs: {len(media)} audio/video, {len(docs)} document(s), "
          f"{len(names)} names")
    common.set_usage_context("build_mapping")
    try:
        found = discover(model, texts, context)
        groups = link_listeners(merge(found, corpus))
        print(f"\n  discovered {len(found)} mention record(s) -> "
              f"{len(groups)} entit(y/ies)")
        decisions = name_entities(model, groups, context)
        note = (f"{len(media)} audio/video and {len(docs)} document "
                f"file(s); {len(groups)} entities found")
        mapping = validated(model, groups, decisions, out, note, context)
    finally:
        common.set_usage_context("")
    report = {"media": len(media), "documents": len(docs),
              "entities": len(groups),
              "replaced": sum(len(mapping.get(s, [])) for s in
                              ("brands", "abbreviations", "brand_facts",
                               "agency")),
              "people_beeped": len(mapping.get("people", [])),
              "kept": len(mapping.get("keep", []))}
    if a.seed:
        seed = yaml.safe_load(Path(a.seed).read_text()) or {}
        mapping, rep_s = reconcile(mapping, seed, add_sections=(
            "brands", "abbreviations", "agency"))
        note += f"; seeded from {Path(a.seed).name}"
        dump(mapping, out, note)
        entity_map.load(out)
        report["seed"] = rep_s
    if a.reconcile:
        other = yaml.safe_load(Path(a.reconcile).read_text()) or {}
        mapping, rep = reconcile(mapping, other)
        dump(mapping, out, f"{note}; reconciled with {Path(a.reconcile).name}")
        entity_map.load(out)
        report["reconcile"] = rep
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
