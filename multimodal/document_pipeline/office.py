#!/usr/bin/env python3
"""Office and text documents — read, detect, redact or replace, verify.

Formats: docx/docm, pptx/pptm, xlsx/xlsm (OOXML, edited IN PLACE in their
own XML so everything not touched stays byte-identical), .doc/.rtf
(converted to .docx by macOS textutil), csv/tsv/txt and survey-tabulation
text files (qdf, qtf, asc, stpn, qpsmran), zip archives (members processed
recursively).

What is read: paragraph text (runs joined, so a name split across runs is
still one span), table cells, shared strings and inline strings, sheet
names, comments and their authors, speaker notes, headers and footers, chart
labels, hyperlink targets, document properties (author, title, company...),
embedded documents.

Detection: the taxonomy pack's prompt over a numbered word stream (the same
word-index contract as the PDF and audio arms) plus deterministic patterns
(emails, phone numbers) and, in Replace mode, the study mapping's matcher.

Rendering, per the pack's action for each span:
  KEEP                     untouched
  Redact mode              every destructive action -> a black bar (█…)
  Replace mode PSEUDONYMIZE  the mapping's invented name (people and brands);
                             IDs that link records -> a consistent code
  Replace mode other       black bar (contact details, addresses, IDs)
Images inside a file are classified and greyed when they show a brand or a
person (pack row BRAND_IMAGE / FACE_PHOTOGRAPH) — in both modes.
"""
import html
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
sys.path.insert(0, str(HERE.parent / "taxonomies"))

OOXML = {"docx", "docm", "pptx", "pptm", "xlsx", "xlsm"}
TEXTISH = {"csv", "tsv", "txt", "asc", "qdf", "qtf", "stpn", "qpsmran"}
CONVERT = {"doc", "rtf"}
IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"}
BAR = "█"
EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RX = re.compile(r"(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d.])"
                      r"|(?<![\d.])\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}")
ID_CATS = {"ASSIGNED_ID", "RECORD_LOCATOR_ID"}
import threading                                      # noqa: E402
CODE_LOCK = threading.Lock()
PERSON_CATS = {"FULL_NAME", "NAME_IN_FREE_TEXT", "IN_CELL_PERSONAL_VALUE"}

# ------------------------------------------------------------- segments
# A segment is one editable piece of text: (part, kind, locator, text,
# pieces). `pieces` are the (start, end) offsets of the XML text nodes that
# make up the segment inside the part's string, so a replacement can be
# written back without touching any other byte.

PARA_RX = {"w": re.compile(r"<w:p[ >].*?</w:p>", re.S),
           "a": re.compile(r"<a:p>.*?</a:p>|<a:p .*?</a:p>", re.S)}
NODE_RX = {"w": re.compile(r"(<w:(?:t|delText)(?: [^>]*)?>)(.*?)"
                           r"(</w:(?:t|delText)>)", re.S),
           "a": re.compile(r"(<a:t>|<a:t [^>]*>)(.*?)(</a:t>)", re.S)}
# SpreadsheetML elements may carry a namespace prefix (<x:si>, <x:t>): some
# exporters write every element that way
SI_RX = re.compile(r"<(?:\w+:)?si>.*?</(?:\w+:)?si>", re.S)
T_RX = re.compile(r"(<(?:\w+:)?t(?: [^>]*)?>)(.*?)(</(?:\w+:)?t>)", re.S)
IS_RX = re.compile(r"<(?:\w+:)?is>.*?</(?:\w+:)?is>", re.S)
STRCACHE_RX = re.compile(r"<c:strCache>.*?</c:strCache>", re.S)
CV_RX = re.compile(r"(<c:v>)(.*?)(</c:v>)", re.S)
SHEET_RX = re.compile(r'(<sheet [^>]*?name=")([^"]*)(")')
AUTHOR_RX = re.compile(r'((?:w:author|w:initials|author)=")([^"]*)(")')
PROP_RX = re.compile(r"(<(dc:creator|cp:lastModifiedBy|dc:title|dc:subject|"
                     r"cp:keywords|dc:description|Company|Manager|"
                     r"cp:category)(?: [^>]*)?>)(.*?)(</\2>)", re.S)
# link targets: web, mail, and local or network file paths (external links
# often carry the folder a file was saved from)
TARGET_RX = re.compile(r'(Target=")((?:https?://|mailto:|file:)[^"]*|'
                       r'[^"]*\\[^"]*)(")')
# hidden copies Office keeps: Excel cell comments and threaded comments,
# comment-author lists, PowerPoint legacy comments, pivot caches
COMMENT_RX = re.compile(r"<comment [^>]*>.*?</comment>", re.S)
TEXT_RX = re.compile(r"(<(?:text|p:text)>)(.*?)(</(?:text|p:text)>)", re.S)
NAMEATTR_RX = re.compile(r'((?<![\w:])(?:displayName|name|initials|userId)'
                         r'|w15:userId)="([^"]*)(")')
PIVOT_S_RX = re.compile(r'(<s v=")([^"]*)(")')
LPSTR_RX = re.compile(r"(<vt:lpw?str>)(.*?)(</vt:lpw?str>)", re.S)
AUTHOR_EL_RX = re.compile(r"(<author>)(.*?)(</author>)", re.S)
ABSPATH_RX = re.compile(r'(<x15ac:absPath [^>]*?url=")([^"]*)(")')
EXT_V_RX = re.compile(r"(<v>)(.*?)(</v>)", re.S)
NAMEATTR_W_RX = re.compile(r'((?:w15:userId|w15:author)=")([^"]*)(")')
STRV_RX = re.compile(r'<(?:\w+:)?c\b[^>]*?\bt="str"[^>]*>'
                     r'(?:(?!</(?:\w+:)?c>).)*?<(?:\w+:)?v>(.*?)</(?:\w+:)?v>',
                     re.S)
HF_RX = re.compile(r"(<(?:\w+:)?(?:oddHeader|oddFooter|evenHeader|evenFooter|"
                   r"firstHeader|firstFooter)>)(.*?)(</)", re.S)


def _nodes_in(xml, start, end, rx):
    for m in rx.finditer(xml, start, end):
        yield m.start(2), m.end(2)


def segments_of(part, xml):
    """Editable text segments of one OOXML part."""
    segs = []

    def add(kind, pieces, **extra):
        text = "".join(html.unescape(xml[a:b]) for a, b in pieces)
        if text.strip():
            segs.append(dict({"part": part, "kind": kind, "pieces": pieces,
                              "text": text}, **extra))
    if part.startswith(("word/", "ppt/")):
        ns = "w" if part.startswith("word/") else "a"
        for pm in PARA_RX[ns].finditer(xml):
            add("para", list(_nodes_in(xml, pm.start(), pm.end(),
                                       NODE_RX[ns])))
        if ns == "w":
            for m in AUTHOR_RX.finditer(xml):
                add("author", [(m.start(2), m.end(2))])
            if part.endswith("people.xml"):
                for m in NAMEATTR_RX.finditer(xml):
                    add("author", [(m.start(2), m.end(2))])
    if part.startswith("xl/"):
        for k, sm in enumerate(SI_RX.finditer(xml)):
            add("shared", list(_nodes_in(xml, sm.start(), sm.end(), T_RX)),
                si=k)
        for im in IS_RX.finditer(xml):
            add("inline", list(_nodes_in(xml, im.start(), im.end(), T_RX)))
        if part.endswith("workbook.xml"):
            for m in SHEET_RX.finditer(xml):
                add("sheet_name", [(m.start(2), m.end(2))])
            # the folder the workbook was saved from (often names the client
            # and the agency)
            for m in ABSPATH_RX.finditer(xml):
                add("property", [(m.start(2), m.end(2))])
        for m in AUTHOR_RX.finditer(xml):
            add("author", [(m.start(2), m.end(2))])
        if part.startswith("xl/comments"):
            for m in AUTHOR_EL_RX.finditer(xml):
                add("author", [(m.start(2), m.end(2))])
            for cm in COMMENT_RX.finditer(xml):
                add("comment", list(_nodes_in(xml, cm.start(), cm.end(),
                                              T_RX)))
        if part.startswith("xl/threadedComments/"):
            for m in TEXT_RX.finditer(xml):
                add("comment", [(m.start(2), m.end(2))])
        if part.startswith("xl/persons/"):
            for m in NAMEATTR_RX.finditer(xml):
                add("author", [(m.start(2), m.end(2))])
        if part.startswith("xl/externalLinks/"):
            # values cached from the linked workbook
            for m in EXT_V_RX.finditer(xml):
                if any(ch.isalpha() for ch in html.unescape(m.group(2))):
                    add("external_value", [(m.start(2), m.end(2))])
        if part.startswith("xl/pivotCache/"):
            for m in PIVOT_S_RX.finditer(xml):
                add("pivot", [(m.start(2), m.end(2))])
        if re.match(r"xl/worksheets/sheet\d*\.xml$", part):
            # a formula's cached text result, and page headers / footers
            if 't="str"' in xml:
                for m in STRV_RX.finditer(xml):
                    add("formula_text", [(m.start(1), m.end(1))])
            for m in HF_RX.finditer(xml):
                add("header_footer", [(m.start(2), m.end(2))])
        if "/drawings/" in part:
            for pm in PARA_RX["a"].finditer(xml):
                add("para", list(_nodes_in(xml, pm.start(), pm.end(),
                                           NODE_RX["a"])))
    if part.startswith("ppt/"):
        if part.endswith("commentAuthors.xml") or "/authors" in part:
            for m in NAMEATTR_RX.finditer(xml):
                add("author", [(m.start(2), m.end(2))])
        if part.startswith("ppt/comments/"):
            for m in TEXT_RX.finditer(xml):
                add("comment", [(m.start(2), m.end(2))])
            for m in AUTHOR_RX.finditer(xml):
                add("author", [(m.start(2), m.end(2))])
    if "/charts/" in part:
        # every cached chart value with letters (string, multi-level and
        # literal caches alike)
        for m in CV_RX.finditer(xml):
            if any(ch.isalpha() for ch in html.unescape(m.group(2))):
                add("chart", [(m.start(2), m.end(2))])
        for pm in PARA_RX["a"].finditer(xml):
            add("para", list(_nodes_in(xml, pm.start(), pm.end(),
                                       NODE_RX["a"])))
    if part.startswith("docProps/"):
        for m in PROP_RX.finditer(xml):
            add("property", [(m.start(3), m.end(3))])
        # app.xml keeps a second copy of sheet / slide titles
        for m in LPSTR_RX.finditer(xml):
            add("property", [(m.start(2), m.end(2))])
    if part.endswith(".rels"):
        for m in TARGET_RX.finditer(xml):
            add("link", [(m.start(2), m.end(2))])
    return segs


TEXT_PART_RX = re.compile(r"\.(xml|rels)$")
# parts that hold typography only (some font names are
# also first names): not read for verification
NO_PEOPLE_PART_RX = re.compile(r"/theme/|themeOverride|fontTable\.xml$|"
                               r"/styles\.xml$|tableStyles\.xml$")


def read_ooxml(blob):
    """-> (parts {name: str}, segments, media names, embedded names, rule
    spans). Worksheet column rules (survey.xlsx_rules) and phone numbers
    stored as numbers add 'numcell' segments after the text segments."""
    parts, media, embedded = {}, [], []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if TEXT_PART_RX.search(n):
                parts[n] = z.read(n).decode("utf-8", "surrogateescape")
            elif ("/media/" in n or n.startswith("docProps/thumbnail")) \
                    and n.rsplit(".", 1)[-1].lower() in IMAGE_EXT | {"jpeg",
                                                                   "wmf",
                                                                   "emf"}:
                media.append(n)
            elif "/embeddings/" in n and n.rsplit(".", 1)[-1].lower() in OOXML:
                embedded.append(n)
    segs = []
    for name, xml in parts.items():
        segs += segments_of(name, xml)
    import survey
    extra, rules = survey.xlsx_rules(parts, segs)
    segs += extra
    extra2, rules2 = survey.phone_numcells(parts, len(segs))
    taken = {(s["part"], s["pieces"][0]) for s in extra}
    keep = [k for k, s in enumerate(extra2)
            if (s["part"], s["pieces"][0]) not in taken]
    remap = {}
    for k in keep:
        remap[len(segs) + k] = len(segs) + len(remap)
    segs += [extra2[k] for k in keep]
    rules += [dict(r, seg=remap[r["seg"]]) for r in rules2
              if r["seg"] in remap]
    return parts, segs, media, embedded, rules


# ------------------------------------------------------------- detection

def words_of(segs):
    """Numbered word stream over segments: [{i, w, seg, a, b}] with char
    offsets into the segment text."""
    words = []
    for si, s in enumerate(segs):
        for m in re.finditer(r"\S+", s["text"]):
            words.append({"i": len(words), "w": m.group(0), "seg": si,
                          "a": m.start(), "b": m.end()})
    return words


SEG_FRAME = """

THE MATERIAL BELOW is the text of ONE document, as numbered words; a line
break separates its pieces (paragraphs, table cells, slide text, notes,
comments, sheet names, chart labels, properties, links). Return each span as
start_word/end_word (inclusive), its categories (the category NAMES above),
spoken_text = the exact words, confidence and a short reason. Never return a
span that crosses a line break.

WORDS:
"""


def _has_letters(t):
    return any(ch.isalpha() for ch in t)


def detect(segs, pack, model, em=None, chunk=4000):
    """Spans over the segments: [{seg, a, b, text, categories, source}]."""
    import common
    import policy
    words = words_of(segs)
    spans = []
    names = set(pack.categories)
    # model pass, skipping pieces with no letters (tab-book figures)
    active = [w for w in words if _has_letters(segs[w["seg"]]["text"])]
    for k in range(0, len(active), chunk):
        sub = active[k:k + chunk]
        lines, last = [], None
        for w in sub:
            if last is not None and w["seg"] != last:
                lines.append("")
            lines.append(f"{w['i']}\t{w['w']}")
            last = w["seg"]
        prompt = pack.prompt + SEG_FRAME + "\n".join(lines)
        raw = json.loads(common.generate(model, [{"text": prompt}],
                                         policy.DETECTION_SCHEMA))
        by_i = {w["i"]: w for w in sub}
        for s in raw.get("spans", []):
            a, b = by_i.get(s.get("start_word")), by_i.get(s.get("end_word"))
            if not a or not b or a["seg"] != b["seg"] or b["i"] < a["i"]:
                continue
            cats = [c for c in s.get("categories", []) if c in names]
            if not cats:
                continue
            seg = segs[a["seg"]]
            spans.append({"seg": a["seg"], "a": a["a"], "b": b["b"],
                          "text": seg["text"][a["a"]:b["b"]],
                          "categories": cats, "source": "model"})
    spans += deterministic(segs, em)
    return _dedupe(spans)


def deterministic(segs, em=None):
    out = []
    for si, s in enumerate(segs):
        t = s["text"]
        for m in EMAIL_RX.finditer(t):
            out.append({"seg": si, "a": m.start(), "b": m.end(),
                        "text": m.group(0), "categories": ["CONTACT_CHANNEL"],
                        "source": "pattern"})
        for m in PHONE_RX.finditer(t):
            out.append({"seg": si, "a": m.start(), "b": m.end(),
                        "text": m.group(0), "categories": ["CONTACT_CHANNEL"],
                        "source": "pattern"})
        if s["kind"] == "property" and s["text"].strip():
            # author / last-modified-by / company are people or the issuer
            pass
        if em is not None:
            toks = [(m.group(0), m.start(), m.end())
                    for m in re.finditer(r"\S+", t)]
            for i0, i1, e, how in em.match_tokens([x[0] for x in toks]):
                if e.get("context_only"):
                    continue     # an ordinary word: never matched blind
                cat = ("FULL_NAME" if e["category"] == "PERSON"
                       else _map_cat(e["category"]))
                out.append({"seg": si, "a": toks[i0][1], "b": toks[i1][2],
                            "text": t[toks[i0][1]:toks[i1][2]],
                            "categories": [cat], "source": "mapping",
                            "mapping_id": e["id"]})
    return out


def _map_cat(c):
    return {"BRAND": "BRAND", "PRODUCT": "PRODUCT_OR_MODEL",
            "MODEL": "PRODUCT_OR_MODEL", "DEALER": "DEALER_OR_OUTLET",
            "AGENCY": "RESEARCH_OR_AD_AGENCY", "SLOGAN":
            "BRAND_IDENTIFYING_DETAIL", "BRAND_FACT": "BRAND_IDENTIFYING_DETAIL",
            "BANK": "BRAND", "ORGANISATION": "BRAND"}.get(c, "BRAND")


def _dedupe(spans):
    """Overlapping spans in one segment merge into their union; categories
    are unioned (the most protective action then wins)."""
    spans.sort(key=lambda s: (s["seg"], s["a"], -s["b"]))
    out = []
    for s in spans:
        if out and out[-1]["seg"] == s["seg"] and s["a"] < out[-1]["b"]:
            o = out[-1]
            if (o["a"], o["b"]) != (s["a"], s["b"]):
                # a different extent: a mapping id names only its own words
                # (the lookup re-matches every piece of the union)
                o.pop("mapping_id", None)
                s = {k: v for k, v in s.items() if k != "mapping_id"}
            if s["b"] > o["b"]:
                o["text"] = o["text"] + s["text"][o["b"] - s["a"]:]
                o["b"] = s["b"]
            o["categories"] = sorted(set(o["categories"]) | set(s["categories"]))
            o["source"] = "+".join(sorted(set(o["source"].split("+"))
                                          | {s["source"]}))
            if s.get("mapping_id") and (o["a"], o["b"]) == (s["a"], s["b"]):
                o["mapping_id"] = s["mapping_id"]
            if s.get("code_prefix") and not o.get("code_prefix"):
                o["code_prefix"] = s["code_prefix"]
            if s.get("source", "").startswith("column_rule"):
                o["column_rule"] = True
        else:
            out.append(dict(s))
    return out


# ------------------------------------------------------------- rendering

class Replacer:
    """Decides the text that replaces one span, given the mode, the pack
    and (Replace mode) the study mapping and a shared code book."""

    def __init__(self, mode, pack, em=None, codes=None, bar=BAR,
                 fixed_width=False):
        self.mode, self.pack, self.em = mode, pack, em
        self.codes = codes if codes is not None else {}
        self.bar, self.fixed_width = bar, fixed_width
        self.unmapped = []

    def code_for(self, value, prefix=None):
        """One code per distinct linking value: R0001 (respondent), INT001
        (interviewer / user / login), ID0001 (any other linking id)."""
        prefix = prefix or "ID"
        k = (prefix, re.sub(r"\s+", " ", str(value).strip().lower()))
        with CODE_LOCK:
            if k not in self.codes:
                n = self.codes.get(("#", prefix), 0) + 1
                self.codes[("#", prefix)] = n
                self.codes[k] = f"{prefix}{n:03d}" if prefix == "INT" \
                    else f"{prefix}{n:04d}"
            return self.codes[k]

    def __call__(self, span):
        new = self._decide(span)
        if new is None or not self.fixed_width:
            return new
        n = len(span["text"])
        return (new + " " * n)[:n]

    def _decide(self, span):
        act = self.pack.action_for(span["categories"])
        if act == "KEEP":
            return None
        n = len(span["text"])
        bar = (" " * n if self.fixed_width and self.bar == " "
               else self.bar * max(3 if not self.fixed_width else 1, n))
        cell = (span.get("source", "").startswith("column_rule")
                or span.get("column_rule"))
        if self.mode != "replace":
            return bar
        if act != "PSEUDONYMIZE":
            # Replace mode removes contact data, addresses and other ids;
            # a survey column value becomes an empty cell
            return "" if cell else bar
        if set(span["categories"]) & ID_CATS and not (
                set(span["categories"]) - ID_CATS - {"OTHER"}):
            return self.code_for(span["text"], span.get("code_prefix"))
        rep = self.lookup(span)
        if rep is None:
            self.unmapped.append(span["text"])
            return "" if cell else bar
        return _case_like(span["text"], rep)

    def lookup(self, span):
        if self.em is None:
            return None
        if span.get("mapping_id") and span["mapping_id"] in self.em.by_id:
            e = self.em.by_id[span["mapping_id"]]
            return e["replace"] or None
        toks = span["text"].split()
        hits = self.em.match_tokens(toks, with_context=True)
        if not hits:
            return None
        # replace every matched piece inside the span; keep the rest
        out, last = [], 0
        for i0, i1, e, _how in hits:
            out += toks[last:i0] + [e["replace"] or BAR * 3]
            last = i1 + 1
        out += toks[last:]
        return " ".join(out)


def _case_like(src, rep):
    if src.isupper() and len(src) > 1:
        return rep.upper()
    return rep


def apply_to_xml(xml, segs, spans, replacer):
    """Write replacements back into one part. Works right to left over
    each segment's text nodes so earlier offsets stay valid."""
    edits = []   # (node_start, node_end, new_node_text)
    by_seg = {}
    for sp in spans:
        by_seg.setdefault(sp["seg"], []).append(sp)
    for si, sps in by_seg.items():
        seg = segs[si]
        pieces = seg["pieces"]
        if seg["kind"] == "numcell":
            # a number cell: the whole cell is rewritten as a text cell
            import survey
            new = replacer(sorted(sps, key=lambda s: s["a"])[0])
            if new is not None:
                a, b = pieces[0]
                edits.append((a, b, survey.numcell_xml(xml[a:b], new)))
            continue
        texts = [html.unescape(xml[a:b]) for a, b in pieces]
        starts, pos = [], 0
        for t in texts:
            starts.append(pos)
            pos += len(t)
        chars = [list(t) for t in texts]
        for sp in sorted(sps, key=lambda s: -s["a"]):
            new = replacer(sp)
            if new is None:
                continue
            first = True
            for ni, st in enumerate(starts):
                en = st + len(texts[ni])
                lo, hi = max(sp["a"], st), min(sp["b"], en)
                if lo >= hi:
                    continue
                seg_chars = chars[ni]
                rel_lo, rel_hi = lo - st, hi - st
                seg_chars[rel_lo:rel_hi] = list(new) if first else []
                first = False
        for (a, b), t, cs in zip(pieces, texts, chars):
            nt = "".join(cs)
            if nt != t:
                edits.append((a, b, html.escape(nt, quote=True)))
    return _splice(xml, edits)


def _splice(text, edits):
    """Apply (start, end, new) edits in ONE pass (a copy per edit is
    quadratic on a sheet with tens of thousands of cells)."""
    if not edits:
        return text
    out, pos = [], 0
    for a, b, nt in sorted(edits, key=lambda e: e[0]):
        if a < pos:
            continue                 # overlapping edit: the first one wins
        out.append(text[pos:a])
        out.append(nt)
        pos = b
    out.append(text[pos:])
    return "".join(out)


def rename_sheet_refs(parts, renames):
    """A renamed sheet must be renamed in every formula and defined name
    that refers to it, or the workbook's references break."""
    if not renames:
        return parts
    for name, xml in parts.items():
        for old, new in renames.items():
            q_old, q_new = html.escape(old, quote=False), html.escape(new,
                                                                      quote=False)
            xml = xml.replace(f"'{q_old}'!", f"'{q_new}'!")
            if re.fullmatch(r"\w+", old):
                xml = re.sub(rf"(?<![\w']){re.escape(old)}!", f"'{q_new}'!",
                             xml)
        parts[name] = xml
    return parts


def write_ooxml(blob, parts, media_out, embedded_out):
    """New zip: every member copied verbatim unless replaced."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(blob)) as zin, \
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            n = info.filename
            if n in parts:
                data = parts[n].encode("utf-8", "surrogateescape")
            elif n in media_out:
                data = media_out[n]
            elif n in embedded_out:
                data = embedded_out[n]
            else:
                data = zin.read(n)
            zout.writestr(info, data)
    return out.getvalue()


# ------------------------------------------------------------- images

def grey_image(data, ext):
    """Same-size neutral grey replacement for an image (a vector picture
    PIL cannot open becomes a small grey PNG under the same name)."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        size = im.size
    except Exception:
        size, ext = (64, 64), "png"
    g = Image.new("RGB", size, (200, 200, 200))
    buf = io.BytesIO()
    fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "gif": "GIF",
           "bmp": "BMP", "tif": "TIFF", "tiff": "TIFF"}.get(ext, "PNG")
    g.save(buf, fmt)
    return buf.getvalue()


IMAGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"kind": {"type": "string", "enum": [
        "brand_logo", "brand_product_or_store", "person_photo",
        "screenshot_or_ad", "chart_or_table", "generic_icon_or_decoration",
        "other"]},
        "brand_or_company_text": {"type": "boolean"},
        "identifiable_person": {"type": "boolean"}},
    "required": ["kind", "brand_or_company_text", "identifiable_person"]}
IMAGE_PROMPT = ("Classify this image from a market-research document. "
                "brand_logo = any company, brand or agency logo or wordmark; "
                "brand_product_or_store = a product, packaging, storefront, "
                "menu or signage of a specific brand; screenshot_or_ad = a "
                "screenshot, social post, advert or document image; "
                "brand_or_company_text = any visible company or brand name; "
                "identifiable_person = a recognisable human face.")
GREY_KINDS = {"brand_logo", "brand_product_or_store", "person_photo",
              "screenshot_or_ad"}


def classify_image(data, ext, cache, model):
    """-> (grey?, kind). Cached by content hash; a failed classification
    greys the image (fail closed)."""
    import hashlib
    h = hashlib.md5(data).hexdigest()
    if h in cache:
        return cache[h]["grey"], cache[h]["kind"]
    try:
        from PIL import Image
        import judge_client
        im = Image.open(io.BytesIO(data))
        if im.width < 24 or im.height < 24:
            res = {"grey": False, "kind": "tiny"}
        else:
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                im.convert("RGB").save(f.name, "PNG")
                raw = judge_client.judge_call(IMAGE_PROMPT, IMAGE_SCHEMA,
                                              model=model, images=[f.name],
                                              native_gemini=model.startswith(
                                                  "gemini"),
                                              log_engine="doc_images",
                                              max_tokens=400)
            grey = (raw.get("kind") in GREY_KINDS
                    or raw.get("brand_or_company_text")
                    or raw.get("identifiable_person"))
            res = {"grey": bool(grey), "kind": raw.get("kind")}
    except Exception as exc:
        res = {"grey": True, "kind": f"unclassified ({str(exc)[:40]})"}
    cache[h] = res
    return res["grey"], res["kind"]


# ------------------------------------------------------------- one file

def process_ooxml(blob, pack, model, mode, em, codes, img_cache,
                  img_model, record=None, detections=None, embedded_fn=None,
                  replacer=None):
    """Return (new blob, report). `detections`, when given, are reused
    (Replace mode renders from the detection pass that built the mapping
    plus a mapping-guided pass)."""
    parts, segs, media, embedded, rules = read_ooxml(blob)
    spans = (detections if detections is not None
             else _dedupe(detect(segs, pack, model, em) + rules))
    if detections is not None and em is not None:
        spans = _dedupe(list(spans) + deterministic(segs, em))
    rep = replacer or Replacer(mode, pack, em, codes)
    by_part = {}
    for sp in spans:
        by_part.setdefault(segs[sp["seg"]]["part"], []).append(sp)
    renames = {}
    for part, sps in by_part.items():
        # sheet renames, recorded before the workbook XML changes
        for sp in sps:
            seg = segs[sp["seg"]]
            if seg["kind"] == "sheet_name":
                new = rep(sp)
                if new is not None:
                    renames[seg["text"]] = (seg["text"][:sp["a"]] + new
                                            + seg["text"][sp["b"]:])[:31]
        local = {}
        for sp in sps:
            local.setdefault(sp["seg"], []).append(sp)
        idx = {si: segs[si] for si in local}
        parts[part] = apply_to_xml(parts[part], segs,
                                   [s for s in sps], rep) if idx else parts[part]
    parts = rename_sheet_refs(parts, renames)
    media_out, greyed = {}, []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in media:
            data = z.read(n)
            ext = n.rsplit(".", 1)[-1].lower()
            if n.startswith("docProps/thumbnail"):
                # a picture of page one: always replaced
                grey, kind = True, "thumbnail"
            elif ext in ("wmf", "emf"):
                grey, kind = True, "vector picture (not classifiable)"
            else:
                grey, kind = classify_image(data, ext, img_cache, img_model)
            if grey:
                media_out[n] = grey_image(data, ext)
                greyed.append({"image": n, "kind": kind})
        embedded_out = {}
        for n in embedded:
            if embedded_fn is not None:
                embedded_out[n] = embedded_fn(n, z.read(n))
                continue
            sub, _subrep = process_ooxml(z.read(n), pack, model, mode, em,
                                         codes, img_cache, img_model)
            embedded_out[n] = sub
    out = write_ooxml(blob, parts, media_out, embedded_out)
    report = {"segments": len(segs), "spans": len(spans),
              "images_greyed_list": greyed,
              "by_action": _count_actions(spans, pack),
              "images": len(media), "images_greyed": len(greyed),
              "embedded": len(embedded), "sheet_renames": len(renames),
              "unmapped_pseudonyms": len(rep.unmapped)}
    return out, report


def _count_actions(spans, pack):
    c = {}
    for s in spans:
        a = pack.action_for(s["categories"])
        c[a] = c.get(a, 0) + 1
    return c


def text_segments(text):
    """Line-level segments of a plain text file."""
    segs, pos = [], 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            segs.append({"part": "text", "kind": "line",
                         "pieces": [(pos, pos + len(line.rstrip("\r\n")))],
                         "text": line.rstrip("\r\n")})
        pos += len(line)
    return segs


DELIMITED = {"csv", "tsv"}
FIXED_WIDTH = {"asc"}


FW_RUN_RX = re.compile(r"[^\W\d_][\w.,'&/@+-]*(?: {1,2}[\w.,'&/@+-]+)*")


def fixed_width_runs(text):
    """Fixed-width survey data: the runs of text inside each record (open
    answers, names, models) as segments — the digits between them are codes
    the tab files read by position and are left to patterns and the net."""
    segs, pos = [], 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        taken = []
        for m in FW_RUN_RX.finditer(body):
            t = m.group(0).rstrip()
            if len(t) >= 2:
                segs.append({"part": "text", "kind": "run",
                             "pieces": [(pos + m.start(),
                                         pos + m.start() + len(t))],
                             "text": t})
                taken.append((m.start(), m.start() + len(t)))
        # long digit runs (phones and ids glued to other fields) for the
        # patterns and the net; the model never reads them (no letters)
        for m in re.finditer(r"\d{8,}", body):
            if not any(a <= m.start() < b for a, b in taken):
                segs.append({"part": "text", "kind": "digits",
                             "pieces": [(pos + m.start(), pos + m.end())],
                             "text": m.group(0)})
        pos += len(line)
    segs.sort(key=lambda s: s["pieces"][0][0])
    return segs


def read_text(raw, ext):
    """-> (text, encoding, bom, segments, rule spans) for a text file.
    Delimited files are cut into cells (column rules apply); other text
    files into lines."""
    import survey
    text, enc, bom = survey.decode(raw)
    if ext in DELIMITED:
        segs, rules = survey.csv_segments(text)
    elif ext in FIXED_WIDTH:
        segs, rules = fixed_width_runs(text), []
    else:
        segs, rules = text_segments(text), []
    return text, enc, bom, segs, rules


def process_text(raw, pack, model, mode, em, codes, detections=None,
                 ext="txt"):
    import survey
    text, enc, bom, segs, rules = read_text(raw, ext)
    spans = (detections if detections is not None
             else _dedupe(detect(segs, pack, model, em) + rules))
    if detections is not None and em is not None:
        spans = _dedupe(list(spans) + deterministic(segs, em))
    fixed = ext in FIXED_WIDTH
    rep = Replacer(mode, pack, em, codes,
                   bar=" " if fixed else survey.bar_char(enc, text),
                   fixed_width=fixed)
    edits = []
    for sp in sorted(spans, key=lambda s: (s["seg"], -s["a"])):
        new = rep(sp)
        if new is None:
            continue
        a0 = segs[sp["seg"]]["pieces"][0][0]
        edits.append((a0 + sp["a"], a0 + sp["b"], new))
    text = _splice(text, edits)
    out = survey.encode(text, enc, bom)
    return out, {"segments": len(segs), "spans": len(spans),
                 "encoding": enc, "by_action": _count_actions(spans, pack),
                 "unmapped_pseudonyms": len(rep.unmapped)}


def convert_to_docx(path: Path, workdir: Path) -> Path:
    out = workdir / (path.stem + ".docx")
    subprocess.run(["textutil", "-convert", "docx", "-output", str(out),
                    str(path)], check=True, capture_output=True)
    return out


def segments_for(path: Path, workdir: Path):
    """(kind, blob-or-path, segments) for detection, any supported type."""
    ext = path.suffix.lower().lstrip(".")
    if ext in CONVERT:
        path = convert_to_docx(path, workdir)
        ext = "docx"
    if ext in OOXML:
        blob = path.read_bytes()
        _parts, segs, _m, _e, _r = read_ooxml(blob)
        return "ooxml", path, segs
    if ext in TEXTISH:
        _t, _e, _b, segs, _r = read_text(path.read_bytes(), ext)
        return "text", path, segs
    return None, path, []


# ------------------------------------------------------------- verify

def raw_ooxml_text(blob):
    """Every XML part's text and attribute values, tags removed — the
    verification view, independent of what the reader knows how to edit."""
    out = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if TEXT_PART_RX.search(n) and not NO_PEOPLE_PART_RX.search(n):
                x = z.read(n).decode("utf-8", "replace")
                # namespace and schema URIs are format plumbing, not content
                x = re.sub(r'\s(?:xmlns(?::\w+)?|mc:Ignorable|xsi:\w+)="[^"]*"',
                           " ", x)
                x = re.sub(r'="https?://schemas\.[^"]*"', '=""', x)
                x = re.sub(r"</w:p>|</a:p>|</si>|</row>|</c>", "\n", x)
                vals = ([] if n.startswith(("xl/worksheets/", "xl/styles",
                                            "xl/calcChain", "xl/tables/"))
                        else re.findall(r'="([^"]*)"', x))
                # a tag becomes a separator no pattern spans (a space would
                # join neighbouring numbers into a phone-like run)
                out.append(html.unescape(re.sub(r"<[^>]+>", "\x00", x)))
                out.append(html.unescape("\n".join(vals)))
    return "\n".join(out)


def output_text(blob, ext):
    if ext in OOXML:
        # joined segments (a name split across runs reads whole) AND the
        # raw text of every part (what the reader does not edit)
        _p, segs, _m, _e, _r = read_ooxml(blob)
        return "\n".join(s["text"] for s in segs) + "\n" + \
            raw_ooxml_text(blob)
    import survey
    return survey.decode(blob)[0]


def verify(blob, ext, em=None):
    """Leftover real names (mapping matcher) and contact details in an
    output. Returns a list of findings (no text quoted beyond the entity
    id and kind)."""
    t = output_text(blob, ext)
    found = []
    if em is not None:
        for h in em.scan_text(t):
            found.append({"kind": "mapped_name", "entity": h["entity"],
                          "line": h["line"]})
    for m in EMAIL_RX.finditer(t):
        found.append({"kind": "email"})
    for m in PHONE_RX.finditer(t):
        found.append({"kind": "phone"})
    return found


def _self_test():
    import base
    pack = base.load(HERE.parent / "taxonomies" / "market_research")
    # a docx whose brand is split across two runs, plus a comment author
    doc = ('<w:document><w:body><w:p><w:r><w:t>We tested Zor</w:t></w:r>'
           '<w:r><w:t xml:space="preserve">vex Vision with Meera</w:t></w:r>'
           '</w:p><w:p><w:r><w:t>Call 98224 15736 or a@b.com</w:t></w:r>'
           '</w:p></w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", doc)
        z.writestr("word/comments.xml", '<w:comments><w:comment w:author='
                   '"Ravi Kumar"><w:p><w:r><w:t>check</w:t></w:r></w:p>'
                   '</w:comment></w:comments>')
    blob = buf.getvalue()
    parts, segs, _m, _e, _r = read_ooxml(blob)
    texts = [s["text"] for s in segs]
    assert "We tested Zorvex Vision with Meera" in texts, texts
    assert "Ravi Kumar" in texts, texts
    si = texts.index("We tested Zorvex Vision with Meera")
    t = texts[si]
    spans = [{"seg": si, "a": t.index("Zorvex"), "b": t.index("Zorvex") + 6,
              "text": "Zorvex", "categories": ["BRAND"]},
             {"seg": si, "a": t.index("Meera"), "b": len(t),
              "text": "Meera", "categories": ["FULL_NAME"]}]
    spans += deterministic(segs)

    class EM:     # minimal mapping stand-in
        by_id = {"b:0": {"replace": "Kelmora"}, "p:0": {"replace": "Anita"}}

        def match_tokens(self, toks, **kw):
            out = []
            for i, w in enumerate(toks):
                if w.startswith("Zorvex"):
                    out.append((i, i, {"replace": "Kelmora"}, "regex"))
                if w.startswith("Meera"):
                    out.append((i, i, {"replace": "Anita"}, "regex"))
            return out
    red = Replacer("redact", pack)
    new = apply_to_xml(parts["word/document.xml"], segs,
                       [s for s in spans if segs[s["seg"]]["part"]
                        == "word/document.xml"], red)
    assert "Zor" not in new and "Meera" not in new and "98224" not in new
    assert BAR in new and "We tested" in new
    rp = Replacer("replace", pack, EM())
    new2 = apply_to_xml(parts["word/document.xml"], segs,
                        [s for s in spans if segs[s["seg"]]["part"]
                         == "word/document.xml"], rp)
    assert "Kelmora" in new2 and "Anita" in new2 and "Zor" not in new2, new2
    assert "98224" not in new2 and "a@b.com" not in new2
    # a code book gives one code per distinct ID
    rp2 = Replacer("replace", pack, EM(), codes={})
    c1 = rp2({"text": "R-17", "categories": ["ASSIGNED_ID"], "seg": 0,
              "a": 0, "b": 4})
    c2 = rp2({"text": "r-17 ", "categories": ["ASSIGNED_ID"], "seg": 0,
              "a": 0, "b": 4})
    assert c1 == c2 == "ID0001"
    # untouched bytes stay identical
    assert new2.startswith("<w:document><w:body><w:p><w:r><w:t>We tested ")
    print("office self-test: OK")


if __name__ == "__main__":
    _self_test()
