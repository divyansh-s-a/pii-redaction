#!/usr/bin/env python3
"""Survey data — text encodings, CSV cells, fixed-width data, column rules.

Survey exports carry identity in COLUMNS: a header that says the column is a
name, phone, email, address, GPS point, device id or a linking id puts every
value below it under that category, whatever the value looks like. Column
rules are deterministic and run in both modes, beside the model pass that
reads free text (open-ended answers, labels, question wording).

  name / respondent name / contact name      -> FULL_NAME   (person)
  phone / mobile / contact no / email        -> CONTACT_CHANNEL (removed)
  address / pincode / gps / latitude ...     -> ADDRESS     (removed)
  deviceid / imei / _uuid / instanceid       -> DEVICE_VEHICLE_NETWORK_ID
  respondent id / panel id                   -> ASSIGNED_ID, code R0001
  interviewer id / usercode / _submitted_by  -> ASSIGNED_ID, code INT01
A plain running serial (SERIAL, 10001, 10002 ...) is a row key the tab files
depend on and identifies no one: kept.

Text files keep their encoding (a BOM-marked UTF-16 file stays UTF-16, CRLF
stays CRLF). Fixed-width data (.asc) keeps every column: a replacement is
padded or cut to the original width, and a removed value becomes spaces (the
survey tools read blank as missing).
"""
import re

# (header regex, category, code prefix or None). Anchored on SHORT headers:
# a question ("Did you contact the agent by phone?") is not a column rule.
HEADER_RULES = [
    (r"(respondent|contact|customer|full|first|last|participant|"
     r"interviewer|recruiter|moderator)?[ _]*name", "FULL_NAME", None),
    (r"(contact |mobile |phone |cell |whatsapp )?(mobile|phone|cell|"
     r"contact)[ _]*(no\.?|number|num|#)?( \(contact\))?", "CONTACT_CHANNEL",
     None),
    (r"e-?mail( id| address)?", "CONTACT_CHANNEL", None),
    (r"(full |home |residential |postal |street )?address( line \d)?",
     "ADDRESS", None),
    (r"pin ?code|postcode|zip ?code", "ADDRESS", None),
    (r"(gps|geo)?[ _]*(lat|latitude|long|longitude|lng|gps|geolocation|"
     r"location_gps)[ _\w]*", "ADDRESS", None),
    (r"_?(device ?id|imei|uuid|instance ?id|meta.?instanceid|deviceid|"
     r"android.?id|ip.?address)", "DEVICE_VEHICLE_NETWORK_ID", None),
    (r"(respondent|resp|panel|participant|member|customer)[ _]*(id|code|no\.?)",
     "ASSIGNED_ID", "R"),
    (r"u?id|user[ _]?id|record[ _]?id|response[ _]?id|unique[ _]?id",
     "ASSIGNED_ID", "R"),
    (r"(interviewer|recruiter|supervisor|agent|user|enumerator)[ _]*"
     r"(id|code|login|name)?|usercode|_?submitted_by|username|login",
     "ASSIGNED_ID", "INT"),
]
_RULES = [(re.compile(rf"^\s*{rx}\s*$", re.I), cat, pre)
          for rx, cat, pre in HEADER_RULES]
NAMEISH_EXCLUDE = re.compile(r"brand|product|store|shop|share|city|town|"
                             r"state|area|zone|region|file|sheet|"
                             r"variable|question|label|scheme|plan|model",
                             re.I)


def column_rule(header):
    """-> (category, code prefix) for a short identity header, else None."""
    h = str(header or "").strip()
    if not h or len(h) > 40 or "?" in h:
        return None
    for rx, cat, pre in _RULES:
        if rx.match(h):
            if cat == "FULL_NAME" and NAMEISH_EXCLUDE.search(h):
                return None
            if pre == "INT" and cat == "ASSIGNED_ID" and \
                    re.search(r"name", h, re.I):
                return ("FULL_NAME", None)
            return (cat, pre)
    return None


# ------------------------------------------------------------- encodings

def decode(raw: bytes):
    """-> (text, encoding, bom). BOM first, then UTF-8, then cp1252."""
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", "surrogatepass"), "utf-16-le", \
            b"\xff\xfe"
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "surrogatepass"), "utf-16-be", \
            b"\xfe\xff"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "surrogateescape"), "utf-8", \
            b"\xef\xbb\xbf"
    # UTF-16 without a BOM: every other byte zero
    if len(raw) > 8 and raw[1:40:2].count(0) > 15:
        return raw.decode("utf-16-le", "surrogatepass"), "utf-16-le", b""
    try:
        return raw.decode("utf-8"), "utf-8", b""
    except UnicodeDecodeError:
        return raw.decode("cp1252", "replace"), "cp1252", b""


def encode(text, enc, bom):
    if enc.startswith("utf-16"):
        return bom + text.encode(enc, "surrogatepass")
    if enc == "utf-8":
        return bom + text.encode("utf-8", "surrogateescape")
    return bom + text.encode(enc, "replace")


def bar_char(enc, raw_text):
    """The black-box character a file can hold: █ in Unicode files; in a
    file that is pure ASCII (survey tools often choke on anything else) an
    'X' of the same length."""
    if enc.startswith("utf-16"):
        return "█"
    if enc == "utf-8" and not raw_text.isascii():
        return "█"
    return "X"


# ------------------------------------------------------------- CSV cells

def _split_line(line, delim):
    """Cells of one CSV line as (start, end) offsets of the cell VALUE
    (quotes excluded). None when the line has an open quote (a cell that
    continues on the next line) — the caller falls back to a line segment."""
    cells, i, n = [], 0, len(line)
    while i <= n:
        if i < n and line[i] == '"':
            j = i + 1
            while True:
                k = line.find('"', j)
                if k < 0:
                    return None
                if k + 1 < n and line[k + 1] == '"':
                    j = k + 2
                    continue
                break
            cells.append((i + 1, k))
            i = k + 1
            if i < n and line[i] != delim:
                nxt = line.find(delim, i)
                i = n if nxt < 0 else nxt
            i += 1
        else:
            k = line.find(delim, i)
            if k < 0:
                k = n
            cells.append((i, k))
            i = k + 1
            if k == n:
                break
    return cells


def sniff_delim(text):
    head = text[:4000]
    return max([",", ";", "\t", "|"], key=head.count)


def csv_segments(text, part="text"):
    """Cell-level segments + column-rule spans for a delimited file.
    -> (segments, rule_spans). Each rule span carries code_prefix when the
    column links records."""
    delim = sniff_delim(text)
    segs, rules, pos = [], [], 0
    header, col_rules = None, {}
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        cells = _split_line(body, delim)
        if cells is None:
            if body.strip():
                segs.append({"part": part, "kind": "line",
                             "pieces": [(pos, pos + len(body))], "text": body})
        else:
            if header is None and any(body[a:b].strip() for a, b in cells):
                header = [body[a:b] for a, b in cells]
                for ci, h in enumerate(header):
                    r = column_rule(h)
                    if r:
                        col_rules[ci] = r
                row_is_header = True
            else:
                row_is_header = False
            for ci, (a, b) in enumerate(cells):
                v = body[a:b]
                if not v.strip():
                    continue
                si = len(segs)
                segs.append({"part": part, "kind": "cell", "col": ci,
                             "pieces": [(pos + a, pos + b)], "text": v})
                if not row_is_header and ci in col_rules:
                    cat, pre = col_rules[ci]
                    rules.append({"seg": si, "a": 0, "b": len(v), "text": v,
                                  "categories": [cat], "source": "column_rule",
                                  "code_prefix": pre})
        pos += len(line)
    return segs, rules


# ------------------------------------------------------------- xlsx columns

# a cell is either self-closing or has a body; [^>]* never runs past its
# own tag, so a sheet of empty styled cells is read in linear time
CELL_RX = re.compile(r'<(?:\w+:)?c\b([^>]*?)(/>|>(.*?)</(?:\w+:)?c>)', re.S)
ROW_RX = re.compile(r"<(?:\w+:)?row\b[^>]*?/>|<(?:\w+:)?row\b[^>]*>(.*?)"
                    r"</(?:\w+:)?row>", re.S)
V_RX = re.compile(r"<(?:\w+:)?v>(.*?)</(?:\w+:)?v>", re.S)
REF_RX = re.compile(r'\br="([A-Z]+)(\d+)"')
T_ATTR_RX = re.compile(r'\bt="(\w+)"')


def xlsx_rules(parts, segs):
    """Column rules for every worksheet. Returns (extra segments, spans).

    A header row is the first of the top five rows with at least three text
    cells. Below it, a cell in a rule column becomes a span: on its shared
    string / inline string segment, or — for a NUMBER (phones and ids are
    often stored as numbers) — on a new 'numcell' segment that the writer
    turns into a text cell."""
    import html
    shared = {}
    for si, s in enumerate(segs):
        if s["kind"] == "shared" and "si" in s:
            shared[s["si"]] = si
    import bisect
    inline_by_part = {}
    for si, s in enumerate(segs):
        if s["kind"] == "inline":
            inline_by_part.setdefault(s["part"], []).append(
                (s["pieces"][0][0], si))
    for v in inline_by_part.values():
        v.sort()

    def inline_seg(part, start, end):
        lst = inline_by_part.get(part) or []
        k = bisect.bisect_left(lst, (start, -1))
        if k < len(lst) and lst[k][0] < end:
            return lst[k][1]
        return None
    sst = {}
    for k, si in shared.items():
        sst[k] = segs[si]["text"]
    extra, spans = [], []
    base = len(segs)
    for part, xml in parts.items():
        if not re.match(r"xl/worksheets/sheet\d*\.xml$", part):
            continue
        rows = []
        for rm in ROW_RX.finditer(xml):
            cells = []
            body = rm.group(1) or ""
            off = rm.start(1) if rm.group(1) is not None else rm.start()
            for cm in CELL_RX.finditer(body):
                attrs = cm.group(1).rstrip("/")
                ref = REF_RX.search(" " + attrs)
                if not ref:
                    continue
                t = (T_ATTR_RX.search(attrs) or [None, "n"])[1]
                inner = cm.group(3) or ""
                vm = V_RX.search(inner)
                cells.append({"col": ref.group(1), "t": t,
                              "v": html.unescape(vm.group(1)) if vm else None,
                              "start": off + cm.start(), "end": off + cm.end(),
                              "inner_start": off + cm.start(3)
                              if cm.group(3) is not None else None,
                              "formula": "<f" in inner})
            rows.append(cells)
            if len(rows) > 200000:
                break
        header_i, col_rules = None, {}
        for ri, cells in enumerate(rows[:5]):
            texts = []
            for c in cells:
                if c["t"] == "s" and c["v"] is not None and \
                        c["v"].isdigit() and int(c["v"]) in sst:
                    texts.append((c["col"], sst[int(c["v"])]))
                elif c["t"] == "inlineStr":
                    si = inline_seg(part, c["start"], c["end"])
                    if si is not None:
                        texts.append((c["col"], segs[si]["text"]))
                elif c["t"] == "str" and c["v"]:
                    texts.append((c["col"], c["v"]))
            if len(texts) >= 3:
                header_i = ri
                for col, h in texts:
                    r = column_rule(h)
                    if r:
                        col_rules[col] = r
                break
        if header_i is None or not col_rules:
            continue
        for cells in rows[header_i + 1:]:
            for c in cells:
                r = col_rules.get(c["col"])
                if not r or c["formula"]:
                    continue
                cat, pre = r
                if c["t"] == "s" and c["v"] is not None and c["v"].isdigit():
                    si = shared.get(int(c["v"]))
                    if si is None:
                        continue
                    text = segs[si]["text"]
                    spans.append({"seg": si, "a": 0, "b": len(text),
                                  "text": text, "categories": [cat],
                                  "source": "column_rule",
                                  "code_prefix": pre})
                elif c["t"] == "inlineStr":
                    si = inline_seg(part, c["start"], c["end"])
                    if si is None:
                        continue
                    text = segs[si]["text"]
                    spans.append({"seg": si, "a": 0, "b": len(text),
                                  "text": text, "categories": [cat],
                                  "source": "column_rule",
                                  "code_prefix": pre})
                elif c["v"] not in (None, "") and c["t"] in ("n", "str"):
                    seg = {"part": part, "kind": "numcell",
                           "pieces": [(c["start"], c["end"])],
                           "text": c["v"]}
                    extra.append(seg)
                    spans.append({"seg": base + len(extra) - 1, "a": 0,
                                  "b": len(c["v"]), "text": c["v"],
                                  "categories": [cat], "source": "column_rule",
                                  "code_prefix": pre})
    return extra, spans


PHONE_V_RX = re.compile(r"<(?:\w+:)?v>((?:91)?[6-9]\d{9}(?:\.0)?)"
                        r"</(?:\w+:)?v>")


def phone_numcells(parts, base):
    """Phone numbers stored as numbers anywhere in a sheet (no header
    needed): 10-digit Indian mobiles, optionally with 91."""
    extra, spans = [], []
    for part, xml in parts.items():
        if not re.match(r"xl/worksheets/sheet\d*\.xml$", part):
            continue
        if "v>" not in xml:
            continue
        for m in CELL_RX.finditer(xml):
            body = m.group(3)
            if not body or "<f" in body or 't="s"' in m.group(1):
                continue
            vm = PHONE_V_RX.search(body)
            if not vm:
                continue
            seg = {"part": part, "kind": "numcell",
                   "pieces": [(m.start(), m.end())], "text": vm.group(1)}
            extra.append(seg)
            spans.append({"seg": base + len(extra) - 1, "a": 0,
                          "b": len(vm.group(1)), "text": vm.group(1),
                          "categories": ["CONTACT_CHANNEL"],
                          "source": "pattern"})
    return extra, spans


def numcell_xml(cell_xml, new_text):
    """A numeric <c> rewritten as an inline-string cell holding new_text
    (or an empty cell when new_text is empty)."""
    import html
    m = re.match(r"<((?:\w+:)?)c ([^>]*?)(/>|>)", cell_xml, re.S)
    px = m.group(1)                     # keep the document's prefix
    attrs = re.sub(r'\s*\bt="\w+"', "", m.group(2)).rstrip("/").rstrip()
    if not new_text:
        return f"<{px}c {attrs}/>"
    return (f'<{px}c {attrs} t="inlineStr"><{px}is><{px}t xml:space="preserve">'
            f"{html.escape(new_text, quote=False)}</{px}t></{px}is></{px}c>")


def _self_test():
    assert column_rule("Respondent Name") == ("FULL_NAME", None)
    assert column_rule("Name") == ("FULL_NAME", None)
    assert column_rule("Store Name") is None
    assert column_rule("Phone Number")[0] == "CONTACT_CHANNEL"
    assert column_rule("Contact Mobile No.")[0] == "CONTACT_CHANNEL"
    assert column_rule("Email")[0] == "CONTACT_CHANNEL"
    assert column_rule("Respondent Id") == ("ASSIGNED_ID", "R")
    assert column_rule("Interviewer Id") == ("ASSIGNED_ID", "INT")
    assert column_rule("UserCode") == ("ASSIGNED_ID", "INT")
    assert column_rule("_submitted_by") == ("ASSIGNED_ID", "INT")
    assert column_rule("deviceid")[0] == "DEVICE_VEHICLE_NETWORK_ID"
    assert column_rule("_uuid")[0] == "DEVICE_VEHICLE_NETWORK_ID"
    assert column_rule("SERIAL") is None
    assert column_rule("Who contacted brand support over call") is None
    assert column_rule("Address")[0] == "ADDRESS"
    t = 'SERIAL,Name,Phone,City\r\n10001,"Kavya, R",9822415736,Pune\r\n'
    segs, rules = csv_segments(t)
    vals = {r["text"]: r["categories"][0] for r in rules}
    assert vals == {"Kavya, R": "FULL_NAME", "9822415736": "CONTACT_CHANNEL"}, vals
    raw = "﻿abc".encode("utf-16-le")
    txt, enc, bom = decode(b"\xff\xfe" + "A,B\r\n".encode("utf-16-le"))
    assert enc == "utf-16-le" and txt == "A,B\r\n"
    assert encode(txt, enc, bom) == b"\xff\xfe" + "A,B\r\n".encode("utf-16-le")
    cell = '<c r="C2" s="3"><v>9822415736</v></c>'
    assert numcell_xml(cell, "") == '<c r="C2" s="3"/>'
    assert 't="inlineStr"' in numcell_xml(cell, "XXXX")
    print("survey self-test: OK")


if __name__ == "__main__":
    _self_test()
