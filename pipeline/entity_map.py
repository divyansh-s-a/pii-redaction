#!/usr/bin/env python3
"""Mapped entities — real names a project replaces with invented ones.

Brands, product and model names, dealers, organisations, an agency, and facts
that would re-identify them are not personal data, so the audio taxonomy never
targets them. A project that must hide them supplies a MAPPING file:

    brands:
      - {pattern: "Zorvex(?:\\s+Vision)?", replace: Norvale,
         aliases: [ज़ोरवेक्स, zor vex], category: BRAND}
    agency:
      - {pattern: "Brightpath Research", replace: Insightwell}
    people:                       # replace mode: spoken as the invented name
      - {pattern: "\\bRavindra\\b", replace: Mohan}
    keep:                         # notes: never detected, never changed
      - Pune / Bangalore (fieldwork locations)

`pattern` is a case-insensitive regex (the `abbreviations` section is
case-sensitive), `replace` the invented name, `aliases` optional spoken forms
(Devanagari, transliterations, mis-hearings), `category` an optional label,
`render` optional (voice|beep). Sections default as SECTION_DEFAULTS says; any
other section name is treated as a voiced entity category. Non-dict items
(e.g. `keep:` notes) are recorded, never matched. An optional sidecar
`<mapping>.aliases.yaml` (tools/expand_mapping.py) adds aliases by entry id.

Selection: tools/redact.py --mapping PATH, or the PII_MAPPING_FILE env var.

Detection combines a deterministic net (this module: regex, aliases,
Devanagari->Latin keys, sound-alike fuzzy match) with an LLM pass over the
same numbered word stream (detect_pii). Only CONFIRMED mentions are voiced
with an invented name; a mention found by the fuzzy matcher alone is beeped,
because a wrongly voiced brand puts words in a speaker's mouth.
"""
import difflib
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ENV = "PII_MAPPING_FILE"

# section -> (category, render)
SECTION_DEFAULTS = {
    "brands": ("BRAND", "voice"),
    "abbreviations": ("BRAND", "voice"),
    "brand_facts": ("BRAND_FACT", "voice"),
    "agency": ("AGENCY", "voice"),
    "people": ("PERSON", "voice"),
}
PERSON_CATEGORY = "PERSON"
NOTE_SECTIONS = {"keep"}
FUZZY_MIN = 0.85           # sound-alike ratio on phonetic keys
FUZZY_MIN_LEN = 5          # keys shorter than this match exactly or not at all
MAX_WINDOW = 6             # words per candidate window

# ---------------------------------------------------------------- Devanagari

_DEV_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
               "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
               "ऍ": "e", "ॲ": "a"}
_DEV_MATRAS = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
               "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
_DEV_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch",
             "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th",
             "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
             "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
             "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
             "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
_DEV_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh",
              "फ": "f", "य": "y"}
_DEV_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_VIRAMA, _NUKTA = "्", "़"
_DEV_DIGITS = {chr(0x0966 + k): str(k) for k in range(10)}
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))


def romanize(text: str) -> str:
    """Devanagari (Hindi/Marathi) -> plain Latin, schwa dropped word-finally.
    Other scripts pass through unchanged. Loose by design: the output feeds
    a sound-alike key, not a reader."""
    s = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in _DEV_CONS:
            cons = _DEV_CONS[c]
            j = i + 1
            if j < n and s[j] == _NUKTA:
                cons = _DEV_NUKTA.get(c, cons)
                j += 1
            if j < n and s[j] in _DEV_MATRAS:
                out.append(cons + _DEV_MATRAS[s[j]])
                j += 1
            elif j < n and s[j] == _VIRAMA:
                out.append(cons)
                j += 1
            else:
                # inherent vowel, dropped at the end of a word (schwa)
                nxt = s[j] if j < n else " "
                at_end = not (nxt in _DEV_CONS or nxt in _DEV_VOWELS
                              or nxt in _DEV_SIGNS)
                out.append(cons if at_end else cons + "a")
            i = j
            continue
        if c in _DEV_VOWELS:
            out.append(_DEV_VOWELS[c])
        elif c in _DEV_MATRAS:
            out.append(_DEV_MATRAS[c])
        elif c in _DEV_SIGNS:
            out.append(_DEV_SIGNS[c])
        elif c in _DEV_DIGITS:
            out.append(_DEV_DIGITS[c])
        elif c in (_VIRAMA, _NUKTA):
            pass
        else:
            out.append(c)
        i += 1
    return "".join(out)


_KEY_STEPS = [("chh", "C"), ("ch", "C"), ("ph", "f"), ("bh", "b"),
              ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
              ("jh", "j"), ("sh", "s"), ("ck", "k"), ("q", "k"), ("x", "ks"),
              ("w", "v"), ("z", "j"), ("c", "k"), ("ee", "i"), ("oo", "u"),
              ("y", "i")]


def key(text: str) -> str:
    """Sound-alike key: romanize, lowercase, keep letters/digits, merge the
    spellings ASR engines and transliterators use for the same sound."""
    s = romanize(unicodedata.normalize("NFKC", str(text))).lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if ch.isalnum() and not unicodedata.combining(ch))
    for a, b in _KEY_STEPS:
        s = s.replace(a, b)
    # collapse runs (aa->a, nn->n) so vowel length and gemination never matter
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


# ------------------------------------------------------- regex -> literals

def _parse(p):
    """Tiny regex parser for literal extraction: returns a list of
    alternatives, each a list of nodes. Nodes: ('lit', s) | ('grp', alts) |
    ('opt', node). Character classes become their first member; anchors,
    lookarounds and \\b vanish; \\s becomes a space."""
    i = 0

    def alts():
        nonlocal i
        res, seq = [], []
        while i < len(p):
            c = p[i]
            if c == "|":
                res.append(seq)
                seq = []
                i += 1
                continue
            if c == ")":
                break
            node = atom()
            if node is None:
                continue
            if i < len(p) and p[i] in "?*":
                node = ("opt", node)
                i += 1
            elif i < len(p) and p[i] == "+":
                i += 1
            elif i < len(p) and p[i] == "{":
                j = p.find("}", i)
                lo = p[i + 1:j].split(",")[0] if j > 0 else "1"
                i = j + 1 if j > 0 else i + 1
                if lo.strip() == "0":
                    node = ("opt", node)
            if i < len(p) and p[i] == "?":        # lazy suffix
                i += 1
            seq.append(node)
        res.append(seq)
        return res

    def atom():
        nonlocal i
        c = p[i]
        if c == "(":
            i += 1
            look = False
            if p.startswith("?:", i):
                i += 2
            elif p.startswith("?", i):
                look = True                       # lookaround / flags
                while i < len(p) and p[i] not in ":=!)":
                    i += 1
                if i < len(p) and p[i] != ")":
                    i += 1
            inner = alts()
            i += 1                                # ')'
            return None if look else ("grp", inner)
        if c == "[":
            j = i + 1
            if j < len(p) and p[j] == "^":
                j += 1
            first = p[j]
            if first == "\\" and j + 1 < len(p):
                first = p[j + 1]
            k = p.find("]", j + 1)
            i = (k + 1) if k > 0 else len(p)
            return ("lit", first)
        if c == "\\":
            e = p[i + 1] if i + 1 < len(p) else ""
            i += 2
            if e == "s":
                return ("lit", " ")
            if e in "bBAZdDwW":
                return None
            return ("lit", e)
        i += 1
        if c in "^$.":
            return None
        return ("lit", c)

    return alts()


def _render(seq, mode):
    out = []
    for node in seq:
        kind = node[0]
        if kind == "lit":
            out.append(node[1])
        elif kind == "grp":
            out.append(_render(node[1][0], mode) if node[1] else "")
        elif kind == "opt":
            if mode == "max":
                out.append(_render([node[1]], mode))
    return "".join(out)


def literals(pattern: str):
    """Best-effort literal spellings a regex accepts: per top-level
    alternative, with every optional part kept and with every optional part
    dropped. Used only to seed sound-alike keys."""
    try:
        tree = _parse(pattern)
    except (IndexError, ValueError):
        return []
    found = []
    for alt in tree:
        for mode in ("max", "min"):
            t = " ".join(_render(alt, mode).split()).strip(" ,.'’")
            if sum(ch.isalpha() for ch in t) >= 2 and t not in found:
                found.append(t)
    return found


# ------------------------------------------------------------------ mapping

def _tokens(text: str):
    """(token, start, end) for whitespace-separated tokens of `text`."""
    return [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"\S+", text)]


_PUNCT = ".,!?;:\"'()[]{}<>।॥‘’“”"


def _clean(tok: str) -> str:
    t = tok.strip(_PUNCT)
    return t if t else tok


class EntityMap:
    def __init__(self, path, data, aliases_sidecar=None, validate=True):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.entries = []
        self.keep_terms = []
        side = aliases_sidecar or {}
        for section, items in (data or {}).items():
            if not isinstance(items, list):
                continue
            for k, it in enumerate(items):
                if not isinstance(it, dict) or "pattern" not in it:
                    if section in NOTE_SECTIONS and isinstance(it, str):
                        self.keep_terms += _keep_terms(it)
                    continue
                cat, render = SECTION_DEFAULTS.get(
                    section, (section.upper().rstrip("S") or "ENTITY",
                              "voice"))
                eid = f"{section}:{k}"
                flags = 0 if section == "abbreviations" else re.I
                pat = str(it["pattern"])
                extra = side.get(eid) or {}
                if extra.get("pattern_sha256") not in (
                        None, hashlib.sha256(pat.encode()).hexdigest()):
                    extra = {}           # sidecar written for an older pattern
                aliases = [str(a) for a in (it.get("aliases") or [])]
                aliases += [str(a) for a in extra.get("aliases") or []
                            if str(a) not in aliases]
                name = str(it.get("name") or extra.get("name") or "")
                e = {"id": eid, "section": section, "pattern": pat,
                     "rx": re.compile(pat, flags),
                     "replace": str(it.get("replace") or "").strip(),
                     "category": str(it.get("category") or cat).upper(),
                     "render": str(it.get("render") or render).lower(),
                     # a name that is also an ordinary word ("Vision"):
                     # only the context-reading LLM pass may mark it
                     "context_only": bool(it.get("context_only")),
                     "aliases": aliases, "name": name}
                lits = literals(pat)
                forms = ([name] if name else []) + aliases + lits
                e["forms"] = [f for f in dict.fromkeys(forms) if f.strip()]
                # a case-sensitive abbreviation ("BK") matches by its regex
                # only; sound-alike keys would turn it into a common syllable
                keyed = (e["forms"] if flags else
                         [f for f in e["forms"] if f not in lits])
                e["keys"] = sorted({key(f) for f in keyed
                                    if len(key(f)) >= 2})
                self.entries.append(e)
        self.by_id = {e["id"]: e for e in self.entries}
        # what DETECTION depends on — everything but the invented names. A
        # mapping whose replacements change (e.g. reconciled with another
        # project's names) re-renders, but never re-detects.
        self.detect_sha256 = hashlib.sha256(json.dumps(
            [[e["id"], e["pattern"], e["category"], e["render"],
              e["name"], e["aliases"], e["context_only"]]
             for e in self.entries]
            + [self.keep_terms], ensure_ascii=False).encode()).hexdigest()
        self._index()
        if validate:
            self.validate()
        else:
            self.warnings = []

    # ---------------------------------------------------------- validation
    def validate(self):
        """Hard errors: a voiced entry without a replacement, an invented
        name the real patterns match, or one that SOUNDS identical to a real
        form (same key). A merely similar one ("Concept B" for "Concept P")
        is a warning, recorded in self.warnings and the run summary."""
        problems, self.warnings = [], []
        for e in self.entries:
            if e["render"] == "voice" and not e["replace"]:
                problems.append(f"{e['id']}: voiced entry needs 'replace'")
            if e["render"] not in ("voice", "beep"):
                problems.append(f"{e['id']}: render must be voice|beep")
        for e in self.entries:
            rep = e["replace"]
            if not rep or e["render"] != "voice":
                continue
            for f in self.entries:
                if f["rx"].search(rep):
                    problems.append(f"{e['id']}: invented name '{rep}' "
                                    f"matches real pattern {f['id']}")
                rk = key(rep)
                for k in f["keys"]:
                    if rk == k:
                        problems.append(
                            f"{e['id']}: invented name '{rep}' sounds the "
                            f"same as a real form of {f['id']}")
                        break
                    if (len(rk) >= FUZZY_MIN_LEN and len(k) >= FUZZY_MIN_LEN
                            and difflib.SequenceMatcher(
                                None, rk, k).ratio() >= FUZZY_MIN):
                        self.warnings.append(
                            f"{e['id']}: invented name sounds close to a "
                            f"real form of {f['id']}")
                        break
        if problems:
            raise ValueError("mapping " + str(self.path) + ":\n  "
                             + "\n  ".join(dict.fromkeys(problems)))

    def _index(self):
        self._exact = {}                 # key -> entry
        self._fuzzy = []                 # [(key, entry, bigram set)]
        self._by_bigram = {}             # bigram -> {fuzzy index}
        self.max_window = 1
        for e in self.entries:
            if e["context_only"]:
                continue                 # never matched without context
            for k in e["keys"]:
                self._exact.setdefault(k, e)
                if len(k) >= FUZZY_MIN_LEN:
                    bg = _bigrams(k)
                    for g in bg:
                        self._by_bigram.setdefault(g, set()).add(
                            len(self._fuzzy))
                    self._fuzzy.append((k, e, bg))
            for f in e["forms"]:
                self.max_window = max(self.max_window,
                                      min(MAX_WINDOW, len(f.split()) + 1))
        self._keep_keys = {key(t) for t in self.keep_terms if key(t)}
        inv = [re.escape(e["replace"]) for e in self.entries
               if e["replace"]]
        self._invented_rx = (re.compile(r"(?<!\w)(?:" + "|".join(
            sorted(inv, key=len, reverse=True)) + r")(?!\w)", re.I)
            if inv else None)

    # ------------------------------------------------------------ matching
    def _key_match(self, k, fuzzy_min=FUZZY_MIN):
        """(entry, 'alias'|'fuzzy', ratio) for a window key, or None.
        Fuzzy candidates are found by shared letter pairs, not by first
        letter: a recogniser that drops an initial consonant (प्लमोरा heard
        as ला मोरा) must still be compared with the right name."""
        if not k or k in self._keep_keys:
            return None
        e = self._exact.get(k)
        if e is not None and len(k) >= 3:
            return e, "alias", 1.0
        if len(k) < FUZZY_MIN_LEN or k.isdigit():
            return None
        bg = _bigrams(k)
        counts = {}
        for g in bg:
            for i in self._by_bigram.get(g, ()):
                counts[i] = counts.get(i, 0) + 1
        best = None
        for i, c in counts.items():
            ak, ae, abg = self._fuzzy[i]
            if c < 0.4 * min(len(bg), len(abg)):
                continue
            if abs(len(ak) - len(k)) > max(2, 0.3 * len(ak)):
                continue
            sm = difflib.SequenceMatcher(None, k, ak)
            if sm.quick_ratio() < fuzzy_min:
                continue
            r = sm.ratio()
            if r >= fuzzy_min and (best is None or r > best[2]):
                best = (ae, "fuzzy", r)
        return best

    def match_tokens(self, toks, fuzzy_min=FUZZY_MIN):
        """toks: list of token strings. Returns non-overlapping hits
        [(i0, i1, entry, how)] (inclusive token indices), preferring
        regex > alias > fuzzy and longer spans."""
        cand = []
        # regex over the joined text, mapped back to tokens
        text, spans, pos = "", [], 0
        for t in toks:
            c = _clean(t)
            spans.append((pos, pos + len(c)))
            text += c + " "
            pos += len(c) + 1
        for e in self.entries:
            if e["context_only"]:
                continue
            for m in e["rx"].finditer(text):
                if not m.group(0).strip():
                    continue
                idx = [j for j, (a, b) in enumerate(spans)
                       if a < m.end() and m.start() < b]
                if idx:
                    cand.append((idx[0], idx[-1], e, "regex", 3))
        # sound-alike keys over 1..max_window token windows
        keys = [key(_clean(t)) for t in toks]
        for i in range(len(toks)):
            if not keys[i]:
                continue
            joined = ""
            for w in range(self.max_window):
                j = i + w
                if j >= len(toks):
                    break
                joined += keys[j]
                hit = self._key_match(joined, fuzzy_min)
                if hit:
                    e, how, _r = hit
                    cand.append((i, j, e, how, 2 if how == "alias" else 1))
        cand.sort(key=lambda c: (-c[4], -(c[1] - c[0]), c[0]))
        taken, out = set(), []
        for i0, i1, e, how, _p in cand:
            rng = set(range(i0, i1 + 1))
            if rng & taken:
                continue
            taken |= rng
            out.append((i0, i1, e, how))
        return sorted(out, key=lambda h: h[0])

    def find_spans(self, words, fuzzy_min=FUZZY_MIN):
        """Deterministic net over detect_pii.flatten_words() output.
        Returns spans in detect_pii's span format plus mapping fields."""
        spans = []
        by_utt = {}
        for w in words:
            by_utt.setdefault(w["ui"], []).append(w)
        for ws in by_utt.values():
            for i0, i1, e, how in self.match_tokens([w["w"] for w in ws],
                                                    fuzzy_min):
                spans.append(self.span(ws[i0]["i"], ws[i1]["i"], e,
                                       " ".join(w["w"] for w in ws[i0:i1 + 1]),
                                       ws[i0]["speaker"], "mapping_net", how))
        return spans

    def span(self, a, b, e, spoken, speaker, source, how, reason=""):
        confirmed = how in ("regex", "alias", "llm", "llm+net")
        person = e is not None and e["category"] == PERSON_CATEGORY
        return {"start_word": a, "end_word": b,
                "categories": (["FULL_NAME"] if person
                               else ["MAPPED_ENTITY"]),
                "spoken_text": spoken, "speaker": speaker,
                "public_figure": False,
                "confidence": "high" if confirmed else "low",
                "source": source,
                "reason": reason or (f"mapped entity ({how} match)"
                                     if e else "brand hint"),
                "mapping_id": e["id"] if e else None,
                "mapping_category": e["category"] if e else "BRAND_HINT",
                "match": how,
                # voice only what is confirmed and voiced by policy
                "render": ("voice" if e and confirmed and e["render"] == "voice"
                           else "beep")}

    def scan_text(self, text):
        """Real-name mentions surviving in a text (audit). Returns
        [{'line', 'quoted_text', 'entity', 'match'}]; invented names and
        redaction markers are masked first so they can never count."""
        hits = []
        for ln, line in enumerate(text.splitlines(), 1):
            masked = re.sub(r"<[A-Z_ ]+REDACTED>", " ", line)
            if self._invented_rx is not None:
                masked = self._invented_rx.sub(" ", masked)
            toks = [t for t, _a, _b in _tokens(masked)]
            for i0, i1, e, how in self.match_tokens(toks):
                hits.append({"line": ln,
                             "quoted_text": " ".join(toks[i0:i1 + 1]),
                             "entity": e["id"], "match": how})
        return hits

    # -------------------------------------------------------------- helpers
    def keyterms(self, cap=100):
        """Real spoken forms for transcription keyterm biasing (a listener
        that spells the brand consistently is easier to match)."""
        out = []
        for e in self.entries:
            for f in ([e["name"]] if e["name"] else []) + e["forms"][:2]:
                f = " ".join(f.split())
                if f and len(f) < 50 and f not in out and f.isascii():
                    out.append(f)
        return out[:cap]

    def invented_terms(self):
        return sorted({e["replace"] for e in self.entries if e["replace"]})

    def pseudonymise(self, s: str) -> str:
        """Apply every replacement to a name or path (outputs must not carry
        a real brand in their file names)."""
        for e in sorted(self.entries, key=lambda e: -len(e["pattern"])):
            # a function, not a template: an invented name may hold a
            # backslash (a path-like entry) that re.sub would read as an
            # escape ("bad escape \D")
            rep = e["replace"] or "REDACTED"
            s = e["rx"].sub(lambda _m, rep=rep: rep, s)
        return s

    def prompt_block(self):
        """Entity list for the LLM pass — real forms only; the invented names
        never reach the detector."""
        lines = []
        for k, e in enumerate(self.entries, 1):
            forms = "; ".join(e["forms"][:8])
            lines.append(f"E{k}\t{e['category']}\t{forms}\t"
                         f"regex: {e['pattern']}"
                         + ("\t(also an ordinary word: mark it only where "
                            "the context shows the brand)"
                            if e["context_only"] else ""))
        out = "ENTITIES (id, category, known forms, regex):\n" + "\n".join(lines)
        if self.keep_terms:
            out += ("\n\nKEEP — never mark these, the project keeps them: "
                    + "; ".join(self.keep_terms))
        return out

    def entry_for_prompt_id(self, pid):
        m = re.fullmatch(r"E(\d+)", str(pid or "").strip())
        if not m:
            return None
        k = int(m.group(1))
        return self.entries[k - 1] if 1 <= k <= len(self.entries) else None

    def summary(self):
        cats = {}
        for e in self.entries:
            cats[e["category"]] = cats.get(e["category"], 0) + 1
        return {"mapping_sha256": self.sha256,
                "detect_sha256": self.detect_sha256,
                "entries": len(self.entries),
                "by_category": cats,
                "voiced": sum(e["render"] == "voice" for e in self.entries),
                "warnings": list(self.warnings)}


def _bigrams(k: str):
    return {k[i:i + 2] for i in range(len(k) - 1)}


def _keep_terms(note: str):
    """'Pune / Bangalore (fieldwork locations)' -> ['Pune', 'Bangalore']."""
    core = re.sub(r"\(.*?\)", "", note)
    return [t.strip() for t in re.split(r"[/,]", core) if t.strip()]


def load(path, validate=True) -> EntityMap:
    import yaml           # pyyaml: needed only when a mapping is used
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    side_path = path.with_name(path.stem + ".aliases.yaml")
    side = (yaml.safe_load(side_path.read_text()) or {}
            if side_path.exists() else {})
    return EntityMap(path, data, side.get("entries") if side else None,
                     validate=validate)


_CACHE = {}


def active():
    """The EntityMap selected via PII_MAPPING_FILE, or None (cached)."""
    f = os.environ.get(ENV, "").strip()
    if not f:
        return None
    p = Path(f).resolve()
    stamp = (str(p), p.stat().st_mtime)
    if stamp not in _CACHE:
        _CACHE.clear()
        # PII_MAPPING_TRUSTED=1: a caller that already checked this exact
        # file (the dataset route) spares every arm the quadratic check
        _CACHE[stamp] = load(p, validate=os.environ.get(
            "PII_MAPPING_TRUSTED", "") != "1")
    return _CACHE[stamp]


# ---------------------------------------------------------------- self-test

def _self_test():
    import tempfile
    assert romanize("नोरवालिक") == "noravaalik", romanize("नोरवालिक")
    assert key("Norvalic") == key("norvalik")
    assert key("पुणे") == key("pune"), (key("पुणे"), key("pune"))
    assert "Mac Donald" in " ".join(literals("Ma?c ?Donald[’']?s?"))
    assert literals(r"\bZVX\b") == ["ZVX"]
    assert "in January 2022" in literals(
        r"(launched|entered India)? ?in January,? 2022")
    mapping = {
        "brands": [
            {"pattern": r"Zorvex(?:\s+Vision)?", "replace": "Kelmora",
             "aliases": ["ज़ोरवेक्स", "zor vex"]},
            {"pattern": r"Plumora\s+Bank", "replace": "Tandel Bank",
             "aliases": ["प्लमोरा", "Plumora"]}],
        "abbreviations": [{"pattern": r"\bZVX\b", "replace": "KLM"}],
        "agency": [{"pattern": r"Brightpath(?:\s+Research)?",
                    "replace": "Insightwell"}],
        "people": [{"pattern": r"\bRavindra\b", "replace": "Mohan"}],
        "keep": ["Pune / Nashik (fieldwork locations)"],
    }
    import yaml
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.yaml"
        p.write_text(yaml.safe_dump(mapping, allow_unicode=True))
        em = load(p)
        assert em.by_id["people:0"]["render"] == "voice"
        assert em.keep_terms == ["Pune", "Nashik"]
        toks = "maine ज़ोरवेक्स ka TV liya aur zorveks bhi".split()
        hits = em.match_tokens(toks)
        got = [(i0, i1, e["id"], how) for i0, i1, e, how in hits]
        assert (1, 1, "brands:0", "alias") in got, got
        assert (6, 6, "brands:0", "alias") in got, got
        toks2 = "I bank with Plumora Bank and zor vex is fine".split()
        got2 = [(i0, i1, e["id"], how)
                for i0, i1, e, how in em.match_tokens(toks2)]
        assert (3, 4, "brands:1", "regex") in got2, got2
        assert (6, 7, "brands:0", "alias") in got2, got2
        # a dropped initial consonant still matches (bigram candidates)
        got3 = [(e["id"], how) for _a, _b, e, how in
                em.match_tokens(["पण", "ला", "मोरा", "अनुभव"])]
        assert ("brands:1", "fuzzy") in got3, got3
        # abbreviations are case-sensitive; keep terms never match
        assert not em.match_tokens(["zvx"])
        assert em.match_tokens(["ZVX"])
        assert not em.match_tokens(["Pune"])
        # a fuzzy-only hit is beeped, a regex hit voiced
        sp = em.span(0, 0, em.by_id["brands:0"], "zorvecks", "s0",
                     "mapping_net", "fuzzy")
        assert sp["render"] == "beep" and sp["confidence"] == "low"
        sp = em.span(0, 0, em.by_id["brands:0"], "Zorvex", "s0",
                     "mapping_net", "regex")
        assert sp["render"] == "voice"
        sp = em.span(0, 0, em.by_id["people:0"], "Ravindra", "s0",
                     "mapping_net", "regex")
        assert sp["categories"] == ["FULL_NAME"] and sp["render"] == "voice"
        # audit scan ignores invented names and markers
        text = ("[00:00:01.00 - 00:00:02.00] speaker_0 - Kelmora is good, "
                "<BRAND REDACTED> too\n[x] speaker_1 - but Zorvex Vision no")
        h = em.scan_text(text)
        assert len(h) == 1 and h[0]["line"] == 2, h
        assert em.pseudonymise("Zorvex Vision study.mp4") == "Kelmora study.mp4"
        # an invented name that sounds like a real one is refused
        bad = dict(mapping, brands=[{"pattern": "Zorvex", "replace": "Zorvecks"}])
        p.write_text(yaml.safe_dump(bad, allow_unicode=True))
        try:
            load(p)
            raise AssertionError("near-homophone invented name accepted")
        except ValueError:
            pass
    print("entity_map self-test: OK")


if __name__ == "__main__":
    _self_test()
