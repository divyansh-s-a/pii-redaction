#!/usr/bin/env python3
"""Label-anchored field regions: redact where the VALUE must be, not where
OCR managed to read one.

On a filled paper form the printed labels survive OCR perfectly while the
handwritten values shatter into unusable fragments -- on the claim form in
this corpus the reader returns 'd) Name:' cleanly and then 'R', 'H', 'E',
'S', 'OP', 'N', 'A', '42' for the answers. No detector can find a name in
that, so no value-anchored redactor can cover it. Anchoring on the label
instead inverts the problem: the label says what the field IS, and geometry
says where its answer lives, so the value is destroyed whether or not anyone
could read it.

Generic by construction: the geometry knows nothing about insurance. Only
LEXICON is domain vocabulary, and a document-genre pack can extend it for
bank statements, HR files or CAD title blocks without touching the mechanism.

    python3 document_pipeline/formfields.py --self-test
"""
import re
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))

import fitz

# (pattern, pack category, multiline). Categories are PACK ids, not audio
# taxonomy names: the label says what a field IS, and only the pack decides what
# happens to it. Two splits matter and are easy to get wrong --
#   * bank NAME / BRANCH / IFSC are common infrastructure (T33, kept) while the
#     cheque/DD instrument and the account number are person-unique (T31, gone);
#   * a clinician's name is T29 (pseudonymised) while the patient's is T01.
# multiline fields keep claiming rows until the next label appears in the same
# left gutter (addresses wrap, names do not).
# ---------------------------------------------------------------- lexicon
# BASE vocabulary only: labels that mean the same thing on any document --
# names, addresses, contact, dates of birth, bank rows, government IDs,
# signatures. DOMAIN vocabulary lives in the pack that owns it, under a
# "lexicon" key in pack.json, and is compiled ONLY when that pack is selected.
#
# That split exists because this list used to be global, and a label added to
# serve one pack silently changed the other. Measured: a bare `patient` row
# added for the generic pack matched the END of charge-line descriptions on a
# real hospital bill -- "Amount Received From Patient  0.00", "Patient Payable
# 156559.00" -- and the answer area to the right of the label is the figure.
# Three bill totals were destroyed by a row that had nothing to do with bills.
# A pack-scoped entry cannot do that: it does not exist during another pack's
# run.
#
# Two traps that apply to BOTH base and pack vocabulary:
#   * Bare words are not field labels. 'cell' matched "Red Blood Cell";
#     'surgeon' matched a charge line; 'patient' matched a bill total.
#   * A region must never enter a numeric column (see numeric_columns).
LEXICON = [
    (r"^(company|insurer|insurance\s*company|tpa|employer|firm|organi[sz]ation)"
     r"'?s?\s*name$",                              "sem:org_name", False),
    (r"^(full\s*)?name$",                          "sem:person_name", False),
    (r"^(patient|insured|member|claimant|proposer|employee|customer|"
     r"account\s*holder)'?s?\s*name$",             "sem:person_name", False),
    (r"^name\s*of\s*(the\s*)?(patient|insured|person|claimant|employee|"
     r"applicant|proposer|deceased)$",             "sem:person_name", False),
    (r"^(nominee|witness|attendant|guardian|next\s*of\s*kin)'?s?\s*"
     r"(name)?$",                                  "sem:third_party", False),
    (r"^(father|mother|spouse|husband|wife|s/o|w/o|d/o|care\s*of)"
     r"'?s?\s*(name)?$",                           "sem:related_party", False),
    (r"^address$",                                 "sem:address", True),
    (r"^(residential|permanent|correspondence|communication|billing|"
     r"present|mailing|mail)\s*address$",          "sem:address", True),
    (r"^(city|town|state|district|taluk|village|locality|street)$",
                                                   "sem:geo", False),
    (r"^(pin|zip|postal)\s*code$",                 "sem:geo", False),
    # 'cell' and 'tel' REQUIRE a suffix. Bare 'cell' matched 'Red Blood Cell',
    # 'Packed Cell' and 'Red Cell (RDW)' on a haematology report and blacked out
    # three results rows -- a direct breach of owner rule 4.
    (r"^((phone|mobile|telephone|landline)\s*(no|number|nos)?"
     r"|(cell|tel|mob)\s*(no|number|nos)"
     r"|contact\s*(no|number|nos|details?))$",     "sem:contact", False),
    (r"^e-?\s*mail\s*(id|address)?$",             "sem:contact", False),
    (r"^pan(\s*(no|number|card))?$",               "sem:gov_id", False),
    # 'GST' alone is a bill CHARGE line, not the provider's registration. The
    # 15-character GSTIN is caught by its checksum, which cannot fire on a
    # rupee figure, so the label entry only ever cost us bill rows.
    (r"^(gstin|tin|tan)\s*(no|number)?$",           "sem:org_reg_code", False),
    (r"^(aadhaar|aadhar|uid)\s*(no|number|card)?$", "sem:gov_id", False),
    (r"^(passport)\s*(no|number)?$",               "sem:gov_id", False),
    (r"^(driving\s*licen[cs]e|dl)\s*(no|number)?$", "sem:gov_id", False),
    (r"^(voter\s*id|epic\s*(no|number)?)$",       "sem:gov_id", False),
    (r"^(account|a/c|bank\s*a/?c)\s*(no|number)$", "sem:bank_personal", False),
    (r"^(cheque|dd)\s*(no|number|payable\s*details)?$",  "sem:bank_personal", False),
    (r"^ifsc\s*(code)?$",                          "sem:bank_common", False),
    (r"^(bank\s*name(\s*(and|&)\s*branch)?|branch)$",  "sem:bank_common", False),
    (r"^(date\s*of\s*birth|dob|birth\s*date)$",  "sem:dob", False),
    (r"^age(\s*(years|yrs|in\s*years))?$",        "sem:age_sex", False),
    (r"^(sex|gender)$",                            "sem:age_sex", False),
    (r"^age\s*/\s*(sex|gender)$",                 "sem:age_sex", False),
    (r"^(occupation|profession|designation|employer)$",   "sem:occupation", False),
    (r"^(signature|sign|thumb\s*impression)(\s*of\s*.*)?$",  "sem:signature", False),
    (r"^(vehicle|registration)\s*(no|number)$",    "sem:unresolved", False),
]
COMPILED = [(re.compile(p, re.I), c, m) for p, c, m in LEXICON]

_PACK_COMPILED = {}


def compiled_for(pack):
    """Base vocabulary plus whatever THIS pack owns -- and nothing else.

    A pack with no `lexicon` key behaves exactly as before. Compilation is
    cached per pack id because find_labels runs once per page.
    """
    if pack is None:
        return COMPILED
    pid = getattr(pack, "id", None)
    if pid in _PACK_COMPILED:
        return _PACK_COMPILED[pid]
    rows = list(getattr(pack, "lexicon", ()) or ())
    extra = []
    for r in rows:
        try:
            pat, cat = r[0], r[1]
            multi = bool(r[2]) if len(r) > 2 else False
            extra.append((re.compile(pat, re.I), cat, multi))
        except Exception:
            continue                       # a bad row must not kill the page
    # pack rows FIRST: a domain pack may legitimately override a base reading
    out = extra + COMPILED
    _PACK_COMPILED[pid] = out
    return out


def vocab_for(pack):
    """VOCAB for OCR repair, widened by the pack's own label words."""
    if pack is None:
        return VOCAB, VOCAB_SET
    pid = getattr(pack, "id", None)
    k = ("vocab", pid)
    if k in _PACK_COMPILED:
        return _PACK_COMPILED[k]
    words = set(VOCAB)
    for r in (getattr(pack, "lexicon", ()) or ()):
        try:
            words.update(_vocab_words(r[0]))
        except Exception:
            continue
    # NOT `words -= NEVER_FIX`. NEVER_FIX protects an INPUT token from being
    # rewritten; it is not a list of words the lexicon may not aim at. "date"
    # is in both -- the input "date" must be left alone, and the input "Dale"
    # must still be repaired TO "date". Subtracting it here deleted every such
    # target from the pack vocabulary, so `_fix_token("Addrass")` corrected under
    # the base list and silently stopped correcting under every pack. Invisible
    # on clean text, and it only bites on OCR-damaged scans, which is the only
    # place the repair exists to help. The guard is applied to the input, in
    # _fix_token, which is the correct place.
    out = (sorted(words), words)
    _PACK_COMPILED[k] = out
    return out

# A BARE ONE-WORD LABEL INSIDE RUNNING TEXT IS NOT A LABEL.
# "IN THE CITY CIVIL COURT AT BENGALURU" and "before the Hon'ble XII
# Additional City Civil Judge" both yielded a `City` field whose answer area
# swallowed the rest of the line -- the court's identity and the hearing date.
# A form row is short and telegraphic; a sentence carries function words. Only
# SINGLE-WORD candidates are tested, so multi-word labels ("Date of
# Admission", "Name of Hospital") are never affected by this.
FUNCTION_WORDS = {"the", "of", "at", "in", "on", "to", "for", "and", "or",
                  "is", "are", "was", "were", "be", "been", "by", "with",
                  "from", "that", "this", "which", "before", "after", "under",
                  "shall", "will", "has", "have", "not", "as", "an", "a"}
PROSE_MIN_WORDS = 6
PROSE_MIN_FUNCTION = 2


def _is_prose_line(line_tokens):
    """Does this line read as a sentence rather than a form row?

    The colon veto matters more than it looks. On a scanned form, words from
    several short fields land on one visual row and OCR jumbles them --
    measured on case-48142 p37: "DRAFT GRE I, they Name: Registrar I, are
    Registrar Patient/Husband's/Parent(s) I, admission, det...". That line
    carries enough stray function words to read as prose, and without the veto
    the guard dropped REAL labels from it, including an address field. A
    colon anywhere on the line means at least one field is being named, so the
    line is form structure however badly it scanned.
    """
    toks = [str(t) for t in line_tokens]
    if any(":" in t for t in toks):
        return False
    ws = [re.sub(r"[^A-Za-z']", "", t).lower() for t in toks]
    ws = [w for w in ws if w]
    if len(ws) < PROSE_MIN_WORDS:
        return False
    return sum(1 for w in ws if w in FUNCTION_WORDS) >= PROSE_MIN_FUNCTION


STOP = re.compile(r"^(details|section|note|important|please|total|rs|amount|"
                  r"sl|s\.?\s*no)(\b|$)", re.I)
MAX_LABEL_TOKENS = 5
LEAD = re.compile(r"^[a-z]?[\)\.\-]\s*", re.I)      # "d)", "i.", "2-"


def _clean(s):
    s = LEAD.sub("", str(s).strip())
    return re.sub(r"[\s:：\-\.,;?]+$", "", s).strip()


def _lev(a, b):
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _expand_classes(pat, cap=8):
    """`hospitali[sz]ation` -> both real spellings, not two fragments.

    Rejecting fragments was right but incomplete: a character class in the
    MIDDLE of a word means the whole word never enters the vocabulary at all,
    so `hospitalization`, `organisation`, `licence` and `license` were all
    unknown and an OCR misread of them could never be repaired. Expanding the
    class recovers the real words. Only single-character classes are expanded
    and the result is capped, because this is vocabulary harvesting, not a
    regex engine.
    """
    out = [pat]
    for _ in range(4):
        nxt = []
        for cand in out:
            m = re.search(r"\[([A-Za-z]{2,4})\]", cand)
            if not m:
                nxt.append(cand); continue
            for ch in m.group(1):
                nxt.append(cand[:m.start()] + ch + cand[m.end():])
        if nxt == out:
            break
        out = nxt[:cap]
    return out


def _vocab_words(pat):
    """Literal words in a pattern -- NOT the fragments a regex splices.

    `licen[cs]e` contains no word "licen", but a naive [a-z]{4,} scrape adds
    one, and "Licence" then sits two edits from it and gets SHORTENED to
    "Licen" -- so the driving-licence label stopped matching entirely. Same for
    `organi[sz]ation` -> "organi"/"ation" and `driv(ing|er)` -> "driv".

    A word is a fragment when a character class opens right after it, when it
    follows one, or when a MANDATORY group opens after it. An OPTIONAL group
    (`hospital(i[sz]ation)?`) leaves the preceding word standing on its own.
    """
    out = []
    for variant in _expand_classes(pat):
        out.extend(_literal_words(variant))
    return out


def _literal_words(pat):
    out = []
    for m in re.finditer(r"[a-z]{4,}", pat):
        nxt, prv = pat[m.end():m.end() + 1], pat[m.start() - 1:m.start()]
        if nxt == "[" or prv in "])":
            continue
        if nxt == "(":
            depth, k = 0, m.end()
            while k < len(pat):
                if pat[k] == "(":
                    depth += 1
                elif pat[k] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if pat[k + 1:k + 2] != "?":        # group is mandatory
                continue
        out.append(m.group())
    return out


# every word the lexicon knows, harvested from its own patterns
VOCAB = sorted({w for pat, _, _ in LEXICON for w in _vocab_words(pat)})
VOCAB_SET = set(VOCAB)


# Ordinary English words that sit within one edit of a lexicon word. Snapping
# these is not OCR repair, it is corruption: "In CASE OF any issues, please
# contact ..." became the label "Care Of", and the rest of that sentence --
# including a phone number -- was then carved out as a relative's name field.
# Measured on case-48142: five such phantom labels across pages 28, 36 and 37,
# every one of them in running prose. The guard has to be explicit, because
# the only thing that silenced it was the word "case" incidentally appearing
# in an unrelated pattern, and deleting that pattern would bring it back.
def _system_words():
    """Real words from the OS dictionary, if there is one.

    NEVER_FIX below is a list of words I watched get corrupted on one corpus.
    That is a patch, not a rule: the next corpus contains different words. The
    GENERIC statement is "a correctly-spelled word of the language is not an
    OCR misread", and a dictionary states it without anyone having to predict
    which words appear. NEVER_FIX stays as a floor for machines with no
    dictionary and for domain words a dictionary lacks.
    """
    for cand in ("/usr/share/dict/words", "/usr/dict/words",
                 "/usr/share/dict/american-english"):
        try:
            txt = Path(cand).read_text(errors="ignore")
        except Exception:
            continue
        return {w.strip().lower() for w in txt.splitlines()
                if 4 <= len(w.strip()) <= 20 and w.strip().isalpha()}
    return set()


SYSTEM_WORDS = _system_words()

NEVER_FIX = {
    "case", "cases", "care", "cared", "date", "dates", "late", "rate", "rates",
    "name", "named", "nine", "line", "lines", "time", "times", "role", "role",
    "male", "mile", "note", "notes", "side", "size", "sign", "signs", "fine",
    "file", "files", "type", "types", "term", "terms", "form", "forms",
    "from", "firm", "free", "fees", "year", "years", "cost", "costs", "post",
    "past", "part", "parts", "test", "tests", "rest", "best", "list", "last",
    "page", "pages", "plan", "plans", "paid", "said", "same", "since",
    "state", "stated", "class", "claim", "claims", "value", "values",
    "total", "detail", "details", "number", "member", "matter", "patient",
    "present", "period", "person", "report", "reason", "result", "results",
    # one-edit neighbours of a lexicon word that are themselves common words.
    # "Industrial ESTATE" -> "State" turned a letterhead into a geo field and
    # destroyed the issuer's own city and postcode.
    "estate", "estates", "space", "trace", "grace", "place", "places",
    "scare", "share", "shares", "stage", "stake", "plane", "phone", "phase",
    "prone", "pane", "cane", "mane", "wane", "dare", "hare", "bare", "fare",
    "wire", "hire", "core", "bore", "more", "mode", "made", "mate", "gate",
    "gave", "cave", "wave", "face", "pace", "lace", "race", "races", "rice",
    "city", "cite", "site", "bite", "note", "vote", "rote", "route", "under",
    "order", "orders", "other", "offer", "often", "owner", "owners", "after",
    # longer words the two-edit budget reaches. Measured on case-48142:
    # "addressed" -> "address" turned "inconsistencies to be addressed to the
    # consultant" into an address field, and "Other" -> "Mother" invented a
    # relative on three pages of a discharge summary.
    "addressed", "addresses", "addressing", "brother", "another", "mothers",
    "consultants", "patients", "signature", "signatures", "assigned",
    "relative", "relatives", "district", "distance", "attendant", "attended",
    # "password" is two edits from "passport" and was being read as a
    # government ID; "pass" words generally sit close to one another.
    "password", "passwords", "passport", "passcode", "passenger",
}


# OCR confusions that cost TWO edits but are ONE mistake: the scanner merged
# or split glyphs. "Narne" for "Name" is the canonical one, and at a plain
# edit distance of 2 on a five-letter word it is indistinguishable from an
# unrelated word, so the short-word limit of 1 rejected it. Folding both sides
# before measuring costs nothing -- a real "rn" in "return" folds on both
# sides and still compares equal -- and it never touches output, only the
# comparison.
LIGATURE = [("rn", "m"), ("cl", "d"), ("vv", "w"), ("ii", "n")]


def _ocr_fold(s):
    """Fold confusables for COMPARISON only; never for output.

    Ligatures are folded only AWAY FROM THE WORD START. Folding them anywhere
    made "claim" -> "daim", which is one edit from "daie" (folded "Dale"), so
    "Dale" repaired to "Claim" instead of "Date". A scanner merges glyphs
    mid-word; a leading "cl" or "rn" is where the reader has the most context
    and OCR is most reliable, so the cost of folding there outweighs the gain.
    """
    s = s.lower()
    head, tail = s[:1], s[1:]
    for a, b in LIGATURE:
        tail = tail.replace(a, b)
    return (head + tail).translate(str.maketrans("l1|0", "iioo"))


def _fix_token(t, vocab=None, vocab_set=None):
    """OCR misreads printed labels ('Dale of Dicherge', 'Matemity'). Snap a
    token to the nearest lexicon word when it is within one edit (two for
    words of seven letters or more); leave everything else untouched.

    A correctly-spelled ordinary word is never a misread, so NEVER_FIX is
    checked before the edit-distance search.
    """
    core = re.sub(r"[^A-Za-z]", "", t)
    if len(core) < 4:
        return t
    vocab = VOCAB if vocab is None else vocab
    vocab_set = VOCAB_SET if vocab_set is None else vocab_set
    low = core.lower()
    # A word the lexicon already knows, a word this corpus taught us to leave
    # alone, or a correctly-spelled word of the language, is not a misread.
    # The dictionary check is skipped for words the LEXICON itself uses, so a
    # real label ("name", "address") is still a valid repair TARGET.
    if low in vocab_set or low in NEVER_FIX:
        return t
    if low in SYSTEM_WORDS and low not in vocab_set:
        return t
    lim = 1 if len(low) < 7 else 2
    folded = _ocr_fold(low)
    best, bd = None, lim + 1
    for w in vocab:
        if abs(len(w) - len(low)) > lim + 1:
            continue
        d = _lev(low, w)
        if d > lim:
            # second chance through the confusable fold, same limit
            d = _lev(folded, _ocr_fold(w))
        if d < bd:
            best, bd = w, d
    if best is None:
        return t
    # A candidate that is a strict prefix or suffix of the input is a fragment,
    # not a misread: real OCR damage substitutes and drops characters in the
    # middle ("Dicherge" -> "Discharge"), it does not cleanly truncate a word.
    if len(best) < len(low) and (low.startswith(best) or low.endswith(best)):
        return t
    return t.replace(core, best.capitalize() if core[0].isupper() else best)


def _same_line(a, b):
    ov = min(a.y1, b.y1) - max(a.y0, b.y0)
    return ov > 0.45 * min(a.height, b.height)


def find_labels(words, pack=None):
    """-> [{'category','label','box','multiline'}] for every printed label.

    Longest match wins: 'Name of Hospital where Admitted' must not collapse
    to 'Name', and 'Date of Birth' must not stop at 'Date'.
    """
    out = []
    compiled = compiled_for(pack)
    vocab, vocab_set = vocab_for(pack)
    n = len(words)
    used = set()
    for i in range(n):
        if i in used:
            continue
        acc, box, best = "", None, None
        for j in range(i, min(i + MAX_LABEL_TOKENS, n)):
            _, t, r = words[j]
            if box is not None and not (_same_line(box, r)
                                        and r.x0 - box.x1 <= 1.2 * r.height):
                break
            acc = (acc + " " + _fix_token(str(t), vocab, vocab_set)).strip()
            box = r if box is None else (box | r)
            cand = _clean(acc)
            if not cand or STOP.match(cand):
                continue
            for rx, cat, multi in compiled:
                if rx.match(cand):
                    best = (j, cat, cand, fitz.Rect(box), multi)
                    break
        if best:
            j, cat, cand, bx, multi = best
            # a ONE-WORD label sitting in a sentence is a coincidence, not a
            # field; multi-word labels are trusted as-is (see _is_prose_line)
            if len(cand.split()) == 1:
                line = [t for _, t, r in words if _same_line(bx, r)]
                if _is_prose_line(line):
                    continue
            out.append({"category": cat, "label": cand, "box": bx,
                        "multiline": multi})
            used.update(range(i, j + 1))
    return out


NUMERIC = re.compile(r"^[0-9][0-9.,:%/-]*$")
COL_TOL = 26.0          # points; how close two rows' figures must sit to be a column
COL_MIN_ROWS = 3        # a column is a column when three rows share it


def numeric_columns(words, tol=COL_TOL, min_rows=COL_MIN_ROWS):
    """x-ranges where numeric values stack across several rows.

    A bill, a ledger and a results table all put their figures in columns; a
    form field puts one value beside one label. So an x-range carrying numbers
    on three or more different rows is a DATA COLUMN, and no field-label region
    may extend into it. That single rule is what stops 'Surgeon', 'GST',
    'Other' or 'Package discount' from taking the amount printed beside them --
    those are charge lines, and the figure is the claim agent's working data.
    """
    nums = [(w[2].x0, w[2].x1, (w[2].y0 + w[2].y1) / 2) for w in words
            if NUMERIC.match(str(w[1]))]
    cols, used = [], [False] * len(nums)
    for i, (x0, x1, y) in enumerate(nums):
        if used[i]:
            continue
        group, used[i] = [(x0, x1, y)], True
        for j in range(i + 1, len(nums)):
            if used[j]:
                continue
            ax0, ax1, ay = nums[j]
            if abs(ax0 - x0) <= tol and abs(ay - y) > 3:
                group.append((ax0, ax1, ay))
                used[j] = True
        if len({round(g[2]) for g in group}) >= min_rows:
            cols.append((min(g[0] for g in group), max(g[1] for g in group)))
    return sorted(cols)


# Imperative openers mark INSTRUCTION text, not an answer. A guidance sheet
# pairs every field name with a sentence telling you what to write there, and
# a label-anchored redactor reads that sentence as the answer.
INSTRUCTION = re.compile(
    r"^(enter|indicate|tick|use|mention|specify|write|fill|read|attach|"
    r"provide|select|state|give|refer|please|in\s+case|if\s+(yes|no|any|the)|"
    r"same\s+as|as\s+per|do\s+not)\b", re.I)
GUIDANCE_PAGE = re.compile(
    r"(guidance\s+for\s+filling|instructions?\s+(for|to)\s+fill|"
    r"how\s+to\s+fill|guidelines?\s+for\s+filling|data\s+element)", re.I)


INK_EMPTY = 0.01        # fraction of dark pixels below which a field is blank


def _has_marks(page, box, thresh=INK_EMPTY):
    """Does this answer area contain ANY ink beyond blank paper?

    The decisive test for "is this field filled". OCR tokens and chromatic ink
    both miss BLACK ballpoint: the ink channel is colour-based and OCR cannot
    read handwriting at all, so a black-pen-filled field looked identical to an
    empty one and the region was suppressed -- every value on such a form went
    undetected. Measured on the synthetic cases: an empty ruled box is 0.000%
    dark, a pen-filled field 6.3-6.7%.
    """
    try:
        import numpy as np
        pm = page.get_pixmap(dpi=150, clip=box, colorspace=fitz.csGRAY)
        if not pm.samples:
            return False
        a = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w)
        return float((a < 140).mean()) > thresh
    except Exception:
        return True          # cannot tell -> assume filled, never suppress


def _as_rect(x):
    """Accept either a raw Rect from ink_boxes() or a finding dict."""
    if isinstance(x, dict):
        # value_regions() carries 'bbox'; find_labels() carries 'box'
        return fitz.Rect(x.get("bbox") or x["box"])
    return fitz.Rect(x)


def value_regions(page, words, labels=None, right_pad=2.0, ink=None,
                  pack=None):
    """Carve each label's answer area.

    Horizontally: from just past the label to the next label on the same line,
    else to the right edge of the page's ink. Vertically: the label's own row,
    extended down to the next left-gutter label for multiline fields.
    """
    labels = find_labels(words, pack) if labels is None else labels
    if not labels:
        return []
    # A page explaining how to fill a form contains no answers at all, only
    # field names beside descriptions of them. Nothing on it is redactable.
    page_text = " ".join(str(w[1]) for w in words[:400])
    if GUIDANCE_PAGE.search(page_text):
        return []
    cols = numeric_columns(words)
    ink = ink or []
    ink_r = max((w[2].x1 for w in words), default=page.rect.x1)
    ink_b = max((w[2].y1 for w in words), default=page.rect.y1)
    labels = sorted(labels, key=lambda L: (L["box"].y0, L["box"].x0))
    out = []
    for k, L in enumerate(labels):
        lb = L["box"]
        right = [o["box"].x0 for o in labels
                 if o is not L and _same_line(lb, o["box"])
                 and o["box"].x0 > lb.x1]
        x1 = min(right) - right_pad if right else min(ink_r, page.rect.x1) - 2
        # no vertical expansion: the row is the label's own glyph box.
        # The +/-1pt this used to add was the first half of a 2pt bleed
        # (actions.PAD adds the other) that reached into adjacent lines.
        y0, y1 = lb.y0, lb.y1
        if L["multiline"]:
            below = [o["box"].y0 for o in labels
                     if o["box"].y0 > lb.y1 + 1
                     and abs(o["box"].x0 - lb.x0) < 40]
            y1 = (min(below) if below else min(ink_b, page.rect.y1)) - 1.0
            y1 = min(y1, lb.y1 + 90)            # never run away down the page
        x0 = lb.x1 + right_pad
        # A field's value is one value. A region that would cover three or more
        # separate numeric tokens is not covering a value, it is covering the
        # numeric columns of a table -- so stop it at the first of them. This is
        # the structural backstop for the 'Red Blood Cell' class of false label
        # match: even when the lexicon is wrong, a results row survives.
        # never reach into a data column
        for cx0, _cx1 in cols:
            if x0 < cx0 < x1:
                x1 = min(x1, cx0 - 2)
        # and never swallow a run of figures even outside a detected column
        nums = sorted((w[2].x0 for w in words
                       if len(str(w[1])) >= 2 and NUMERIC.match(str(w[1]))
                       and y0 - 1 <= (w[2].y0 + w[2].y1) / 2 <= y1 + 1
                       and x0 <= w[2].x0 < x1))
        if len(nums) >= 2:
            x1 = min(x1, nums[0] - 2)
        if x1 - x0 < 8 or y1 - y0 < 4:
            continue
        box = fitz.Rect(x0, y0, x1, y1)
        inside = [w for w in words
                  if box.contains(fitz.Point((w[2].x0 + w[2].x1) / 2,
                                             (w[2].y0 + w[2].y1) / 2))]
        has_ink = any(not (box & _as_rect(h)).is_empty for h in ink)
        # An answer area holding no text, no pen ink AND no marks at all is an
        # EMPTY FIELD. Redacting it destroys printed scaffolding and removes
        # nothing -- it is how a blank CKYC form came to be 16% black. The
        # pixel test is what keeps this from also suppressing black-ballpoint
        # entries, which are invisible to the other two channels.
        if not inside and not has_ink and not _has_marks(page, box):
            continue
        # An answer area holding a sentence is a DESCRIPTION, not an answer.
        if inside and not has_ink:
            txt = " ".join(str(w[1]) for w in inside)
            # Count WORDS, not tokens. A boxed-character field renders one
            # glyph per printed cell, so 'ROSALIND' arrives as seven tokens of
            # length 1 -- counting tokens suppressed the very field this
            # mechanism exists for.
            prose = sum(1 for w in inside if len(str(w[1]).strip()) >= 3)
            if INSTRUCTION.match(txt.strip()) or prose >= 6:
                continue
        out.append({
            "categories": [L["category"]],
            "text": f"[{L['label']} field]",
            "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "source": "label_anchored",
            "label": L["label"],
            "locator": "label_anchored",
            "region": True,                     # exempt from the size gate
        })
    return out


def _column_test():
    """A bill: description on the left, three amount columns on the right."""
    import fitz
    W = lambda i, t, x0, y0, x1, y1: (i, t, fitz.Rect(x0, y0, x1, y1))
    words = []
    for k, (desc, amts) in enumerate([("Surgeon", ("12000", "0", "12000")),
                                      ("GST", ("2160", "0", "2160")),
                                      ("Other", ("450", "50", "400")),
                                      ("Patient Gown", ("150", "0", "150"))]):
        y = 300 + k * 20
        words.append(W(len(words), desc, 60, y, 60 + 9 * len(desc), y + 10))
        for c, a in enumerate(amts):
            x = 300 + c * 90
            words.append(W(len(words), a, x, y, x + 34, y + 10))
    cols = numeric_columns(words)
    doc = fitz.open(); pg = doc.new_page()
    regs = value_regions(pg, words)
    print(f"  bill rows: detected {len(cols)} numeric columns at "
          f"x={[round(c[0]) for c in cols]}")
    intrude = [r for r in regs if cols and r["bbox"][2] > cols[0][0]]
    for r in regs:
        print(f"    region {r['categories']} {r['label']!r} "
              f"x{r['bbox'][0]:.0f}-{r['bbox'][2]:.0f}")
    assert len(cols) >= 3, f"amount columns not detected ({len(cols)})"
    assert not intrude, f"{len(intrude)} region(s) reached into an amount column"
    print("  -> no region reaches an amount column")
    doc.close()


def _block_test():
    """Both failure directions, because they need opposite fixes: missing the
    recipient leaks a name and address, claiming the letterhead destroys the
    only clue to where the document came from."""
    def build(lines):
        d = fitz.open(); pg = d.new_page()
        for i, t in enumerate(lines):
            pg.insert_text((60, 90 + i * 14), t, fontsize=10)
        return d, pg
    cases = [
        (["Sunita Rao", "14 MG Road, Indiranagar", "Bengaluru 560038", "",
          "Dear Madam,", "", "We write regarding the matter."], True, 3),
        (["Apollo Diagnostics Pvt Ltd", "14 MG Road", "Bengaluru 560038"],
         False, 0),
        (["Northside Chambers", "14 MG Road", "Bengaluru 560038",
          "www.northside.example"], False, 0),
        (["The claimant attended the review on 4 March and confirmed that",
          "the address held on file at 14 MG Road remained correct 560038.",
          "No further action was identified by the reviewing officer."],
         False, 0),
        (["Ravi Kumar Menon", "Flat 3B, Sunrise Apartments", "Cross Road"],
         True, 3),
    ]
    for lines, want, nlines in cases:
        d, pg = build(lines)
        w = [(i, x[4], fitz.Rect(x[:4]))
             for i, x in enumerate(pg.get_text("words"))]
        got = unlabelled_identity_blocks(pg, w)
        assert bool(got) == want, (lines[0], "detected" if got else "missed")
        if want:
            n = int(got[0]["reason"].split()[0])
            assert n == nlines, f"block spans {n} lines, want {nlines}: annexed neighbours"
        d.close()
    return len(cases)


def _ocr_damage_test():
    """The repair path must be EXERCISED, not merely executed.

    Every other test feeds this module correctly-spelled labels. On clean text
    _fix_token returns its input unchanged, so a completely broken repair is
    indistinguishable from a working one -- which is exactly how a pack
    vocabulary that had lost every repair target passed 22 checks. These
    labels are damaged the way 109-dpi OCR damages them (rn->m, l->t, i->l,
    dropped letters), and the assertion is that they still resolve.
    """
    import base as _b
    pack = _b.load(Path(__file__).resolve().parent.parent
                   / "taxonomies" / "insurance_medical")
    # OCR overwhelmingly produces NON-WORDS, and those are repaired. Where it
    # happens to produce a real word of the language ("Dale" for "Date",
    # "Cocle" for "Code") the dictionary guard protects it and the repair is
    # NOT made -- a deliberate recall cost, taken because the opposite error
    # corrupts prose into phantom field labels and destroys the line beside
    # them. Those cases are listed under `protected` and asserted to be left
    # ALONE rather than mis-repaired.
    cases = [
        ("Datc of Dicherge:  14/08/2026", "sem:event_date"),
        ("Narne:  Sunita Rao", "sem:person_name"),
        ("Addrass:  14 MG Road", "sem:address"),
        ("Hospitalizalion due to:  fever", "sem:clinical"),
        ("Narne of Hospilal:  Apollo", "sem:provider_name"),
    ]
    protected = ["Dale", "Cocle", "Estate", "case", "Other", "born", "claim"]
    for w in protected:
        assert _fix_token(w) == w, \
            f"a real word was rewritten: {w!r} -> {_fix_token(w)!r}"
    missed = []
    for line, want in cases:
        d = fitz.open(); pg = d.new_page()
        pg.insert_text((60, 100), line, fontsize=10)
        w = [(i, x[4], fitz.Rect(x[:4]))
             for i, x in enumerate(pg.get_text("words"))]
        got = [L["category"] for L in find_labels(w, pack)]
        if want not in got:
            missed.append((line.split(":")[0], want, got))
        d.close()
    assert not missed, f"OCR-damaged labels no longer resolve: {missed}"
    return len(cases)


def self_test():
    _column_test()
    nb = _block_test()
    print(f"  unlabelled identity blocks: {nb} cases pass, "
          "both the leak and the letterhead direction")
    import readers
    src = (str(readers.corpus_dir()) + "/case-48142-6f4d11a6-0ce9-4395-a069-b814baf51664.pdf")
    wd = tempfile.mkdtemp(prefix="pii_selftest_")
    Path(wd).mkdir(parents=True, exist_ok=True)
    # The corpus-backed part of this test needs a real bundle. Without one it
    # SKIPS with a clear message; it must never crash on a fresh checkout, and
    # it must never pass silently either -- the count printed at the end says
    # how much was actually asserted.
    if not Path(src).exists():
        print(f"  (no corpus at {src}; set PII_CORPUS_DIR to run the "
              f"corpus-backed assertions -- structural checks above still ran)")
        return True
    d = fitz.open(src)
    r = readers.read_page(d, 1, workdir=wd)
    # pass the ink, exactly as run_document does. Without it every field whose
    # handwritten value OCR could not read looks EMPTY and is correctly
    # suppressed -- so a no-ink self-test measures the wrong call path.
    import handwriting
    ink, _ = handwriting.ink_boxes(d[0])
    labs = find_labels(r["words"])
    print("fuzzy:", _fix_token("Addrass"), "|", _fix_token("Narne"), "|",
          _fix_token("Hospitalization"), "|", _fix_token("Matemity"))
    # OCR repair must not rewrite correctly-spelled prose into a label
    # (Matemity stays as-is: 'maternity' is not in any lexicon pattern, so
    # VOCAB never learned it -- the repair only knows words the lexicon uses.)
    assert _fix_token("Addrass") == "Address", _fix_token("Addrass")
    # "Discharge" is INSURANCE vocabulary now, so base repair must NOT know it
    # and the insurance pack's own vocabulary MUST -- that is the isolation
    # working in both directions, on the OCR-repair path as well as matching.
    assert _fix_token("Dicherge") == "Dicherge", "base knows a pack's word"
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent.parent / "taxonomies"))
    import base as _b
    _ip = _b.load(Path(__file__).resolve().parent.parent
                  / "taxonomies" / "insurance_medical")
    _v, _vs = vocab_for(_ip)
    assert _fix_token("Dicherge", _v, _vs) == "Discharge", \
        _fix_token("Dicherge", _v, _vs)
    # A pack's vocabulary must be a SUPERSET of the base one. Subtracting
    # NEVER_FIX from it once removed every repair TARGET that is also a
    # protected input word, so "Dale" stopped becoming "Date" under every
    # pack while still working on the base list -- invisible on clean text.
    # "Address" is a BASE word; repairing to it through the PACK vocabulary
    # proves the pack list did not lose the base list. Uses a non-dictionary
    # misread, since a real word is protected by design.
    assert _fix_token("Addrass", _v, _vs) == "Address", \
        f"pack vocabulary lost a base repair target: {_fix_token('Addrass', _v, _vs)!r}"
    assert set(VOCAB) <= set(_v), "pack vocabulary is not a superset of base"
    assert _fix_token("case", _v, _vs) == "case", "NEVER_FIX not applied to input"
    print("  OCR repair is pack-scoped too "
          "(base does not know 'discharge'; insurance does)")
    for w in ("case", "CASE", "Case", "claim", "test", "results", "class"):
        assert _fix_token(w) == w, f"_fix_token corrupted {w!r} -> {_fix_token(w)!r}"
    print("  ordinary words survive OCR repair (case/claim/test/results)")
    nd = _ocr_damage_test()
    print(f"  {nd} OCR-DAMAGED labels still resolve "
          "(the repair path is exercised, not just executed)")
    regs = value_regions(d[0], r["words"], labs, ink=ink)
    print(f"page 1: {len(r['words'])} tokens -> {len(labs)} labels, "
          f"{len(regs)} value regions")
    for g in regs:
        b = g["bbox"]
        print(f"   {g['categories'][0]:<24}{g['label']!r:<26}"
              f"x{b[0]:6.0f}-{b[2]:<6.0f} y{b[1]:6.0f}-{b[3]:<6.0f}")
    want = {"sem:person_name", "sem:address", "sem:gov_id",
            "sem:bank_personal"}
    got = {g["categories"][0] for g in regs}
    missing = want - got
    print(f"\nrequired categories present: {sorted(want & got)}")
    if missing:
        print(f"MISSING: {sorted(missing)}")
    info = {"sem:dob", "sem:age_sex", "sem:clinical", "sem:provider_name",
            "sem:event_date", "sem:geo", "sem:bank_common"}
    print(f"policy-dependent categories found: {sorted(info & got)}")
    print("sem:policy_id: expected miss on this page (label unreadable under "
          "the handwriting) -> covered by the chromatic-ink path")
    d.close()
    return not missing





# ---------------------------------------------- unlabelled identity blocks
# THE LEAK THIS CLOSES. Every mechanism so far needs a hook: a printed label, a
# checksum, coloured ink, a visual object, or a value already learned
# elsewhere. A plain letter's recipient block has none of them --
#
#     Sunita Rao
#     14 MG Road, Indiranagar
#     Bengaluru 560038
#
# no colon, no keyword, no checksum, nothing to propagate from. On an
# insurance claim it never mattered because claim forms are labelled
# throughout; on the generic documents this pack exists for -- letters,
# notices, invoices, HR files -- it is the single most common way a name and
# address appear. Position and shape are the only evidence available.
#
# The hard part is not finding the block, it is telling a RECIPIENT block from
# the ISSUER's letterhead, which has identical geometry and sits on the same
# page. Guess wrong towards the issuer and a person leaks; guess wrong towards
# the person and G30 ISSUING_ORGANISATION is destroyed on every letter. So the
# org veto is keyed to the FIRST line only: a letterhead leads with the
# organisation's name, while "opp. City Hospital" in an address's third line
# must not veto anything.
POSTCODE = re.compile(r"(?<!\d)(?:\d{6}|\d{3}\s?\d{3}|\d{5}(?:-\d{4})?)(?!\d)")
STREET = re.compile(
    r"\b(road|rd|street|st|lane|ln|avenue|ave|nagar|colony|sector|block|"
    r"flat|apartments?|apt|plot|house|cross|main|marg|layout|phase|floor|"
    r"tower|villa|society|chawl|gali|p\.?\s?o\.?\s?box|near|opp|opposite|"
    r"behind|beside)\b", re.I)
ORG_MARK = re.compile(
    r"\b(pvt|private|ltd|limited|llp|inc|corp|corporation|company|co\.|gmbh|"
    r"plc|hospital|hospitals|clinic|nursing\s+home|bank|insurance|assurance|"
    r"university|college|school|institute|trust|foundation|society|"
    r"department|ministry|authority|board|council|commission|agency|"
    r"enterprises|industries|solutions|services|technologies|labs?|"
    r"laborator(y|ies)|diagnostics?|centre|center|&\s*co|and\s+sons)\b", re.I)
WEB_MARK = re.compile(r"(www\.|https?://|gstin|\bcin\b|reg\.?\s*no)", re.I)
HONORIFIC = re.compile(
    r"^(mr|mrs|ms|miss|dr|prof|shri|smt|sri|kum|m/s|to)\b[.,: ]*", re.I)

BLOCK_MAX_LINES = 6
BLOCK_MIN_LINES = 2
BLOCK_WIDTH_FRAC = 0.55      # an address column is narrow; prose is not


def _lines(words, tol=3.0):
    """Group words into visual lines -> [(y0, y1, x0, x1, [words])]."""
    # a word is (index, text, rect) -- the rect is LAST, matching find_labels
    rows = []
    for w in sorted(words, key=lambda w: (round(w[2].y0, 1), w[2].x0)):
        r = w[2]
        for L in rows:
            if r.y0 < L[1] - tol and r.y1 > L[0] + tol:
                L[0], L[1] = min(L[0], r.y0), max(L[1], r.y1)
                L[2], L[3] = min(L[2], r.x0), max(L[3], r.x1)
                L[4].append(w)
                break
        else:
            rows.append([r.y0, r.y1, r.x0, r.x1, [w]])
    for L in rows:
        L[4].sort(key=lambda w: w[2].x0)
    return sorted(rows, key=lambda L: L[0])


def _name_shaped(text):
    """Two to four capitalised words and nothing else -- a person's name line.

    Deliberately narrow. Widening it to "any capitalised run" swallows headings
    and section titles, and each of those is a whole block destroyed.
    """
    t = HONORIFIC.sub("", text.strip()).strip()
    parts = t.split()
    if not 2 <= len(parts) <= 4:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    return all(p[:1].isupper() and p[1:].islower() or p.isupper()
               for p in parts if len(p) > 1)


def unlabelled_identity_blocks(page, words, labels=None, cols=None,
                               pack=None):
    """Contiguous short-line blocks that read as a person's name and address.

    -> findings carrying `sem:unlabelled_identity_block`, which each pack maps
    itself (generic: G04 REDACT; insurance: the address row).
    """
    if not words:
        return []
    labels = find_labels(words, pack) if labels is None else labels
    cols = numeric_columns(words) if cols is None else cols
    page_text = " ".join(str(w[1]) for w in words[:400])
    if GUIDANCE_PAGE.search(page_text):
        return []
    lab_rects = [_as_rect(L) for L in labels]

    L = _lines(words)
    if not L:
        return []
    pw = float(page.rect.width) or 1.0

    # Split on BASELINE PITCH, not on the gap between glyph boxes. A text
    # rect runs from ascender to descender, so consecutive 14pt-leaded lines
    # sit only ~3pt apart and a blank separator line only ~17pt -- against any
    # threshold expressed in line HEIGHTS the two are indistinguishable, and
    # the salutation of a letter merged into the address block above it.
    # Pitch is scale-free: one blank line doubles it, whatever the font size.
    # ...and take the 25th percentile, not the median. On a short block half
    # the gaps ARE the doubled ones, so the median lands on the blank-line
    # pitch and nothing ever splits -- an address block then annexed the
    # salutation and the first line of body text below it.
    pitches = sorted(b[0] - a[0] for a, b in zip(L, L[1:]))
    pitch = pitches[max(0, len(pitches) // 4)] if pitches else 12.0
    blocks, cur = [], [L[0]]
    for prev, line in zip(L, L[1:]):
        if line[0] - prev[0] > 1.6 * max(pitch, 1.0):
            blocks.append(cur); cur = [line]
        else:
            cur.append(line)
    blocks.append(cur)

    out = []
    for b in blocks:
        if not BLOCK_MIN_LINES <= len(b) <= BLOCK_MAX_LINES:
            continue
        if max(l[3] - l[2] for l in b) > BLOCK_WIDTH_FRAC * pw:
            continue                      # running prose, not an address column
        texts = [" ".join(str(w[1]) for w in l[4]).strip() for l in b]
        joined = " ".join(texts)
        if len(joined) < 12:
            continue
        # A paragraph that happens to mention a street and a postcode is not
        # an address block. The width filter alone misses short ones: "the
        # address held on file at 14 MG Road remained correct 560038" fits in
        # a narrow column and carries both signals. An address is telegraphic;
        # a sentence carries function words.
        if sum(1 for t in texts if _is_prose_line(t.split())) >= max(
                1, len(texts) // 2):
            continue
        rect = fitz.Rect(min(l[2] for l in b), min(l[0] for l in b),
                         max(l[3] for l in b), max(l[1] for l in b))
        # a labelled field is already someone else's job
        if any(not (rect & r).is_empty for r in lab_rects):
            continue
        if any(rect.x1 > c[0] and rect.x0 < c[1] for c in cols):
            continue                      # inside a figures column
        # ISSUER VETO -- first line only (see the note above)
        if ORG_MARK.search(texts[0]) or WEB_MARK.search(joined):
            continue
        # POSITIVE evidence required; absence of a label is not evidence
        has_post = bool(POSTCODE.search(joined))
        has_street = bool(STREET.search(joined))
        has_name = any(_name_shaped(t) for t in texts[:2])
        if not ((has_post and has_street) or (has_name and (has_post or has_street))):
            continue
        out.append({
            "categories": ["sem:unlabelled_identity_block"],
            "text": joined[:120], "bbox": tuple(rect), "all_boxes": [tuple(rect)],
            "source": "unlabelled_block", "locator": "geometry",
            "reason": f"{len(b)} short lines; "
                      f"name={has_name} street={has_street} postcode={has_post}",
        })
    return out





# ------------------------------------------------- the table guard, extended
# numeric_columns() has always fenced LABEL-anchored regions out of a figures
# column. It did not fence the MODEL, and the model is the one that sweeps.
# Measured on a scanned bank statement: on the born-digital rendition the model
# flagged one value, the labelled 'Closing Balance'; on the 105-dpi scan, with
# the label structure gone, it flagged the entire balance column -- seven
# figures including every running balance and the transaction amounts. That is
# exactly the content G35 TABULAR_NUMERIC_CONTENT exists to keep, and the same
# failure as the original complaint that a bill's total was struck while its
# line items survived.
#
# The veto is deliberately narrow. A figures column is a transaction, quantity
# or results space, so a MONEY reading of a cell inside one is wrong. An
# IDENTIFIER inside one is not: a ledger can legitimately carry a column of
# account or card numbers, and those must still die. So the veto keys on the
# pack's own resolution of the finding, not on a hard-coded category list.
MONEYISH = ("person_money", "in_cell_value", "numeric_column")
# Content categories that a TABLE ROW protects. G35 keeps "line-item
# descriptions" in so many words, and a description is not a claim about the
# person: "Surgeon  1  12000  12000" on a bill is a charge line, yet the model
# read the word as health information and, under a pack that redacts health
# content, destroyed it. The figures beside it all survived, which is the same
# split-brain the original review objected to -- a bill whose totals were
# struck while its line items stood, only inverted.
ROWISH = ("clinical", "org_name", "provider_name", "furniture", "event_date")
ROW_MIN_FIGURES = 2


ROW_GUTTER = 36.0        # points; a gap this wide separates COLUMNS, not cells


def numeric_rows(words, cols, tol=4.0):
    """Table rows as (y0, y1, x0, x1) -- BOUNDED HORIZONTALLY.

    A y-band alone is not a table row. Returning one made the veto fire on
    anything that merely shared a horizontal line with a table, and on a
    two-column page that is a different column entirely: "Diagnosis: HIV
    positive" in a narrative column at x112 was flipped from REDACT to KEEP
    because an unrelated charges table at x350-475 happened to have a row at
    the same height. That is a LEAK, and an invisible one -- the removal gate
    skips KEEP findings, so it stays green.

    The row's true extent is reconstructed by walking LEFT from the first
    figures column across words separated by ordinary cell gaps, and stopping
    at a gutter. A description cell sits beside its figures; a separate column
    sits across a gutter from them.
    """
    if len(cols) < ROW_MIN_FIGURES:
        return []
    bands = []
    for _, t, r in words:
        if not NUMERIC.match(str(t).strip()):
            continue
        mid = (r.x0 + r.x1) / 2.0
        if not any(c[0] - 2 <= mid <= c[1] + 2 for c in cols):
            continue
        ymid = (r.y0 + r.y1) / 2.0
        for b in bands:
            if abs(b[0] - ymid) <= tol:
                b[1] += 1
                b[2] = min(b[2], r.y0); b[3] = max(b[3], r.y1)
                b[4] = min(b[4], r.x0); b[5] = max(b[5], r.x1)
                break
        else:
            bands.append([ymid, 1, r.y0, r.y1, r.x0, r.x1])

    out = []
    for b in bands:
        if b[1] < ROW_MIN_FIGURES:
            continue
        y0, y1, x0, x1 = b[2], b[3], b[4], b[5]
        # walk left from the figures across ordinary cell gaps only
        on_row = sorted((w for w in words
                         if y0 - tol <= (w[2].y0 + w[2].y1) / 2.0 <= y1 + tol
                         and w[2].x1 <= x0 + 1),
                        key=lambda w: w[2].x0, reverse=True)
        edge = x0
        for _, _, r in on_row:
            if edge - r.x1 > ROW_GUTTER:
                break
            edge = min(edge, r.x0)
        out.append((y0, y1, edge, x1))
    return out


def spare_numeric_columns(page, words, findings, pack, cols=None):
    """Enforce the table row: figures AND their descriptions stay.

    Two vetoes, both narrow:
      * a MONEY reading of a cell inside a figures column
      * a CONTENT reading of anything on a row that carries figures
    An identifier is spared by neither -- a ledger may legitimately hold a
    column of account numbers, and a person named in a line item still goes.
    """
    if not findings:
        return findings
    cols = numeric_columns(words) if cols is None else cols
    if not cols:
        return findings
    money_ids = {pack.resolve(f"sem:{k}") for k in MONEYISH}
    row_ids = {pack.resolve(f"sem:{k}") for k in ROWISH}
    # AN IDENTIFIER IS NEVER SPARED, AND AMBIGUITY COUNTS AS IDENTIFIER.
    # A pack may map several semantics onto one row: insurance_medical sends
    # person_money, card AND bank_personal all to T31, so a real account
    # number printed once per statement row -- which makes it a figures column
    # -- resolved to T31, satisfied the money test, and was SPARED. A leak,
    # produced by a guard written to prevent over-redaction. Where a row is
    # reachable from both a money semantic and an identifier semantic the two
    # cannot be told apart, so the veto stands down and the value is redacted.
    ident_ids = {pack.resolve(f"sem:{k}") for k in
                 ("card", "bank_personal", "gov_id", "policy_id",
                  "patient_id", "org_reg_code", "credential", "device_id")}
    money_ids.discard(None); row_ids.discard(None); ident_ids.discard(None)
    money_ids -= ident_ids
    row_ids -= ident_ids
    if not money_ids and not row_ids:
        return findings
    rows = numeric_rows(words, cols)
    for f in findings:
        cats = f.get("categories") or []
        # only a reading that is ABOUT money, and only when that is all it is;
        # a span also carrying an identifier keeps the stronger action
        ids = {c if c in pack.categories else pack.resolve(c) for c in cats}
        ids.discard(None)
        b = f.get("bbox")
        if not ids or not b:
            continue
        mid = (b[0] + b[2]) / 2.0
        ymid = (b[1] + b[3]) / 2.0
        if ids <= money_ids and any(c[0] - 2 <= mid <= c[1] + 2 for c in cols):
            f["action"] = "KEEP"
            f["spared"] = "money reading inside a figures column (G35/T34)"
        elif ids <= row_ids and any(y0 - 2 <= ymid <= y1 + 2
                                    and x0 - 2 <= mid <= x1 + 2
                                    for y0, y1, x0, x1 in rows):
            f["action"] = "KEEP"
            f["spared"] = "line-item description on a figures row (G35/T34)"
    return findings


if __name__ == "__main__":
    ok = self_test()
    sys.exit(0 if ok else 1)
