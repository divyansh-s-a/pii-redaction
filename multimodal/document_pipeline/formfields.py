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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))
# LAST insert wins position 0. This directory must beat the audio arm's
# pipeline/, which has its own redact.py: in the wrong order a later plain
# `import redact` resolved to the AUDIO redactor and crashed the suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    # The qualifier is REQUIRED. It was optional, so the bare word matched the
    # printed boilerplate on an actual Aadhaar letter -- "Aadhaar helps you
    # avail various Government services", "Carry Aadhaar in your smart phone"
    # -- and the rest of each sentence became the field's "value" and was
    # blacked out, while the Aadhaar number itself survived. A real label says
    # which number it wants; prose does not.
    (r"^(aadhaar|aadhar|uid)\s*(no|number|card(\s*(no|number))?)$",
                                                   "sem:gov_id", False),
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
    # The staff who treat the patient, as they are labelled on a patient
    # sticker ("Physician  Dr ...") and on OT and nursing forms ("NAME OF THE
    # ANAESTHETIST", "NAME OF THE SISTER"). Both packs name the role and cut
    # the name; without a label the name survived whenever the model did not
    # run. A bare "Doctor" is deliberately absent: as a column heading its
    # "value" would be the next heading.
    (r"^(physician|name\s*of\s*(the\s*)?(an(a)?esthetist|sister|nurse|"
     r"staff\s*nurse))$",                          "sem:clinician", False),
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
                  "shall", "will", "has", "have", "not", "as", "an", "a",
                  # second-person and modal forms, which printed instructions
                  # are full of: "Carry Aadhaar in your smart phone - use
                  # mAadhaar App" carried only one word from the list above
                  # and so read as a form row, letting bare 'phone' match.
                  "you", "your", "yours", "use", "using", "its", "our",
                  "their", "them", "can", "may", "must", "should", "would",
                  "any", "all", "into", "through", "about", "when", "if",
                  "please", "do", "does", "it", "we", "he", "she", "they"}
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


def find_labels(words, pack=None, prose_veto=True):
    """-> [{'category','label','box','multiline'}] for every printed label.

    Longest match wins: 'Name of Hospital where Admitted' must not collapse
    to 'Name', and 'Date of Birth' must not stop at 'Date'.

    `prose_veto=False` is for callers asking the opposite question -- "is
    there ANY identity label here?" -- where a missed label costs a leak and
    a spurious one costs nothing but a kept default. Two headings on one
    printed line ("NAME OF THE ANAESTHETIST  NAME OF THE SISTER") read as a
    sentence to the veto and vanish.
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
            # A label sitting in a SENTENCE is a coincidence, not a field.
            # This used to apply only to one-word candidates, on the theory
            # that a multi-word label is specific enough to trust. It is not:
            # an Aadhaar letter prints "Keep your mobile number & email ID
            # updated in Aadhaar", and both "mobile number" and "email ID"
            # matched, so the instruction was redacted as if it were a filled
            # field. The colon veto inside _is_prose_line keeps real form rows
            # safe, including OCR-jumbled ones.
            line = [t for _, t, r in words if _same_line(bx, r)]
            if prose_veto and _is_prose_line(line):
                continue
            out.append({"category": cat, "label": cand, "box": bx,
                        "multiline": multi})
            used.update(range(i, j + 1))
    return out


NUMERIC = re.compile(r"^[0-9][0-9.,:%/-]*$")
COL_TOL = 26.0          # points; how close two rows' figures must sit to be a column
COL_MIN_ROWS = 3        # a column is a column when three rows share it


COL_MAX_ROW_GAP = 40.0   # wider than any table row pitch, far narrower
                         # than the scatter of unrelated numbers on a page


def runs_append(cols, run, min_rows):
    """Record a contiguous run as a column if it has enough distinct rows.

    The y-extent is APPENDED to the tuple, so every existing c[0]/c[1] caller
    is unaffected. Without it a figures column vetoed anything sharing its
    x-range ANYWHERE on the page -- the same defect as the table-row guard
    that had no horizontal bound, turned through ninety degrees.
    """
    if len({round(g[2]) for g in run}) >= min_rows:
        cols.append((min(g[0] for g in run), max(g[1] for g in run),
                     min(g[2] for g in run), max(g[2] for g in run)))


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
        # A COLUMN IS CONTIGUOUS. Sharing a left edge is not enough: on an
        # Aadhaar letter, '202/' at y=176, a phone at y=276 and '3000' at
        # y=414 share an x and were called a column spanning the whole page,
        # which then vetoed the recipient address block sitting between them.
        # A real figures column -- a bill, a ledger, a results table -- has a
        # row pitch of 10-30pt, so split the group wherever the gap is far
        # larger than any row pitch and judge each run on its own.
        group.sort(key=lambda g: g[2])
        run = [group[0]]
        for prev, cur in zip(group, group[1:]):
            if cur[2] - prev[2] > COL_MAX_ROW_GAP:
                runs_append(cols, run, min_rows)
                run = [cur]
            else:
                run.append(cur)
        runs_append(cols, run, min_rows)
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
        for cx0, _cx1, *_yextent in cols:
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
            "multiline": bool(L.get("multiline")),
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

BLOCK_MAX_LINES = 12     # an Indian recipient block is name + relation +
                         # street + area + landmark + city + district + state
                         # + PIN + phone. At 6 every genuine one was rejected
                         # for being too long, which is why the mechanism
                         # never fired on the ID letters it exists for. The
                         # positive-evidence gate below is what keeps a longer
                         # cap honest: a block still needs street+postcode, or
                         # a name plus one of them.
BLOCK_MIN_LINES = 2
BLOCK_WIDTH_FRAC = 0.55      # an address column is narrow; prose is not


def _rotated(w):
    """A word whose glyphs run vertically rather than across.

    Vision reports a rect in PAGE space, so a 90-degree rotated word comes
    back narrow and tall. Grouping by vertical overlap then merges it into
    every horizontal line it spans: on an Aadhaar letter the rotated
    "Download / Date: / 16/11/2019" stamp running down the margin landed
    INSIDE the recipient address rows, so unlabelled_identity_blocks saw rows
    that were half stamp and half address. The address split across three
    blocks, no block carried a name, a street AND a postcode together, and
    the mechanism returned nothing on the page it exists for.

    Horizontal text more than two characters long is always far wider than
    tall -- about 4pt per character against a 7pt line height -- while these
    rotated words measure 0.6 to 1.4pt per character and are two to five
    times taller than wide. Wide margin either side of the threshold.
    """
    r = w[2]
    return (r.y1 - r.y0) > 1.5 * (r.x1 - r.x0) and len(str(w[1]).strip()) >= 3


def _lines(words, tol=3.0):
    """Group words into visual lines -> [(y0, y1, x0, x1, [words])]."""
    # a word is (index, text, rect) -- the rect is LAST, matching find_labels
    rows = []
    upright, turned = [], []
    for w in words:
        (turned if _rotated(w) else upright).append(w)
    for w in sorted(upright, key=lambda w: (round(w[2].y0, 1), w[2].x0)):
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

    # Rotated words are DROPPED, not returned as one-word lines. Returning
    # them keeps them interleaved by y with the horizontal text, and the
    # block builder -- which requires consecutive lines to share a column --
    # then breaks the address at every margin word it passes. The caller
    # wants horizontal blocks; reading a vertical stamp belongs to the
    # carrier pass. _lines() has a single caller, so this is not a loss of
    # information anyone else depends on.
    return sorted(rows, key=lambda L: (L[0], L[2]))


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
        # Lines are ordered by (y, x), so two side-by-side columns arrive
        # interleaved. Vertical pitch alone would stitch them into one block;
        # a block must also stay within one column, so require the next line
        # to overlap the running block horizontally.
        bx0 = min(l[2] for l in cur)
        bx1 = max(l[3] for l in cur)
        overlap = min(bx1, line[3]) - max(bx0, line[2])
        same_col = overlap > 0.35 * min(bx1 - bx0, line[3] - line[2])
        if line[0] - prev[0] > 1.6 * max(pitch, 1.0) or not same_col:
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
        # A labelled field is already someone else's job -- but only when the
        # labels actually CLAIM this block. Vetoing on any intersection meant
        # a single labelled line inside a nine-line address ("D/O:" on the
        # father's-name row) abandoned the whole block, leaving the other
        # eight lines -- name, street, city, PIN, phone -- with no cover at
        # all. Require the labelled regions to cover most of the block.
        def _area(q):
            return max(0.0, q.x1 - q.x0) * max(0.0, q.y1 - q.y0)
        area = max(_area(rect), 1.0)
        claimed = sum(_area(rect & r) for r in lab_rects
                      if not (rect & r).is_empty)
        if claimed >= 0.5 * area:
            continue
        # inside a figures column -- horizontally AND vertically. A column
        # elsewhere on the page says nothing about this block.
        if any(rect.x1 > c[0] and rect.x0 < c[1]
               and (len(c) < 4 or (rect.y1 > c[2] - 6 and rect.y0 < c[3] + 6))
               for c in cols):
            continue
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


# A BILL ROW CARRIES MONEY. A CLINICAL RESULTS ROW CARRIES MEASUREMENTS, and
# the two are indistinguishable to a guard that only asks "does this row have
# figures on it". That is how the veto came to protect an A-scan biometry
# chart: every measurement the model correctly flagged as clinical sat on a
# row of numbers and was flipped back to KEEP, silently overriding a pack
# that had deliberately chosen to remove it.
#
# A money amount is thousands-separated, or it is a two-decimal figure on a
# page that says somewhere that it is about money. A measurement -- 3.49,
# 321, 23.17 -- is neither.
THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?$")
DEC2 = re.compile(r"^\d+\.\d{2}$")
CURRENCY = re.compile(
    r"(?i)(?:^|\W)(?:rs\.?|inr|amount|amt|total|charges?|payable|invoice|"
    r"bill|subtotal|gross|nett?|tariff|rate)(?:\W|$)|\u20b9")


def _money_context(words):
    """Does this page claim to be about money at all?"""
    return bool(CURRENCY.search(" ".join(str(w[1]) for w in words[:600])))


def _money_row(words, y0, y1, x0, x1, money_ctx, need=2):
    """At least `need` money-shaped figures inside this row's own bounds."""
    n = 0
    for _i, t, r in words:
        ym, xm = (r.y0 + r.y1) / 2.0, (r.x0 + r.x1) / 2.0
        if not (y0 - 2 <= ym <= y1 + 2 and x0 - 2 <= xm <= x1 + 2):
            continue
        s = str(t).strip().lstrip("\u20b9").replace("Rs.", "").strip()
        if THOUSANDS.match(s) or (money_ctx and DEC2.match(s)):
            n += 1
            if n >= need:
                return True
    return False


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
    # IDENTITY IS NEVER A LINE-ITEM DESCRIPTION. An organisation or provider
    # name on a bill row is the provider's identity, which both packs now
    # destroy; sparing it here resurrected exactly what the issuer decision
    # was meant to remove. "Surgeon" is a charge line. "Dr Agarwal's Eye
    # Hospital" is not.
    row_ids -= {pack.resolve("sem:org_name"), pack.resolve("sem:provider_name")}
    row_ids.discard(None)
    if not money_ids and not row_ids:
        return findings
    rows = numeric_rows(words, cols)
    money_ctx = _money_context(words)
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
        elif ids <= row_ids and any(
                y0 - 2 <= ymid <= y1 + 2 and x0 - 2 <= mid <= x1 + 2
                and _money_row(words, y0, y1, x0, x1, money_ctx)
                for y0, y1, x0, x1 in rows):
            f["action"] = "KEEP"
            f["spared"] = "line-item description on a MONEY row (G35/T34)"
    return findings


if __name__ == "__main__":
    ok = self_test()
    sys.exit(0 if ok else 1)


# ------------------------------------------------ identity COLUMNS in a table
# A medication administration record, a nursing chart or a verification log is
# a ruled grid whose "Signature", "Administered by" or "Name & ID No." column
# holds nothing but staff sign-offs: a short handwritten name, a scrawl and a
# staff code in every cell. Each cell is small, so no single mark looks like
# much, and a classifier shown one cell sees "part of a clinical record" and
# keeps it -- on the review bundle the nurse names and staff codes in such a
# grid survived almost untouched. The column HEADER is the reliable evidence:
# it says what every cell below it is. Clinical content never lives in a
# signature column, so cutting the column body cannot destroy a clinical note.
IDENTITY_COLUMN = re.compile(
    r"(?i)\b(signature|sign\b|initials?|administered\s*by|verified\s*by|"
    r"verification|checked\s*by|given\s*by|name\s*(&|and)?\s*id|id\s*no|"
    r"emp(loyee)?\s*(id|code|no)|staff\s*(id|code|no)|nurse'?s?\s*name)")


RULE_CONTRAST = 35      # grey levels darker than the paper beside it. A
                        # photographed chart washed out by the light keeps
                        # its upper rules at grey 200-220 on white: at 50
                        # the "NURSE SIGN" column lost both rules above
                        # mid-page and its sign-offs were never found.


def _vertical_rules(page, dpi=100):
    """Long printed vertical lines -> [(x_pt, y0_pt, y1_pt)].

    A phone photo of a ring-bound chart is neither square nor flat: its
    column rules lean a degree or two and fade where the page curves, so a
    rule is rarely one straight run of dark pixels 8% of the page tall. A
    single tall opening kept only 2 of the 5 sign-off columns' rules on a
    medication order sheet. Short vertical runs survive the lean; bridging
    them sideways by a pixel and stitching collinear pieces across small gaps
    rebuilds the rule. A stitched rule must still span 8% of the page and be
    mostly ink along its length, so a column of stacked handwriting is not
    one.
    """
    import numpy as _np
    pm = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    g = _np.frombuffer(pm.samples, _np.uint8).reshape(pm.h, pm.w)
    return _rules_along_rows(g, 72.0 / dpi)


def _horizontal_rules(page, dpi=100):
    """Long printed horizontal lines -> [(y_pt, x0_pt, x1_pt)]; the same
    lean- and fade-tolerant search as _vertical_rules, on the transposed page."""
    import numpy as _np
    pm = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    g = _np.frombuffer(pm.samples, _np.uint8).reshape(pm.h, pm.w)
    return _rules_along_rows(_np.ascontiguousarray(g.T), 72.0 / dpi)


def _rules_along_rows(g, sc):
    """Lines running down axis 0 of grey image g -> [(pos, start, end)] in
    points, where pos is the line's position along axis 1."""
    import numpy as _np
    from scipy import ndimage as _ndi
    class _pm:                                   # the shape the body reads
        h, w = g.shape
    pm = _pm
    # dark, OR clearly darker than the paper beside it on the same row: a
    # hairline rule at 100 dpi is one anti-aliased pixel of grey 150-180,
    # which a fixed threshold drops wherever the print is light
    bg = _ndi.median_filter(g, size=(1, 15))
    dark = (g < 140) | (bg.astype(_np.int16) - g > RULE_CONTRAST)
    seg = max(12, int(pm.h * 0.025))
    runs = _ndi.binary_opening(dark, _np.ones((seg, 1)))
    runs = _ndi.binary_closing(runs, _np.ones((seg, 3)))
    lab, _n = _ndi.label(runs)
    pieces = []
    for i, sl in enumerate(_ndi.find_objects(lab)):
        if sl is None:
            continue
        ys, xs = sl
        if xs.stop - xs.start > 0.2 * (ys.stop - ys.start) + 6:
            continue                               # a blob, not a line
        yy, xx = _np.nonzero(lab[sl] == i + 1)
        top = xx[yy <= yy.min() + 2].mean() + xs.start
        bot = xx[yy >= yy.max() - 2].mean() + xs.start
        pieces.append([ys.start, ys.stop, top, bot, ys.stop - ys.start])
    pieces.sort(key=lambda q: q[0])
    gap_max, tol = 0.06 * pm.h, 4.0
    merged = []
    for q in pieces:
        for m in merged:
            if 0 <= q[0] - m[1] <= gap_max or (q[0] < m[1] and q[1] > m[0]):
                # extend m's lean to q's top and compare
                lean = (m[3] - m[2]) / max(m[1] - m[0], 1)
                if abs(m[3] + lean * (q[0] - m[1]) - q[2]) <= tol:
                    m[1] = max(m[1], q[1]); m[3] = q[3] if q[1] >= m[1] else m[3]
                    m[4] += q[4]
                    break
        else:
            merged.append(list(q))
    out = []
    for y0, y1, top, bot, ink in merged:
        span = y1 - y0
        if span < pm.h * 0.08 or ink < 0.6 * span:
            continue
        out.append(((top + bot) / 2 * sc, y0 * sc, y1 * sc))
    return out


def _header_chains(words, max_len=4):
    """Runs of up to `max_len` words read the way a header cell is read:
    rightward along a line, or down to the next line of the same cell.

    A narrow column's header wraps -- "Administered / by", "Doctor's /
    signature / with time" -- and a page-wide line grouping interleaves the
    wrapped words with their neighbours' by x ("... administration by
    Administered ..."), so a two-word header never matched.
    -> [(text, [(start, end) per word], [words])]"""
    up = [w for w in words if not _rotated(w) and str(w[1]).strip()]

    def nxt(w):
        r = w[2]; h = max(r.height, 1.0)
        right = [v for v in up if v is not w and _same_line(r, v[2])
                 and 0 <= v[2].x0 - r.x1 <= 1.2 * h]
        if right:
            return min(right, key=lambda v: v[2].x0)
        # the next line of the cell; OCR boxes of stacked lines overlap a
        # little, so "below" is judged by the top edge, not a positive gap
        below = [v for v in up if v is not w
                 and v[2].y0 > r.y0 + 0.5 * h
                 and -0.3 * h <= v[2].y0 - r.y1 <= 0.9 * h
                 and min(r.x1, v[2].x1) - max(r.x0, v[2].x0)
                 >= 0.3 * min(r.width, v[2].width)]
        return min(below, key=lambda v: v[2].y0) if below else None

    succ = {id(w): nxt(w) for w in up}
    out = []
    for w in up:
        chain, v = [w], succ[id(w)]
        while v is not None and len(chain) < max_len and v not in chain:
            chain.append(v); v = succ.get(id(v))
        text, spans, pos = "", [], 0
        for c in chain:
            t = str(c[1]); spans.append((pos, pos + len(t)))
            text += t + " "; pos += len(t) + 1
        out.append((text, spans, chain))
    return out


def identity_columns(page, words, pack=None):
    """Bodies of table columns whose header marks them as staff identity."""
    if not words:
        return []
    rules = _vertical_rules(page)
    if len(rules) < 3:
        return []                                  # not a ruled table
    H = page.rect.height
    hrules = None                                  # rendered only if needed
    out = []
    seen = set()
    for text, spans, chain in _header_chains(words):
        # Map each regex match back to the EXACT words that produced it, by
        # character offset. Locating a header by the whole line put every
        # "Signature" and "verification" header of a four-column sign-off row
        # into the SAME column, and missed multi-word headers like
        # "Name & ID No." whose words do not match the pattern one by one.
        for m in IDENTITY_COLUMN.finditer(text):
            if spans and m.start() >= spans[0][1]:
                continue           # every match is found from its own first word
            span = [chain[i] for i, (a, b) in enumerate(spans)
                    if a < m.end() and b > m.start()]
            if not span:
                continue
            hx0 = min(w[2].x0 for w in span); hx1 = max(w[2].x1 for w in span)
            hy0 = min(w[2].y0 for w in span); hy1 = max(w[2].y1 for w in span)
            hmid = (hx0 + hx1) / 2
            # The two rules that bracket the header. On a photographed chart
            # a rule can fade out across the header row itself, so it may
            # also START just below it -- but it must then run a good way
            # down the table: a signature's own upright stroke is a short
            # "rule" too.
            def brackets(r, reach=30.0):
                return (r[1] <= hy1 + reach and r[2] >= hy0
                        and r[2] - max(r[1], hy1) >= 0.15 * H)
            cand = [r for r in rules if brackets(r)]
            if not ([r for r in cand if r[0] <= hmid]
                    and [r for r in cand if r[0] > hmid]):
                # Where light washes out the top of a photographed chart,
                # its column rules are only found from half way down. Accept
                # rules that start well below the header -- but only when the
                # header sits in a real header ROW: a ruled line spanning it
                # at its foot. A caption above some unrelated table has none.
                if hrules is None:
                    hrules = _horizontal_rules(page)
                hh0 = max(w[2].height for w in span)
                floor = any(h[1] <= hx0 + 2 and h[2] >= hx1 - 2
                            and -hh0 <= h[0] - hy1 <= 2.5 * hh0
                            for h in hrules)
                if floor:
                    cand = [r for r in rules if brackets(r, reach=0.25 * H)]
            left = [r for r in cand if r[0] <= hmid]
            right = [r for r in cand if r[0] > hmid]
            if not left or not right:
                continue
            lx = max(left, key=lambda r: r[0]); rx = min(right, key=lambda r: r[0])
            if rx[0] - lx[0] > 0.35 * page.rect.width:
                continue                           # a wide text column, not a cell column
            # A column HEADER heads the column: the rules bracketing it begin
            # within a few text lines above it. "c) Company TPA ID No." half
            # way down a claim form's instructions table is a ROW label, and
            # taking it for a header blacked out 1500pt of printed guidance.
            hh = max(w[2].height for w in span)
            if hy0 - max(lx[1], rx[1]) > max(50.0, 4 * hh):
                continue
            bottom = min(lx[2], rx[2])
            body = fitz.Rect(lx[0] + 1.5, hy1 + 1.0, rx[0] - 1.5, bottom - 1.0)
            if body.height < 20 or body.is_empty:
                continue
            # A sign-off column holds pen: signatures, initials, staff codes,
            # names. Printed SENTENCES below the header mean the header was a
            # row label in a column of text -- on the instructions page of a
            # claim form the bodies held 6-18 prose lines; real sign-off
            # columns across the corpus held none.
            inside = [w for w in words
                      if body.contains(fitz.Point((w[2].x0 + w[2].x1) / 2,
                                                  (w[2].y0 + w[2].y1) / 2))]
            if sum(_is_prose_line([str(w[1]) for w in L[4]])
                   for L in _lines(inside)) >= 2:
                continue
            key = (round(body.x0), round(body.x1))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "categories": ["sem:signature"], "text": f"[{m.group(0)} column]",
                "bbox": [round(v, 2) for v in (body.x0, body.y0, body.x1, body.y1)],
                "all_boxes": [[round(v, 2) for v in (body.x0, body.y0, body.x1, body.y1)]],
                "n_boxes": 1, "source": "identity_column", "locator": "table_column",
                "region": True,
                "action": (pack.action_for(["sem:signature"]) if pack is not None
                           else "REDACT"),
                "reason": f"body of a table column headed {m.group(0)!r}",
            })
    return out
