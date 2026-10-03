#!/usr/bin/env python3
"""OCR-tolerant value matching.

The pipeline's central promise is that a value learned anywhere in a bundle is
struck everywhere in it. The first implementation normalised case and
whitespace and then did a literal substring test -- which at this corpus's
median 109 dpi matches the one clean instance and misses every degraded one,
so the promise failed silently and every PSEUDONYMIZE row degraded into a
partial redaction.

What actually goes wrong at 109 dpi, from the corpus:
  'MARTIN GRIEVES'  read as  'MARTlN GRlEVES'   (I/l confusion)
  'BCAPT9189E'     read as  'BCAPT9I89E'      (1 -> I)
  'Dr. S. Menon'   read as  'Dr S Menon'      (punctuation dropped)
  '50100094060529' read as  '5O1OOO94O6O529'  (0 -> O)

So: fold confusables, strip punctuation and honorifics, then compare by
normalised edit distance over CONTIGUOUS token runs -- contiguous because a
value spanning non-adjacent tokens is not one value, and a box drawn around
non-adjacent tokens swallows everything between them.

    python3 document_pipeline/matching.py --self-test
"""
import re
import sys
import unicodedata

MAX_DIST = 0.2          # normalised edit distance; 1 error per 5 characters
MIN_LEN = 4             # below this, edit distance is noise

# Two folding regimes, because names and identifiers fail differently.
#
# A NAME degrades by segmentation and letter confusion, and a one-character
# difference almost always means the same person, so letters fold within their
# own class and a 20% edit distance is allowed on top.
#
# An IDENTIFIER is dense: every character carries information, and two real
# account numbers can differ by a single digit. Letters fold across to digits
# (that IS the OCR failure mode -- O for 0, I for 1) but NO edit distance is
# allowed afterwards. Folding does the work; slack would merge distinct accounts.
LETTER_FOLD = {"l": "i", "1": "i", "|": "i"}          # names: within-class only
ID_FOLD = {}
for _grp in ("0o", "1li", "5s", "8b", "2z"):
    for _ch in _grp:
        ID_FOLD[_ch] = _grp[0]

HONORIFIC = re.compile(
    r"^(dr|doctor|mr|mrs|ms|miss|master|mx|prof|professor|shri|smt|sri|"
    r"kumari|baby\s*of|b/o|s/o|w/o|d/o|c/o|late)\b[\.\s]*", re.I)
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")
HAS_DIGIT = re.compile(r"\d")


def is_identifier(*values):
    """Identifier regime if ANY side carries a digit.

    Deliberately asymmetric-safe: a value read as 'BCAPT9I89E' has no digit in
    the misread character but still has others, and treating the pair strictly
    is the safe direction -- a missed propagation over-redacts nothing and is
    caught by the carrier ledger, whereas a false merge silently unifies two
    different people's accounts.
    """
    return any(HAS_DIGIT.search(str(v) or "") for v in values)


def normalize(s, mode="name"):
    """Lowercase, strip accents / punctuation / honorifics / whitespace, then
    fold per regime. Whitespace goes entirely: OCR splits and joins words
    unpredictably inside boxed-character form fields."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = HONORIFIC.sub("", s.strip())
    s = PUNCT.sub("", s)
    s = WS.sub("", s).lower()
    table = ID_FOLD if mode == "id" else LETTER_FOLD
    return "".join(table.get(c, c) for c in s)


def edit_distance(a, b, cap=None):
    """Levenshtein with an early-exit band. cap=None means no early exit."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def similarity(a, b):
    """1.0 is identical; 0.0 is unrelated. Operates on normalised strings."""
    mode = "id" if is_identifier(a, b) else "name"
    na, nb = normalize(a, mode), normalize(b, mode)
    if not na or not nb:
        return 0.0
    d = edit_distance(na, nb)
    return 1.0 - d / max(len(na), len(nb))


def matches(a, b, max_dist=MAX_DIST):
    """True if `a` and `b` are the same value seen through OCR."""
    if is_identifier(a, b):
        na, nb = normalize(a, "id"), normalize(b, "id")
        return bool(na) and na == nb          # folding only; no slack
    na, nb = normalize(a, "name"), normalize(b, "name")
    if len(na) < MIN_LEN or len(nb) < MIN_LEN:
        return na == nb and bool(na)
    cap = int(max_dist * max(len(na), len(nb)))
    return edit_distance(na, nb, cap=cap) <= cap


def _adjacent(p, r, gap_mult=1.6):
    ov = min(p.y1, r.y1) - max(p.y0, r.y0)
    if ov <= 0.5 * min(p.height, r.height):
        return False
    return r.x0 - p.x1 <= gap_mult * max(p.height, r.height)


def find_runs(words, target, max_dist=MAX_DIST, gap_mult=1.6):
    """Every CONTIGUOUS run of tokens that reads as `target`.

    words: [(index, text, Rect)] in reading order -> [Rect, ...], one per
    occurrence. Runs only ever extend across tokens that are on the same line
    and adjacent, so a value printed three times yields three boxes rather than
    one rectangle spanning all three and everything between them.
    """
    mode = "id" if is_identifier(target) else "name"
    tgt = normalize(target, mode)
    if len(tgt) < MIN_LEN:
        return []
    cap = 0 if mode == "id" else int(max_dist * len(tgt))
    out, n = [], len(words)
    for i in range(n):
        acc, run = "", []
        best = None
        for j in range(i, n):
            w = words[j]
            if run and not _adjacent(run[-1][2], w[2], gap_mult):
                break
            piece = normalize(w[1], mode)
            if not piece and run:
                break
            acc += piece
            run.append(w)
            if len(acc) > len(tgt) + cap + 2:
                break
            if len(acc) >= len(tgt) - cap:
                # An identifier often arrives with its label glued on by OCR
                # ('IP#:IPMH2627000566'), so whole-run equality misses it and
                # the value then goes un-propagated across the whole bundle.
                # Containment is the right test for a dense identifier; the
                # length guard keeps it from swallowing a long neighbour.
                if mode == "id" and tgt in acc and len(acc) <= len(tgt) + 12:
                    best = (0, list(run))
                    break
                d = edit_distance(acc, tgt, cap=cap)
                if d <= cap and (best is None or d < best[0]):
                    best = (d, list(run))
                    if d == 0:
                        break
        if best:
            b = best[1][0][2]
            for _, _, r in best[1][1:]:
                b = b | r
            out.append(b)
    # one hit can be found from several start indices; keep the first of each
    uniq = []
    for b in out:
        if not any(abs(b.x0 - u.x0) < 1.5 and abs(b.y0 - u.y0) < 1.5
                   for u in uniq):
            uniq.append(b)
    return uniq


def self_test():
    import fitz
    # the confusions actually observed in this corpus
    pairs = [("MARTIN GRIEVES", "MARTlN GRlEVES"), ("BCAPT9189E", "BCAPT9I89E"),
             ("Dr. S. Menon", "S Menon"), ("50100094060529", "5O1OOO94O6O529"),
             ("HDFC0001869", "HDFCOOO1869"), ("RIVERBEND", "R1VERBEND")]
    for a, b in pairs:
        assert matches(a, b), f"should match: {a!r} vs {b!r}"
        print(f"  match   {a!r:<22} ~ {b!r}")
    # and the ones that must NOT match
    for a, b in [("MARTIN GRIEVES", "ROSALIND"), ("Riverbend", "Apollo"),
                 ("50100094060529", "50100094060530"),   # one digit apart
                 ("MRN 13677", "MRN 13699"),             # two digits apart
                 ("BCAPT9189E", "BCAPT9189F"),           # PAN last char differs
                 ("HDFC0001869", "HDFC0001868")]:
        assert not matches(a, b), f"should NOT match: {a!r} vs {b!r}"
    print("  non-matches correctly rejected")

    T = lambda i, t, x0, y0, x1, y1: (i, t, fitz.Rect(x0, y0, x1, y1))
    # 'MARTIN GRIEVES' split into boxed characters, with two OCR errors, and a
    # decoy 'GRIEVES' far away on another line that must not be joined in
    words = [T(0, "MARTlN", 100, 90, 140, 100), T(1, "GRlEVES", 142, 90, 195, 100),
             T(2, "ROSALIND", 100, 200, 150, 210),
             T(3, "MARTIN", 300, 400, 340, 410), T(4, "GRIEVES", 342, 400, 395, 410)]
    # an identifier with its label glued on by OCR must still be found
    glued = [T(0, "IP#:IPMH2627000566", 100, 50, 260, 60),
             T(1, "MRN", 300, 50, 330, 60)]
    gh = find_runs(glued, "IPMH2627000566")
    assert len(gh) == 1, f"glued-prefix identifier missed ({len(gh)} hits)"
    print(f"  glued 'IP#:IPMH2627000566' -> found the IP number ({len(gh)} box)")
    # but a DIFFERENT number sharing a prefix must not match
    assert not find_runs(glued, "IPMH2627000567"), "near-miss id matched"

    hits = find_runs(words, "MARTIN GRIEVES")
    print(f"  find_runs found {len(hits)} occurrences: "
          f"{[[round(v) for v in h] for h in hits]}")
    assert len(hits) == 2, f"expected 2 occurrences, got {len(hits)}"
    assert all(h.width < 120 for h in hits), "a run spanned unrelated tokens"

    # the old literal-substring behaviour would have found only the clean one
    literal = [w for w in words
               if "martingrieves".find(normalize(w[1], "name")) >= 0]
    print(f"  (literal substring test would have matched {len(literal)} bare tokens "
          f"- the bug this module replaces)")
    print("matching self-test OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
