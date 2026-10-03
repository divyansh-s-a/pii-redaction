#!/usr/bin/env python3
"""Spatial matching — the page/bbox analogue of the audio project's temporal
clustering. Same job, two dimensions instead of one.

Two rules, and the distinction between them matters:
  iou    symmetric overlap. Used to decide two readers found the SAME item.
  cover  how much of A lies inside B. Used to decide a redaction box actually
         COVERS a value. Coverage is deliberately asymmetric: a box ten times
         too large still covers, and is a separate (over-redaction) problem.
"""
from collections import defaultdict


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def iou(a, b):
    i = intersect(a, b)
    if not i:
        return 0.0
    ia = _area(i)
    u = _area(a) + _area(b) - ia
    return ia / u if u > 0 else 0.0


def cover(inner, outer):
    """Fraction of `inner` that lies inside `outer`."""
    i = intersect(inner, outer)
    if not i:
        return 0.0
    a = _area(inner)
    return _area(i) / a if a > 0 else 0.0


def covered_by_any(box, boxes, threshold=0.98):
    """Is this value hidden by the union of redaction boxes?

    Threshold 0.98, stricter than the audio project's 0.95: a 95%-covered word
    still shows readable glyph tops, and a half-covered name is still a name -
    the exact failure the human redactor left in patient-*.pdf.
    Union-aware, because two adjacent boxes may jointly cover one value.
    """
    if not boxes:
        return False
    best = max(cover(box, b) for b in boxes)
    if best >= threshold:
        return True
    # union: sum the covered area across boxes, subtracting pairwise overlaps
    parts = [intersect(box, b) for b in boxes]
    parts = [p for p in parts if p]
    if not parts:
        return False
    total = sum(_area(p) for p in parts)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            ov = intersect(parts[i], parts[j])
            if ov:
                total -= _area(ov)
    a = _area(box)
    return (total / a) >= threshold if a > 0 else False


def cluster(items, iou_min=0.3, cover_min=0.7, key=lambda x: x["bbox"]):
    """Union-find over boxes on ONE page -> list of clusters (lists of items).

    Repeated occurrences at DIFFERENT places stay separate clusters on purpose:
    Taxonomy v0.2 says every repetition is its own interval and a leak of any
    one is a leak. Merging them would hide exactly the failure this corpus
    exhibits - a name redacted in the body and left in the header.
    """
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            a, b = key(items[i]), key(items[j])
            if not a or not b:
                continue
            if iou(a, b) >= iou_min or cover(a, b) >= cover_min \
                    or cover(b, a) >= cover_min:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(items[i])
    return list(groups.values())


if __name__ == "__main__":
    a = [0, 0, 10, 10]
    assert iou(a, a) == 1.0
    assert iou(a, [20, 20, 30, 30]) == 0.0
    assert abs(cover([2, 2, 4, 4], a) - 1.0) < 1e-9
    assert abs(cover(a, [2, 2, 4, 4]) - 0.04) < 1e-9      # asymmetric
    assert covered_by_any([2, 2, 4, 4], [a])
    assert not covered_by_any([0, 0, 10, 10], [[0, 0, 5, 10]])   # half only
    # union of two halves does cover
    assert covered_by_any([0, 0, 10, 10], [[0, 0, 5.1, 10], [4.9, 0, 10, 10]])
    cl = cluster([{"bbox": [0, 0, 10, 10]}, {"bbox": [1, 1, 9, 9]},
                  {"bbox": [50, 50, 60, 60]}])
    assert len(cl) == 2 and sorted(len(c) for c in cl) == [1, 2]
    print("geometry self-test OK (iou, asymmetric cover, union coverage, "
          "clustering)")


def line_runs(tokens, gap_mult=1.6):
    """Split tokens into CONTIGUOUS runs: same visual line, horizontally close.

    tokens: [(index, text, Rect), ...]  ->  [[(index, text, Rect), ...], ...]

    A PII value is a contiguous piece of text. One rectangle spanning tokens
    that are not adjacent covers everything between them -- which is how a
    footer email address came to black out 74% of a lab report, and how the
    tick labels of a growth chart came to black out the whole plot. Emitting
    one box per run keeps the redaction on the ink and off everything else.

    Lines are formed by VERTICAL OVERLAP, never by sorting on a rounded
    y-centre: handwritten glyphs and skewed scans jitter by a few points per
    token, and a y-sort interleaves them out of x-order so that no two
    neighbours are ever adjacent and every run collapses to one token.
    """
    if not tokens:
        return []
    lines = []                       # [y0, y1, heights, [tokens]]
    for t in sorted(tokens, key=lambda t: (t[2].y0, t[2].x0)):
        r = t[2]
        for L in lines:
            med = sorted(L[2])[len(L[2]) // 2]
            ov = min(L[1], r.y1) - max(L[0], r.y0)
            if ov > 0.5 * min(med, r.height):
                L[3].append(t)
                L[2].append(r.height)
                if r.height <= 1.6 * med:      # outliers must not widen the band
                    L[0], L[1] = min(L[0], r.y0), max(L[1], r.y1)
                break
        else:
            lines.append([r.y0, r.y1, [r.height], [t]])
    runs = []
    for _, _, _, toks in lines:
        toks.sort(key=lambda t: t[2].x0)
        cur = [toks[0]]
        for t in toks[1:]:
            p, r = cur[-1][2], t[2]
            if r.x0 - p.x1 <= gap_mult * max(p.height, r.height):
                cur.append(t)
            else:
                runs.append(cur)
                cur = [t]
        runs.append(cur)
    return runs


def run_boxes(tokens, gap_mult=1.6):
    """line_runs() collapsed to one Rect per run."""
    out = []
    for run in line_runs(tokens, gap_mult):
        b = run[0][2]
        for _, _, r in run[1:]:
            b = b | r
        out.append(b)
    return out


def too_large(box, page_rect, max_frac=0.06):
    """A single value-anchored PII box larger than this is a localisation
    failure, not a large secret. Region redactions (QR, photo, label-anchored
    form fields) are deliberate and exempt -- callers skip the check."""
    pa = page_rect.width * page_rect.height
    if pa <= 0:
        return False
    return ((box[2] - box[0]) * (box[3] - box[1])) / pa > max_frac
