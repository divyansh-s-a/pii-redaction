#!/usr/bin/env python3
"""Put the second listener's words on the primary listener's timeline.

--assurance unions intervals detected on a SECOND transcript (Gemini) with
the primary (ElevenLabs). The two transcripts share the audio, not their
clocks or their word numbering:
  * an interval's start_word/end_word index the SECOND transcript, but the
    redacted transcript is the primary's, so used as-is they hid whatever
    primary words happened to sit at those positions;
  * its start_ms/end_ms come from the second listener's clock — measured on a
    synthetic Marathi clip, up to ~1 s early — so a value only it found was
    beeped before it was spoken and stayed audible.

This aligns the two word sequences on a cross-script spelling key
(textnorm.fold_indic, so the same word written in Devanagari by both
listeners compares equal) and maps every second-listener word to primary
time and words:
  equal     the matching primary word
  replace   the primary words it replaces, time split in proportion (the
            primary heard something else there, but it is the same stretch
            of audio)
  insert    the gap between the neighbouring primary words (no primary word)
Long files are aligned in windows, with a margin for clock drift.

place() then CHECKS each interval: its own value text is compared with the
primary words at the aligned position and at the second listener's own time.
The clear winner is used; when neither wins, BOTH stretches are beeped (the
transcript marks the aligned words) — so the audio is never covered less
than by the second listener's own times alone. Replayed on 137 cached runs
(5,628 second-listener intervals), the aligned position matched the value
better in 75% of intervals, its own time in 6%, neither in 19%.
"""
import difflib
import unicodedata

import textnorm

WINDOW = 1500          # primary words per window
MARGIN_MS = 30000      # second-listener words considered around a window
_VARIANTS = (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ph", "f"), ("w", "v"),
             ("z", "j"), ("ck", "k"), ("c", "k"), ("q", "k"))


def key(word):
    """Loose cross-script spelling key for alignment (not for display)."""
    s = textnorm.fold_indic(unicodedata.normalize("NFC", str(word))).lower()
    s = "".join(ch for ch in s if ch.isalnum())
    for a, b in _VARIANTS:
        s = s.replace(a, b)
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def align(primary, second):
    """primary/second: detect_pii.flatten_words() lists. Returns
    {second_index: (start_ms, end_ms, [primary_indices])}."""
    pk = [key(w["w"]) for w in primary]
    sk = [key(w["w"]) for w in second]
    out = {}
    for w0 in range(0, len(primary), WINDOW):
        w1 = min(len(primary), w0 + WINDOW)
        t0 = primary[w0]["start_ms"] - MARGIN_MS
        t1 = primary[w1 - 1]["end_ms"] + MARGIN_MS
        js = [j for j in range(len(second))
              if t0 <= second[j]["start_ms"] <= t1]
        if not js:
            continue
        j0, j1 = js[0], js[-1] + 1
        sm = difflib.SequenceMatcher(None, pk[w0:w1], sk[j0:j1],
                                     autojunk=False)
        for tag, i1, i2, k1, k2 in sm.get_opcodes():
            i1, i2, k1, k2 = i1 + w0, i2 + w0, k1 + j0, k2 + j0
            if tag == "equal":
                for d in range(i2 - i1):
                    p = primary[i1 + d]
                    out.setdefault(k1 + d, (p["start_ms"], p["end_ms"],
                                            [i1 + d]))
            elif tag == "replace":
                s, e = primary[i1]["start_ms"], primary[i2 - 1]["end_ms"]
                n = k2 - k1
                for d in range(n):
                    out.setdefault(k1 + d, (int(s + (e - s) * d / n),
                                            int(s + (e - s) * (d + 1) / n),
                                            list(range(i1, i2))))
            elif tag == "insert":
                s = primary[i1 - 1]["end_ms"] if i1 > 0 else 0
                e = (primary[i1]["start_ms"] if i1 < len(primary)
                     else s + 500)
                for d in range(k2 - k1):
                    out.setdefault(k1 + d, (s, max(e, s + 1), []))
    return out


def retime(interval, mapping):
    """Copy of a second-listener interval on primary time, with primary word
    indices (None when no primary word lies in it), or None if unmapped."""
    got = [mapping[j] for j in range(interval["start_word"],
                                     interval["end_word"] + 1)
           if j in mapping]
    if not got:
        return None
    s = min(g[0] for g in got)
    e = max(g[1] for g in got)
    idx = sorted({i for g in got for i in g[2]})
    return dict(interval, start_ms=s, end_ms=e,
                start_word=idx[0] if idx else None,
                end_word=idx[-1] if idx else None,
                retimed_from=[interval["start_ms"], interval["end_ms"]])


MARGIN = 0.1           # similarity lead that counts as a clear winner
STRETCH_MAX = 3.0      # aligned span at most this many times its own span
STRETCH_ABS_MS = 5000  # ... or its own span plus this, whichever is larger


NARROW_MIN = 0.6       # best-run similarity needed to narrow a span


def _best(value_key, words, n_tokens):
    """(similarity, i0, i1) of the run of words that best matches the
    value. Scoring the best RUN, not the whole span, matters: a second
    listener that stamps coarse 5-second spans would otherwise lose every
    comparison, its own span diluted by the words around the value."""
    best = (0.0, None, None)
    if not value_key or not words:
        return best
    keys = [key(w["w"]) for w in words]
    for size in sorted({max(1, n_tokens - 1), n_tokens, n_tokens + 1}):
        for i in range(0, len(words) - size + 1):
            k = "".join(keys[i:i + size])
            if not k:
                continue
            r = difflib.SequenceMatcher(None, value_key, k).ratio()
            if r > best[0]:
                best = (r, i, i + size - 1)
    return best


def place(interval, mapping, primary, second):
    """-> list of 1-2 intervals on primary time for one second-listener
    interval, each tagged placed_by: aligned | own_time | both."""
    value = interval.get("spoken_text") or " ".join(
        w["w"] for w in second[interval["start_word"]:
                               interval["end_word"] + 1])
    vk = key(value)
    s0, e0 = interval["start_ms"], interval["end_ms"]

    def own_words():
        idx = [w_i for w_i, w in enumerate(primary)
               if min(w["end_ms"], e0) - max(w["start_ms"], s0)
               >= 0.5 * max(1, w["end_ms"] - w["start_ms"])]
        return idx
    own_idx = own_words()
    own = dict(interval, start_word=own_idx[0] if own_idx else None,
               end_word=own_idx[-1] if own_idx else None)
    r = retime(interval, mapping)
    if r is None:
        return [dict(own, placed_by="own_time")]
    # an alignment that stretches a value far beyond its own span (where the
    # transcripts diverge for a while, or the primary's word times are
    # themselves stretched) is not trusted: keep the second listener's own
    # placement. Observed: a 3-word value stretched to 47 s.
    own_dur = max(1, e0 - s0)
    if r["end_ms"] - r["start_ms"] > max(STRETCH_MAX * own_dur,
                                         own_dur + STRETCH_ABS_MS):
        return [dict(own, placed_by="own_time")]
    al_words = (primary[r["start_word"]: r["end_word"] + 1]
                if r["start_word"] is not None else [])
    ov = min(r["end_ms"], e0) - max(r["start_ms"], s0)
    if ov >= 0.5 * max(1, min(r["end_ms"] - r["start_ms"], e0 - s0)):
        return [dict(r, placed_by="aligned")]   # both clocks agree
    n = max(1, len(value.split()))
    sa = _best(vk, al_words, n)[0]
    own_w = [primary[i] for i in own_idx]
    so, o0, o1 = _best(vk, own_w, n)
    if so >= NARROW_MIN:
        # beep the matching words, not the whole (often coarse) own span
        own = dict(interval, start_ms=own_w[o0]["start_ms"],
                   end_ms=own_w[o1]["end_ms"], start_word=own_idx[o0],
                   end_word=own_idx[o1])
    if sa >= so + MARGIN:
        return [dict(r, placed_by="aligned")]
    if so >= sa + MARGIN:
        return [dict(own, placed_by="own_time")]
    second_copy = dict(own, placed_by="both")
    if so < NARROW_MIN:
        second_copy.update(start_word=None, end_word=None)  # audio only
    if "annotation_id" in second_copy:
        second_copy["annotation_id"] += "_own_time"
    return [dict(r, placed_by="both"), second_copy]


def _self_test():
    # second listener: a second early, heard the brand the primary misheard
    P = [("namaskar", 100, 900), ("mi", 980, 1100), ("Doorways", 3240, 3900),
         ("cha", 3920, 4100), ("tv", 4140, 4500)]
    G = [("नमस्कार", 0, 700), ("मी", 800, 950), ("धोरवेक्स", 2400, 3100),
         ("चा", 3200, 3350), ("टीव्ही", 3400, 3700)]
    pw = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in P]
    gw = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in G]
    m = align(pw, gw)
    r = retime({"start_word": 2, "end_word": 2, "start_ms": 2400,
                "end_ms": 3100}, m)
    assert (r["start_ms"], r["end_ms"]) == (3240, 3900), r
    assert (r["start_word"], r["end_word"]) == (2, 2), r
    # Devanagari in both transcripts aligns word for word
    assert key("नमस्कार") == key("नमस्कार,") and key("Mee") == key("mi")
    # a word only the second listener heard maps to the gap, no words
    P2 = [("call", 0, 300), ("me", 320, 500), ("today", 900, 1300)]
    G2 = [("call", 0, 280), ("me", 300, 450), ("Ravi", 500, 800),
          ("today", 850, 1250)]
    m2 = align([{"w": w, "start_ms": s, "end_ms": e} for w, s, e in P2],
               [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in G2])
    r2 = retime({"start_word": 2, "end_word": 2, "start_ms": 500,
                 "end_ms": 800}, m2)
    assert (r2["start_ms"], r2["end_ms"], r2["start_word"]) == (500, 900,
                                                                None), r2
    # place(): the clear winner is used; a tie beeps both stretches
    iv = {"start_word": 2, "end_word": 2, "start_ms": 2400, "end_ms": 3100,
          "spoken_text": "धोरवेक्स", "annotation_id": "x"}
    got = place(iv, m, pw, gw)
    assert [g["placed_by"] for g in got] in (["aligned"], ["both", "both"])
    assert got[0]["start_ms"] == 3240
    P3 = [("my", 0, 200), ("number", 220, 600), ("is", 620, 700),
          ("nine", 720, 1000), ("eight", 1020, 1300)]
    G3 = [("my", 0, 200), ("number", 220, 600), ("is", 620, 700),
          ("nine", 720, 1000), ("eight", 1020, 1300)]
    m3 = align([{"w": w, "start_ms": s, "end_ms": e} for w, s, e in P3],
               [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in G3])
    got3 = place({"start_word": 3, "end_word": 4, "start_ms": 720,
                  "end_ms": 1300, "spoken_text": "nine eight"}, m3,
                 [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in P3],
                 [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in G3])
    assert len(got3) == 1 and (got3[0]["start_word"], got3[0]["end_word"]) \
        == (3, 4), got3
    # a coarse own span (5 s) holding the value, alignment pointing at a
    # repeat of the word elsewhere: both are beeped, the own one narrowed
    P5 = [("ok", 0, 400), ("Dewi", 500, 900), ("bank", 1000, 1300),
          ("Mandala", 1400, 1900), ("yes", 2000, 2300), ("so", 9000, 9300),
          ("Mandala", 9400, 9900), ("again", 10000, 10300)]
    G5 = [("ok", 0, 400), ("Dewi", 500, 900), ("bank", 1000, 1300),
          ("Mandala", 1400, 1900), ("yes", 2000, 2300), ("so", 9000, 9300),
          ("Mandala", 9400, 9900), ("again", 10000, 10300)]
    pw5 = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in P5]
    gw5 = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in G5]
    got5 = place({"start_word": 6, "end_word": 6, "start_ms": 0,
                  "end_ms": 5000, "spoken_text": "Mandala"},
                 align(pw5, gw5), pw5, gw5)
    assert [g["placed_by"] for g in got5] == ["both", "both"], got5
    assert (got5[1]["start_ms"], got5[1]["end_ms"]) == (1400, 1900), got5
    # an implausibly stretched alignment falls back to the own placement
    P4 = [("pass", 0, 300), ("word", 320, 47000), ("is", 47100, 47300),
          ("x", 47400, 47600)]
    G4 = [("pass", 0, 300), ("wrd", 46000, 46400), ("is", 47100, 47300),
          ("x", 47400, 47600)]
    pw4 = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in P4]
    gw4 = [{"w": w, "start_ms": s, "end_ms": e} for w, s, e in G4]
    got4 = place({"start_word": 1, "end_word": 1, "start_ms": 46000,
                  "end_ms": 46400, "spoken_text": "wrd"},
                 align(pw4, gw4), pw4, gw4)
    assert [g["placed_by"] for g in got4] == ["own_time"], got4
    print("align_listeners self-test: OK")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _self_test()
