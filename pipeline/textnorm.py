#!/usr/bin/env python3
"""Shared text normalization helpers (pipeline-level so both the
pipeline and the judge tools can use them)."""

# Indic -> rough Latin skeleton, for cross-script value matching. Devanagari,
# Bengali, Telugu and Kannada share one Unicode block layout, so a single
# per-offset table transliterates the main consonants/vowels/matras of all
# four ("ಸುಧಾ" -> "sudha", "ఆయుష్" -> "ayush"). Deliberately lossy: it only
# has to push difflib similarity of same-name spellings across scripts over
# the matcher threshold, not romanize correctly.
_INDIC_BASES = (0x0900, 0x0980, 0x0C00, 0x0C80)   # Deva, Beng, Telu, Knda
_INDIC_OFFSETS = {
    0x01: "n", 0x02: "n", 0x03: "",                            # nasal marks
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u",     # vowels
    0x0A: "u", 0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",   # consonants
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3C: "", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u",      # nukta, matras
    0x42: "u", 0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4D: "",                # 4D = virama
    0x60: "ri", 0x62: "", 0x63: "",
    0x66: "0", 0x67: "1", 0x68: "2", 0x69: "3", 0x6A: "4",     # digits
    0x6B: "5", 0x6C: "6", 0x6D: "7", 0x6E: "8", 0x6F: "9",
}
_INDIC_FOLD = {}
for _b in _INDIC_BASES:
    for _off, _lat in _INDIC_OFFSETS.items():
        _INDIC_FOLD[_b + _off] = _lat


def fold_indic(text):
    """Fold Devanagari/Bengali/Telugu/Kannada chars in `text` to the Latin
    skeleton above; unmapped chars inside those blocks are dropped, everything
    else passes through unchanged."""
    out = []
    for ch in str(text or ""):
        cp = ord(ch)
        if any(b <= cp < b + 0x80 for b in _INDIC_BASES):
            out.append(_INDIC_FOLD.get(cp, ""))
        else:
            out.append(ch)
    return "".join(out)
