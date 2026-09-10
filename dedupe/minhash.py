#!/usr/bin/env python3
"""MinHash signatures + LSH banding (numpy, vectorized).

A permutation family h_i(x) = a_i * x + b_i (mod 2^64, a_i odd) — the
"multiply-shift" universal family — approximates random permutations of the
64-bit token space. The MinHash of a set is the per-permutation minimum; the
fraction of equal signature entries between two sets estimates their Jaccard
similarity. LSH banding turns those signatures into bucket keys so only
plausibly-similar pairs are ever compared (near-linear, not O(n^2)).
"""
import hashlib

import numpy as np


class MinHasher:
    def __init__(self, num_perm: int, seed: int = 20260910):
        rng = np.random.default_rng(seed)
        # odd multipliers -> bijections on 2^64 (proper permutations)
        a = rng.integers(0, 1 << 63, size=num_perm, dtype=np.uint64) * 2 + 1
        b = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm,
                         dtype=np.uint64)
        self.a = a.astype(np.uint64)
        self.b = b.astype(np.uint64)
        self.num_perm = int(num_perm)

    def sign(self, tokens):
        """MinHash signature (uint64[num_perm]) for a token iterable, or None
        when there are no tokens."""
        if tokens is None or len(tokens) == 0:
            return None
        t = np.asarray(list(tokens), dtype=np.uint64)
        with np.errstate(over="ignore"):            # uint64 wraparound = mod 2^64
            hashed = self.a[:, None] * t[None, :] + self.b[:, None]
        return hashed.min(axis=1).astype(np.uint64)


def signatures(records, token_key, hasher: MinHasher):
    """path -> signature for every record that carries a non-empty token set."""
    sigs = {}
    for r in records:
        s = hasher.sign(r.get(token_key))
        if s is not None:
            sigs[r["path"]] = s
    return sigs


def lsh_candidate_pairs(sigs: dict, bands: int, rows: int):
    """Candidate pairs: any two signatures colliding in >=1 band bucket."""
    buckets = {}
    for path, sig in sigs.items():
        for band in range(bands):
            chunk = sig[band * rows:(band + 1) * rows].tobytes()
            key = (band, hashlib.blake2b(chunk, digest_size=8).digest())
            buckets.setdefault(key, []).append(path)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        m = sorted(set(members))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                pairs.add((m[i], m[j]))
    return pairs


def estimated_jaccard(sig_a, sig_b) -> float:
    return float(np.mean(sig_a == sig_b))
