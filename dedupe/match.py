#!/usr/bin/env python3
"""Turn fingerprint records into duplicate edges (a, b, score, tier).

Exact/decoded tiers group by a hash column (linear). Acoustic/transcript tiers
build MinHash signatures, gather LSH candidate pairs, then verify each pair by
estimated Jaccard against the tier threshold. Every returned edge is an
undirected (path_a < path_b) pair with a similarity score and the tier that
produced it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minhash import (MinHasher, signatures, lsh_candidate_pairs,
                     estimated_jaccard)


def hash_group_edges(records, field: str, tier: str):
    """Edges between all records sharing a non-empty value of `field`."""
    groups = {}
    for r in records:
        v = r.get(field)
        if v:
            groups.setdefault(v, []).append(r["path"])
    edges = []
    for members in groups.values():
        m = sorted(set(members))
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                edges.append((m[i], m[j], 1.0, tier))
    return edges


def minhash_edges(records, token_key: str, tier: str, num_perm: int,
                  bands: int, rows: int, threshold: float):
    """LSH-blocked near-duplicate edges for a token set (audio or transcript)."""
    if bands * rows > num_perm:
        raise ValueError(f"{tier}: bands*rows ({bands*rows}) > num_perm "
                         f"({num_perm})")
    hasher = MinHasher(num_perm)
    sigs = signatures(records, token_key, hasher)
    if len(sigs) < 2:
        return []
    pairs = lsh_candidate_pairs(sigs, bands, rows)
    edges = []
    for a, b in pairs:
        score = estimated_jaccard(sigs[a], sigs[b])
        if score >= threshold:
            edges.append((a, b, round(score, 4), tier))
    return edges


def all_edges(records, cfg: dict, tiers):
    """Collect edges across the requested tiers. `records` must already carry
    audio_tokens (and transcript_tokens when the transcript tier runs)."""
    edges = []
    if "exact" in tiers:
        edges += hash_group_edges(records, "file_sha", "exact")
        edges += hash_group_edges(records, "pcm_sha", "decoded")
    if "audio" in tiers:
        usable = [r for r in records
                  if len(r.get("audio_tokens", [])) >= cfg["min_audio_tokens"]]
        edges += minhash_edges(
            usable, "audio_tokens", "audio", cfg["num_perm"],
            cfg["audio_lsh_bands"], cfg["audio_lsh_rows"],
            cfg["audio_threshold"])
    if "transcript" in tiers:
        usable = [r for r in records if r.get("transcript_tokens")]
        if len(usable) >= 2:
            edges += minhash_edges(
                usable, "transcript_tokens", "transcript", cfg["num_perm"],
                cfg["transcript_lsh_bands"], cfg["transcript_lsh_rows"],
                cfg["transcript_threshold"])
    return edges
