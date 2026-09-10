"""Dataset dedupe for audio clips — a scalable pre-filter for the redaction
pipeline.

Every file is reduced to a tiny record (byte hash, decoded-PCM hash, an
acoustic sub-fingerprint token set, and — when a transcript exists — a
shingle token set). Exact/decoded tiers are hash-grouped (linear); the
acoustic and transcript tiers become integer token sets fed through one
MinHash + LSH matcher, so near-duplicate detection stays near-linear and
shardable across machines rather than O(n^2). Union-find clusters the matched
pairs and picks one canonical representative per cluster.

Modules:
  fingerprint.py  file -> {file_sha, pcm_sha, audio_tokens, duration_ms}
  minhash.py      MinHash signatures + LSH banding
  match.py        hash-group edges + LSH candidate pairs + verification
  cluster.py      union-find clustering + canonical-rep selection
  run_dedupe.py   CLI orchestrator (parallel, resumable cache, report, --apply)
"""
