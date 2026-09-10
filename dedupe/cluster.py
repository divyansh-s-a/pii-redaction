#!/usr/bin/env python3
"""Union-find clustering of duplicate edges + canonical-representative pick.

Any two files joined (directly or transitively) by an edge land in one
cluster. The representative kept from each cluster is the longest recording,
then the largest file, then the lexicographically smallest path — a fully
deterministic choice so re-runs and shards agree.
"""


class _UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster(paths, edges):
    """Return list of clusters (each a sorted list of paths)."""
    uf = _UF()
    for p in paths:
        uf.find(p)
    for e in edges:
        uf.union(e[0], e[1])
    groups = {}
    for p in paths:
        groups.setdefault(uf.find(p), []).append(p)
    return [sorted(m) for m in groups.values()]


def pick_representative(members, recmap):
    def key(p):
        r = recmap.get(p, {})
        return (-int(r.get("duration_ms", 0)), -int(r.get("size", 0)), p)
    return sorted(members, key=key)[0]
