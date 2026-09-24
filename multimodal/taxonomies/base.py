#!/usr/bin/env python3
"""Taxonomy packs: what counts as PII in a given context.

Deliberately separate from `router/`, which answers a different question. The
router asks "which engine can physically read this file"; a pack asks "what
counts as PII here". Conflating them is what produced the first failed
deliverable -- the engine changed from audio to documents while the policy
stayed audio-shaped, so every judgement collapsed to "black box it".

A pack is a DIRECTORY, not code: `pack.json` carries the categories, the action
matrix and the signals that select it, `prompt.txt` carries the detection
prompt, and `hooks.py` is optional and only for packs that need custom logic.
Adding a taxonomy is therefore a data change -- which is the same rule
router/registry.py already sets for modalities.

    python3 taxonomies/base.py --self-test
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# How much of the original value survives the action. Ordering matters: when a
# span carries several categories the MOST protective action wins, so a patient
# name that also sits inside kept clinical content is still pseudonymised.
# PSEUDONYMIZE ranks below MASK_PARTIAL because both destroy the value outright,
# but a stable token additionally discloses that two pages concern one person.
ACTION_RANK = {
    "KEEP": 0,           # everything survives
    "GENERALIZE": 1,     # a coarsened value survives
    "PSEUDONYMIZE": 2,   # value destroyed, stable token written back
    "MASK_PARTIAL": 3,   # value destroyed, only shape and position survive
    "REDACT": 4,         # nothing survives
}
ACTIONS = tuple(ACTION_RANK)
DESTRUCTIVE = {"REDACT", "MASK_PARTIAL", "PSEUDONYMIZE", "GENERALIZE"}


class PackError(ValueError):
    pass


class Pack:
    """One taxonomy, loaded from a directory."""

    def __init__(self, data, root=None):
        self.root = Path(root) if root else None
        self.id = data["id"]
        self.version = data.get("version", "0")
        self.title = data.get("title", self.id)
        self.description = data.get("description", "")
        self.owner_rules = data.get("owner_rules", [])
        self.signals = data.get("signals", {})
        self.conservative = bool(data.get("conservative", False))
        self.notes = {k: data.get(k) for k in
                      ("bank_line", "provider_deid_spec", "consistency_rules",
                       "residual_risks") if data.get(k)}
        self.semantics = data.get("semantics", {})
        self.categories = {}
        for c in data.get("categories", []):
            act = c.get("action")
            if act not in ACTION_RANK:
                raise PackError(f"{self.id}/{c.get('id')}: unknown action {act!r}; "
                                f"expected one of {ACTIONS}")
            self.categories[c["id"]] = c
            # let a detector emit either the stable id (T01) or the human name
            self.categories.setdefault(c["name"], c)
        self._raw = data

    # ---------- selection ----------
    def score(self, analysis):
        """How well this pack fits an analyzed input. 0 means 'does not apply'.

        Declarative and deterministic: it reads signals out of pack.json and
        counts hits in whatever router/analyze.py extracted. No model involved
        -- a model tie-break, if any, happens in registry.select().
        """
        s = self.signals or {}
        if not s:
            return 0.0
        hay = " ".join(str(v).lower() for v in _flatten(analysis))
        total = 0.0
        for group, spec in s.items():
            terms = spec.get("any") or []
            weight = float(spec.get("weight", 1.0))
            hits = sum(1 for t in terms if str(t).lower() in hay)
            if hits and spec.get("required_min", 1) <= hits:
                total += weight * min(hits, spec.get("cap", 3))
        return total

    # ---------- the action matrix ----------
    def resolve(self, key):
        """A mechanism's SEMANTIC key -> this pack's own category id.

        Mechanisms (label lexicon, ink zoning, the objective visual pass) must
        not hard-code one pack's ids. They did, and the consequence was severe:
        under any other pack those ids are unknown, action_for fails closed to
        REDACT, and every KEEP rule INVERTS -- clinical content, dates, bank
        names and claim amounts all destroyed the moment selection fell back to
        the conservative pack.
        """
        if key.startswith("sem:"):
            key = key[4:]
        return self.semantics.get(key)

    def action_for(self, categories, default="REDACT"):
        """Most protective action across the categories on one span.

        An unknown category does NOT fall through to KEEP. Detector silence and
        detector confusion are both failures, and the safe direction for a
        failure is to over-redact -- which is visible and recoverable -- rather
        than to leak, which is neither.
        """
        mapped = []
        for c in (categories or ()):
            if c in self.categories:
                mapped.append(c)
            else:
                r = self.resolve(c)
                if r in self.categories:
                    mapped.append(r)
        known = [self.categories[c]["action"] for c in mapped]
        if not known:
            return default
        return max(known, key=lambda a: ACTION_RANK[a])

    def unknown_categories(self, categories):
        return [c for c in (categories or ())
                if c not in self.categories
                and self.resolve(c) not in self.categories]

    def rows(self, action=None):
        out, seen = [], set()
        for c in self.categories.values():
            if id(c) in seen:
                continue
            seen.add(id(c))
            if action is None or c["action"] == action:
                out.append(c)
        return sorted(out, key=lambda c: c["id"])

    @property
    def lexicon(self):
        """Label patterns this pack OWNS. Compiled only during its own run,
        so vocabulary added for one domain cannot reach another."""
        return self._raw.get("lexicon", []) or []

    @property
    def prompt(self):
        """Hand-written guidance plus a category table rendered from pack.json.

        Generated rather than hand-maintained so the prompt cannot drift from
        the matrix the redactor actually applies -- a detector told to look for
        a category that no longer exists, or unaware of one that does, produces
        exactly the inconsistency this taxonomy was built to remove.
        """
        for name in ("prompt_v1.txt", "prompt.txt"):
            p = (self.root or HERE) / name
            if p.exists():
                return p.read_text().rstrip() + "\n\n" + self.category_table()
        raise PackError(f"{self.id}: no prompt file in {self.root}")

    def category_table(self):
        lines = [f"CATEGORIES ({self.id} v{self.version}). Return one or more "
                 "of these ids per span.", ""]
        keep = [c for c in self.rows() if c["action"] == "KEEP"]
        flag = [c for c in self.rows() if c["action"] != "KEEP"]
        for c in flag:
            lines.append(f"  {c['id']}  {c['name']}")
            lines.append(f"        {_one_line(c['definition'], 300)}")
            if c.get("corpus_examples"):
                lines.append(f"        e.g. {_one_line(c['corpus_examples'], 150)}")
        lines += ["", "THESE ARE KEPT. Do not return spans for them; they are "
                      "listed so you can recognise and step over them:"]
        for c in keep:
            lines.append(f"  {c['id']}  {c['name']} - "
                         f"{_one_line(c['definition'], 180)}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"<Pack {self.id} v{self.version}: {len(self.rows())} categories>")


def _one_line(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


def _flatten(obj, depth=0):
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _flatten(v, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _flatten(v, depth + 1)
    else:
        yield obj


def load(path):
    """Load one pack directory (or its pack.json), resolving `extends`."""
    path = Path(path)
    jf = path if path.suffix == ".json" else path / "pack.json"
    if not jf.exists():
        raise PackError(f"no pack.json at {jf}")
    data = json.loads(jf.read_text())
    parent = data.get("extends")
    if parent:
        base = load(jf.parent.parent / parent)._raw
        merged = dict(base)
        merged.update({k: v for k, v in data.items() if k != "categories"})
        # child rows override parent rows of the same id; the rest are inherited
        by_id = {c["id"]: c for c in base.get("categories", [])}
        for c in data.get("categories", []):
            by_id[c["id"]] = c
        merged["categories"] = list(by_id.values())
        # lexicon APPENDS, it does not replace. A tenant pack that extends
        # insurance_medical and adds two labels must not silently lose the
        # eleven it inherited -- and `update` above would have done exactly
        # that, leaving every claim-form label unmatched with no error.
        merged["lexicon"] = (list(base.get("lexicon", []))
                             + list(data.get("lexicon", [])))
        # SEMANTICS MERGE KEY BY KEY. `update` above replaced the parent's
        # whole block, so a tenant pack adding ONE mapping inherited all 40
        # categories and none of the 46 semantic keys -- every mechanism key
        # then failed closed to REDACT and EVERY KEEP ROW INVERTED. Clinical
        # content, dates, bank names, letterheads and empty fields all
        # destroyed, from adding one line to a pack.
        merged["semantics"] = {**base.get("semantics", {}),
                               **data.get("semantics", {})}
        data = merged
    return Pack(data, root=jf.parent)


def self_test():
    p = load(HERE / "insurance_medical")
    print(f"loaded {p!r}")
    assert len(p.rows()) == 40, f"expected 40 categories, got {len(p.rows())}"

    # owner rule 4: clinical content is kept
    assert p.action_for(["T37"]) == "KEEP", "clinical content must be KEEP"
    # owner rule 1: patient identity is destroyed
    assert p.action_for(["T01"]) == "PSEUDONYMIZE"
    assert p.action_for(["T08"]) == "REDACT"
    # owner override 2026-09-23: dates kept exact
    assert p.action_for(["T17"]) == "KEEP", "DOB override not applied"
    assert p.action_for(["T21"]) == "KEEP", "event-date override not applied"
    # most protective wins: a patient name inside kept clinical prose
    assert p.action_for(["T37", "T01"]) == "PSEUDONYMIZE", "precedence wrong"
    assert p.action_for(["T01", "T13"]) == "REDACT", "REDACT must dominate"
    # fail-closed: an unknown category must not become KEEP
    assert p.action_for(["NOT_A_CATEGORY"]) == "REDACT", "unknown must fail closed"
    assert p.unknown_categories(["T01", "NOPE"]) == ["NOPE"]
    # a detector may emit the human name instead of the id
    assert p.action_for(["PATIENT_NAME"]) == "PSEUDONYMIZE"
    # a MECHANISM emits a semantic key, which every pack maps to its own row
    assert p.action_for(["sem:clinical"]) == "KEEP"
    assert p.action_for(["sem:person_name"]) == "PSEUDONYMIZE"
    g = load(HERE / "generic_document")
    # THE DELIBERATE DIVERGENCE, asserted in both directions so neither pack
    # can drift into the other's rule. Insurance keeps clinical content
    # because an adjudicator cannot work without it (owner rule 4). A generic
    # document carries no such justification, so health content is redacted
    # there, mirroring the audio taxonomy's HEALTH_MEDICAL_INFO.
    assert p.action_for(["sem:clinical"]) == "KEEP", "insurance must keep clinical"
    assert g.action_for(["sem:clinical"]) == "REDACT", \
        "generic must redact health content (owner, 2026-09-22)"
    # ...but UNZONED PEN INK is not a clinical claim, and emitting it as one
    # made "keep this ink" mean "destroy this ink" the moment a pack redacted
    # health content. Both packs must keep it.
    assert p.action_for(["sem:ink_unzoned"]) == "KEEP"
    assert g.action_for(["sem:ink_unzoned"]) == "KEEP", \
        "unzoned ink INVERTS -- handwriting.py is emitting sem:clinical again"
    # the objective pass must not destroy what a pack protects
    assert g.action_for(["sem:bank_common"]) == "KEEP"
    assert p.action_for(["sem:bank_common"]) == "KEEP", \
        "IFSC destroyed under insurance -- scan_text emitting legacy names?"
    assert g.action_for(["sem:org_contact"]) == "KEEP", "letterhead destroyed"
    assert p.action_for(["sem:org_contact"]) == "REDACT", "provider carrier kept"
    assert g.action_for(["sem:event_date"]) == "KEEP"
    assert g.action_for(["sem:person_name"]) == "PSEUDONYMIZE"
    assert not p.unknown_categories(["sem:clinical"])

    # EVERY key any mechanism can emit must be mapped by EVERY pack. An
    # unmapped key fails closed to REDACT, which is safe for an identifier and
    # catastrophic for a KEEP row -- that is how clinical content, dates, bank
    # names and letterheads all got destroyed at various points.
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / "document_pipeline"))
    import formfields as _ff
    emitted = {c for _, c, _ in _ff.LEXICON} | {
        "sem:card", "sem:org_contact", "sem:ink_unzoned", "sem:face",
        "sem:machine_code", "sem:handwriting_ink", "sem:gov_id",
        "sem:org_reg_code", "sem:bank_common", "sem:bank_personal",
        "sem:unlabelled_identity_block", "sem:clinical"}
    for pack in (p, g):
        missing = sorted(k for k in emitted if not pack.resolve(k))
        assert not missing, f"{pack.id} does not map {missing}"
    print(f"  semantic keys resolve in both packs ({len(emitted)} emitted keys);"
          f" no KEEP rule inverts")

    from collections import Counter
    print("  actions:", dict(Counter(c["action"] for c in p.rows())))
    print("  only model_judged row(s):",
          [c["id"] for c in p.rows() if c["enforcement"] == "model_judged"])
    print("base self-test OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
