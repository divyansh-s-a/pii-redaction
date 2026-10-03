#!/usr/bin/env python3
"""Stable pseudonym tokens: [PATIENT_1], [DR_3], [HOSPITAL_1].

PSEUDONYMIZE only earns its place over REDACT if the token is the SAME
everywhere the value appears -- across pages, across carrier classes (printed,
pen, sticker, barcode), and across re-runs after a pipeline change. A token
that renumbers is worse than a black box, because it looks like linkage
information while being noise.

So a token is a keyed hash of (secret, bundle, role, normalised value), and the
readable sequence number is assigned on first sight and PERSISTED in a sidecar
keyed by that hash. Re-running after a code change reuses the sidecar and
renumbers nothing.

Matching is delegated to matching.py, so a value read cleanly on a born-digital
invoice and smeared on a 90 dpi ward photocopy collapses to one token.

The sidecar maps token -> real value. It is the re-identification key for the
whole bundle and must be stored separately from the redacted output, never
shipped with it.

    python3 document_pipeline/tokens.py --self-test
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matching

# role -> token prefix. A pack's category names the role; unknown roles get a
# generic prefix rather than an error, so a new pack cannot crash the redactor.
ROLE_PREFIX = {
    "PATIENT": "PATIENT", "INSURED": "INSURED", "MEMBER": "MEMBER",
    "THIRD_PARTY": "PERSON", "CLINICIAN": "DR", "HOSPITAL": "HOSPITAL",
    "LAB": "LAB", "PROVIDER": "PROVIDER", "POLICY_ID": "POLICY",
    "PATIENT_ID": "MRN", "DOC_SERIAL": "DOC", "DEVICE": "DEVICE",
    "EMPLOYER": "EMPLOYER",
}
DEFAULT_PREFIX = "ID"

# Roles whose values are ORGANISATIONS. One provider writes its name a dozen
# ways across a bundle -- 'Riverbend Hospital CHILDRENS CLINIC BHARAT LIMITED.',
# 'riverbend', 'CHILDRENS CLINIC BHARAT LTD', 'Riverbend Maternity' -- and string
# equality mints a new token for each, which is worse than useless: it tells a
# reader there were nine providers when there was one.
ORG_ROLES = {"HOSPITAL", "LAB", "PROVIDER", "EMPLOYER"}
ORG_STOP = {
    "hospital", "hospitals", "clinic", "clinics", "ltd", "limited", "pvt",
    "private", "india", "indian", "centre", "center", "care", "services",
    "service", "diagnostics", "diagnostic", "lab", "labs", "laboratory",
    "laboratories", "medical", "healthcare", "health", "multispeciality",
    "speciality", "specialty", "maternity", "paediatrics", "pediatrics",
    "fertility", "mother", "child", "and", "the", "of", "regd", "office",
    "branch", "unit", "institute", "nursing", "home", "trust", "society",
}
# A clinician's honorific is decisive evidence of role, whatever category the
# detector assigned. Arithmetic, not judgement.
CLINICIAN_HONORIFIC = ("dr", "dr.", "doctor", "prof", "prof.", "professor")


def core_tokens(value):
    """Distinctive words of an organisation name, legal furniture removed."""
    import re as _re
    words = _re.findall(r"[A-Za-z]{3,}", str(value).lower())
    return {w for w in words if w not in ORG_STOP}


def clinician_role(value, role):
    """Force CLINICIAN when the value is titled, so 'Dr Alina' cannot be
    minted as a patient because one page's detector called it a patient name."""
    parts = str(value).strip().split()
    if parts and parts[0].lower() in CLINICIAN_HONORIFIC:
        return "CLINICIAN"
    return role


class TokenStore:
    """One bundle's pseudonyms. Deterministic, persisted, collision-checked."""

    def __init__(self, bundle, secret="", path=None):
        self.bundle = str(bundle)
        self.secret = str(secret)
        self.path = Path(path) if path else None
        self._by_hash = {}      # hash -> {"token","role","value","aliases"}
        self._counts = {}       # prefix -> highest N issued
        if self.path and self.path.exists():
            self._load()

    # ---------- persistence ----------
    def _load(self):
        d = json.loads(self.path.read_text())
        if d.get("bundle") != self.bundle:
            raise ValueError(
                f"token sidecar is for bundle {d.get('bundle')!r}, not "
                f"{self.bundle!r}; refusing to reuse another bundle's tokens")
        self._by_hash = d.get("entries", {})
        self._counts = d.get("counts", {})

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "bundle": self.bundle, "counts": self._counts,
            "entries": self._by_hash,
            "_warning": "RE-IDENTIFICATION KEY. Store apart from the redacted "
                        "output; never ship alongside it.",
        }, indent=1, ensure_ascii=False))

    # ---------- the map ----------
    def _hash(self, value, role):
        mode = "id" if matching.is_identifier(value) else "name"
        norm = matching.normalize(value, mode)
        h = hashlib.blake2b(digest_size=12)
        h.update(self.secret.encode())
        h.update(b"\x00" + self.bundle.encode())
        h.update(b"\x00" + role.encode())
        h.update(b"\x00" + norm.encode())
        return h.hexdigest(), norm

    def _find_alias(self, value, role):
        """An existing entry of the same role that is the same entity.

        For people and identifiers that means an OCR-tolerant string match.
        For organisations it additionally means sharing a distinctive core
        word, because a provider's renderings differ by far more than OCR
        noise and no edit-distance threshold will ever join 'riverbend' to
        'Riverbend Hospital CHILDRENS CLINIC BHARAT LIMITED.'
        """
        org = role in ORG_ROLES
        core = core_tokens(value) if org else None
        for hk, e in self._by_hash.items():
            if e["role"] != role:
                continue
            if matching.matches(e["value"], value):
                return hk, e
            if org and core:
                other = core_tokens(e["value"])
                if core & other:
                    return hk, e
                # one side may be a fuzzy misread of the other's core word
                if any(matching.matches(a, b) for a in core for b in other
                       if len(a) >= 5 and len(b) >= 5):
                    return hk, e
        return None, None

    def token(self, value, role):
        """Stable token for this value in this role. Idempotent."""
        value = str(value).strip()
        if not value:
            return None
        role = clinician_role(value, role)
        hk, norm = self._hash(value, role)
        e = self._by_hash.get(hk)
        if e is None:
            # a degraded reading of a value already seen collapses onto it
            ahk, ae = self._find_alias(value, role)
            if ae is not None:
                if norm not in ae["aliases"]:
                    ae["aliases"].append(norm)
                    # keep the longest reading as canonical: it is the least
                    # likely to be a truncated OCR fragment
                    if len(value) > len(ae["value"]):
                        ae["value"] = value
                return ae["token"]
            prefix = ROLE_PREFIX.get(role, DEFAULT_PREFIX)
            n = self._counts.get(prefix, 0) + 1
            self._counts[prefix] = n
            e = {"token": f"[{prefix}_{n}]", "role": role, "value": value,
                 "aliases": [norm]}
            self._by_hash[hk] = e
        return e["token"]

    def resolve(self, value, default_role):
        """Token for a value whose ROLE is unknown at this occurrence.

        A name read from running prose ('discussed with the patient, Martin
        Grieves') must carry the token it was given where it appeared under a
        printed label -- otherwise the same person picks up two identities and
        the pseudonyms stop meaning anything. So search every role first, and
        only mint under the default role when nothing matches.
        """
        value = str(value).strip()
        if not value:
            return None
        forced = clinician_role(value, default_role)
        for e in self._by_hash.values():
            if e["role"] == forced and matching.matches(e["value"], value):
                return e["token"]
        for e in self._by_hash.values():
            if matching.matches(e["value"], value):
                return e["token"]
        return self.token(value, forced)

    def lookup(self, token):
        for e in self._by_hash.values():
            if e["token"] == token:
                return e["value"]
        return None

    def merge_roles(self, value_a, role_a, value_b, role_b):
        """Collapse two roles that turn out to be one person.

        Arithmetic, not model judgement: if the insured string reads the same
        as the patient string, they are one person and must carry one token,
        or a reviewer counting distinct people in the bundle gets it wrong.
        """
        if not matching.matches(value_a, value_b):
            return False
        ta = self.token(value_a, role_a)
        hb, _ = self._hash(value_b, role_b)
        eb = self._by_hash.get(hb)
        if eb is None:
            self._by_hash[hb] = {"token": ta, "role": role_a, "value": value_a,
                                 "aliases": [self._hash(value_b, role_b)[1]]}
        else:
            eb["token"] = ta
        return True

    @property
    def entries(self):
        return list(self._by_hash.values())

    def __len__(self):
        return len({e["token"] for e in self._by_hash.values()})


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tok.json"
        s = TokenStore("case-48142", secret="k", path=p)

        t1 = s.token("MARTIN GRIEVES", "PATIENT")
        assert t1 == "[PATIENT_1]", t1
        # idempotent
        assert s.token("MARTIN GRIEVES", "PATIENT") == t1
        # a degraded reading of the same name collapses onto the same token
        assert s.token("MARTlN GRlEVES", "PATIENT") == t1, "OCR variant split"
        assert s.token("Martin  Grieves", "PATIENT") == t1, "spacing split"
        # a different person gets a different token
        assert s.token("ROSALIND", "PATIENT") == "[PATIENT_2]"
        # roles have their own sequences and prefixes
        assert s.token("S Menon", "CLINICIAN") == "[DR_1]"
        assert s.token("Dr. S. Menon", "CLINICIAN") == "[DR_1]", "honorific split"
        assert s.token("Riverbend Childrens Clinic", "HOSPITAL") == "[HOSPITAL_1]"
        # identifiers are strict: one digit apart is a different account
        a = s.token("50100094060529", "PATIENT_ID")
        b = s.token("50100094060530", "PATIENT_ID")
        assert a != b, "two different account numbers collapsed to one token"
        assert s.token("5O1OOO94O6O529", "PATIENT_ID") == a, "O/0 variant split"
        print(f"  issued {len(s)} distinct tokens across "
              f"{len(set(e['role'] for e in s.entries))} roles")

        # role collapse when the filer IS the patient
        assert s.merge_roles("MARTIN GRIEVES", "PATIENT", "Martin Grieves", "INSURED")
        assert s.token("Martin Grieves", "INSURED") == t1, "role collapse failed"

        # a name met in prose under no label picks up the token it already has
        assert s.resolve("Martin Grieves", "THIRD_PARTY") == t1, \
            "free-text name minted a second identity"
        assert s.resolve("S Menon", "THIRD_PARTY") == "[DR_1]", \
            "clinician in prose split from the labelled occurrence"
        assert s.resolve("BRAND NEW NAME", "THIRD_PARTY") == "[PERSON_1]"

        # one provider written nine ways is one provider
        forms = ["Riverbend Hospital CHILDRENS CLINIC BHARAT LIMITED.", "riverbend",
                 "CHILDRENS CLINIC BHARAT LTD", "Riverbend Maternity",
                 "Riverbend Paediatrics.", "Rivebend Mother and Child Care,",
                 "Riverbend Clinic", "Riverbend Fertility:"]
        got = {s.token(f, "HOSPITAL") for f in forms}
        print(f"  {len(forms)} renderings of one provider -> {len(got)} token(s): {got}")
        assert len(got) == 1, f"provider fragmented into {len(got)} tokens: {got}"
        # a genuinely different provider must NOT be merged into it
        assert s.token("Apollo Speciality Hospitals", "HOSPITAL") != got.pop()
        # an honorific overrides whatever role the detector guessed: a titled
        # name must land in the clinician sequence, never the patient one
        t = s.token("Dr Alina", "PATIENT")
        assert t.startswith("[DR_"), f"titled clinician minted as {t}"
        assert t != "[DR_1]", "distinct doctors collapsed onto one token"
        assert s.token("Dr. Alina Vasquez", "PATIENT").startswith("[DR_")
        print(f"  'Dr Alina' under a PATIENT label -> {t}")

        s.save()
        # a re-run reuses the sidecar and renumbers nothing
        s2 = TokenStore("case-48142", secret="k", path=p)
        assert s2.token("MARTIN GRIEVES", "PATIENT") == t1, "renumbered on reload"
        assert s2.token("ROSALIND", "PATIENT") == "[PATIENT_2]", "renumbered"
        assert s2.token("NEW PERSON", "PATIENT") == "[PATIENT_3]", "bad sequence"
        assert s2.lookup(t1) == "MARTIN GRIEVES"
        # a sidecar from another bundle must be refused, not silently reused
        try:
            TokenStore("case-48000", secret="k", path=p)
        except ValueError:
            pass
        else:
            raise AssertionError("cross-bundle sidecar reuse was allowed")
    print("tokens self-test OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
