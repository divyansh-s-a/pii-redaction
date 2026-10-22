#!/usr/bin/env python3
"""Code labels for the insurance route's --replace mode (owner's request, 2026-10-22): the coded claim files of the claim benchmark.

Redaction is unchanged and finished first; this module then writes, on the truly redacted output, ONE short code per value for the
whole file -- the same code on every page and in every box that holds that value -- on a small white label over its black box:

    Pt1 patient            Prop1 insured / proposer   Per1 another person     Dr1 clinician
    Hosp1 provider         Ins1 insurer or TPA         Empr1 employer
    HPID1 ...4821 hospital patient ID                  Pol1 ...4410 policy or membership number
    Aad1 ...3312 Aadhaar   PAN1 ...674P PAN            ID1 ...0091 other government ID
    Acct1 ...5521 bank     Reg1 ...2222 registration   Emp1 ...4444 employee code
    Doc1 ...2234 document serial                       Dev1 ...7710 implant or device identifier
    Ph1 phone              Em1 email                   Addr1 address           DOB1 date of birth

The same value written another way (initials, another word order, a misread letter or digit) keeps its code with (v2), (v3), ...
An identifier shows only its last four characters, the way a masked Aadhaar is printed. A signature gets [sig], a non-Latin name
[nm], an age / sex / relationship field [age/sex]; a face, QR code, stamp, logo, pen-ink region or any box whose value the detector
did not read stays plain black. A box too small for a legible label (7 pt or more) stays black.

Which boxes share a code is decided by the route's token store (tokens.TokenStore): one entity per value per role, OCR-tolerant for
people and identifiers, by distinctive core word for organisations; a name read in running prose takes the code of the person it
matches. Codes are numbered per family in the order values are first seen (page order).

A legend (code = what kind of value it stands for; never the value) is appended AFTER the last page, so page numbers do not change.
The code -> real value key goes to a sidecar (`--codes-sidecar`): a re-identification key that must be stored apart from the output
and never shipped with it. Nothing written on a page holds a real value beyond an identifier's last four characters; a label that
would is not written and its box stays black.

    python3 code_labels.py --self-test
"""
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matching
import tokens as tokenstore

# category -> (code family, token-store role, value kind). A family in GOV is decided from the value itself.
FAMILY = {
    "PATIENT_NAME": ("Pt", "PATIENT", "name"), "NAME_IN_FREE_TEXT": ("Pt", "PATIENT", "prose"),
    "RELATED_PARTY_NAME": ("Prop", "INSURED", "name"), "THIRD_PARTY_PERSON_NAME": ("Per", "THIRD_PARTY", "name"),
    "UNSEEN_NAME_IN_PROSE": ("Per", "THIRD_PARTY", "prose"), "CLINICIAN_IDENTITY": ("Dr", "CLINICIAN", "name"),
    "PROVIDER_NAME_TEXT": ("Hosp", "HOSPITAL", "name"), "INSURER_TPA_IDENTITY": ("Ins", "INSURER_TPA", "name"),
    "POLICY_MEMBERSHIP_ID": ("Pol", "POLICY_ID", "id"), "PROVIDER_ASSIGNED_PATIENT_ID": ("HPID", "PATIENT_ID", "id"),
    "GOVERNMENT_ID_NUMBER": ("GOV", "GOV_ID", "id"), "EMPLOYEE_CODE": ("Emp", "EMPLOYEE_CODE", "id"),
    "PROVIDER_REGISTRATION_CODE": ("Reg", "REGISTRATION", "id"), "PERSON_BANK_INSTRUMENT": ("Acct", "BANK_INSTRUMENT", "id"),
    "DOCUMENT_SERIAL": ("Doc", "DOC_SERIAL", "id"), "IMPLANT_DEVICE_IDENTIFIER": ("Dev", "DEVICE", "id"),
    "PERSONAL_CONTACT_CHANNEL": ("CONTACT", "CONTACT", "id"), "PERSONAL_POSTAL_ADDRESS": ("Addr", "ADDRESS", "name"),
    "DATE_OF_BIRTH": ("DOB", "DOB", "id"),
}
GENERIC = {"SIGNATURE_AND_THUMBPRINT": "[sig]", "NON_LATIN_IDENTITY_TEXT": "[nm]", "AGE_SEX_RELATIONSHIP": "[age/sex]"}
ID_FAMILIES = {"Pol", "HPID", "Aad", "PAN", "ID", "Emp", "Reg", "Acct", "Doc", "Dev"}
MEANING = {
    "Pt": "the patient's name", "Prop": "the insured person's or proposer's name", "Per": "another person's name",
    "Dr": "a clinician's name", "Hosp": "a hospital, clinic, laboratory or other provider", "Ins": "an insurer or TPA",
    "HPID": "a hospital patient ID (MRN, UHID, IP or OP number)", "Pol": "a policy, membership or card number",
    "Aad": "an Aadhaar number", "PAN": "a PAN", "ID": "another government ID number", "Emp": "an employee code",
    "Reg": "a provider's registration number", "Acct": "a bank account, card or cheque number",
    "Doc": "a document serial (bill, receipt, report or visit number)", "Dev": "an implant or device identifier",
    "Ph": "a phone number", "Em": "an email address", "Addr": "a postal address", "DOB": "a date of birth",
}
MIN_FS = 7.0
_PAN = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")


def _cats(f):
    """category NAMES of a finding (findings may carry ids like T01 or names like PATIENT_NAME)"""
    return [str(c).split(":")[-1] for c in (f.get("categories") or [])]


def family_of(f, by_id):
    """(family, role, kind) for a finding, or (generic label, None, None), or None (the box stays black)"""
    for c in _cats(f):
        name = by_id.get(c, c)
        if name in FAMILY:
            fam, role, kind = FAMILY[name]
            if fam == "GOV":
                v = re.sub(r"[^A-Za-z0-9]", "", f.get("text") or "").upper()
                fam = "Aad" if re.fullmatch(r"(X{0,8}|\d{0,8})\d{4}", v) and len(v) == 12 else "PAN" if _PAN.match(v) else "ID"
            elif fam == "CONTACT":
                fam = "Em" if "@" in (f.get("text") or "") else "Ph"
            return fam, role, kind
        if name in GENERIC:
            return GENERIC[name], None, None
    return None


def last_four(value):
    v = re.sub(r"[^A-Za-z0-9]", "", value or "")
    return v[-4:].upper() if len(v) >= 6 else ""


class Codes:
    """value -> code for one file, numbered per family in order of first sight"""

    def __init__(self, bundle):
        self.store = tokenstore.TokenStore(bundle, path=None)
        self.code_of_token = {}                 # store token -> base code ("Pt1", "HPID1 ...4821")
        self.n = defaultdict(int)               # family -> highest number issued
        self.forms = defaultdict(OrderedDict)   # base code -> {normalised form: variant number}
        self.real = defaultdict(list)           # base code -> real values (sidecar only)
        self.family = {}

    def code(self, value, fam, role, kind):
        value = str(value or "").strip()
        if not value:
            return None
        tok = self.store.resolve(value, role) if kind == "prose" else self.store.token(value, role)
        if tok is None:
            return None
        if tok not in self.code_of_token:
            self.n[fam] += 1
            base = f"{fam}{self.n[fam]}"
            if fam in ID_FAMILIES and last_four(value):
                base += f" ...{last_four(value)}"
            self.code_of_token[tok] = base; self.family[base] = fam
        base = self.code_of_token[tok]
        norm = matching.normalize(value, "id" if fam in ID_FAMILIES | {"Ph", "DOB"} else "name")
        fs = self.forms[base]
        if norm not in fs:
            fs[norm] = len(fs) + 1
        if value not in self.real[base]:
            self.real[base].append(value)
        v = fs[norm]
        return base if v == 1 else f"{base} (v{v})"


def _safe(label, reals):
    """True when the label shows no real value beyond an identifier's last four characters"""
    low = label.lower()
    for r in reals:
        for w in re.findall(r"[^\W\d_]{4,}", r.lower()):
            if w in low:
                return False
        d = re.sub(r"\D", "", r)
        if len(d) >= 5 and d[:-4] and d[:-4] in re.sub(r"\D", "", label):
            return False
    return True


def _patch(page, r, s):
    """a white patch just big enough for the label at the box's top-left (the rest of the box stays black)"""
    for fs in [x / 2 for x in range(20, int(MIN_FS * 2) - 1, -1)]:
        w = fitz.get_text_length(s, fontname="helv", fontsize=fs) * 1.08 + 6
        if w <= r.width and fs * 1.25 <= r.height + 0.5:
            p = fitz.Rect(r.x0, r.y0, r.x0 + w, r.y0 + min(r.height, fs * 1.3))
            page.draw_rect(p, color=None, fill=(1, 1, 1), overlay=True)
            page.insert_text((p.x0 + 1.5, p.y0 + fs * 0.98), s, fontsize=fs, fontname="helv", color=(0, 0, 0))
            return True
    return False


def label_document(pdf, per_page, out, pack=None, sidecar=None, bundle=None):
    """per_page: {page number (1-based): [finding, ...]} as the route redacted them; pdf: the redacted output -> stats"""
    raw = getattr(pack, "_raw", None) or {}
    by_id = {c["id"]: c["name"] for c in raw.get("categories", [])}      # findings may carry category ids (T01) or names
    codes = Codes(bundle or Path(pdf).stem)
    items = defaultdict(list)                   # page -> [(rect, label, base code or None)]
    st = defaultdict(int)
    for pno in sorted(per_page):
        for f in per_page[pno]:
            if str(f.get("action") or "").upper() == "KEEP":
                continue
            fam = family_of(f, by_id)
            rects = [fitz.Rect(b) for b in (f.get("all_boxes") or ([f["bbox"]] if f.get("bbox") else [])) if b]
            if fam is None or not rects:
                st["black"] += len(rects); continue
            if fam[1] is None:
                lab, base = fam[0], None
            else:
                lab = codes.code(f.get("text"), *fam)
                if not lab:
                    st["black"] += len(rects); continue
                base = lab.split(" (v")[0]
            for r in rects:
                items[pno].append((r, lab, base))
    doc = fitz.open(pdf)
    used = set()
    for pno, its in items.items():
        page = doc[pno - 1]
        groups = []                             # overlapping boxes: one label per group, in its largest box
        for it in sorted(its, key=lambda x: -abs(x[0])):
            g = next((g for g in groups if any(abs(it[0] & m[0]) > 0.2 * min(abs(it[0]), abs(m[0])) for m in g)), None)
            (g.append(it) if g else groups.append([it]))
        for g in groups:
            labs = list(dict.fromkeys(m[1] for m in g if m[2])) or list(dict.fromkeys(m[1] for m in g))
            lab = " · ".join(labs)
            reals = [v for m in g if m[2] for v in codes.real[m[2]]]
            if not _safe(lab, reals):
                st["unsafe_left_black"] += 1; continue
            if page.rotation:                   # a label would run sideways on a rotated page: the box stays black
                st["rotated_page_left_black"] += 1; continue
            r = g[0][0]
            if not _patch(page, r, lab):
                st["no_room"] += 1; continue
            st["labelled" if any(m[2] for m in g) else "generic"] += 1
            used.update(m[2] for m in g if m[2])
            used.update(m[1] for m in g if m[2] and m[1] != m[2])
    _legend(doc, sorted(used, key=_order), codes)
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    if sidecar:
        Path(sidecar).parent.mkdir(parents=True, exist_ok=True)
        Path(sidecar).write_text(json.dumps({
            "file": Path(pdf).name,
            "_warning": "RE-IDENTIFICATION KEY: code -> real values. Store apart from the redacted output; never ship it with it.",
            "codes": {b: {"meaning": MEANING.get(codes.family[b], ""), "real_values": codes.real[b],
                          "variants": {f"{b} (v{n})": None for n in codes.forms[b].values() if n > 1}}
                      for b in codes.code_of_token.values()}}, indent=1, ensure_ascii=False))
    st["codes"] = len(codes.code_of_token)
    return dict(st)


def _order(c):
    m = re.match(r"([A-Za-z]+)(\d+)", c)
    return (m.group(1), int(m.group(2)), c) if m else (c, 0, c)


def _legend(doc, used, codes):
    if not used:
        return
    pg = doc.new_page(width=595, height=842); y = 60
    pg.insert_text((50, y), "Legend of codes", fontsize=14, fontname="hebo"); y += 22
    for line in ["Each personal value in this file is shown as a short code on a white label over its black box: the same code",
                 "wherever that value appears, a different code for a different value. '...3333' = the last four characters of",
                 "an identifier; '(v2)' = the same value written another way. [sig] signature, [nm] name in another script,",
                 "[age/sex] age, sex or relationship. A black box without a label is redacted content.", ""]:
        pg.insert_text((50, y), line, fontsize=9, fontname="helv"); y += 13
    for c in used:
        base = c.split(" (v")[0]
        meaning = MEANING.get(codes.family.get(base, ""), "") + (" - written another way" if " (v" in c else "")
        if y > 800:
            pg = doc.new_page(width=595, height=842); y = 60
        pg.insert_text((50, y), f"{c}   =   {meaning}", fontsize=8, fontname="helv"); y += 11


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src, red, out, side = (Path(td) / n for n in ("a.pdf", "r.pdf", "o.pdf", "k.json"))
        d = fitz.open(); p = d.new_page()
        p.insert_text((72, 100), "Patient: Meera Iyer   UHID: 0041237788", fontsize=11)
        p.insert_text((72, 130), "Seen by Dr Arjun Rao. Meera Iyar was stable.", fontsize=11)
        d.save(src)
        pg = fitz.open(src)[0]
        def box(t):
            return list(pg.search_for(t)[0])
        per = {1: [{"categories": ["PATIENT_NAME"], "text": "Meera Iyer", "all_boxes": [box("Meera Iyer")], "action": "PSEUDONYMIZE"},
                   {"categories": ["PROVIDER_ASSIGNED_PATIENT_ID"], "text": "0041237788", "all_boxes": [box("0041237788")]},
                   {"categories": ["CLINICIAN_IDENTITY"], "text": "Dr Arjun Rao", "all_boxes": [box("Arjun Rao")]},
                   {"categories": ["NAME_IN_FREE_TEXT"], "text": "Meera Iyar", "all_boxes": [box("Meera Iyar")]}]}
        rd = fitz.open(src); pr = rd[0]
        for fs in per[1]:
            for b in fs["all_boxes"]:
                pr.add_redact_annot(fitz.Rect(b), fill=(0, 0, 0))
        pr.apply_redactions(); rd.save(red)
        st = label_document(red, per, out, sidecar=side)
        o = fitz.open(out); text = o[0].get_text()
        assert "Pt1" in text and "HPID1 ...7788" in text and "Dr1" in text, text
        assert "Pt1 (v2)" in text, "a misread of the same name keeps the code as a variant"
        assert "Meera" not in text and "0041237" not in text and "Arjun" not in text, "no real value on the page"
        assert o.page_count == 2 and "Legend of codes" in o[1].get_text(), "legend appended after the last page"
        assert "Meera" not in o[1].get_text(), "the legend holds no value"
        k = json.loads(side.read_text())
        assert "Meera Iyer" in k["codes"]["Pt1"]["real_values"], "the key is in the sidecar only"
        print("code_labels self-test passed:", st)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
