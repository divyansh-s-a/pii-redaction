#!/usr/bin/env python3
"""Which taxonomy applies to this input.

Separate from router/registry.py on purpose. That one answers "which engine can
physically read this file"; this one answers "what counts as PII in it". They
are orthogonal: the same PDF engine serves an insurance bundle and a CAD
drawing, and the same insurance taxonomy could arrive as a PDF, a spreadsheet
or an email.

Selection is deliberately NOT a fixed second axis. Domain is only one of the
things that legitimately change the answer -- jurisdiction (DPDP vs GDPR vs
HIPAA), tenant contract, downstream task and document genre all do too, and
hard-coding any one of them forces a rewrite for the next. So a pack declares
its own signals and the registry scores them.

    python3 taxonomies/registry.py --self-test
    python3 taxonomies/registry.py --list
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base

# How much the top pack must beat the runner-up by before we trust the gap.
# Below it, the choice is ambiguous and falls to the conservative pack.
MARGIN = 1.5
MIN_SCORE = 2.0


def discover(root=HERE):
    packs = {}
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and (d / "pack.json").exists():
            p = base.load(d)
            packs[p.id] = p
    return packs


def conservative(packs):
    """The pack to fall back to. Explicitly flagged, never guessed."""
    for p in packs.values():
        if p.conservative:
            return p
    return None


def select(analysis, override=None, packs=None, root=HERE):
    """-> (pack, decision) where decision explains itself.

    The fallback direction is the whole point: choosing a permissive pack
    wrongly LEAKS, while choosing a conservative one wrongly over-redacts,
    which is visible in the over-redaction metric and recoverable by a re-run.
    So ambiguity resolves to the conservative pack, and the decision is
    recorded rather than silently applied.
    """
    packs = packs or discover(root)
    if not packs:
        raise base.PackError(f"no taxonomy packs found under {root}")

    if override:
        if override not in packs:
            raise base.PackError(
                f"unknown taxonomy {override!r}; have {sorted(packs)}")
        return packs[override], {"basis": "explicit_override", "pack": override,
                                 "scores": {}, "ambiguous": False}

    scores = {pid: round(p.score(analysis), 2) for pid, p in packs.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    fallback = conservative(packs)

    if top_score < MIN_SCORE:
        chosen, basis = (fallback or packs[top]), "no_signal_fallback"
        ambiguous = True
    elif top_score - second_score < MARGIN and fallback is not None \
            and ranked[1][1] >= MIN_SCORE:
        chosen, basis = fallback, "ambiguous_fallback"
        ambiguous = True
    else:
        chosen, basis = packs[top], "signal_score"
        ambiguous = False

    return chosen, {"basis": basis, "pack": chosen.id, "scores": scores,
                    "runner_up": ranked[1][0] if len(ranked) > 1 else None,
                    "ambiguous": ambiguous}


def self_test():
    packs = discover()
    print(f"discovered {len(packs)} packs: {sorted(packs)}")
    assert "insurance_medical" in packs
    assert conservative(packs) is not None, "no pack is flagged conservative"

    # a claim bundle, as router/analyze.py would describe it
    claim = {"genres": ["claim_form_partA", "discharge_summary", "lab_report"],
             "labels": ["a) Policy No", "Patient Name", "Diagnosis",
                        "Name of Hospital where Admitted", "TPA ID", "IFSC Code",
                        "Sum Insured", "UHID", "MRN No"]}
    p, d = select(claim)
    print(f"  claim bundle  -> {p.id:<20} {d['basis']:<22} {d['scores']}")
    assert p.id == "insurance_medical", f"claim routed to {p.id}"

    # something with no insurance signal at all must NOT get the medical pack
    cad = {"genres": ["cad_drawing"],
           "labels": ["Drawing No", "Revision", "Scale", "Drawn By", "Title Block"]}
    p2, d2 = select(cad)
    print(f"  cad drawing   -> {p2.id:<20} {d2['basis']:<22} {d2['scores']}")
    assert p2.conservative, "an unknown genre must fall to the conservative pack"

    # explicit override always wins, even against the signals
    p3, d3 = select(claim, override="generic_document")
    assert p3.id == "generic_document" and d3["basis"] == "explicit_override"
    print(f"  override      -> {p3.id:<20} {d3['basis']}")

    try:
        select(claim, override="no_such_pack")
    except base.PackError:
        print("  unknown override rejected")
    else:
        raise AssertionError("unknown override was accepted")
    print("registry self-test OK")
    return True


if __name__ == "__main__":
    if "--list" in sys.argv:
        for pid, p in discover().items():
            flag = " (conservative default)" if p.conservative else ""
            print(f"{pid:<22} v{p.version:<8} {len(p.rows()):>3} categories{flag}")
            print(f"  {p.description[:100]}")
        sys.exit(0)
    sys.exit(0 if self_test() else 1)
