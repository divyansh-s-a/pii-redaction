#!/usr/bin/env python3
"""One entry point for both arms — route automatically, or say exactly which.

    # let it decide (modality, then taxonomy signals)
    python3 router/dispatch.py INPUT

    # force an arm / a taxonomy
    python3 router/dispatch.py INPUT --audio
    python3 router/dispatch.py INPUT --video
    python3 router/dispatch.py INPUT --insurance
    python3 router/dispatch.py INPUT --generic

    # the long forms, equivalent to the shorthands above
    python3 router/dispatch.py INPUT --pipeline audio
    python3 router/dispatch.py INPUT --pipeline document --taxonomy insurance_medical

    # see the decision without running anything
    python3 router/dispatch.py INPUT --dry-run

Anything after `--` is passed through verbatim to the underlying tool, so the
arm-specific flags keep working:

    python3 router/dispatch.py call.mp3 --audio -- --sweep --assurance
    python3 router/dispatch.py scan.pdf --generic -- --workers 8

WHY A FORCED MODE EXISTS. Routing is a guess, and a wrong guess is expensive in
both directions: an insurance bundle sent to the generic pack loses the clinical
record, and a generic document sent to the insurance pack keeps health content it
should have removed. Whenever the operator already knows what the document is,
saying so is strictly better than letting signals infer it -- and the manifest
then records `explicit_override` rather than a score, so a reader can tell the
difference later.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                      # the multimodal arm
PROJ = PKG.parent                      # repository root (audio lives here)

# TWO modules are called `registry` -- router/registry.py (modalities) and
# taxonomies/registry.py (taxonomy packs) -- so a bare `import registry` here
# poisons the one analyze.py resolves and it fails with a missing attribute.
# Load each by path and keep them under distinct names. (Same trap as
# pipeline/redact.py shadowing document_pipeline/redact.py.)
sys.path.insert(0, str(HERE))          # so analyze.py finds ITS registry first


def _by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tax_registry = _by_path("tax_registry", PKG / "taxonomies" / "registry.py")
import analyze                          # noqa: E402

AUDIO_TOOL = PROJ / "tools" / "redact.py"
VIDEO_TOOL = PKG / "video_pipeline" / "run_video.py"
DOC_TOOL = PKG / "document_pipeline" / "run_document.py"


def decide(path, pipeline, taxonomy):
    """-> (arm, taxonomy_id_or_None, basis) without running anything."""
    if pipeline == "audio":
        return "audio", None, "explicit_override"
    if pipeline == "video":
        return "video", None, "explicit_override"
    if pipeline == "document":
        return "document", (None if taxonomy == "auto" else taxonomy), \
               "explicit_override"

    info = analyze.analyze(str(path))
    modality = info.get("modality", "unknown")
    arm = {"audio": "audio", "video": "video"}.get(modality, "document")
    if arm in ("audio", "video"):
        return arm, None, f"modality={modality}"
    if taxonomy != "auto":
        return "document", taxonomy, "explicit_override"
    return "document", None, f"modality={modality}; taxonomy chosen at run time"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--pipeline", choices=["auto", "audio", "video",
                                           "document"],
                    default="auto")
    ap.add_argument("--taxonomy", default="auto",
                    help="pack id, or 'auto' to select from the document's signals")
    ap.add_argument("--audio", action="store_true", help="= --pipeline audio")
    ap.add_argument("--video", action="store_true", help="= --pipeline video")
    ap.add_argument("--insurance", action="store_true",
                    help="= --pipeline document --taxonomy insurance_medical")
    ap.add_argument("--generic", action="store_true",
                    help="= --pipeline document --taxonomy generic_document")
    ap.add_argument("--out", help="output PDF (document arm)")
    ap.add_argument("--outdir", help="output directory (audio/video arm)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the decision and the command; run nothing")
    # Split on the literal `--` OURSELVES. argparse.REMAINDER after a
    # positional swallows this parser's own flags too, so `--insurance
    # --dry-run` got forwarded to the child instead of being acted on.
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        mine, passthrough = argv[:i], argv[i + 1:]
    else:
        mine, passthrough = argv, []
    a = ap.parse_args(mine)

    shorthand = sum((a.audio, a.video, a.insurance, a.generic))
    if shorthand > 1:
        ap.error("--audio, --video, --insurance and --generic are mutually "
                 "exclusive")
    if a.audio:
        a.pipeline = "audio"
    elif a.video:
        a.pipeline = "video"
    elif a.insurance:
        a.pipeline, a.taxonomy = "document", "insurance_medical"
    elif a.generic:
        a.pipeline, a.taxonomy = "document", "generic_document"

    src = Path(a.input)
    if not src.exists():
        ap.error(f"no such input: {src}")

    # fail early and clearly on an unknown pack rather than deep inside a run
    if a.taxonomy != "auto":
        known = set(tax_registry.discover())
        if a.taxonomy not in known:
            ap.error(f"unknown taxonomy {a.taxonomy!r}; have {sorted(known)}")

    arm, pack, basis = decide(src, a.pipeline, a.taxonomy)

    if arm in ("audio", "video"):
        tool = AUDIO_TOOL if arm == "audio" else VIDEO_TOOL
        cmd = [sys.executable, str(tool), str(src)]
        if a.outdir:
            cmd += ["--outdir", a.outdir]
        cwd = PROJ
    else:
        # Default into a `redacted/` SUBFOLDER, never beside the input.
        # Writing outputs into the source directory mixes originals with
        # redactions, and a later batch that globs *.pdf picks up its own
        # output as an input and redacts it again. A subfolder keeps the
        # corpus clean and makes "what did this produce" answerable at a
        # glance. --out still overrides.
        out = a.out or str(src.parent / "redacted" / (src.stem + "_redacted.pdf"))
        cmd = [sys.executable, str(DOC_TOOL), str(src), "--out", out,
               "--findings-out", str(Path(out).with_suffix(".jsonl"))]
        if pack:
            cmd += ["--taxonomy", pack]
        cwd = PKG
    cmd += passthrough

    print(f"  input     : {src}")
    print(f"  arm       : {arm}   ({basis})")
    print(f"  taxonomy  : {pack or ('n/a' if arm in ('audio', 'video') else 'selected at run time')}")
    print(f"  command   : {' '.join(cmd)}")
    if a.dry_run:
        return 0
    print()
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def self_test():
    """Decisions only -- nothing is executed."""
    import tempfile
    import fitz
    ok = True
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "x.pdf"
        d = fitz.open(); d.new_page().insert_text((60, 60), "hello"); d.save(pdf); d.close()
        mp3 = Path(td) / "y.mp3"
        mp3.write_bytes(b"ID3" + b"\0" * 64)
        mp4 = Path(td) / "z.mp4"
        mp4.write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 64)
        cases = [
            (pdf, "auto", "auto", "document", None),
            (pdf, "document", "generic_document", "document", "generic_document"),
            (pdf, "document", "insurance_medical", "document", "insurance_medical"),
            (pdf, "audio", "auto", "audio", None),       # override beats modality
            (mp3, "auto", "auto", "audio", None),
            (mp4, "auto", "auto", "video", None),
            (mp4, "audio", "auto", "audio", None),      # audio-only override
        ]
        for path, pl, tx, want_arm, want_pack in cases:
            arm, pack, basis = decide(path, pl, tx)
            good = arm == want_arm and pack == want_pack
            ok &= good
            print(f"  {path.suffix:<5} pipeline={pl:<9} taxonomy={tx:<18}"
                  f"-> {arm:<9}{str(pack):<18}{basis:<34}{'' if good else 'WRONG'}")
    print("dispatch self-test " + ("OK" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    sys.exit(main())
