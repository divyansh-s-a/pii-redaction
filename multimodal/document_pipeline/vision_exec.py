#!/usr/bin/env python3
"""One compiled Vision helper, shared by the reader and the objective pass.

`swift file.swift args...` COMPILES the helper before every run. The document
pipeline calls it several times per page -- OCR, QR, and one batch of face
tiles -- across concurrent workers, so a 29-page bundle paid for hundreds of
compiles and ran them many-at-once. That contention is not theoretical: it
made `swift` spawns fail and write nothing to stdout, and both call sites
read an empty result as "this page has no text / no faces". Pages were then
redacted with no text coverage at all, silently.

Compiling once and invoking the binary removes the cause rather than
retrying around it. The build is cached next to the source and rebuilt only
when the source is newer, and every failure path falls back to `swift` so a
machine without `swiftc` still works.
"""
import os
import subprocess
import threading
from pathlib import Path

SRC = Path(__file__).resolve().parent / "vision_helper.swift"
BIN = SRC.with_suffix(".bin")
_LOCK = threading.Lock()
_STATE = {"checked": False, "argv0": None}


def _build():
    """-> [executable] to prefix arguments with. Falls back to swift."""
    if BIN.exists() and BIN.stat().st_mtime >= SRC.stat().st_mtime:
        return [str(BIN)]
    tmp = BIN.with_suffix(f".bin.{os.getpid()}")
    try:
        r = subprocess.run(["swiftc", "-O", str(SRC), "-o", str(tmp)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and tmp.exists():
            tmp.replace(BIN)              # atomic: concurrent builds are safe
            return [str(BIN)]
    except Exception:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    return ["swift", str(SRC)]


def argv(rest):
    """Full argument vector for a helper call, compiling once per process."""
    with _LOCK:
        if not _STATE["checked"]:
            _STATE["argv0"] = _build()
            _STATE["checked"] = True
    return list(_STATE["argv0"]) + list(rest)
