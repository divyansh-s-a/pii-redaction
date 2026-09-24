#!/usr/bin/env python3
"""Modality registry — the table stage zero consults to route an input.

Design rule (owner decision 2026-09-17): adding support for a new kind of data
must be a REGISTRY ENTRY, not a rewrite. Everything modality-specific about
routing lives in MODALITIES below; analyze.py contains no per-format branching
beyond what it reads from here.

A modality maps to a PIPELINE, not to code: `audio_pii` is the existing audio
workflow at the project root; `document_pipeline` is the document workflow in
this folder. `unsupported` means we can identify it but have nothing to run.
"""

# extension -> modality. Lowercase, no leading dot.
EXTENSIONS = {
    # --- audio: the existing pipeline ---
    "mp3": "audio", "wav": "audio", "m4a": "audio", "ogg": "audio",
    "flac": "audio", "aac": "audio", "opus": "audio", "wma": "audio",
    "mpeg": "audio", "mpga": "audio", "mp2": "audio", "amr": "audio",
    # --- video: audio pipeline extracts the track ---
    "mp4": "video", "mov": "video", "m4v": "video", "avi": "video",
    "mkv": "video", "webm": "video",
    # --- page-based documents ---
    "pdf": "document",
    "docx": "document", "doc": "document", "rtf": "document",
    "odt": "document", "pages": "document",
    # --- standalone images (a scan without a PDF wrapper) ---
    "png": "image", "jpg": "image", "jpeg": "image", "tif": "image",
    "tiff": "image", "bmp": "image", "heic": "image", "webp": "image",
    "gif": "image",
    # --- tabular ---
    "xlsx": "spreadsheet", "xls": "spreadsheet", "ods": "spreadsheet",
    "csv": "spreadsheet", "tsv": "spreadsheet",
    # --- slides ---
    "pptx": "presentation", "ppt": "presentation", "odp": "presentation",
    # --- engineering drawings ---
    "dwg": "cad", "dxf": "cad", "step": "cad", "stp": "cad", "iges": "cad",
    "igs": "cad",
    # --- mail ---
    "eml": "email", "msg": "email",
    # --- plain text / structured text ---
    "txt": "text", "md": "text", "json": "text", "xml": "text",
    "html": "text", "htm": "text", "log": "text",
    # --- containers: expand, then route each member ---
    "zip": "archive", "tar": "archive", "gz": "archive", "7z": "archive",
}

# leading bytes -> modality, for inputs whose extension lies or is missing.
MAGIC = [
    (b"%PDF-", "document"),
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"II*\x00", "image"), (b"MM\x00*", "image"),          # TIFF
    (b"RIFF", "audio"),                                     # WAV (or AVI)
    (b"ID3", "audio"), (b"\xff\xfb", "audio"),              # MP3
    (b"OggS", "audio"),
    (b"fLaC", "audio"),
    (b"\x00\x01\x00\x00", "cad"),                           # some DWG/TTF
    (b"AC10", "cad"),                                       # DWG version tag
    (b"PK\x03\x04", None),                                  # OOXML/ODF/zip
]

MODALITIES = {
    "audio":        {"pipeline": "audio_pii",        "ready": True},
    "video":        {"pipeline": "audio_pii",        "ready": True,
                     "note": "audio track extracted; no video-frame redaction"},
    "document":     {"pipeline": "document_pipeline",  "ready": True},
    "image":        {"pipeline": "document_pipeline",  "ready": True,
                     "note": "wrapped as a single-page document"},
    "spreadsheet":  {"pipeline": "document_pipeline",  "ready": False,
                     "note": "cell-level extraction not implemented"},
    "presentation": {"pipeline": "document_pipeline",  "ready": False,
                     "note": "slide extraction not implemented"},
    "cad":          {"pipeline": "document_pipeline",  "ready": False,
                     "note": "title-block/annotation text not implemented"},
    "email":        {"pipeline": "document_pipeline",  "ready": False,
                     "note": "header/body/attachment split not implemented"},
    "text":         {"pipeline": "document_pipeline",  "ready": True,
                     "note": "text-only path; no localisation needed"},
    "archive":      {"pipeline": "expand",           "ready": False,
                     "note": "expand and route members individually"},
    "unknown":      {"pipeline": "unsupported",      "ready": False},
}


def modality_for(path, sniff_bytes=b""):
    """Extension first (cheap and usually right), magic bytes as the check.

    Magic WINS on disagreement: a mislabelled .txt that is really a PDF must
    route to the document pipeline, and a PDF renamed .dat must still be found.
    OOXML/ODF share the zip magic, so PK falls back to the extension.
    """
    ext = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else ""
    by_ext = EXTENSIONS.get(ext, "unknown")
    by_magic = None
    for sig, mod in MAGIC:
        if sniff_bytes.startswith(sig):
            by_magic = mod
            break
    if by_magic and by_magic != by_ext:
        return by_magic, ("magic bytes override extension "
                          f"({ext or 'none'} -> {by_magic})")
    return by_ext, ("extension" if by_ext != "unknown" else "unrecognised")
