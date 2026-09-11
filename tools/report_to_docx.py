#!/usr/bin/env python3
"""Render a generated report (the markdown subset generate_final_report.py
emits: #/##/### headings, paragraphs, - bullets, | tables |, > quotes,
**bold** markers) into a client-presentable Word document via docx_lib.

  python3 tools/report_to_docx.py FINAL_REPORT_client.md [out.docx]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_lib import para, h1, h2, table, write_docx, MUTED  # noqa: E402


def _clean(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


def _flush_table(body, tbl):
    if not tbl:
        return
    header, rows = tbl[0], [r for r in tbl[1:]
                            if not all(set(c) <= {"-", ":", " "} for c in r)]
    ncol = max(len(header), *(len(r) for r in rows)) if rows else len(header)
    widths = [int(9600 / ncol)] * ncol
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    body.append(table(widths, [_clean(c) for c in header] + [""] *
                      (ncol - len(header)), [[_clean(c) for c in r]
                                             for r in rows]))
    tbl.clear()


def md_to_docx(md_path: Path, out_path: Path):
    body = []
    tbl = []
    first_h1 = True
    for raw in md_path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            tbl.append([c.strip() for c in line.strip("|").split("|")])
            continue
        _flush_table(body, tbl)
        if not line.strip():
            continue
        if line.startswith("### "):
            body.append(para(_clean(line[4:]), bold=True, size=22,
                             space_before=200))
        elif line.startswith("## "):
            body.append(h2(_clean(line[3:])))
        elif line.startswith("# "):
            if first_h1:
                body.append(para(_clean(line[2:]), bold=True, size=40,
                                 center=True, space_before=600,
                                 space_after=300))
                first_h1 = False
            else:
                body.append(h1(_clean(line[2:])))
        elif line.startswith("> "):
            body.append(para(_clean(line[2:]), italic=True, color=MUTED))
        elif line.startswith(("- ", "* ")):
            body.append(para(_clean(line[2:]), bullet=True))
        elif line.startswith("---"):
            continue
        else:
            body.append(para(_clean(line)))
    _flush_table(body, tbl)
    write_docx(out_path, "".join(body))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".docx")
    md_to_docx(src, dst)
