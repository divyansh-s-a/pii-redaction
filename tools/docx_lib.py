#!/usr/bin/env python3
"""Minimal stdlib WordprocessingML builder shared by report generators.
A .docx is a ZIP of XML parts; these helpers emit paragraphs, tables and
sections and package them. See tools/make_taxonomy_docx.py for origin."""
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

GREY, HDR, RED, GREEN, MUTED = "F2F2F2", "D9E2F3", "C00000", "1F7A33", "666666"


def run(text, bold=False, italic=False, size=20, color=None):
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    return (f"<w:r><w:rPr>{''.join(props)}</w:rPr>"
            f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>')


def para(text="", style=None, bold=False, italic=False, size=20, color=None,
         bullet=False, center=False, space_after=120, space_before=0,
         page_break=False, sect=None):
    pprops = []
    if style:
        pprops.append(f'<w:pStyle w:val="{style}"/>')
    if bullet:
        pprops.append('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>')
    if sect:
        pprops.append(sect)
    pprops.append(f'<w:spacing w:before="{space_before}" w:after="{space_after}"/>')
    if center:
        pprops.append('<w:jc w:val="center"/>')
    body = run(text, bold, italic, size, color) if text else ""
    if page_break:
        body += '<w:r><w:br w:type="page"/></w:r>'
    return f"<w:p><w:pPr>{''.join(pprops)}</w:pPr>{body}</w:p>"


def h1(t):
    return para(t, style="Heading1", space_before=300, space_after=120)


def h2(t):
    return para(t, style="Heading2", space_before=240, space_after=100)


def cell(text, width, bold=False, shade=None, color=None, size=18):
    shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>' if shade else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shd}'
            '<w:tcMar><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/>'
            '<w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar></w:tcPr>'
            f'<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>{run(text, bold=bold, size=size, color=color)}</w:p></w:tc>')


def table(widths, header, rows, row_color=None):
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    b = ('<w:tblBorders>' + ''.join(
        f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')
    xml = [f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>{b}'
           '<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>' + grid + '</w:tblGrid>']
    xml.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>' + "".join(
        cell(h, widths[i], bold=True, shade=HDR) for i, h in enumerate(header)) + '</w:tr>')
    for ri, r in enumerate(rows):
        shade = GREY if ri % 2 else None
        cells = []
        for ci, t in enumerate(r):
            color = row_color(ri, ci, t) if row_color else None
            cells.append(cell(t, widths[ci], shade=shade, color=color,
                              bold=(ci == 0 and len(widths) == 5)))
        xml.append("<w:tr>" + "".join(cells) + "</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def sectpr(landscape=False, in_ppr=True):
    w, h = (16838, 11906) if landscape else (11906, 16838)
    orient = ' w:orient="landscape"' if landscape else ""
    s = (f'<w:sectPr><w:pgSz w:w="{w}" w:h="{h}"{orient}/>'
         '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
         'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>')
    return s




STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="300" w:after="120"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="30"/><w:szCs w:val="30"/><w:color w:val="1F3864"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
<w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="240" w:after="100"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/><w:color w:val="2E5395"/></w:rPr></w:style>
</w:styles>"""

NUMBERING = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/>
<w:lvlText w:val="•"/><w:lvlJc w:val="left"/>
<w:pPr><w:ind w:left="360" w:hanging="200"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>"""

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""


EMU_PER_PX_96DPI = 9525


def img_para(rel_id, img_id, width_px, height_px, target_w_emu=5943600):
    """Centered inline image paragraph. target_w_emu default = 6.5 inches;
    height scales by aspect ratio."""
    cy = int(target_w_emu * height_px / width_px)
    return (
        '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="160"/></w:pPr>'
        '<w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f'<wp:extent cx="{target_w_emu}" cy="{cy}"/>'
        f'<wp:docPr id="{img_id}" name="image{img_id}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{img_id}" name="image{img_id}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{target_w_emu}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def write_docx(out_path, body_xml, images=None):
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>" + body_xml + sectpr(landscape=False) + "</w:body></w:document>")
    out = Path(out_path)
    images = images or {}   # {rel_id: png_path}
    ctypes = CONTENT_TYPES
    if images:
        ctypes = ctypes.replace(
            "</Types>",
            '<Default Extension="png" ContentType="image/png"/></Types>')
    doc_rels = DOC_RELS
    media = {}
    for i, (rid, path) in enumerate(sorted(images.items()), 1):
        fn = f"media/image{i}.png"
        media[f"word/{fn}"] = Path(path).read_bytes()
        doc_rels = doc_rels.replace(
            "</Relationships>",
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
            f'officeDocument/2006/relationships/image" Target="{fn}"/></Relationships>')
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/numbering.xml", NUMBERING)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        for name, blob in media.items():
            z.writestr(name, blob)
    import xml.dom.minidom
    with zipfile.ZipFile(out) as z:
        for name in z.namelist():
            if name.endswith((".xml", ".rels")):
                xml.dom.minidom.parseString(z.read(name))
    return out
