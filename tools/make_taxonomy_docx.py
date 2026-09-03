#!/usr/bin/env python3
"""Generate outputs/Audio_PII_Taxonomy_v0.2.docx with the stdlib only.

This machine has no node/pandoc/LibreOffice/python-docx, so the document is
built as raw WordprocessingML inside a ZIP (which is all a .docx is). Content
mirrors outputs/Audio_PII_Taxonomy_v0.2.md + Action_Matrix_v1.0.md (normative).
Verify with: textutil -convert txt outputs/Audio_PII_Taxonomy_v0.2.docx
"""
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


# ------------------------- content ------------------------------------------
CATS = [
    ("FULL_NAME", "Identifying real-person name (multi-part, or a lone first name that clearly identifies)", "Beep - except established public figures in their public capacity (kept audible, flagged & audited)", "“Ramesh Kumar”; “Dr. Priya Anand” (name only, not “Dr.”)", "Fictional characters; initials; roles (“the doctor”)"),
    ("PHONE", "Complete phone/fax number, written or spoken", "Beep", "“98765 43210”; “plus nine one double nine…”; “ek do teen…”", "Partial suffixes; extensions alone; obviously fake numbers"),
    ("EMAIL", "Complete email address incl. spoken form", "Beep", "“priya dot anand at outlook dot com”", "Domain alone; incomplete addresses; username without domain"),
    ("ADDRESS", "Street/residence-level personal address", "Beep", "“12 MG Road, Indiranagar”; PIN code spoken as part of the full address", "City/state alone; workplaces/venues; place of birth"),
    ("DATE_OF_BIRTH", "Explicitly stated birth date", "Beep", "“born seventeenth September nineteen ninety”", "Ordinary dates; expiry dates; birth year alone"),
    ("AGE", "Explicit numerical age of a person", "Beep", "“I am 42”; “she is turning eighty” (number only)", "Durations (“3-4 months of pain” is NOT age); grade levels"),
    ("PASSPORT_NUMBER", "Complete passport number, any country", "Beep", "“P eight three nine two two eight four one”", "Label without value"),
    ("DRIVERS_LICENSE", "Complete driving licence number, any jurisdiction", "Beep", "“KA zero five two zero one one…”", "Vehicle registration (separate category); licence type"),
    ("GOVERNMENT_ID", "Any national identity/citizen/voter/social-security-type ID, any jurisdiction", "Beep", "Aadhaar (“4321 8765 2109”); US SSN; UK NINO; voter ID; ration card ID", "Corporate registrations; unclear unlabeled numbers; patient IDs (see ASSIGNED_ID)"),
    ("TAX_ID", "Personal tax identifier", "Beep", "Personal PAN (“ABCDE1234F”); ITIN; UTR", "GSTIN and corporate tax numbers"),
    ("INSURANCE_ID", "Insurance policy/member/subscriber ID", "Beep", "“policy number HLT three three four…”", "Insurer name; plan type; claim topics"),
    ("STUDENT_ID", "Student/enrolment identifier", "Beep", "“roll number 44821”", "School name (see ORG_PERSON_LINKED); class/grade"),
    ("LEGAL_CASE_ID", "Person-linked court/docket/case identifier", "Beep", "“case CV 2023 08841”", "Court names; legal topics; unlinked case numbers"),
    ("IMMIGRATION_STATUS", "Explicitly stated visa/asylum/residency status", "Beep", "“I have refugee status”", "Reference numbers (see ASSIGNED_ID); agency mentions"),
    ("PAYMENT_CARD_NUMBER", "Complete raw card number; card+expiry+CVV block = ONE interval", "Beep", "“4916 3385 0608 2832, expiry oh nine twenty-eight, CVV…”", "“card ending 4832” (masked - excluded entirely); expiry alone"),
    ("BANK_ACCOUNT_NUMBER", "Complete personal account number / IBAN", "Beep", "“account number eight three nine two one zero four four”", "Masked suffixes; bank names; IFSC/SWIFT alone"),
    ("FINANCIAL_ACCOUNT_INFO", "Personal brokerage/loan/pension refs; UPI IDs / VPAs", "Beep", "“ramesh at oksbi”; loan ref; pension member no.", "Bank infrastructure beside a full account number (one combined interval)"),
    ("FINANCIAL_INFO", "Clearly personal substantive financial fact", "Beep", "“my salary is 12 lakh”; “bill was 30,635 rupees”", "Incidental prices; limits; organization balances"),
    ("PASSWORD_CREDENTIAL", "Password, PIN, OTP, recovery code", "Beep", "“OTP is 4 5 2 9 1 1”", "Credential labels; reset instructions; CVV inside a card block"),
    ("IP_ADDRESS", "Complete IP linked to a person/account/device", "Beep", "“203.0.113.45”", "Examples; network-range discussion"),
    ("DEVICE_IDENTIFIER", "IMEI, MAC, serial, UDID with a personal-device link", "Beep", "“IMEI 356938035643809”", "Product models; OS versions"),
    ("VEHICLE_REGISTRATION", "Person-linked plate/registration/VIN", "Beep", "“KA 01 AB 1234”", "Vehicle models; incomplete fragments"),
    ("HEALTH_MEDICAL_INFO", "Person's diagnosis, symptom, medication, treatment, test result, blood type", "Beep in strict - KEPT AUDIBLE in medical_consult (annotated only)", "“type 2 diabetes”; “Metformin 500mg”", "Generic words (doctor, clinic, appointment); casual diet/stress talk"),
    ("BIOMETRIC_ID", "Biometric / genetic-kit identifier value", "Beep", "“kit K10023488-2”", "Voice sound itself; generic biometrics talk"),
    ("SENSITIVE_DEMOGRAPHIC", "Explicitly stated religion, ethnicity, caste, orientation, gender identity, political view", "Beep", "“I am trans”; explicit caste statement", "Anything inferred from name/accent/location; isolated form markers"),
    ("CHILD_INFO", "Concrete personal info of an identifiable child (stacks with the specific category)", "Beep", "“my daughter Lily, born 05/03/2018” [CHILD_INFO + DATE_OF_BIRTH]", "Generic mentions of children/classes/schools"),
    ("NATIONALITY", "Explicit person-linked nationality/citizenship", "Beep", "“she is a Nepali citizen”", "Country codes; document boilerplate; place of birth"),
    ("USERNAME_HANDLE", "Complete personal social/platform handle (NEW in v0.2)", "Beep", "“at priya underscore anand”; “my insta is…”", "Org/brand accounts; fragments; emails (see EMAIL)"),
    ("ASSIGNED_ID", "Person-assigned org identifier: patient/employee/customer/membership/application no. (NEW)", "Beep", "“patient ID AP-33821”; NHS number; asylum reference", "Order/claim/workflow numbers; org-only IDs"),
    ("ORGANIZATION_PERSON_LINKED", "Named clinic/employer/school/provider tied to a person's identity — incl. their insurer, TPA, lender (D8)", "Beep", "“my doctor at Apollo Jayanagar”; “my claim with Medi Assist”; “the loan from Orix”", "Organizations discussed generically; brands; bare bank names"),
    ("OTHER", "Concrete person-linked identifier fitting no category (rare; needs reviewer note)", "Beep", "- (discovery label)", "Anything that fits a specific category"),
]

ROUTING = [
    ("Aadhaar, US SSN, UK NINO, NRIC, voter ID, ration card", "GOVERNMENT_ID"),
    ("Personal PAN, TIN, ITIN, UTR, TFN", "TAX_ID"),
    ("GSTIN, EIN, corporate registrations", "excluded (organization-only)"),
    ("NHS number, patient/medical-record number, hospital UHID", "ASSIGNED_ID"),
    ("Employee / customer / membership / application number", "ASSIGNED_ID"),
    ("Immigration or asylum reference number", "ASSIGNED_ID (+ IMMIGRATION_STATUS only if a status is stated)"),
    ("UPI ID / VPA (“name@oksbi”), loan/pension refs", "FINANCIAL_ACCOUNT_INFO"),
    ("IFSC, sort code, routing number, SWIFT/BIC - alone", "excluded (bank infrastructure)"),
    ("IMEI, MAC, serial, UDID", "DEVICE_IDENTIFIER"),
    ("Licence plate, VIN", "VEHICLE_REGISTRATION"),
    ("Social handle, platform username", "USERNAME_HANDLE"),
]

EX_W = [3200, 2600, 1600, 1626]
EX_HDR = ["Spoken value", "Category", "strict", "medical_consult"]
EX_MED = [
    ("“Ramesh Kumar”", "FULL_NAME", "BEEPED", "BEEPED"),
    ("“Jayanagar, Bangalore” (his locality)", "ADDRESS", "BEEPED", "BEEPED"),
    ("“42” (years old)", "AGE", "BEEPED", "BEEPED"),
    ("“type 2 diabetes”", "HEALTH_MEDICAL_INFO", "BEEPED", "KEPT audible"),
    ("“Metformin 500mg”", "HEALTH_MEDICAL_INFO", "BEEPED", "KEPT audible"),
    ("“Apollo Clinic” (his clinic)", "ORGANIZATION_PERSON_LINKED", "BEEPED", "BEEPED"),
    ("“AP-33821” (patient ID)", "ASSIGNED_ID", "BEEPED", "BEEPED"),
    ("“98765 43210”", "PHONE", "BEEPED", "BEEPED"),
    ("“years old”, “I'm taking” (lead-ins)", "- not a value", "kept", "kept"),
]
EX_BANK = [
    ("“Arjun Mehta”", "FULL_NAME", "BEEPED", "BEEPED"),
    ("“nine eight seven six five four three two one zero”", "PHONE", "BEEPED", "BEEPED"),
    ("“card ending four eight three two”", "- masked/partial: excluded by policy", "kept", "kept"),
    ("“arjun at okhdfc” (UPI)", "FINANCIAL_ACCOUNT_INFO", "BEEPED", "BEEPED"),
    ("“12 MG Road, Indiranagar, 560038”", "ADDRESS", "BEEPED", "BEEPED"),
    ("“IFSC HDFC0001234” (alone)", "- bank infrastructure: excluded", "kept", "kept"),
    ("“OTP is 4 5 2 9 1 1”", "PASSWORD_CREDENTIAL", "BEEPED", "BEEPED"),
    ("“my salary is 12 lakh”", "FINANCIAL_INFO", "BEEPED", "BEEPED"),
]
EX_POD = [
    ("“the Prime Minister said yesterday…” (public figure, public capacity)", "FULL_NAME + public_figure", "kept (flagged)", "kept (flagged)"),
    ("“our guest, Meera Krishnan” (private person)", "FULL_NAME", "BEEPED", "BEEPED"),
    ("“follow me at meera underscore k”", "USERNAME_HANDLE", "BEEPED", "BEEPED"),
    ("“Nidhi” (fictional story character)", "- fiction: excluded", "kept", "kept"),
    ("“Infosys announced results” (generic org)", "- not person-linked", "kept", "kept"),
]


def ex_color(ri, ci, t):
    if ci >= 2:
        if t.startswith("BEEPED"):
            return RED
        if t.startswith(("KEPT", "kept")):
            return GREEN
    return None


MASTER_W = [2050, 3550, 2900, 3300, 2900]


def master_color(ri, ci, t):
    if ci == 2 and "KEPT AUDIBLE" in t:
        return GREEN
    return None


body = []
# ---- section 1: portrait intro (ends with section break to landscape) ----
body.append(para("Audio PII Redaction - Taxonomy v0.2", bold=True, size=44,
                 center=True, space_before=2400, space_after=200))
body.append(para("What counts as PII in spoken audio, and what gets beeped - with examples",
                 size=24, color=MUTED, center=True, space_after=2000))
body.append(para("Deccan.ai - Audio PII Pilot  |  Version 0.2  |  2026-09-01  |  Owner: Shivangi",
                 size=20, color=MUTED, center=True))
body.append(para(page_break=True))
body.append(h1("1. Purpose and scope"))
body.append(para("This document defines the policy for detecting and redacting spoken personal information (PII) in audio, or the audio track of video. Detected PII intervals are replaced with a 1 kHz beep in the audio, and with “<CATEGORY REDACTED>” markers in the transcript - both driven by the same detection record, so they can never disagree."))
body.append(para("In scope: spoken PII in any language or accent, including code-switched speech (e.g. Hindi-English).", bullet=True))
body.append(para("Out of scope: visual PII (faces, on-screen text), and identifying people by their voice.", bullet=True))
body.append(para("Jurisdiction-neutral: no category is country-specific. Country identifiers (Aadhaar, SSN, NINO…) route into neutral categories - see the routing table (section 5).", bullet=True))
body.append(h1("2. Profiles - the same taxonomy, different actions per domain"))
body.append(para("Annotation never changes across domains: every PII value is always detected and recorded. What changes per domain profile is the ACTION - beep it, or keep it audible (annotate-only). Adding a domain later means adding one profile column; the taxonomy itself does not change."))
body.append(para("strict (default): every detected category is beeped. For maximum-privacy outputs (e.g. public clips).", bullet=True))
body.append(para("medical_consult: identical to strict EXCEPT health/medical content (diagnoses, symptoms, medicines) stays audible - in doctor-patient recordings the clinical content is the purpose of the recording; identity is removed, medicine stays.", bullet=True))
body.append(para("Public figures (all profiles): an established public figure named in their public capacity is NOT beeped - the interval is flagged and audited. Their private details (phone, address, health) are still beeped. When in doubt: beep.", bullet=True))
body.append(para("Three rules that apply everywhere: (1) every repetition of a value is redacted - one audible leak is a leak; (2) masked/partial values (“card ending 4832”) are excluded entirely; (3) fictional, test, and placeholder values are not PII."))
body.append(para(sect=sectpr(landscape=False)))  # end portrait section

# ---- section 2: landscape master table ----
body.append(h1("3. The 31 categories"))
body.append(para("“Beep” means beeped in every current profile. The one profile-dependent category is highlighted in green."))
body.append(table(MASTER_W, ["Category code", "What it covers", "Action (per profile)", "Examples", "Key exclusions"],
                  [list(c) for c in CATS], row_color=master_color))
body.append(para(sect=sectpr(landscape=True)))  # end landscape section

# ---- section 3: portrait examples/routing/history (final sectPr in body) ----
body.append(h1("4. Worked examples - what is beeped, what is kept"))
body.append(h2("4.1 Medical consultation (profile: medical_consult)"))
body.append(para("“My name is Ramesh Kumar, I'm calling from Jayanagar, Bangalore. I am 42 years old. I have type 2 diabetes and I'm taking Metformin 500mg. My Apollo Clinic patient ID is AP-33821. Call me back on 98765 43210.”", italic=True))
body.append(table(EX_W, EX_HDR, [list(r) for r in EX_MED], row_color=ex_color))
body.append(para("The only difference between profiles is the clinical content: medical_consult keeps “type 2 diabetes” and “Metformin” audible because that is what the recording exists to capture. Everything identifying the patient is beeped in both.", space_before=120))
body.append(h2("4.2 Banking support call (either profile - identical here)"))
body.append(para("“Am I speaking with Arjun Mehta? Your registered mobile is nine eight seven six five four three two one zero, card ending four eight three two. Your UPI arjun at okhdfc… the branch IFSC is HDFC0001234. The OTP is 4 5 2 9 1 1.”", italic=True))
body.append(table(EX_W, EX_HDR, [list(r) for r in EX_BANK], row_color=ex_color))
body.append(para("Note the two deliberate “kept” rows: a masked value (“ending 4832”) is excluded by policy decision D5, and a bank IFSC alone is infrastructure, not personal data.", space_before=120))
body.append(h2("4.3 Podcast / general content"))
body.append(table(EX_W, EX_HDR, [list(r) for r in EX_POD], row_color=ex_color))
body.append(h1("5. Routing table - country-specific identifiers"))
body.append(para("This table is normative: it fixes where known identifier types land, so the same value is never coded two ways."))
body.append(table([5426, 3600], ["Identifier", "Category"], [list(r) for r in ROUTING]))
body.append(h1("6. Never beeped"))
for t in [
    "Lead-ins and labels: “my number is…”, “email me at…”, “passport number” without a value.",
    "Masked or partial values: “card ending 4832”, “last four of my Aadhaar”.",
    "Fictional, test, sample or placeholder values - including fictional characters' names.",
    "Public figures' names in their public capacity (flagged and audited, never silently ignored).",
    "Generic organizations, broad locations, ordinary dates, incidental amounts.",
    "Bank infrastructure alone: IFSC, sort code, routing number, SWIFT/BIC.",
]:
    body.append(para(t, bullet=True))
body.append(h1("7. Version history"))
body.append(para("v0.2 clarifications (2026-09-01, D8): ORGANIZATION_PERSON_LINKED explicitly covers the person's insurer/TPA/lender; financial values count in any language or script.", bullet=True))
body.append(para("v0.2 (2026-09-01): SSN folded into GOVERNMENT_ID (jurisdiction neutrality); added USERNAME_HANDLE, ASSIGNED_ID, ORGANIZATION_PERSON_LINKED; masked/partial values excluded entirely; public-figure flag; routing table and India/audio edge cases added; actions moved to the profile-based Action Matrix v1.0.", bullet=True))
body.append(para("v0.1 (2026-09-01): initial 29-category taxonomy adapted from the Mindy text pilot.", bullet=True))
body.append(para("Normative sources: outputs/Audio_PII_Taxonomy_v0.2.md and outputs/Action_Matrix_v1.0.md (this document is the readable companion). Change control: every change needs a real example, expected impact, owner approval, and a version bump.", size=18, color=MUTED))

DOCUMENT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>" + "".join(body) + sectpr(landscape=False) + "</w:body></w:document>")

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

out = Path("outputs/Audio_PII_Taxonomy_v0.2.docx")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", CONTENT_TYPES)
    z.writestr("_rels/.rels", RELS)
    z.writestr("word/document.xml", DOCUMENT)
    z.writestr("word/styles.xml", STYLES)
    z.writestr("word/numbering.xml", NUMBERING)
    z.writestr("word/_rels/document.xml.rels", DOC_RELS)
print(f"written {out} ({out.stat().st_size} bytes)")

# well-formedness check of every XML part
import xml.dom.minidom
with zipfile.ZipFile(out) as z:
    for name in z.namelist():
        xml.dom.minidom.parseString(z.read(name))
print("all XML parts well-formed")
