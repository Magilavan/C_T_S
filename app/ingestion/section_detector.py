"""Detects FDA prescribing-information section headings and appendices (Medication Guides,
Patient Information, Instructions for Use) so chunks can be split on true section
boundaries instead of arbitrary token windows.

FDA labels and RxAbbVie prescribing documents follow standard structured sections:
  HIGHLIGHTS OF PRESCRIBING INFORMATION
  FULL PRESCRIBING INFORMATION: CONTENTS*
  BOXED WARNING
  1 INDICATIONS AND USAGE
  2 DOSAGE AND ADMINISTRATION
    2.1 Recommended Evaluations ...
  3 DOSAGE FORMS AND STRENGTHS
  4 CONTRAINDICATIONS
  5 WARNINGS AND PRECAUTIONS
    5.1 Serious Infections
  6 ADVERSE REACTIONS
  7 DRUG INTERACTIONS
  8 USE IN SPECIFIC POPULATIONS
  ...
  17 PATIENT COUNSELING INFORMATION
  MEDICATION GUIDE
  PATIENT INFORMATION / PATIENT PACKAGE INSERT
  INSTRUCTIONS FOR USE
"""
import re
from dataclasses import dataclass

# Matches numbered subsections, e.g. "5.1 Serious Infections" or "2.4 Recommended Dosage in Psoriatic Arthritis"
SUBSECTION_RE = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2})\s+([A-Z][A-Za-z0-9 ,/\-'’\(\)]{2,90})\s*$",
    re.MULTILINE,
)

# Matches top-level numbered sections, e.g. "5 WARNINGS AND PRECAUTIONS" or "1 INDICATIONS AND USAGE"
SECTION_RE = re.compile(
    r"^\s*(\d{1,2})\s+([A-Z][A-Z0-9 ,/\-'\(\)]{2,60})\s*$",
    re.MULTILINE,
)

# Matches Boxed Warnings and Contents tables
BOXED_WARNING_RE = re.compile(
    r"^\s*(WARNING:\s*[A-Z].*|BOXED WARNING|FULL PRESCRIBING INFORMATION:\s*CONTENTS\*?)",
    re.MULTILINE | re.IGNORECASE,
)

# Matches top-level FDA Highlights & Full PI headers
HIGHLIGHTS_RE = re.compile(
    r"^\s*(HIGHLIGHTS OF PRESCRIBING INFORMATION|FULL PRESCRIBING INFORMATION(?!\s*:\s*CONTENTS))",
    re.MULTILINE | re.IGNORECASE,
)

# Matches Patient-facing Appendices found on rxabbvie.com and FDA labels
PATIENT_APPENDICES_RE = re.compile(
    r"^\s*(MEDICATION\s+GUIDE(?:\s+FOR\s+[A-Z0-9\s\-]+)?|"
    r"PATIENT\s+INFORMATION(?:\s+FOR\s+[A-Z0-9\s\-]+)?|"
    r"PATIENT\s+PACKAGE\s+INSERT|"
    r"INSTRUCTIONS\s+FOR\s+USE(?:\s+FOR\s+[A-Z0-9\s\-]+)?(?:\s*-\s*[A-Za-z0-9\s\-]+)?|"
    r"PRINCIPAL\s+DISPLAY\s+PANEL|"
    r"HOW\s+SUPPLIED\s*/\s*STORAGE\s+AND\s+HANDLING)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class SectionMarker:
    heading: str          # e.g. "2.4 Recommended Dosage in Psoriatic Arthritis" or "MEDICATION GUIDE"
    start_offset: int     # character offset into the page/document text
    is_boxed_warning: bool = False
    is_patient_guide: bool = False


def detect_sections(text: str) -> list[SectionMarker]:
    """Returns section markers found in a block of text, in document order."""
    markers: list[SectionMarker] = []

    # 1. Boxed Warning
    for m in BOXED_WARNING_RE.finditer(text):
        matched_str = m.group(1).strip()
        is_boxed = "WARNING" in matched_str.upper()
        heading = "BOXED WARNING" if is_boxed else "FULL PRESCRIBING INFORMATION: CONTENTS"
        markers.append(SectionMarker(
            heading=heading,
            start_offset=m.start(),
            is_boxed_warning=is_boxed,
        ))

    # 2. Highlights / Full PI
    for m in HIGHLIGHTS_RE.finditer(text):
        heading = m.group(1).strip().upper()
        if not any(abs(mk.start_offset - m.start()) < 15 for mk in markers):
            markers.append(SectionMarker(heading=heading, start_offset=m.start()))

    # 3. Patient Appendices (Medication Guide, IFU, Patient Information)
    for m in PATIENT_APPENDICES_RE.finditer(text):
        heading = m.group(1).strip().upper()
        if not any(abs(mk.start_offset - m.start()) < 15 for mk in markers):
            markers.append(SectionMarker(
                heading=heading,
                start_offset=m.start(),
                is_patient_guide=True,
            ))

    # 4. Numbered Subsections (e.g. 5.1 Serious Infections)
    for m in SUBSECTION_RE.finditer(text):
        heading = f"{m.group(1)} {m.group(2).strip()}"
        markers.append(SectionMarker(heading=heading, start_offset=m.start()))

    # 5. Numbered Top-Level Sections (e.g. 5 WARNINGS AND PRECAUTIONS)
    for m in SECTION_RE.finditer(text):
        sec_num = m.group(1)
        # Avoid double counting if this line matches a subsection or is adjacent
        heading = f"{sec_num} {m.group(2).strip()}"
        if not any(mk.heading.startswith(sec_num + ".") and abs(mk.start_offset - m.start()) < 10 for mk in markers):
            markers.append(SectionMarker(heading=heading, start_offset=m.start()))

    markers.sort(key=lambda x: x.start_offset)
    return markers


def current_section_for_offset(
    markers: list[SectionMarker],
    offset: int,
    default_section: str = "UNSPECIFIED",
) -> str:
    """Given a character offset, return the most recent section heading before it."""
    current = default_section
    for m in markers:
        if m.start_offset <= offset:
            current = m.heading
        else:
            break
    return current

