"""Medical & pharmaceutical domain content validator for uploaded PDFs.

Validates that uploaded documents represent legitimate medical, pharmaceutical,
pharmacological, or clinical prescribing literature before chunking, embedding,
and indexing into the vector database.
"""
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.pdf_extractor import PageContent

logger = logging.getLogger(__name__)

# Key FDA Prescribing Information / Package Insert / Clinical Document Markers
_MEDICAL_SECTIONS = [
    r"\bindications?\s+(and|&)\s+usage\b",
    r"\bdosage\s+(and|&)\s+administration\b",
    r"\bdosage\s+forms?\s+(and|&)\s+strengths\b",
    r"\bcontraindications?\b",
    r"\bwarnings?\s+(and|&)\s+precautions?\b",
    r"\badverse\s+reactions?\b",
    r"\bdrug\s+interactions?\b",
    r"\buse\s+in\s+specific\s+populations?\b",
    r"\bclinical\s+pharmacology\b",
    r"\bmechanism\s+of\s+action\b",
    r"\bpharmacokinetics\b",
    r"\bpharmacodynamics\b",
    r"\bclinical\s+studies?\b",
    r"\bnonclinical\s+toxicology\b",
    r"\bpatient\s+counseling\s+information\b",
    r"\bhighlights\s+of\s+prescribing\s+information\b",
    r"\bprescribing\s+information\b",
    r"\bboxed\s+warning\b",
    r"\bhow\s+supplied\b",
    r"\boverdosage\b",
    r"\bpediatric\s+use\b",
    r"\bgeriatric\s+use\b",
    r"\bpregnancy\b",
    r"\blactation\b",
    r"\bmedication\s+guide\b",
    r"\binstructions\s+for\s+use\b",
    r"\bpatient\s+package\s+insert\b",
    r"\bpatient\s+information\b",
    r"\bprincipal\s+display\s+panel\b",
]

_MEDICAL_TERMS = [
    r"\b(mg|mcg|ml|g|kg|iu|mmol)\b",
    r"\b(tablet|capsule|injection|subcutaneous|intravenous|oral|solution|suspension|infusion|vial|pen)\b",
    r"\b(patient|patients|dosing|dose|dosage|administered|efficacy|adverse\s+events?|side\s+effects?)\b",
    r"\b(contraindicated|warning|precaution|toxicity|clearance|half-life|plasma|serum|bioavailability)\b",
    r"\b(clinical\s+trial|placebo|monotherapy|concomitant|pharmacology|pharmacokinetic|pharmacodynamic)\b",
    r"\b(renal|hepatic|cardiovascular|pulmonary|gastrointestinal|dermatologic|neurologic)\b",
    r"\b(fda|rx\s+only|ndc|package\s+insert|monograph|drug\s+label)\b",
]

_SECTION_REGEXES = [re.compile(p, re.IGNORECASE) for p in _MEDICAL_SECTIONS]
_TERM_REGEXES = [re.compile(p, re.IGNORECASE) for p in _MEDICAL_TERMS]


class PDFDomainValidationError(ValueError):
    """Raised when an uploaded PDF does not meet medical/pharmaceutical domain criteria."""
    pass


def validate_pdf_domain(pages: list["PageContent"], drug_name: str = "") -> dict:
    """Analyze extracted PDF pages to verify medical/pharmaceutical domain relevance.

    Returns a dict with domain validation metrics if valid, or raises
    PDFDomainValidationError if the document is off-topic.
    """
    if not pages:
        raise PDFDomainValidationError("Uploaded document is empty or could not be read.")

    # Combine text from up to first 5 pages (where prescribing headers & indications reside)
    combined_sample = "\n".join(p.text for p in pages[:5]).strip()
    full_sample = "\n".join(p.text for p in pages).strip()

    if not full_sample or len(full_sample) < 50:
        raise PDFDomainValidationError(
            "Uploaded document contains insufficient readable text to verify medical domain."
        )

    # 1. Count medical section matches
    section_matches = []
    for rx in _SECTION_REGEXES:
        if rx.search(full_sample):
            section_matches.append(rx.pattern)

    # 2. Count medical term matches
    term_matches = []
    for rx in _TERM_REGEXES:
        if rx.search(full_sample):
            term_matches.append(rx.pattern)

    # 3. Check drug name match if provided
    drug_name_found = False
    if drug_name and len(drug_name.strip()) >= 2:
        norm_name = re.escape(drug_name.strip())
        if re.search(rf"\b{norm_name}\b", full_sample, re.IGNORECASE):
            drug_name_found = True

    section_count = len(section_matches)
    term_count = len(term_matches)

    # Evaluation rule:
    # - Standard PI/label typically matches 3+ section headings and 4+ term categories.
    # - Clinical monographs/papers match at least 1 section heading and 4+ term categories, or 5+ terms.
    is_valid = (
        section_count >= 2
        or (section_count >= 1 and term_count >= 3)
        or (term_count >= 5)
        or (drug_name_found and term_count >= 3)
    )

    logger.info(
        "PDF Domain validation for '%s': pages=%d sections=%d terms=%d drug_found=%s -> valid=%s",
        drug_name, len(pages), section_count, term_count, drug_name_found, is_valid
    )

    if not is_valid:
        raise PDFDomainValidationError(
            "The uploaded document does not appear to be a medical, pharmaceutical, "
            "or prescribing information document. Please upload official FDA prescribing "
            "labels, package inserts, or clinical healthcare documents."
        )

    return {
        "valid": True,
        "section_matches": section_count,
        "term_matches": term_count,
        "drug_name_found": drug_name_found,
    }
