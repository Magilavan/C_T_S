"""Citation extraction and formatting.

Every citation object returned to the frontend has:
  document      – drug name / document title
  section       – section number + title (never "UNSPECIFIED")
  page          – page number or "Not available"
  chunk_ref     – internal reference
"""
import re


def _clean_section(raw: str | None) -> str:
    """Return a display-safe section string; never return UNSPECIFIED."""
    if not raw or raw.strip().upper() in ("UNSPECIFIED", "", "NONE"):
        return "Not available"
    return raw.strip()


def _abbreviate_section_name(section: str) -> str:
    """Abbreviates lengthy FDA section headings for compact inline citation display.
    e.g. '17 PATIENT COUNSELING INFORMATION' -> '§17'
         '2.1 Recommended Dosage and Administration' -> '§2.1'
         'HIGHLIGHTS OF PRESCRIBING INFORMATION' -> 'Highlights'
    """
    if not section or section == "Not available":
        return section

    s = section.strip()
    # Match numbered section e.g. "17 PATIENT COUNSELING INFORMATION" -> "§17"
    m_num = re.match(r"^§?\s*(\d{1,2}(?:\.\d{1,2})?)(?:\s+.*)?$", s)
    if m_num:
        return f"§{m_num.group(1)}"

    upper_s = s.upper()
    if "HIGHLIGHTS" in upper_s:
        return "Highlights"
    if "BOXED WARNING" in upper_s:
        return "Boxed Warning"
    if "MEDICATION GUIDE" in upper_s:
        return "Med Guide"
    if "INSTRUCTIONS FOR USE" in upper_s or "IFU" in upper_s:
        return "IFU"
    if "PATIENT INFORMATION" in upper_s or "PATIENT PACKAGE INSERT" in upper_s:
        return "Patient Info"
    if "PATIENT COUNSELING" in upper_s:
        return "§17"

    return f"§{s}" if not s.startswith("§") else s


def _format_single_source(meta: dict, compact: bool = True) -> str:
    section_raw = meta.get("section") or ""
    section = _clean_section(section_raw)
    if compact and section != "Not available":
        section = _abbreviate_section_name(section)
    page = meta.get("page_number")

    parts = []
    if section and section != "Not available":
        if not section.startswith("§") and not section.lower().startswith("section") and not section in ("Highlights", "Boxed Warning", "Med Guide", "IFU", "Patient Info"):
            parts.append(f"§{section}")
        else:
            parts.append(section)
    if page and page != "Not available":
        parts.append(f"p.{page}")

    if parts:
        return ", ".join(parts)
    return meta.get("drug_name") or "Prescribing Information"


def extract_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    """Resolve every chunk reference in the answer (e.g. [chunk_1], [chunk_1, chunk_2]) to citation objects."""
    used_indices = sorted(
        set(int(m) for m in re.findall(r"chunk_(\d+)", answer_text, re.IGNORECASE))
    )
    citations = []
    for idx in used_indices:
        if idx < 0 or idx >= len(chunks):
            continue
        meta = chunks[idx]["metadata"]
        section_raw = meta.get("section") or ""
        citations.append({
            "chunk_ref": f"chunk_{idx}",
            "document": meta.get("drug_name", "Prescribing Information"),
            "section": _clean_section(section_raw),
            "page": meta.get("page_number") or "Not available",
            "source_label": _format_single_source(meta),
        })
    return citations


def replace_chunk_markers_with_sources(answer_text: str, chunks: list[dict]) -> str:
    """Replace all [chunk_N], [chunk_N, chunk_M], etc. with human-readable PDF source citations:
    e.g. '[§BOXED WARNING, p.1]' or '[§4 CONTRAINDICATIONS, p.2]'."""
    def _replacer(match: re.Match) -> str:
        content = match.group(0)
        indices = [int(x) for x in re.findall(r"chunk_(\d+)", content, re.IGNORECASE)]
        if not indices:
            return content

        sources = []
        seen = set()
        for idx in indices:
            if 0 <= idx < len(chunks):
                meta = chunks[idx].get("metadata", {})
                src = _format_single_source(meta)
                if src and src not in seen:
                    seen.add(src)
                    sources.append(src)

        if sources:
            return f" [{'; '.join(sources)}]"
        return ""

    # Match brackets containing chunk references
    pattern = re.compile(r"\[(?:[^\]]*\bchunk_\d+\b[^\]]*)\]", re.IGNORECASE)
    res = pattern.sub(_replacer, answer_text)
    # Clean up double spaces or space before punctuation
    res = re.sub(r"\s+([.,;:!?])", r"\1", res)
    res = re.sub(r" +", " ", res)
    return res.strip()


def strip_citation_markers(answer_text: str) -> str:
    """Strip all raw chunk markers and bracketed chunk citations from the text."""
    pattern = re.compile(r"\s*\[(?:[^\]]*\bchunk_\d+\b[^\]]*)\]", re.IGNORECASE)
    res = pattern.sub("", answer_text)
    res = re.sub(r"\s+([.,;:!?])", r"\1", res)
    res = re.sub(r" +", " ", res)
    return res.strip()


def format_citations_text(citations: list[dict]) -> str:
    """Produce a plain-text source block for embedding in the answer when needed."""
    if not citations:
        return ""
    lines = []
    seen = set()
    for c in citations:
        key = (c["document"], c["section"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        section_str = f"§{c['section']}" if c["section"] != "Not available" else "Section: Not available"
        page_str = f"p. {c['page']}" if c["page"] != "Not available" else "Page: Not available"
        lines.append(f"{c['document']} — {section_str}, {page_str}")
    return "\n".join(lines)
