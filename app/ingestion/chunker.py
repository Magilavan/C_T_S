"""Section-aware chunking. Rules (see technical design doc, section 5):
  - split on section/subsection boundaries, never mid-sentence within a rule
  - keep tables intact as their own chunk
  - cap chunk size, but never split a table
  - each chunk carries the section heading + page number for citation
"""
from dataclasses import dataclass, field
from app.ingestion.section_detector import detect_sections, current_section_for_offset

MAX_CHUNK_CHARS = 3200  # ~700-800 tokens


@dataclass
class Chunk:
    text: str
    section: str
    page_number: int
    drug_name: str
    is_table: bool = False
    is_boxed_warning: bool = False


def _split_long_section(text: str, max_chars: int) -> list[str]:
    """Split an over-long section at paragraph boundaries, prefixing each
    sub-chunk with nothing extra — the caller attaches the section heading
    to every sub-chunk so each remains independently answerable."""
    if len(text) <= max_chars:
        return [text]
    paras = text.split("\n\n")
    out, buf = [], ""
    for para in paras:
        if len(buf) + len(para) + 2 > max_chars and buf:
            out.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf.strip())
    return out


def _section_for_table(table_top_y: float, heading_positions: list[tuple[float, str]], fallback: str = "UNSPECIFIED") -> str:
    """Picks the nearest heading whose vertical position is ABOVE the table
    on the page — i.e. the section the table visually sits under. Falls
    back to default section if no heading precedes it."""
    current = fallback
    for top_y, heading_text in heading_positions:
        if top_y <= table_top_y:
            current = heading_text
        else:
            break
    return current


def chunk_page(
    page_text: str,
    page_tables: list,
    page_number: int,
    drug_name: str,
    table_positions: list[float] | None = None,
    heading_positions: list[tuple[float, str]] | None = None,
    initial_section: str = "UNSPECIFIED",
) -> tuple[list[Chunk], str]:
    chunks = []
    markers = detect_sections(page_text)
    table_positions = table_positions or []
    heading_positions = heading_positions or []

    active_section = initial_section

    # 1. table chunks — kept intact, never merged with surrounding prose.
    for i, table in enumerate(page_tables):
        rendered = "\n".join(" | ".join(cell or "" for cell in row) for row in table if any(cell for cell in row))
        if not rendered.strip():
            continue
        if i < len(table_positions) and heading_positions:
            sec = _section_for_table(table_positions[i], heading_positions, fallback=active_section)
        else:
            sec = markers[-1].heading if markers else active_section
        chunks.append(Chunk(text=rendered, section=sec, page_number=page_number,
                             drug_name=drug_name, is_table=True))

    if not page_text.strip():
        return chunks, active_section

    if not markers:
        # Whole page belongs to the section active from previous page
        for piece in _split_long_section(page_text, MAX_CHUNK_CHARS):
            chunks.append(Chunk(text=piece, section=active_section, page_number=page_number, drug_name=drug_name))
        return chunks, active_section

    # 2. split page text at each section boundary
    bounds = [m.start_offset for m in markers] + [len(page_text)]

    # 3. any text before the first marker on this page belongs to initial_section
    lead = page_text[:bounds[0]].strip()
    if lead:
        for piece in _split_long_section(lead, MAX_CHUNK_CHARS):
            chunks.insert(0, Chunk(text=piece, section=active_section, page_number=page_number, drug_name=drug_name))

    for i, marker in enumerate(markers):
        active_section = marker.heading
        segment = page_text[bounds[i]:bounds[i + 1]].strip()
        if not segment:
            continue
        for piece in _split_long_section(segment, MAX_CHUNK_CHARS):
            chunks.append(Chunk(
                text=piece,
                section=marker.heading,
                page_number=page_number,
                drug_name=drug_name,
                is_boxed_warning=marker.is_boxed_warning,
            ))

    return chunks, active_section
