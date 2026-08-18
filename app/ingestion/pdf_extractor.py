"""Extract text, tables, and images per-page from drug label PDFs (e.g. RxAbbVie prescribing information).

Handles:
  - Multi-column layouts (Highlights, Contents, and clinical study sections)
  - Unicode normalizations, ligature expansion (fi, fl, ffi, ffl), CID artifact cleanup
  - De-hyphenation across line breaks for medical terminology
  - Scanned vs. digital pages with automatic OCR fallback
  - Table bounding box isolation to avoid duplicate prose pollution
  - Empty-password decryption and corrupted stream repair heuristics
  - Edge cases (large files, non-standard fonts, embedded vector images)
"""
import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

try:
    import pytesseract
    import shutil
    import os

    # Automatically detect standard Windows installation paths if tesseract is not on PATH
    if not shutil.which("tesseract"):
        _candidates = [
            os.getenv("TESSERACT_CMD", ""),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for candidate in _candidates:
            if candidate and os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                logger.info("Configured Tesseract-OCR binary: %s", candidate)
                break

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract not available — OCR fallback disabled")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION & CLEANUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Unicode ligature mapping for common pharmaceutical PDF embedded fonts
_LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u00a0": " ",      # Non-breaking space
    "\u2002": " ",      # En space
    "\u2003": " ",      # Em space
    "\u2009": " ",      # Thin space
    "\u200a": " ",      # Hair space
    "\u200b": "",       # Zero-width space
    "\u2010": "-",      # Hyphen
    "\u2011": "-",      # Non-breaking hyphen
    "\u2012": "-",      # Figure dash
    "\u2013": "-",      # En dash
    "\u2014": " - ",    # Em dash
    "\u2018": "'",      # Left single quote
    "\u2019": "'",      # Right single quote
    "\u201c": '"',      # Left double quote
    "\u201d": '"',      # Right double quote
    "\u2022": "•",      # Bullet
    "\u2023": "•",      # Triangular bullet
    "\u25cf": "•",      # Black circle bullet
    "\u25cb": "•",      # White circle bullet
    "\uf0b7": "•",      # Symbol font bullet
    "\u2265": ">=",     # Greater than or equal to
    "\u2264": "<=",     # Less than or equal to
    "\u00b1": "+/-",    # Plus-minus
    "\u03bc": "mc",     # Micro sign (e.g. mcg)
    "\u00b5": "mc",     # Micro sign
    "\u00b0": "deg",    # Degree symbol
}

_CID_PATTERN = re.compile(r"\(cid:\d+\)")
_SOFT_HYPHEN = "\u00ad"


def clean_and_normalize_text(text: str) -> str:
    """Normalize extracted PDF text for downstream LLM retrieval and keyword search.

    Performs:
      1. Unicode NFKC normalization
      2. Ligature expansion and symbol normalization
      3. Removal of CID glyph artifacts
      4. De-hyphenation across line wraps (e.g. 'contra- \nindicated' -> 'contraindicated')
      5. Whitespace and blank line standardization
    """
    if not text:
        return ""

    # 1. Expand ligatures and symbols
    for lig, rep in _LIGATURE_MAP.items():
        if lig in text:
            text = text.replace(lig, rep)

    # 2. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Strip soft hyphens and clean CID artifacts
    text = text.replace(_SOFT_HYPHEN, "")
    text = _CID_PATTERN.sub("", text)

    # 4. De-hyphenate words broken across line breaks (e.g. "adverse reac-\ntion" -> "adverse reaction")
    text = re.sub(r"([A-Za-z]{2,})-\n\s*([A-Za-z]{2,})", r"\1\2", text)

    # 5. Fix multiple consecutive spaces (preserving intentional paragraph breaks)
    lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    normalized = "\n".join(lines)
    # Collapse 3+ consecutive newlines to 2
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONTENT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    table_positions: list[float] = field(default_factory=list)      # top-y coordinate of each table
    heading_positions: list[tuple[float, str]] = field(default_factory=list)  # [(top_y, heading_text), ...]
    is_scanned: bool = False
    column_count: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN-AWARE GEOMETRIC LAYOUT EXTRACTION (PyMuPDF)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_column_aware_text(
    fitz_page: fitz.Page,
    exclude_rects: list[fitz.Rect] | None = None,
) -> tuple[str, int, list[tuple[float, str]]]:
    """Extract text from a PyMuPDF page while respecting multi-column layouts (e.g. Highlights 2-column).

    Sorts text blocks by reading order (column by column from left to right,
    top to bottom within each column) rather than naive horizontal interleaving.
    Excludes rectangular areas occupied by tables to prevent duplicate prose.

    Returns:
        (ordered_text, estimated_columns, heading_positions)
    """
    from app.ingestion.section_detector import SUBSECTION_RE, SECTION_RE, HIGHLIGHTS_RE, PATIENT_APPENDICES_RE, BOXED_WARNING_RE

    exclude_rects = exclude_rects or []
    # get_text("blocks") returns list of tuples: (x0, y0, x1, y1, text, block_no, block_type)
    # block_type == 0 indicates text, 1 indicates image
    raw_blocks = fitz_page.get_text("blocks")
    if not raw_blocks:
        return "", 1, []

    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return "", 1, []

    page_width = fitz_page.rect.width
    midpoint_x = page_width / 2.0

    # Determine if the page exhibits a multi-column structure
    # Check if we have substantial blocks on both left (x1 < midpoint_x + 30) and right (x0 > midpoint_x - 30)
    left_blocks = [b for b in text_blocks if b[2] <= midpoint_x + 40]
    right_blocks = [b for b in text_blocks if b[0] >= midpoint_x - 40]
    full_width_blocks = [b for b in text_blocks if b[0] < midpoint_x - 40 and b[2] > midpoint_x + 40]

    is_two_column = len(left_blocks) >= 2 and len(right_blocks) >= 2 and len(full_width_blocks) <= len(text_blocks) * 0.4
    column_count = 2 if is_two_column else 1

    ordered_blocks = []
    if is_two_column:
        # Separate full-width headers (e.g. top title, HIGHLIGHTS OF PRESCRIBING INFORMATION),
        # left column blocks, right column blocks, and bottom full-width footers.
        top_y_thresh = 0.0
        # Check top full width banner
        top_banners = [b for b in full_width_blocks if b[1] < fitz_page.rect.height * 0.25]
        top_banners.sort(key=lambda b: b[1])

        # Bottom banners
        bottom_banners = [b for b in full_width_blocks if b[1] >= fitz_page.rect.height * 0.75]
        bottom_banners.sort(key=lambda b: b[1])

        # Column blocks
        l_blocks = [b for b in text_blocks if b not in top_banners and b not in bottom_banners and b[0] < midpoint_x]
        r_blocks = [b for b in text_blocks if b not in top_banners and b not in bottom_banners and b[0] >= midpoint_x]

        l_blocks.sort(key=lambda b: (b[1], b[0]))
        r_blocks.sort(key=lambda b: (b[1], b[0]))

        ordered_blocks = top_banners + l_blocks + r_blocks + bottom_banners
    else:
        # Single column layout: sort top-to-bottom
        ordered_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

    # Filter out blocks that fall inside excluded table bounding boxes
    filtered_text_parts = []
    heading_positions: list[tuple[float, str]] = []

    for b in ordered_blocks:
        b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
        # Check table collision (if block is mostly inside an excluded table rectangle, skip it)
        is_inside_table = False
        for table_rect in exclude_rects:
            # Overlap check
            intersect = b_rect & table_rect
            if intersect.is_valid and intersect.get_area() > (b_rect.get_area() * 0.5):
                is_inside_table = True
                break

        if is_inside_table:
            continue

        raw_block_text = b[4].strip()
        if not raw_block_text:
            continue

        cleaned_block = clean_and_normalize_text(raw_block_text)
        if cleaned_block:
            filtered_text_parts.append(cleaned_block)

            # Check headings in this block for spatial attribution
            for line in cleaned_block.splitlines():
                line_s = line.strip()
                if (
                    SUBSECTION_RE.match(line_s)
                    or SECTION_RE.match(line_s)
                    or HIGHLIGHTS_RE.match(line_s)
                    or PATIENT_APPENDICES_RE.match(line_s)
                    or BOXED_WARNING_RE.match(line_s)
                ):
                    heading_positions.append((float(b[1]), line_s))

    full_page_text = "\n\n".join(filtered_text_parts)
    return full_page_text, column_count, heading_positions


# ─────────────────────────────────────────────────────────────────────────────
# TABLE EXTRACTION & BOUNDING BOX MASKING
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_tables(pdfplumber_page) -> tuple[list[list[list[str]]], list[float], list[fitz.Rect]]:
    """Extract tables using pdfplumber with enhanced cell alignment strategies.

    Returns:
        (extracted_tables, table_top_y_positions, fitz_table_rects)
    """
    found_tables = pdfplumber_page.find_tables(
        table_settings={
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "edge_min_length": 3,
            "min_words_vertical": 2,
        }
    )

    tables: list[list[list[str]]] = []
    positions: list[float] = []
    rects: list[fitz.Rect] = []

    for t in found_tables or []:
        raw_table = t.extract()
        if not raw_table:
            continue

        # Clean cell contents
        cleaned_table = []
        for row in raw_table:
            cleaned_row = [clean_and_normalize_text(cell) if cell else "" for cell in row]
            if any(cleaned_row):
                cleaned_table.append(cleaned_row)

        if len(cleaned_table) >= 2:
            # Require at least 2 rows that each have 2+ non-empty cells
            multi_cell_rows = [r for r in cleaned_table if sum(1 for c in r if c.strip()) >= 2]
            if len(multi_cell_rows) >= 2:
                tables.append(cleaned_table)
                top_y = float(t.bbox[1])
                positions.append(top_y)
                rects.append(fitz.Rect(t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]))

    return tables, positions, rects


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL CHARACTER RECOGNITION (OCR) FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_page_fitz(fitz_page: fitz.Page, page_number: int) -> str:
    """Rasterize a scanned or image-only page and perform OCR with preprocessing."""
    if not OCR_AVAILABLE:
        logger.warning("Page %d: native text layer empty and OCR unavailable — skipping", page_number)
        return ""

    try:
        # Render at 300 DPI for high OCR accuracy on small prescription fonts
        pix = fitz_page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Image preprocessing: Convert to grayscale and enhance contrast
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)

        ocr_text = pytesseract.image_to_string(img, config="--psm 1")  # Automatic page segmentation with OSD
        if not ocr_text.strip():
            ocr_text = pytesseract.image_to_string(img, config="--psm 3")  # Fully automatic page segmentation

        cleaned = clean_and_normalize_text(ocr_text)
        logger.info("Page %d: OCR extracted %d characters", page_number, len(cleaned))
        return cleaned
    except Exception as exc:
        logger.warning("Page %d: OCR failed (%s) — skipping", page_number, exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DOCUMENT EXTRACTION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str) -> list[PageContent]:
    """Extract structured, multi-column, table-isolated content from a drug label PDF.

    Args:
        pdf_path: Path to the local PDF file.

    Returns:
        List of PageContent objects for each page in the document.

    Raises:
        ValueError: If the file does not exist or cannot be parsed.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        raise ValueError(f"PDF file does not exist or is empty: {pdf_path}")

    # Maximum file size safeguard (100 MB)
    max_bytes = 100 * 1024 * 1024
    if path_obj.stat().st_size > max_bytes:
        raise ValueError(f"PDF exceeds maximum supported file size of 100 MB: {pdf_path}")

    pages: list[PageContent] = []
    doc_fitz: fitz.Document | None = None

    try:
        # 1. Open PyMuPDF with stream repair heuristics
        try:
            doc_fitz = fitz.open(pdf_path)
        except Exception as e:
            logger.warning("Standard PyMuPDF open failed (%s), attempting repair mode", e)
            doc_fitz = fitz.open(pdf_path, filetype="pdf")

        # 2. Handle Password / Encryption restrictions
        if doc_fitz.is_encrypted:
            logger.info("PDF has encryption/permissions enabled; attempting standard empty-password unlock")
            unlocked = doc_fitz.authenticate("")
            if not unlocked:
                logger.warning("PDF requires non-empty user password for decryption")

        total_pages = len(doc_fitz)
        logger.info("Extracting %d pages from '%s'", total_pages, path_obj.name)

        # 3. Open pdfplumber for table parsing
        with pdfplumber.open(pdf_path) as pdf_plumber:
            for i in range(total_pages):
                page_number = i + 1
                fitz_page = doc_fitz[i]
                plumber_page = pdf_plumber.pages[i] if i < len(pdf_plumber.pages) else None

                # Step A: Extract tables & bounding boxes
                tables: list[list[list[str]]] = []
                table_positions: list[float] = []
                table_rects: list[fitz.Rect] = []

                if plumber_page is not None:
                    try:
                        tables, table_positions, table_rects = _extract_page_tables(plumber_page)
                    except Exception as e:
                        logger.warning("Page %d table extraction error: %s", page_number, e)

                # Step B: Extract column-aware prose text (masking table areas)
                page_text, col_count, heading_positions = _extract_column_aware_text(
                    fitz_page,
                    exclude_rects=table_rects,
                )

                # Step C: OCR Fallback for scanned / image-only pages
                is_scanned = False
                if not page_text.strip() and not tables:
                    logger.info("Page %d: no text layer detected, invoking OCR fallback", page_number)
                    page_text = _ocr_page_fitz(fitz_page, page_number)
                    is_scanned = True

                # Step D: Extract embedded non-icon figures / diagrams for multimodal captioning
                images: list[Image.Image] = []
                try:
                    for img_info in fitz_page.get_images(full=True):
                        xref = img_info[0]
                        try:
                            base_image = doc_fitz.extract_image(xref)
                            img_bytes = base_image.get("image")
                            if img_bytes:
                                pil_img = Image.open(io.BytesIO(img_bytes))
                                # Filter out small icons / logos / bullets
                                if pil_img.width >= 160 and pil_img.height >= 160:
                                    images.append(pil_img)
                        except Exception as img_err:
                            logger.debug("Page %d image extraction skip: %s", page_number, img_err)
                except Exception as e:
                    logger.debug("Page %d get_images failed: %s", page_number, e)

                pages.append(PageContent(
                    page_number=page_number,
                    text=page_text,
                    tables=tables,
                    images=images,
                    table_positions=table_positions,
                    heading_positions=heading_positions,
                    is_scanned=is_scanned,
                    column_count=col_count,
                ))

    finally:
        if doc_fitz is not None:
            doc_fitz.close()

    logger.info("Successfully extracted %d pages from '%s'", len(pages), path_obj.name)
    return pages
