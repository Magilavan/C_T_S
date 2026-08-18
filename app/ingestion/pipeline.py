"""Runs the full ingestion pipeline for one PDF: extract -> chunk (incl.
tables) -> caption images (multimodal) -> embed -> index (vector + BM25)."""
import logging
import uuid

from app.ingestion.pdf_extractor import extract_pdf
from app.ingestion.chunker import chunk_page, Chunk
from app.ingestion.image_captioner import caption_image
from app.ingestion.domain_filter import validate_pdf_domain, PDFDomainValidationError
from app.core.privacy_logger import log_pdf_rejection
from app.retrieval.vector_store import upsert_chunks
from app.retrieval.keyword_index import build_index

logger = logging.getLogger(__name__)


def _chunk_to_record(chunk: Chunk, document_id: str, user_id: int | None = None) -> dict:
    """Convert chunk dataclass to vector storage record with isolated user_id metadata."""
    meta = {
        "document_id": document_id,
        "drug_name": chunk.drug_name,
        "section": chunk.section,
        "page_number": chunk.page_number,
        "is_table": chunk.is_table,
        "is_boxed_warning": chunk.is_boxed_warning,
    }
    if user_id is not None:
        meta["user_id"] = user_id

    return {
        "id": str(uuid.uuid4()),
        "text": chunk.text,
        "metadata": meta,
    }


def ingest_pdf(
    pdf_path: str,
    drug_name: str,
    caption_images: bool = True,
    user_id: int | None = None,
    document_id: str | None = None,
) -> dict:
    if not document_id:
        document_id = str(uuid.uuid4())
    logger.info(f"Starting ingestion: {pdf_path} (drug={drug_name}, doc_id={document_id}, user_id={user_id})")

    pages = extract_pdf(pdf_path)

    # Validate medical/pharmaceutical domain before processing
    try:
        validate_pdf_domain(pages, drug_name=drug_name)
    except PDFDomainValidationError as e:
        log_pdf_rejection(
            file_name=pdf_path,
            drug_name=drug_name,
            page_count=len(pages),
            rejection_reason=str(e),
        )
        raise

    all_chunks: list[Chunk] = []

    active_section = "PRESCRIBING INFORMATION"
    for page in pages:
        page_chunks, active_section = chunk_page(
            page.text,
            page.tables,
            page.page_number,
            drug_name,
            table_positions=page.table_positions,
            heading_positions=page.heading_positions,
            initial_section=active_section,
        )
        all_chunks.extend(page_chunks)

        # multimodal: caption each embedded image and add it as its own
        # searchable text chunk, tagged to the same page/section context
        if caption_images and page.images:
            for img in page.images:
                caption = caption_image(img)
                if caption:
                    all_chunks.append(Chunk(
                        text=f"[Image on page {page.page_number}] {caption}",
                        section=f"{active_section} (image)",
                        page_number=page.page_number,
                        drug_name=drug_name,
                    ))

    records = [_chunk_to_record(c, document_id, user_id=user_id) for c in all_chunks]

    from app.retrieval.keyword_index import ensure_bm25_index
    upsert_chunks(records)
    ensure_bm25_index(force_rebuild=True)

    logger.info(f"Ingestion complete: {len(records)} chunks indexed for {drug_name} (user_id={user_id})")
    return {
        "document_id": document_id,
        "user_id": user_id,
        "drug_name": drug_name,
        "page_count": len(pages),
        "chunk_count": len(records),
        "table_chunks": sum(1 for c in all_chunks if c.is_table),
        "image_chunks": sum(1 for c in all_chunks if "(image)" in c.section),
        "boxed_warning_chunks": sum(1 for c in all_chunks if c.is_boxed_warning),
    }

