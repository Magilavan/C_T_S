import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingestion.pipeline import ingest_pdf
from app.rag.chain import handle_chat_message
from app.auth.routes import router as auth_router
from app.auth.models import (
    init_db,
    SessionLocal,
    User,
    UserDocument,
    ChatSession,
    ChatMessage,
    AuditLog,
)
from app.auth.security import get_current_user, log_audit_event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Drug Information Chatbot")

# CORS — allow the frontend to call the API from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize auth database tables ──
init_db()

# ── Include auth API routes ──
app.include_router(auth_router)


@app.on_event("startup")
async def startup_event():
    from app.retrieval.keyword_index import ensure_bm25_index
    logging.info("Building/restoring BM25 index from Chroma database on startup...")
    ensure_bm25_index(force_rebuild=True)


# Serve the frontend static files
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    drug_name_hint: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    active_drug: str | None
    confidence: str
    scores: dict | None = None
    safety_notice: str | None = None
    question_category: str | None = None
    model_used: str | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = "New chat"


class FetchUrlRequest(BaseModel):
    url: str
    drug_name: str | None = None


from app.ingestion.domain_filter import PDFDomainValidationError
from app.ingestion.url_fetcher import fetch_pdf_from_url, PDFDownloadError

# ════════════════════════════════════════════════════════════════════════════
#  PDF DOCUMENT ACCESS CONTROL & ISOLATED STORAGE
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/documents/upload")
async def upload_document(
    request: Request,
    drug_name: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload and ingest a drug prescribing PDF strictly isolated to the authenticated user.
    - PDF file is stored in isolated directory: data/user_pdfs/{user_id}/{document_id}.pdf
    - Vector embeddings in Chroma are tagged with user_id metadata
    - Audit log records upload event
    """
    if not file.filename.lower().endswith(".pdf"):
        log_audit_event(
            action="PDF_UPLOAD_INVALID_FORMAT",
            resource_type="PDF_DOCUMENT",
            resource_id=file.filename,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="FAILED",
            details=f"Non-PDF file rejected: {file.filename}",
        )
        raise HTTPException(400, "Only PDF files are supported")

    document_id = str(uuid.uuid4())
    user_pdf_dir = Path(f"./data/user_pdfs/{current_user.id}")
    user_pdf_dir.mkdir(parents=True, exist_ok=True)
    permanent_pdf_path = user_pdf_dir / f"{document_id}.pdf"

    # Save uploaded file to user's isolated storage
    with open(permanent_pdf_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    file_size = permanent_pdf_path.stat().st_size

    try:
        result = ingest_pdf(
            str(permanent_pdf_path),
            drug_name=drug_name.strip().upper(),
            user_id=current_user.id,
            document_id=document_id,
        )

        # Store metadata in DB
        db = SessionLocal()
        try:
            doc_record = UserDocument(
                user_id=current_user.id,
                document_id=document_id,
                drug_name=drug_name.strip().upper(),
                filename=file.filename,
                file_path=str(permanent_pdf_path),
                file_size=file_size,
                page_count=result.get("page_count", 0),
                chunk_count=result.get("chunk_count", 0),
            )
            db.add(doc_record)
            db.commit()
        finally:
            db.close()

        log_audit_event(
            action="PDF_UPLOAD_SUCCESS",
            resource_type="PDF_DOCUMENT",
            resource_id=document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Uploaded {file.filename} for drug '{drug_name}' ({result.get('chunk_count', 0)} chunks)",
        )

        result["document_id"] = document_id
        return result

    except PDFDomainValidationError as e:
        logger.warning("PDF domain validation rejected upload: %s", e)
        # Clean up rejected file
        permanent_pdf_path.unlink(missing_ok=True)
        log_audit_event(
            action="PDF_UPLOAD_DOMAIN_REJECTED",
            resource_type="PDF_DOCUMENT",
            resource_id=document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="DENIED",
            details=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error ingesting PDF: %s", e)
        permanent_pdf_path.unlink(missing_ok=True)
        log_audit_event(
            action="PDF_UPLOAD_FAILED",
            resource_type="PDF_DOCUMENT",
            resource_id=document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="FAILED",
            details=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to process and ingest document: {str(e)}")


@app.post("/api/documents/fetch-url")
async def fetch_url_document(
    req: FetchUrlRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Download and ingest a prescribing information PDF directly from a URL (e.g. https://www.rxabbvie.com/pdf/rinvoq_pi.pdf).
    Enforces user isolation, PDF magic-byte validation, domain checks, and audit logging.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    document_id = f"doc-{uuid.uuid4().hex[:12]}"
    user_dir = USER_PDF_DIR / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    permanent_pdf_path = user_dir / f"{document_id}.pdf"

    try:
        temp_path, inferred_drug, file_size = fetch_pdf_from_url(url)
        # Move downloaded file to permanent isolated location
        import shutil
        shutil.move(temp_path, str(permanent_pdf_path))

        drug_name = (req.drug_name.strip().upper() if req.drug_name and req.drug_name.strip() else inferred_drug.upper())

        result = ingest_pdf(
            str(permanent_pdf_path),
            drug_name=drug_name,
            user_id=current_user.id,
            document_id=document_id,
        )

        # Store metadata in DB
        db = SessionLocal()
        try:
            filename = Path(urllib.parse.urlparse(url).path).name or f"{drug_name.lower()}_pi.pdf"
            doc_record = UserDocument(
                user_id=current_user.id,
                document_id=document_id,
                drug_name=drug_name,
                filename=filename,
                file_path=str(permanent_pdf_path),
                file_size=file_size,
                page_count=result.get("page_count", 0),
                chunk_count=result.get("chunk_count", 0),
            )
            db.add(doc_record)
            db.commit()
        finally:
            db.close()

        log_audit_event(
            action="PDF_URL_INGEST_SUCCESS",
            resource_type="PDF_DOCUMENT",
            resource_id=document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Fetched and ingested '{url}' for drug '{drug_name}' ({result.get('chunk_count', 0)} chunks)",
        )

        result["document_id"] = document_id
        return result

    except (PDFDownloadError, PDFDomainValidationError) as e:
        permanent_pdf_path.unlink(missing_ok=True)
        log_audit_event(
            action="PDF_URL_INGEST_REJECTED",
            resource_type="PDF_DOCUMENT",
            resource_id=document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="DENIED",
            details=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error ingesting PDF from URL %s: %s", url, e)
        permanent_pdf_path.unlink(missing_ok=True)
        log_audit_event(
            action="PDF_URL_INGEST_FAILED",
            resource_type="PDF_DOCUMENT",
            resource_id=document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="FAILED",
            details=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Failed to fetch and ingest document from URL: {str(e)}")


@app.get("/api/documents")
async def get_documents(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    List ONLY the documents and indexed drugs belonging to the authenticated user.
    Strictly isolated — no cross-user visibility.
    """
    db = SessionLocal()
    try:
        user_docs = db.query(UserDocument).filter(
            UserDocument.user_id == current_user.id,
            UserDocument.is_deleted == False,
        ).order_by(UserDocument.uploaded_at.desc()).all()

        from app.retrieval.vector_store import list_indexed_drugs
        drugs_stats = list_indexed_drugs(user_id=current_user.id)

        doc_list = []
        for d in user_docs:
            doc_list.append({
                "id": d.id,
                "document_id": d.document_id,
                "drug_name": d.drug_name,
                "filename": d.filename,
                "file_size": d.file_size,
                "page_count": d.page_count,
                "chunk_count": d.chunk_count,
                "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            })

        log_audit_event(
            action="PDF_LIST_ACCESSED",
            resource_type="PDF_CATALOG",
            resource_id=str(current_user.id),
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Retrieved {len(doc_list)} documents for user {current_user.id}",
        )

        return {
            "documents": drugs_stats,
            "user_documents": doc_list,
        }
    finally:
        db.close()


@app.get("/api/documents/{doc_id}/download")
async def download_document(
    doc_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Download or view a specific PDF.
    Server-side validation strictly enforces ownership — users can ONLY access their own PDFs.
    """
    db = SessionLocal()
    try:
        # Search by document_id or database id
        doc = db.query(UserDocument).filter(
            (UserDocument.document_id == doc_id) | (UserDocument.id == (int(doc_id) if doc_id.isdigit() else -1)),
            UserDocument.is_deleted == False,
        ).first()

        if not doc:
            log_audit_event(
                action="PDF_DOWNLOAD_NOT_FOUND",
                resource_type="PDF_DOCUMENT",
                resource_id=doc_id,
                user_id=current_user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                status="FAILED",
                details=f"Document {doc_id} not found",
            )
            raise HTTPException(status_code=404, detail="Document not found.")

        # CRITICAL OWNERSHIP CHECK: Ensure current user owns this document
        if doc.user_id != current_user.id:
            log_audit_event(
                action="PDF_CROSS_USER_ACCESS_BLOCKED",
                resource_type="PDF_DOCUMENT",
                resource_id=doc_id,
                user_id=current_user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                status="DENIED",
                details=f"User {current_user.id} attempted unauthorized access to document owned by user {doc.user_id}",
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied. You do not have permission to view or download this document.",
            )

        file_path = Path(doc.file_path)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="PDF physical file not found on disk.")

        log_audit_event(
            action="PDF_DOWNLOAD_SUCCESS",
            resource_type="PDF_DOCUMENT",
            resource_id=doc.document_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"User {current_user.id} downloaded {doc.filename}",
        )

        return FileResponse(
            path=str(file_path),
            filename=doc.filename,
            media_type="application/pdf",
        )
    finally:
        db.close()


@app.delete("/api/documents/{identifier}")
async def delete_document(
    identifier: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Delete an indexed drug document owned by the authenticated user.
    Server-side validation ensures only the owner can delete their documents.
    """
    db = SessionLocal()
    try:
        # Match by document_id or drug_name belonging to current_user
        docs = db.query(UserDocument).filter(
            UserDocument.user_id == current_user.id,
            UserDocument.is_deleted == False,
            (UserDocument.document_id == identifier) | (UserDocument.drug_name == identifier.upper()),
        ).all()

        if not docs:
            # Check if document exists under another user to log unauthorized delete attempt
            foreign_doc = db.query(UserDocument).filter(
                (UserDocument.document_id == identifier) | (UserDocument.drug_name == identifier.upper()),
                UserDocument.is_deleted == False,
            ).first()
            if foreign_doc and foreign_doc.user_id != current_user.id:
                log_audit_event(
                    action="PDF_CROSS_USER_DELETE_BLOCKED",
                    resource_type="PDF_DOCUMENT",
                    resource_id=identifier,
                    user_id=current_user.id,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    status="DENIED",
                    details=f"Unauthorized deletion attempt on document owned by user {foreign_doc.user_id}",
                )
                raise HTTPException(status_code=403, detail="Access denied. You do not own this document.")
            raise HTTPException(status_code=404, detail=f"No document found matching '{identifier}' for your account.")

        from app.retrieval.vector_store import delete_drug_documents
        total_chunks_deleted = 0

        for doc in docs:
            # Delete user's vector chunks
            chunks_deleted = delete_drug_documents(
                doc.drug_name,
                user_id=current_user.id,
                document_id=doc.document_id,
            )
            total_chunks_deleted += chunks_deleted

            # Mark deleted in DB
            doc.is_deleted = True

            # Delete physical file
            try:
                Path(doc.file_path).unlink(missing_ok=True)
            except Exception:
                pass

        db.commit()

        log_audit_event(
            action="PDF_DELETE_SUCCESS",
            resource_type="PDF_DOCUMENT",
            resource_id=identifier,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Deleted document '{identifier}' ({total_chunks_deleted} chunks removed)",
        )

        return {
            "status": "success",
            "identifier": identifier,
            "chunks_deleted": total_chunks_deleted,
            "message": f"Successfully deleted document and {total_chunks_deleted} chunks for '{identifier}'",
        }
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY & SESSION ACCESS CONTROL
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/chat/sessions")
async def get_chat_sessions(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve all conversation sessions belonging strictly to the authenticated user.
    """
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == current_user.id,
        ).order_by(ChatSession.updated_at.desc()).all()

        out = []
        for s in sessions:
            out.append({
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            })

        log_audit_event(
            action="CHAT_SESSIONS_LIST",
            resource_type="CHAT_SESSION",
            resource_id=str(current_user.id),
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"User {current_user.id} retrieved {len(out)} chat sessions",
        )

        return {"sessions": out}
    finally:
        db.close()


@app.post("/api/chat/sessions")
async def create_chat_session(
    req: CreateSessionRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Explicitly create a new conversation session for the authenticated user.
    """
    session_id = f"s-{uuid.uuid4().hex[:12]}"
    db = SessionLocal()
    try:
        session = ChatSession(
            id=session_id,
            user_id=current_user.id,
            title=req.title or "New chat",
        )
        db.add(session)
        db.commit()

        log_audit_event(
            action="CHAT_SESSION_CREATED",
            resource_type="CHAT_SESSION",
            resource_id=session_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Created session {session_id}",
        )

        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
        }
    finally:
        db.close()


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session_history(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve message history for a conversation session.
    Server-side validation strictly enforces that users can ONLY access their own chat history.
    """
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")

        # CRITICAL AUTHORIZATION CHECK
        if session.user_id != current_user.id:
            log_audit_event(
                action="CHAT_HISTORY_CROSS_USER_ACCESS_BLOCKED",
                resource_type="CHAT_SESSION",
                resource_id=session_id,
                user_id=current_user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                status="DENIED",
                details=f"User {current_user.id} attempted unauthorized access to session owned by user {session.user_id}",
            )
            raise HTTPException(status_code=403, detail="Access denied. You do not own this chat history.")

        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == current_user.id,
        ).order_by(ChatMessage.created_at.asc()).all()

        msg_list = []
        for m in messages:
            citations = []
            if m.citations:
                try:
                    citations = json.loads(m.citations)
                except Exception:
                    pass

            scores = None
            if m.scores:
                try:
                    scores = json.loads(m.scores)
                except Exception:
                    pass

            msg_list.append({
                "id": m.id,
                "role": m.role,
                "text": m.content,
                "citations": citations,
                "confidence": m.confidence,
                "scores": scores,
                "safety_notice": m.safety_notice,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })

        log_audit_event(
            action="CHAT_HISTORY_ACCESSED",
            resource_type="CHAT_SESSION",
            resource_id=session_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Retrieved {len(msg_list)} messages for session {session_id}",
        )

        return {
            "session": {
                "id": session.id,
                "title": session.title,
                "created_at": session.created_at.isoformat() if session.created_at else None,
            },
            "messages": msg_list,
        }
    finally:
        db.close()


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Delete a conversation session and all its messages.
    Server-side validation ensures only the session owner can delete it.
    """
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found.")

        if session.user_id != current_user.id:
            log_audit_event(
                action="CHAT_SESSION_DELETE_BLOCKED",
                resource_type="CHAT_SESSION",
                resource_id=session_id,
                user_id=current_user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                status="DENIED",
                details=f"Unauthorized delete attempt on session owned by user {session.user_id}",
            )
            raise HTTPException(status_code=403, detail="Access denied. You do not own this chat session.")

        db.delete(session)
        db.commit()

        log_audit_event(
            action="CHAT_SESSION_DELETED",
            resource_type="CHAT_SESSION",
            resource_id=session_id,
            user_id=current_user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="SUCCESS",
            details=f"Deleted session {session_id}",
        )

        return {"status": "success", "message": "Session deleted successfully."}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
#  CHAT INFERENCE ENDPOINT (AUTHENTICATED & ISOLATED)
# ════════════════════════════════════════════════════════════════════════════

def _classify_llm_error(exc: Exception) -> tuple[int, str]:
    """Map a raw Groq/LLM exception to a safe status code + user-facing message."""
    msg = str(exc).lower()
    if any(t in msg for t in ("401", "unauthorized", "invalid api key", "authentication")):
        return 503, "LLM backend unavailable: authentication error. Check GROQ_API_KEY."
    if any(t in msg for t in ("429", "rate limit", "rate_limit_exceeded")):
        return 503, "LLM backend is rate-limited right now. Please try again shortly."
    if any(t in msg for t in ("timeout", "timed out")):
        return 504, "LLM backend timed out. Please try again."
    if any(t in msg for t in ("502", "503", "504", "unavailable", "overloaded")):
        return 503, "LLM backend is temporarily unavailable. Please try again shortly."
    if any(t in msg for t in ("400", "invalid request", "model_not_found", "does not exist")):
        return 503, "LLM backend rejected the request (invalid model or request format)."
    return 503, "LLM backend unavailable due to an unexpected error."


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Process a chat message strictly isolated to the authenticated user.
    - Validates session ownership
    - Retrieves ONLY prescribing documents owned by current_user
    - Persists turn-by-turn history in database
    """
    db = SessionLocal()
    session = None
    try:
        # Verify or initialize session
        session = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
        if session:
            if session.user_id != current_user.id:
                log_audit_event(
                    action="CHAT_CROSS_USER_INFERENCE_BLOCKED",
                    resource_type="CHAT_SESSION",
                    resource_id=req.session_id,
                    user_id=current_user.id,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    status="DENIED",
                    details=f"User {current_user.id} attempted to send message to session owned by {session.user_id}",
                )
                raise HTTPException(status_code=403, detail="Access denied. You do not own this session.")
        else:
            # Create session tied to current user
            title = req.message[:50] + ("…" if len(req.message) > 50 else "")
            session = ChatSession(
                id=req.session_id,
                user_id=current_user.id,
                title=title or "New chat",
            )
            db.add(session)
            db.commit()

        # Save user message in DB
        user_msg = ChatMessage(
            session_id=req.session_id,
            user_id=current_user.id,
            role="user",
            content=req.message,
        )
        db.add(user_msg)
        db.commit()

    finally:
        db.close()

    try:
        result = handle_chat_message(
            req.session_id,
            req.message,
            req.drug_name_hint,
            user_id=current_user.id,
        )

        # Persist assistant response in DB
        db = SessionLocal()
        try:
            bot_msg = ChatMessage(
                session_id=req.session_id,
                user_id=current_user.id,
                role="bot",
                content=result.get("answer", ""),
                citations=json.dumps(result.get("citations", [])),
                confidence=result.get("confidence"),
                scores=json.dumps(result.get("scores")) if result.get("scores") else None,
                safety_notice=result.get("safety_notice"),
            )
            db.add(bot_msg)

            # Update session timestamp
            sess = db.query(ChatSession).filter(ChatSession.id == req.session_id).first()
            if sess:
                sess.updated_at = datetime.utcnow()

            db.commit()
        finally:
            db.close()

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Error handling chat request")
        status_code, detail = _classify_llm_error(exc)
        raise HTTPException(status_code=status_code, detail=detail)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    """Redirect root to the login page."""
    return RedirectResponse(url="/frontend/login.html")

