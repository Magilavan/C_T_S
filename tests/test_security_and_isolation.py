"""Comprehensive automated tests for PDF Access Control, Chat History Isolation,
Server-side Ownership Enforcement, and Security Audit Logging.
"""

import io
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.auth.models import (
    engine,
    SessionLocal,
    init_db,
    User,
    UserDocument,
    ChatSession,
    ChatMessage,
    AuditLog,
)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def user_a_credentials():
    return {"email": "user_a@example.com", "password": "UserAPassword123!"}


@pytest.fixture
def user_b_credentials():
    return {"email": "user_b@example.com", "password": "UserBPassword123!"}


@pytest.fixture
def user_a_token(client, user_a_credentials):
    client.post("/api/auth/signup", json=user_a_credentials)
    res = client.post("/api/auth/login", json=user_a_credentials)
    return res.json()["token"]


@pytest.fixture
def user_b_token(client, user_b_credentials):
    client.post("/api/auth/signup", json=user_b_credentials)
    res = client.post("/api/auth/login", json=user_b_credentials)
    return res.json()["token"]


# ════════════════════════════════════════════════════════════════════════════
# 1. PDF ACCESS CONTROL & ISOLATION TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestPdfAccessControlAndIsolation:
    """Verifies strict per-user PDF isolation, ownership enforcement, and cross-user blocking."""

    @patch("app.main.ingest_pdf")
    def test_pdf_upload_and_cross_user_isolation(
        self, mock_ingest, client, user_a_token, user_b_token
    ):
        mock_ingest.return_value = {
            "chunk_count": 8,
            "page_count": 3,
            "drug_name": "LIPITOR",
            "file_hash": "dummy_hash_123",
        }

        # User A uploads Lipitor PDF
        fake_pdf = io.BytesIO(b"%PDF-1.4 simulated pdf document data for Lipitor")
        res_upload = client.post(
            "/api/documents/upload",
            headers={"Authorization": f"Bearer {user_a_token}"},
            data={"drug_name": "Lipitor"},
            files={"file": ("lipitor_label.pdf", fake_pdf, "application/pdf")},
        )
        assert res_upload.status_code == 200
        upload_data = res_upload.json()
        doc_id = upload_data["document_id"]
        assert doc_id is not None
        assert upload_data["drug_name"] == "LIPITOR"

        # ── Check User A can see their own document in GET /api/documents ──
        res_a_docs = client.get(
            "/api/documents",
            headers={"Authorization": f"Bearer {user_a_token}"},
        )
        assert res_a_docs.status_code == 200
        docs_a = res_a_docs.json()["user_documents"]
        assert any(d["document_id"] == doc_id for d in docs_a)

        # ── Verify User B CANNOT see User A's document in GET /api/documents ──
        res_b_docs = client.get(
            "/api/documents",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )
        assert res_b_docs.status_code == 200
        docs_b = res_b_docs.json()["user_documents"]
        assert not any(d["document_id"] == doc_id for d in docs_b)

        # ── Verify User A CAN download their own document ──
        res_a_dl = client.get(
            f"/api/documents/{doc_id}/download",
            headers={"Authorization": f"Bearer {user_a_token}"},
        )
        assert res_a_dl.status_code == 200

        # ── Verify User B receives 403 Forbidden on attempting to download User A's document ──
        res_b_dl = client.get(
            f"/api/documents/{doc_id}/download",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )
        assert res_b_dl.status_code == 403
        assert "Forbidden" in res_b_dl.json()["detail"] or "Access denied" in res_b_dl.json()["detail"]

        # ── Verify User B receives 403 Forbidden on attempting to delete User A's document ──
        res_b_del = client.delete(
            f"/api/documents/{doc_id}",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )
        assert res_b_del.status_code == 403

        # ── Verify User A CAN delete their own document ──
        with patch("app.retrieval.vector_store.delete_drug_documents", return_value=8):
            res_a_del = client.delete(
                f"/api/documents/{doc_id}",
                headers={"Authorization": f"Bearer {user_a_token}"},
            )
            assert res_a_del.status_code == 200


# ════════════════════════════════════════════════════════════════════════════
# 2. CHAT HISTORY ACCESS CONTROL & SESSION ISOLATION TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestChatHistoryAccessControl:
    """Verifies that chat sessions and turns are strictly visible only to their authenticated owner."""

    def test_session_isolation_and_cross_user_protection(
        self, client, user_a_token, user_b_token
    ):
        # ── User A creates a chat session ──
        res_create = client.post(
            "/api/chat/sessions",
            headers={"Authorization": f"Bearer {user_a_token}"},
            json={"title": "Lipitor Dosage Inquiry"},
        )
        assert res_create.status_code == 200
        session_a_id = res_create.json()["id"]

        # ── User A posts a message in their session ──
        with patch("app.main.handle_chat_message") as mock_chat:
            mock_chat.return_value = {
                "answer": "Lipitor standard starting dose is 10-20 mg daily.",
                "citations": [{"document": "Lipitor", "section": "Dosage", "page": 2}],
                "confidence": "high",
                "question_category": "medical_qa",
                "active_drug": "LIPITOR",
            }
            res_msg = client.post(
                "/api/chat",
                headers={"Authorization": f"Bearer {user_a_token}"},
                json={"session_id": session_a_id, "message": "What is the recommended dose?"},
            )
            assert res_msg.status_code == 200

        # ── User A lists their sessions ──
        res_a_list = client.get(
            "/api/chat/sessions",
            headers={"Authorization": f"Bearer {user_a_token}"},
        )
        assert res_a_list.status_code == 200
        assert any(s["id"] == session_a_id for s in res_a_list.json()["sessions"])

        # ── User B lists their sessions — User A's session MUST NOT appear ──
        res_b_list = client.get(
            "/api/chat/sessions",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )
        assert res_b_list.status_code == 200
        assert not any(s["id"] == session_a_id for s in res_b_list.json()["sessions"])

        # ── User B attempts to access User A's session details -> 403 Forbidden ──
        res_b_get = client.get(
            f"/api/chat/sessions/{session_a_id}",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )
        assert res_b_get.status_code == 403

        # ── User B attempts to send a message into User A's session -> 403 Forbidden ──
        res_b_send = client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {user_b_token}"},
            json={"session_id": session_a_id, "message": "Malicious attempt to write to User A's session"},
        )
        assert res_b_send.status_code == 403

        # ── User B attempts to delete User A's session -> 403 Forbidden ──
        res_b_del = client.delete(
            f"/api/chat/sessions/{session_a_id}",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )
        assert res_b_del.status_code == 403

        # ── User A CAN delete their own session ──
        res_a_del = client.delete(
            f"/api/chat/sessions/{session_a_id}",
            headers={"Authorization": f"Bearer {user_a_token}"},
        )
        assert res_a_del.status_code == 200


# ════════════════════════════════════════════════════════════════════════════
# 3. SECURITY AUDIT LOGGING TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestAuditLogging:
    """Verifies that security events, downloads, and unauthorized access attempts are logged."""

    def test_audit_log_records_events(self, client, user_a_token, user_b_token):
        # 1. Create a session as User A
        res_create = client.post(
            "/api/chat/sessions",
            headers={"Authorization": f"Bearer {user_a_token}"},
            json={"title": "Audit Test Session"},
        )
        session_id = res_create.json()["id"]

        # 2. Make unauthorized attempt as User B on User A's session
        client.get(
            f"/api/chat/sessions/{session_id}",
            headers={"Authorization": f"Bearer {user_b_token}"},
        )

        # 3. Check DB audit table
        db = SessionLocal()
        try:
            logs = db.query(AuditLog).all()
            assert len(logs) > 0

            actions = [log.action for log in logs]
            assert "CHAT_SESSION_CREATED" in actions
            assert "CHAT_HISTORY_CROSS_USER_ACCESS_BLOCKED" in actions

            # Check that DENIED status was recorded
            denied_logs = [log for log in logs if log.status in ["DENIED", "FORBIDDEN"]]
            assert len(denied_logs) >= 1
            for flog in denied_logs:
                assert flog.action in [
                    "PDF_DOWNLOAD_FORBIDDEN",
                    "PDF_CROSS_USER_ACCESS_BLOCKED",
                    "PDF_CROSS_USER_DELETE_BLOCKED",
                    "PDF_UPLOAD_DOMAIN_REJECTED",
                    "PDF_URL_INGEST_REJECTED",
                    "PDF_URL_INGEST_FAILED",
                    "CHAT_HISTORY_CROSS_USER_ACCESS_BLOCKED",
                    "CHAT_CROSS_USER_INFERENCE_BLOCKED",
                    "CHAT_SESSION_DELETE_BLOCKED",
                    "CHAT_QUERY_FORBIDDEN",
                    "CHAT_SESSION_DELETE_FORBIDDEN",
                    "UNAUTHORIZED_ACCESS_ATTEMPT",
                ]
        finally:
            db.close()
