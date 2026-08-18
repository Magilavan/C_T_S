"""
SQLAlchemy models for authentication.
- User: stores hashed email (for lookup), encrypted email, and bcrypt-hashed password.
- LoginAttempt: tracks failed login attempts per user for rate limiting.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Database setup ──
_db_path = os.getenv("SQLITE_DB_PATH", "./data/app.db")
# Ensure the data directory exists
os.makedirs(os.path.dirname(os.path.abspath(_db_path)), exist_ok=True)

engine = create_engine(f"sqlite:///{_db_path}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """
    Stores user credentials.
    - email_hash: SHA-256 hash of the normalized email (used for lookups & dedup)
    - email_display: the original email in plain text (for display / recovery)
    - password_hash: bcrypt-hashed password
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email_hash = Column(String(64), unique=True, nullable=False, index=True)
    email_display = Column(String(255), nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    documents = relationship("UserDocument", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class LoginAttempt(Base):
    """
    Tracks failed login attempts per email for rate limiting.
    - attempt_count resets on successful login or after the lockout window expires.
    """
    __tablename__ = "login_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email_hash = Column(String(64), unique=True, nullable=False, index=True)
    attempt_count = Column(Integer, default=0)
    first_attempt_at = Column(DateTime, default=datetime.utcnow)
    locked_until = Column(DateTime, nullable=True)


class UserDocument(Base):
    """
    Tracks uploaded drug PDFs per user for strict multi-user isolation.
    - user_id: Foreign key to User, enforcing strict ownership
    - document_id: Unique UUID for vector chunk tagging & file naming
    - file_path: Isolated disk storage path under data/user_pdfs/{user_id}/
    """
    __tablename__ = "user_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(String(64), unique=True, nullable=False, index=True)
    drug_name = Column(String(128), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)

    user = relationship("User", back_populates="documents")


class ChatSession(Base):
    """
    Tracks conversational chat sessions tied strictly to a user account.
    """
    __tablename__ = "chat_sessions"

    id = Column(String(64), primary_key=True)  # Session UUID / slug (e.g. s-xxxx)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """
    Stores turn-by-turn chat history messages isolated by session and user.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" | "bot" | "assistant"
    content = Column(Text, nullable=False)
    citations = Column(Text, nullable=True)  # JSON-encoded array of citation objects
    confidence = Column(String(64), nullable=True)
    scores = Column(Text, nullable=True)  # JSON-encoded dict of grounding/retrieval scores
    safety_notice = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class AuditLog(Base):
    """
    Audit logging for security-sensitive events:
    - PDF uploads, PDF views/downloads, PDF deletions, and unauthorized access attempts.
    - Chat history access and authorization checks.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)  # null if unauthenticated
    action = Column(String(64), nullable=False, index=True)
    resource_type = Column(String(64), nullable=False)  # e.g., "PDF_DOCUMENT", "CHAT_SESSION"
    resource_id = Column(String(128), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False)  # "SUCCESS", "DENIED", "FAILED"
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

