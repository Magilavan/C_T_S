"""
Security utilities for DrugBot authentication.
- Password hashing (bcrypt)
- Email hashing (SHA-256)
- JWT token creation & verification
"""

import hashlib
import os
from datetime import datetime, timedelta

import bcrypt
import jwt

from app.core.config import settings

# ── Configuration ──
JWT_SECRET = settings.jwt_secret_key
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


# ════════════════════════════════════════════════════════════════════════════
#  PASSWORD HASHING (bcrypt)
# ════════════════════════════════════════════════════════════════════════════

def hash_password(plain_password: str) -> str:
    """
    Hash a plain-text password using bcrypt with 12 salt rounds.
    Returns the hashed string (includes salt automatically).
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against a bcrypt hash.
    Returns True if they match, False otherwise.
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ════════════════════════════════════════════════════════════════════════════
#  EMAIL HASHING (SHA-256)
# ════════════════════════════════════════════════════════════════════════════

def hash_email(email: str) -> str:
    """
    Create a SHA-256 hash of the normalized (lowercased, stripped) email.
    Used for lookups and duplicate detection — not reversible.
    """
    normalized = email.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
#  JWT TOKENS
# ════════════════════════════════════════════════════════════════════════════

def create_access_token(user_id: int, email: str) -> str:
    """
    Create a signed JWT token with the user's id and email.
    Expires after JWT_EXPIRY_HOURS.
    """
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_access_token(token: str) -> dict | None:
    """
    Verify and decode a JWT token.
    Returns the payload dict on success, None on failure (expired / invalid).
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ════════════════════════════════════════════════════════════════════════════
#  FASTAPI DEPENDENCIES & AUTH ENFORCEMENT
# ════════════════════════════════════════════════════════════════════════════

from fastapi import Depends, HTTPException, Request
from app.auth.models import SessionLocal, User, AuditLog


def get_current_user(request: Request) -> User:
    """
    FastAPI dependency to extract and verify the JWT bearer token from the
    Authorization header, ensuring the user exists and is currently active.
    Raises HTTPException(401) on missing/invalid token or deactivated user.
    """
    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    elif "token" in request.query_params:
        # Fallback for file downloads/viewing via browser tabs
        token = request.query_params["token"]

    if not token:
        # Log unauthorized attempt
        log_audit_event(
            action="UNAUTHORIZED_REQUEST",
            resource_type="API_ENDPOINT",
            resource_id=str(request.url.path),
            user_id=None,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="DENIED",
            details="Missing Bearer token in request header",
        )
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_access_token(token)
    if not payload or "sub" not in payload:
        log_audit_event(
            action="INVALID_TOKEN_ATTEMPT",
            resource_type="API_ENDPOINT",
            resource_id=str(request.url.path),
            user_id=None,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="DENIED",
            details="Expired or signature-mismatched JWT token",
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(payload["sub"])
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Malformed token subject.")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            log_audit_event(
                action="INACTIVE_OR_DELETED_USER_ACCESS",
                resource_type="API_ENDPOINT",
                resource_id=str(request.url.path),
                user_id=user_id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                status="DENIED",
                details="User record not found or account deactivated",
            )
            raise HTTPException(status_code=401, detail="Account is deactivated or not found.")
        return user
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
#  AUDIT LOGGING
# ════════════════════════════════════════════════════════════════════════════

import logging
audit_logger = logging.getLogger("drugbot.audit")


def log_audit_event(
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    user_id: int | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    status: str = "SUCCESS",
    details: str | None = None,
):
    """
    Record an audit log entry in the SQLite database and output to the audit log stream.
    Used for tracking PDF access/upload/download events, authentication, and authorization checks.
    """
    audit_logger.info(
        "AUDIT_EVENT: action=%s status=%s user_id=%s resource_type=%s resource_id=%s ip=%s details=%s",
        action, status, user_id, resource_type, resource_id, ip_address, details
    )

    db = SessionLocal()
    try:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent[:255] if user_agent else None,
            status=status,
            details=details,
        )
        db.add(entry)
        db.commit()
    except Exception as exc:
        audit_logger.error("Failed to write audit log to database: %s", exc)
    finally:
        db.close()

