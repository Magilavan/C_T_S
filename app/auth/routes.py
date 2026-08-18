"""
Authentication API routes for DrugBot.
- POST /api/auth/signup  — Register a new user
- POST /api/auth/login   — Login with rate limiting
- GET  /api/auth/me      — Get current user info (requires valid JWT)
"""

import re
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr

from app.auth.models import SessionLocal, User, LoginAttempt
from app.auth.security import (
    hash_password,
    verify_password,
    hash_email,
    create_access_token,
    verify_access_token,
    MAX_LOGIN_ATTEMPTS,
    LOCKOUT_MINUTES,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Request / Response Schemas ──

class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str
    message: str


class UserInfo(BaseModel):
    id: int
    email: str


# ── Validation helpers ──

def _validate_email(email: str) -> str:
    """Validate and normalize email."""
    email = email.strip().lower()
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise HTTPException(400, "Invalid email format.")
    return email


def _validate_password(password: str):
    """Enforce password strength rules."""
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters long.")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(400, "Password must contain at least one uppercase letter.")
    if not re.search(r'[a-z]', password):
        raise HTTPException(400, "Password must contain at least one lowercase letter.")
    if not re.search(r'[0-9]', password):
        raise HTTPException(400, "Password must contain at least one digit.")
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':\"\\|,.<>\/?]', password):
        raise HTTPException(400, "Password must contain at least one special character.")


# ════════════════════════════════════════════════════════════════════════════
#  SIGNUP
# ════════════════════════════════════════════════════════════════════════════

@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignupRequest):
    """Register a new user account."""
    email = _validate_email(req.email)
    _validate_password(req.password)

    email_h = hash_email(email)
    pwd_hash = hash_password(req.password)

    db = SessionLocal()
    try:
        # Check if email already registered
        existing = db.query(User).filter(User.email_hash == email_h).first()
        if existing:
            raise HTTPException(409, "An account with this email already exists.")

        # Create user
        user = User(
            email_hash=email_h,
            email_display=email,
            password_hash=pwd_hash,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Issue token
        token = create_access_token(user.id, email)
        logger.info(f"New user registered: {email}")

        return AuthResponse(
            token=token,
            email=email,
            message="Account created successfully.",
        )
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
#  LOGIN (with rate limiting)
# ════════════════════════════════════════════════════════════════════════════

@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """
    Authenticate a user.
    - Max 5 attempts allowed.
    - After 5 failures → 15-minute lockout.
    """
    email = _validate_email(req.email)
    email_h = hash_email(email)

    db = SessionLocal()
    try:
        # ── Check rate limit ──
        attempt_record = db.query(LoginAttempt).filter(
            LoginAttempt.email_hash == email_h
        ).first()

        now = datetime.utcnow()

        if attempt_record:
            # Check if currently locked out
            if attempt_record.locked_until and attempt_record.locked_until > now:
                remaining = int((attempt_record.locked_until - now).total_seconds() / 60) + 1
                raise HTTPException(
                    429,
                    f"Account temporarily locked. Try again in {remaining} minute(s)."
                )

            # Reset if lockout has expired
            if attempt_record.locked_until and attempt_record.locked_until <= now:
                attempt_record.attempt_count = 0
                attempt_record.locked_until = None
                attempt_record.first_attempt_at = now
                db.commit()

        # ── Find user ──
        user = db.query(User).filter(User.email_hash == email_h).first()

        if not user or not verify_password(req.password, user.password_hash):
            # Track failed attempt
            if not attempt_record:
                attempt_record = LoginAttempt(
                    email_hash=email_h,
                    attempt_count=1,
                    first_attempt_at=now,
                )
                db.add(attempt_record)
            else:
                attempt_record.attempt_count += 1

            # Lock if max attempts reached
            if attempt_record.attempt_count >= MAX_LOGIN_ATTEMPTS:
                attempt_record.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
                db.commit()
                raise HTTPException(
                    429,
                    f"Too many failed attempts. Account locked for {LOCKOUT_MINUTES} minutes."
                )

            db.commit()
            remaining = MAX_LOGIN_ATTEMPTS - attempt_record.attempt_count
            raise HTTPException(
                401,
                f"Invalid email or password. {remaining} attempt(s) remaining."
            )

        if not user.is_active:
            raise HTTPException(403, "This account has been deactivated.")

        # ── Success — reset attempts ──
        if attempt_record:
            attempt_record.attempt_count = 0
            attempt_record.locked_until = None
            db.commit()

        token = create_access_token(user.id, email)
        logger.info(f"User logged in: {email}")

        return AuthResponse(
            token=token,
            email=email,
            message="Login successful.",
        )
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
#  ME (get current user)
# ════════════════════════════════════════════════════════════════════════════

@router.get("/me", response_model=UserInfo)
async def get_me(request: Request):
    """Return current user info from a valid JWT in the Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header.")

    token = auth_header.split(" ", 1)[1]
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(401, "Token expired or invalid. Please log in again.")

    return UserInfo(
        id=int(payload["sub"]),
        email=payload["email"],
    )
