"""Privacy-preserving logging for out-of-scope queries and security events.

This module logs metrics and cryptographically hashed identifiers (salted SHA-256)
without storing sensitive query strings, personally identifiable information (PII),
or protected health information (PHI).
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger("drugbot.privacy_metrics")

# Internal salt for query hashing — avoids rainbow table lookups on common phrases
_HASH_SALT = "drugbot-privacy-salt-v1"


def compute_privacy_hash(text: str, prefix_len: int = 16) -> str:
    """Compute a truncated salted SHA-256 hash of a string."""
    if not text:
        return "empty"
    salted = f"{_HASH_SALT}:{text.strip().lower()}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()[:prefix_len]


def mask_session_id(session_id: str) -> str:
    """Mask a session identifier so individual users cannot be re-identified."""
    if not session_id:
        return "anon"
    return f"sess_{compute_privacy_hash(session_id, prefix_len=10)}"


def log_out_of_scope_query(
    session_id: str,
    raw_message: str,
    classification_source: str = "regex",
    reason: str = "non_medical_topic",
) -> dict:
    """Log an out-of-scope query event with strict privacy guarantees.

    Raw query text is NEVER logged. Only cryptographic hash, character length,
    approximate token count, and category metadata are recorded.
    """
    approx_tokens = max(1, len(raw_message.split()))
    event = {
        "event_type": "out_of_scope_query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_hash": mask_session_id(session_id),
        "query_hash": compute_privacy_hash(raw_message),
        "query_length_chars": len(raw_message),
        "approx_tokens": approx_tokens,
        "classification_source": classification_source,
        "reason": reason,
    }
    logger.info("PRIVACY_METRIC %s", json.dumps(event))
    return event


def log_pdf_rejection(
    file_name: str,
    drug_name: str,
    page_count: int,
    rejection_reason: str,
) -> dict:
    """Log a PDF domain rejection event without recording document text content."""
    event = {
        "event_type": "pdf_domain_rejected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file_name_hash": compute_privacy_hash(file_name, prefix_len=12),
        "drug_name_hash": compute_privacy_hash(drug_name, prefix_len=12),
        "page_count": page_count,
        "rejection_reason": rejection_reason,
    }
    logger.info("PRIVACY_METRIC %s", json.dumps(event))
    return event


def log_safety_event(
    session_id: str,
    category: str,
    risk_level: str,
    source: str,
) -> dict:
    """Log safety classification event metrics without sensitive details."""
    event = {
        "event_type": "safety_classification",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_hash": mask_session_id(session_id),
        "category": category,
        "risk_level": risk_level,
        "source": source,
    }
    logger.info("PRIVACY_METRIC %s", json.dumps(event))
    return event
