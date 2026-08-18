"""Groundedness check — optimized for token efficiency.

retrieval_score   : fraction of top chunks with distance < 0.5 (local)
grounding_score   : fraction of answer sentences supported by context (LOCAL — no LLM)
citation_score    : fraction of cited chunks that actually exist (local)

Also exposes verify_document_evidence() — a broad keyword sweep used by
chain.py before issuing a "not found" response.

NOTE: The grounding_score is now computed locally using sentence-level
overlap checking instead of an LLM call, saving ~2000-4000 tokens per request.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GroundednessResult:
    retrieval_score: float      # 0–1, local proxy
    grounding_score: float      # 0–1, local overlap-based
    citation_score: float       # 0–1, local
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return self.grounding_score >= settings.groundedness_min_support

    @property
    def overall(self) -> float:
        """Weighted composite for display."""
        return round(
            0.3 * self.retrieval_score
            + 0.5 * self.grounding_score
            + 0.2 * self.citation_score,
            3,
        )


def _retrieval_score(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    good = sum(1 for c in chunks if c.get("distance", 1.0) < 0.5)
    return good / len(chunks)


def _citation_score(answer: str, chunks: list[dict]) -> float:
    cited = set(int(m) for m in re.findall(r"\[chunk_(\d+)\]", answer))
    if not cited:
        return 1.0  # no citations expected (e.g. safety response)
    valid = sum(1 for i in cited if 0 <= i < len(chunks))
    return valid / len(cited)


def _local_grounding_score(answer: str, chunks: list[dict]) -> tuple[float, list[str]]:
    """Compute grounding score locally using sentence-level keyword overlap.

    For each sentence in the answer, check if key terms appear in the context.
    Filters out conversational framing words and includes section names to prevent
    false rejections on concise or negative assertion sections.
    """
    # Split answer into sentences
    sentences = re.split(r'[.!?]\s+', answer.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return 1.0, []

    # Build context word set for fast lookup from text AND section metadata
    context_text = " ".join(c["text"].lower() for c in chunks)
    section_names = " ".join((c.get("metadata") or {}).get("section", "").lower() for c in chunks)
    full_context_str = context_text + " " + section_names
    context_words = set(re.findall(r"[a-z0-9]{2,}", full_context_str))

    # Framing / stop words commonly added by LLM in structured responses
    stop = {
        "the", "and", "for", "are", "this", "that", "with", "from",
        "has", "have", "been", "was", "were", "can", "may", "which",
        "should", "not", "also", "than", "each", "per", "its", "there",
        "according", "section", "prescribing", "information", "listed",
        "stated", "provided", "context", "label", "document", "mentioned",
        "described", "noted", "indicates", "shows", "states", "humira",
        "drug", "fda", "table", "above", "below", "following", "summary",
        "details", "general", "patient", "usage", "recommended", "dose",
    }

    supported = 0
    unsupported = []

    for sent in sentences:
        sent_lower = sent.lower()

        # Check 1: Valid explicit citation marker in sentence (e.g. [chunk_0])
        cited_chunks = re.findall(r"\[chunk_(\d+)\]", sent)
        if cited_chunks:
            valid_cite = any(0 <= int(idx) < len(chunks) for idx in cited_chunks)
            if valid_cite:
                supported += 1
                continue

        # Check 2: Negative assertions for sections stating "none" or "no contraindications"
        if any(neg in sent_lower for neg in (
            "no contraindications", "none listed", "no known contraindications",
            "not listed", "no listed contraindications", "no formal contraindications",
            "there are no contraindications", "states none", "states \"none\"",
            "section states none", "are no listed",
        )):
            if "none" in context_words or "contraindications" in context_words:
                supported += 1
                continue

        # Check 3: Word overlap after filtering out framing stop words
        sent_words = set(re.findall(r"[a-z0-9]{3,}", sent_lower))
        meaningful = sent_words - stop

        if not meaningful:
            supported += 1
            continue

        overlap = meaningful & context_words
        ratio = len(overlap) / len(meaningful) if meaningful else 1.0

        # Overlap threshold: 25% of content words OR at least 2 key terms
        if ratio >= 0.25 or len(overlap) >= 2:
            supported += 1
        else:
            # Phrase-level fallback match (2-word subsequences)
            words = [w for w in sent_lower.split() if len(w) >= 3]
            phrase_found = False
            for i in range(len(words) - 1):
                phrase = " ".join(words[i:i+2])
                if phrase in full_context_str:
                    phrase_found = True
                    break
            if phrase_found:
                supported += 1
            else:
                unsupported.append(sent[:80])

    score = supported / len(sentences) if sentences else 1.0
    return round(score, 3), unsupported


# ── Document evidence verification ───────────────────────────────────────

@dataclass
class EvidenceResult:
    evidence_found: bool
    evidence_strength: float          # 0–1
    matched_terms: list[str] = field(default_factory=list)
    source_sections: list[str] = field(default_factory=list)


def _expand_query_terms(query: str) -> list[str]:
    """Return the original query plus normalised variants and key sub-phrases."""
    q = query.lower()
    terms = [q]
    for prefix in (
        "does the label mention ", "does the label say ", "does rinvoq ",
        "what does the label say about ", "what is the ", "is there a ",
        "is rinvoq approved for ", "what is ",
    ):
        if q.startswith(prefix):
            terms.append(q[len(prefix):])

    _STOP = {"what", "does", "label", "mention", "about", "rinvoq",
             "prescribing", "information", "recommended", "dose", "dosage"}
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 4 and w not in _STOP]
    terms.extend(words)

    _ABBREV = {
        "nr-axspa": "non-radiographic axial spondyloarthritis",
        "axspa": "axial spondyloarthritis",
        "ra": "rheumatoid arthritis",
        "psa": "psoriatic arthritis",
        "as": "ankylosing spondylitis",
        "uc": "ulcerative colitis",
        "cd": "crohn",
        "ad": "atopic dermatitis",
    }
    for abbr, expansion in _ABBREV.items():
        if abbr in q:
            terms.append(expansion)
        if expansion in q:
            terms.append(abbr)

    return list(dict.fromkeys(t.strip() for t in terms if t.strip()))


def verify_document_evidence(
    query: str, retrieved_chunks: list[dict], drug_name: str | None = None
) -> EvidenceResult:
    """Broad keyword sweep across ALL stored chunks to verify evidence exists."""
    from app.retrieval.keyword_index import keyword_search

    terms = _expand_query_terms(query)
    matched_terms: list[str] = []
    source_sections: list[str] = []

    # 1. BM25 sweep with expanded terms
    for term in terms:
        hits = keyword_search(term, top_k=5, drug_name=drug_name)
        for h in hits:
            bm25 = h.get("bm25_score", 0)
            if bm25 > 0:
                text_lower = h["text"].lower()
                if any(t in text_lower for t in terms):
                    if term not in matched_terms:
                        matched_terms.append(term)
                    sec = h["metadata"].get("section") or ""
                    if sec and sec not in source_sections and sec.upper() != "UNSPECIFIED":
                        source_sections.append(sec)

    # 2. Also scan the already-retrieved chunks directly (free, no extra call)
    for chunk in retrieved_chunks:
        text_lower = chunk["text"].lower()
        for term in terms:
            if term in text_lower and term not in matched_terms:
                matched_terms.append(term)
                sec = chunk["metadata"].get("section") or ""
                if sec and sec not in source_sections and sec.upper() != "UNSPECIFIED":
                    source_sections.append(sec)

    if not matched_terms:
        return EvidenceResult(evidence_found=False, evidence_strength=0.0)

    strength = min(len(matched_terms) / max(len(terms), 1), 1.0)
    return EvidenceResult(
        evidence_found=True,
        evidence_strength=round(strength, 3),
        matched_terms=matched_terms,
        source_sections=source_sections,
    )


def check_groundedness(
    answer: str, chunks: list[dict], llm=None
) -> GroundednessResult:
    """Local-only groundedness check — no LLM call."""
    r_score = _retrieval_score(chunks)
    c_score = _citation_score(answer, chunks)

    if not chunks:
        return GroundednessResult(
            retrieval_score=0.0, grounding_score=0.0,
            citation_score=0.0, unsupported_claims=["no context retrieved"],
        )

    g_score, unsupported = _local_grounding_score(answer, chunks)

    return GroundednessResult(
        retrieval_score=r_score,
        grounding_score=g_score,
        citation_score=c_score,
        unsupported_claims=unsupported,
    )
