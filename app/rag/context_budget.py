"""Adaptive context selection — picks the most relevant chunks within a token budget.

Takes the CrossEncoder-reranked top-k and selects which chunks to actually send
to the LLM, respecting MAX_CONTEXT_TOKENS while preserving section-critical evidence.
"""
import re
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Rough token estimation: ~4 chars per token for English text
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(len(text) // _CHARS_PER_TOKEN, 1)


def _classify_query_complexity(query: str) -> str:
    """Classify query complexity to set context budget scaling."""
    q = query.lower()
    # Complex: comparisons, tables, multiple indications
    if any(w in q for w in ("compare", "versus", "vs", "difference", "table", "all indications")):
        return "complex"
    # Count question aspects
    aspects = sum(1 for w in ("dosage", "dose", "warning", "precaution", "contraindication",
                               "indication", "adverse", "side effect", "interaction",
                               "description", "active ingredient", "mechanism")
                  if w in q)
    if aspects >= 2:
        return "medium"
    # Simple: single-section factual question
    return "simple"


# Budget multipliers by complexity
_BUDGET_SCALE = {
    "simple": 0.5,    # ~3000 tokens
    "medium": 0.75,   # ~4500 tokens
    "complex": 1.0,   # ~6000 tokens (full budget)
}


def _is_section_critical(query: str, section: str) -> bool:
    """Check if a chunk's section is directly relevant to the query intent."""
    if not section:
        return False
    sec_upper = section.upper()
    q = query.lower()

    checks = [
        (r"contraindicat", ["CONTRAINDICATION", "4 CONTRAINDICATION"]),
        (r"active ingredient|description|what is|composition", ["DESCRIPTION", "11 DESCRIPTION"]),
        (r"dosag|dosing|dose|how to take|administration", ["DOSAGE", "RECOMMENDED DOSAGE"]),
        (r"warning|precaution|risk", ["WARNING", "PRECAUTION", "BOXED WARNING"]),
        (r"side effect|adverse", ["ADVERSE REACTION"]),
        (r"indication|used for|treatment|approved", ["INDICATIONS AND USAGE"]),
        (r"how supplied|storage", ["HOW SUPPLIED"]),
        (r"mechanism|pharmacol", ["MECHANISM", "PHARMACOL", "CLINICAL PHARMACOLOGY"]),
    ]
    for pattern, targets in checks:
        if re.search(pattern, q, re.I):
            if any(t in sec_upper for t in targets):
                return True
    return False


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks with substantially overlapping text content."""
    if len(chunks) <= 1:
        return chunks

    result = []
    seen_texts = []
    for c in chunks:
        text = c["text"].strip()
        is_dup = False
        for prev_text in seen_texts:
            # If one text is substantially contained in another, skip
            shorter, longer = (text, prev_text) if len(text) < len(prev_text) else (prev_text, text)
            if len(shorter) > 50 and shorter[:100] in longer:
                is_dup = True
                break
        if not is_dup:
            result.append(c)
            seen_texts.append(text)
    return result


def select_context_chunks(
    query: str,
    reranked_chunks: list[dict],
    max_tokens: int | None = None,
) -> list[dict]:
    """Select chunks to send to the LLM within a token budget.

    Strategy:
    1. Classify query complexity → set token budget
    2. Always include section-critical chunks first
    3. Fill remaining budget with highest-ranked chunks
    4. Deduplicate overlapping text
    """
    if not reranked_chunks:
        return []

    # Determine budget
    base_budget = max_tokens or settings.max_context_tokens
    complexity = _classify_query_complexity(query)
    token_budget = int(base_budget * _BUDGET_SCALE[complexity])

    logger.info(
        "Context selection: complexity=%s budget=%d tokens (base=%d)",
        complexity, token_budget, base_budget,
    )

    # Deduplicate first
    chunks = _deduplicate_chunks(reranked_chunks)

    # Partition into critical and non-critical
    critical = []
    others = []
    for c in chunks:
        sec = c.get("metadata", {}).get("section", "")
        if _is_section_critical(query, sec):
            critical.append(c)
        else:
            others.append(c)

    # Build selection: critical first, then fill with others
    selected = []
    used_tokens = 0

    for c in critical:
        tok = _estimate_tokens(c["text"])
        if used_tokens + tok <= token_budget or not selected:
            selected.append(c)
            used_tokens += tok

    for c in others:
        tok = _estimate_tokens(c["text"])
        if used_tokens + tok <= token_budget:
            selected.append(c)
            used_tokens += tok

    # Always include at least 2 chunks for grounding even if budget is tight
    if len(selected) < 2 and len(chunks) >= 2:
        for c in chunks[:2]:
            if c not in selected:
                selected.append(c)
                used_tokens += _estimate_tokens(c["text"])

    logger.info(
        "Context selection: %d/%d chunks selected, ~%d tokens",
        len(selected), len(reranked_chunks), used_tokens,
    )
    return selected
