"""Main RAG pipeline — optimized for token efficiency.

Flow:
  User message
    → conversational shortcut (greetings) — no LLM
    → safety classification  (regex first, LLM fallback)
    → query rewrite (skip for standalone queries)
    → hybrid retrieval + rerank
    → adaptive context selection (token budget)
    → generate answer (single LLM call with max_tokens)
    → local groundedness check (no LLM)
    → deterministic state extraction (no LLM)
    → return structured response

LLM calls per request:
  Best case (standalone + regex safety): 1 (generate only)
  Typical (follow-up): 2 (resolve + generate)
  Worst case (ambiguous + patient_specific): 3
"""
import logging
import re
import json

from langchain_groq import ChatGroq

from app.core.config import settings, groq_llm_kwargs
from app.core.llm_retry import get_last_used_model, DailyTokenLimitError
from app.rag.memory import memory
from app.rag.context_resolver import resolve_context, update_conversation_state, decompose_multi_indication_query
from app.retrieval.hybrid_retriever import hybrid_retrieve, rerank
from app.rag.generator import generate_answer
from app.rag.groundedness import check_groundedness, verify_document_evidence
from app.rag.safety_classifier import classify_question, EMPATHETIC_OUT_OF_SCOPE_REFUSAL
from app.core.privacy_logger import log_out_of_scope_query, log_safety_event
from app.rag.citations import extract_citations, strip_citation_markers, replace_chunk_markers_with_sources
from app.rag.context_budget import select_context_chunks

logger = logging.getLogger(__name__)
_token_logger = logging.getLogger("drugbot.tokens")

# ── Static responses ───────────────────────────────────────────────────────

_HIGH_RISK_RESPONSE = (
    "⚠️ This sounds like it may be a medical emergency.\n\n"
    "Please call emergency services (911) or Poison Control (1-800-222-1222) "
    "immediately if you are in the United States, or your local emergency number.\n\n"
    "Do not wait for an online response in an emergency situation. "
    "A healthcare professional needs to evaluate you in person."
)

# Category-aware not-found messages — no medical escalation for document questions
_NOT_FOUND = {
    "general_label": (
        "I couldn't find sufficient evidence in the provided prescribing information "
        "to answer this question confidently. I don't want to infer information that "
        "isn't supported by the document.\n\n"
        "You can try rephrasing the question or asking about a specific section of "
        "the prescribing information."
    ),
    "unsupported": (
        "I couldn't find this information in the provided prescribing information."
    ),
    "patient_specific": (
        "I couldn't find sufficient information in the provided prescribing information "
        "to answer this confidently. Because this question relates to your individual "
        "situation, please consult an appropriate healthcare professional for "
        "personalised guidance."
    ),
}

_ABOUT_BOT_REPLY = (
    "I can analyze official drug prescribing information (PDFs) and answer "
    "questions about dosages, drug interactions, contraindications, warnings, "
    "and pediatric guidelines with direct page citations."
)

_CONVERSATIONAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(hi|hello|hey|hiya|yo|greetings)( there)?$", re.I),
     "Hello! I am DrugBot, your AI drug information assistant. How can I help you with prescribing information today?"),
    (re.compile(r"^good (morning|afternoon|evening)$", re.I),
     "Hello! How can I assist you with drug prescribing information today?"),
    (re.compile(r"^(who are you|what are you)$", re.I),
     "I am DrugBot, an AI assistant designed to answer questions based on official FDA prescribing information documents."),
    (re.compile(
        r".*\b(kind of pdf|type of pdf|pdf file|pdf format|what pdf|upload a pdf|document format|supported document|what document|pdf can you|analyze pdf|read pdf|parse pdf)\b.*", re.I,
     ), "I can analyze official FDA drug prescribing information PDFs (such as HUMIRA, Rinvoq, Skyrizi, etc.). You can upload drug PDFs using the sidebar, and I will answer questions about indications, dosages, contraindications, warnings, side effects, and drug interactions with citations."),
    (re.compile(
        r"^(what (can|do|will) you do|what are you (doing|for)|"
        r"what('?s| is) your (purpose|job|function|deal)|"
        r"how can you help( me)?|what is this( bot| app)?)$", re.I,
     ), _ABOUT_BOT_REPLY),
    (re.compile(r"^(how are you( doing)?|how'?s it going|what'?s up|sup)$", re.I),
     "I'm running fine, thanks for asking! I'm ready to help with any drug prescribing information questions you have."),
    (re.compile(r"^(are you (a bot|human|real|an ai|alive)|is this a bot)$", re.I),
     "I'm an AI assistant — DrugBot — built to answer questions from official drug prescribing information, not a human."),
    (re.compile(r"^what('?s| is) your name$", re.I),
     "I'm DrugBot, an AI assistant for drug prescribing information."),
    (re.compile(r"^who (made|built|created|trained) you$", re.I),
     "I was built as an AI assistant to answer questions from official FDA prescribing information documents."),
    (re.compile(r"^help( me)?$", re.I),
     "You can ask me questions about any ingested drug prescribing information (such as dosage guidelines, contraindications, or side effects) or upload a drug PDF using the sidebar."),
    (re.compile(r"^(thanks|thank you|thx|ty|cheers)$", re.I),
     "You're very welcome! Feel free to ask if you need any more drug information."),
    (re.compile(r"^(bye|goodbye|see you|see ya|later|take care)$", re.I),
     "Goodbye! Feel free to come back anytime you have questions about prescribing information."),
    (re.compile(r"^(ok|okay|cool|nice|great|got it|sounds good)$", re.I),
     "Sounds good! Let me know if you have a question about drug prescribing information."),
]


def _conversational_reply(message: str) -> str | None:
    normalized = message.strip().rstrip("!.,? ").strip()
    if not normalized:
        return None
    for pattern, reply in _CONVERSATIONAL_PATTERNS:
        if pattern.match(normalized):
            return reply
    return None


def _not_found_for(category: str) -> str:
    return _NOT_FOUND.get(category, _NOT_FOUND["general_label"])


# ── Main handler ───────────────────────────────────────────────────────────

def handle_chat_message(
    session_id: str,
    message: str,
    drug_name_hint: str | None = None,
    user_id: int | None = None,
) -> dict:
    llm = ChatGroq(**groq_llm_kwargs(temperature=0.1, max_tokens=settings.max_output_tokens))

    # 1. Active drug context
    active_drug = drug_name_hint or memory.get_active_drug(session_id, user_id=user_id)

    # 2. Conversational shortcut — zero LLM cost
    reply = _conversational_reply(message)
    if reply:
        memory.add_turn(session_id, "user", message, user_id=user_id)
        memory.add_turn(session_id, "assistant", reply, user_id=user_id)
        return {
            "answer": reply,
            "citations": [],
            "active_drug": active_drug,
            "confidence": "conversational",
            "scores": None,
            "safety_notice": None,
            "question_category": "conversational",
        }

    # 3. Safety classification — regex first (free), LLM fallback
    classification = classify_question(message, llm=llm)
    logger.info(
        "[%s] category=%s risk=%s source=%s (user_id=%s)",
        session_id, classification.category, classification.risk_level, classification.source, user_id,
    )

    # 4. High-risk shortcut — no retrieval, no generation
    if classification.category == "high_risk":
        log_safety_event(
            session_id=session_id,
            category="high_risk",
            risk_level="critical",
            source=classification.source,
        )
        memory.add_turn(session_id, "user", message, user_id=user_id)
        memory.add_turn(session_id, "assistant", _HIGH_RISK_RESPONSE, user_id=user_id)
        return {
            "answer": _HIGH_RISK_RESPONSE,
            "citations": [],
            "active_drug": active_drug,
            "confidence": "high_risk",
            "scores": None,
            "safety_notice": None,
            "question_category": "high_risk",
        }

    # 4b. Out-of-scope shortcut — gentle empathetic refusal + privacy-preserving telemetry
    if classification.category == "out_of_scope":
        log_out_of_scope_query(
            session_id=session_id,
            raw_message=message,
            classification_source=classification.source,
            reason=classification.reason,
        )
        memory.add_turn(session_id, "user", message, user_id=user_id)
        memory.add_turn(session_id, "assistant", EMPATHETIC_OUT_OF_SCOPE_REFUSAL, user_id=user_id)
        return {
            "answer": EMPATHETIC_OUT_OF_SCOPE_REFUSAL,
            "citations": [],
            "active_drug": active_drug,
            "confidence": "out_of_scope",
            "scores": None,
            "safety_notice": None,
            "question_category": "out_of_scope",
        }

    # Get conversation state
    state = memory.get_state(session_id, user_id=user_id)

    # 5. Context Resolver / Query Rewriting
    resolved = resolve_context(message, state, llm=llm)
    logger.info(
        "[%s] Context Resolution: is_followup=%s is_ambiguous=%s intent=%s resolved_query='%s'",
        session_id, resolved.is_followup, resolved.is_ambiguous, resolved.intent, resolved.resolved_query
    )

    # Handle ambiguous query
    if resolved.is_ambiguous:
        clarification_reply = resolved.clarification_question or "Could you please clarify your question?"
        memory.add_turn(session_id, "user", message, user_id=user_id)
        memory.add_turn(session_id, "assistant", clarification_reply, user_id=user_id)
        return {
            "answer": clarification_reply,
            "citations": [],
            "active_drug": active_drug,
            "confidence": "clarification",
            "scores": None,
            "safety_notice": None,
            "question_category": classification.category,
        }

    # Update active drug and target drug filters
    from app.retrieval.vector_store import normalize_drug_name, get_known_drugs
    target_drugs = []
    resolved_drug = resolved.entities.get("drug") or resolved.entities.get("drugs")
    if isinstance(resolved_drug, (list, tuple)):
        for d in resolved_drug:
            nd = normalize_drug_name(d, user_id=user_id)
            if nd and nd not in target_drugs:
                target_drugs.append(nd)
    elif resolved_drug:
        nd = normalize_drug_name(resolved_drug, user_id=user_id)
        if nd and nd not in target_drugs:
            target_drugs.append(nd)

    # Also check known drugs mentioned directly in resolved query
    for known in get_known_drugs(user_id=user_id):
        if re.search(rf"\b{re.escape(known)}\b", resolved.resolved_query, re.I):
            if known not in target_drugs:
                target_drugs.append(known)

    if target_drugs:
        active_drug = target_drugs[0]
        memory.set_active_drug(session_id, active_drug, user_id=user_id)

    # For multi-drug or comparison queries, search across all relevant drugs
    retrieval_drug_filter = target_drugs if len(target_drugs) > 1 else active_drug

    history_text = memory.format_history(session_id, user_id=user_id)

    # 6. Multi-Hop Retrieval (Query Decomposition) + rerank
    retrieval_qs = list(resolved.retrieval_queries)
    for sub in decompose_multi_indication_query(resolved.resolved_query, drug_name=active_drug):
        if sub not in retrieval_qs:
            retrieval_qs.append(sub)

    all_candidates = []
    seen_candidate_ids = set()
    for q_text in retrieval_qs:
        candidates = hybrid_retrieve(q_text, drug_name=retrieval_drug_filter, user_id=user_id)
        for c in candidates:
            if c["id"] not in seen_candidate_ids:
                seen_candidate_ids.add(c["id"])
                all_candidates.append(c)

    top_chunks = rerank(resolved.resolved_query, all_candidates)

    if not top_chunks:
        not_found = _not_found_for(classification.category)
        memory.add_turn(session_id, "user", message, user_id=user_id)
        memory.add_turn(session_id, "assistant", not_found, user_id=user_id)
        return {
            "answer": not_found,
            "citations": [],
            "active_drug": active_drug,
            "confidence": "not_found",
            "scores": None,
            "safety_notice": None,
            "question_category": classification.category,
        }

    # Update active drug from top chunk if still not set
    if not active_drug:
        active_drug = top_chunks[0]["metadata"]["drug_name"]
        memory.set_active_drug(session_id, active_drug, user_id=user_id)

    # 7. ADAPTIVE CONTEXT SELECTION — token budgeting
    context_chunks = select_context_chunks(resolved.resolved_query, top_chunks)

    _token_logger.info(
        "Query: '%s' | Reranked: %d chunks | Context: %d chunks sent to LLM",
        message[:60], len(top_chunks), len(context_chunks),
    )

    # 8. Route by category/intent
    if resolved.intent == "comparison":
        gen_mode = "comparison"
    elif classification.category == "patient_specific":
        gen_mode = "patient_specific"
    else:
        gen_mode = "general_label"

    # 9. Generate answer (SINGLE LLM call with token limit)
    draft = generate_answer(resolved.resolved_query, history_text, context_chunks, gen_mode, llm)

    # 10. LOCAL groundedness check — NO LLM call
    groundedness = check_groundedness(draft, context_chunks)

    # 11. Groundedness gate
    if not groundedness.passes:
        logger.info(
            "[%s] low groundedness (%.2f) — running evidence verification",
            session_id, groundedness.grounding_score,
        )
        evidence = verify_document_evidence(resolved.resolved_query, top_chunks, drug_name=active_drug)
        logger.info(
            "[%s] evidence_found=%s strength=%.2f matched=%s",
            session_id, evidence.evidence_found, evidence.evidence_strength, evidence.matched_terms,
        )

        if evidence.evidence_found:
            # Re-generate with the same context (already token-budgeted)
            draft = generate_answer(resolved.resolved_query, history_text, context_chunks, gen_mode, llm)
            groundedness = check_groundedness(draft, context_chunks)
            if not groundedness.passes:
                not_found = _not_found_for(classification.category)
                memory.add_turn(session_id, "user", message, user_id=user_id)
                memory.add_turn(session_id, "assistant", not_found, user_id=user_id)
                return {
                    "answer": not_found,
                    "citations": [],
                    "active_drug": active_drug,
                    "confidence": "limited_evidence",
                    "scores": {
                        "retrieval_score": groundedness.retrieval_score,
                        "grounding_score": groundedness.grounding_score,
                        "citation_score": groundedness.citation_score,
                    },
                    "safety_notice": (
                        "Because this question relates to your individual situation, "
                        "please consult an appropriate healthcare professional."
                    ) if classification.category == "patient_specific" else None,
                    "question_category": classification.category,
                }
        else:
            not_found = _not_found_for(classification.category)
            memory.add_turn(session_id, "user", message, user_id=user_id)
            memory.add_turn(session_id, "assistant", not_found, user_id=user_id)
            return {
                "answer": not_found,
                "citations": [],
                "active_drug": active_drug,
                "confidence": "not_found",
                "scores": {
                    "retrieval_score": groundedness.retrieval_score,
                    "grounding_score": groundedness.grounding_score,
                    "citation_score": groundedness.citation_score,
                },
                "safety_notice": (
                    "Because this question relates to your individual situation, "
                    "please consult an appropriate healthcare professional."
                ) if classification.category == "patient_specific" else None,
                "question_category": classification.category,
            }

    # 12. Citations (uses context_chunks for correct chunk_N mapping)
    citations = extract_citations(draft, context_chunks)
    display_answer = replace_chunk_markers_with_sources(draft, context_chunks)

    # 13. Safety notice — only for patient_specific
    safety_notice = None
    if classification.category == "patient_specific":
        safety_notice = (
            "The information above is taken directly from the official prescribing "
            "information and is not personalised medical advice. Please contact your "
            "healthcare professional for guidance specific to your situation."
        )

    # 14. Memory
    memory.add_turn(session_id, "user", message, user_id=user_id)
    memory.add_turn(session_id, "assistant", display_answer, user_id=user_id)

    # 15. Deterministic state extraction — NO LLM call
    try:
        updated_state = update_conversation_state(
            session_id, message, resolved.resolved_query, display_answer, context_chunks, llm=llm
        )
    except Exception as exc:
        logger.warning("[%s] Failed to update conversation state: %s", session_id, exc)
        updated_state = state

    # Log summary
    logger.info(
        "\n--- Chat Message Log ---\n"
        "Original Query: %s\n"
        "Detected Follow-Up: %s\n"
        "Resolved Query: %s\n"
        "Retrieved Sections: %s\n"
        "Context Chunks Sent: %d\n"
        "------------------------",
        message,
        resolved.is_followup,
        resolved.resolved_query,
        [c["metadata"].get("section") for c in context_chunks],
        len(context_chunks),
    )

    return {
        "answer": display_answer,
        "citations": citations,
        "active_drug": active_drug,
        "confidence": "grounded",
        "scores": {
            "retrieval_score": groundedness.retrieval_score,
            "grounding_score": groundedness.grounding_score,
            "citation_score": groundedness.citation_score,
        },
        "safety_notice": safety_notice,
        "question_category": classification.category,
        "model_used": get_last_used_model(),
    }

