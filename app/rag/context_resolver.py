import json
import logging
import re
from dataclasses import dataclass, field
from langchain_groq import ChatGroq
from app.core.config import settings, groq_llm_kwargs

logger = logging.getLogger(__name__)

@dataclass
class ResolutionResult:
    is_followup: bool
    is_ambiguous: bool
    clarification_question: str | None
    intent: str  # "comparison" | "standard"
    resolved_query: str
    entities: dict = field(default_factory=dict)
    retrieval_queries: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)

# ── Indication Patterns for Decomposition ──────────────────────────────

_INDICATION_PATTERNS = [
    ("rheumatoid arthritis", r"\b(rheumatoid arthritis|ra)\b"),
    ("psoriatic arthritis", r"\b(psoriatic arthritis|psa)\b"),
    ("ankylosing spondylitis", r"\b(ankylosing spondylitis|as)\b"),
    ("Crohn's disease", r"\b(crohn'?s( disease)?|cd)\b"),
    ("ulcerative colitis", r"\b(ulcerative colitis|uc)\b"),
    ("plaque psoriasis", r"\b(plaque psoriasis|psoriasis)\b"),
    ("hidradenitis suppurativa", r"\b(hidradenitis suppurativa|hs)\b"),
    ("uveitis", r"\b(uveitis)\b"),
    ("juvenile idiopathic arthritis", r"\b(juvenile idiopathic arthritis|jia)\b"),
    ("atopic dermatitis", r"\b(atopic dermatitis|ad)\b"),
]


def decompose_multi_indication_query(query: str, drug_name: str | None = None) -> list[str]:
    """Detect multi-indication comparison queries and generate targeted sub-queries.

    NO LLM call — 100% deterministic entity extraction.
    """
    q_lower = query.lower()

    detected = []
    if re.search(r"\b(all indications|8 indications|table of indications|every indication|each indication)\b", q_lower):
        detected = [
            "rheumatoid arthritis", "psoriatic arthritis", "ankylosing spondylitis",
            "Crohn's disease", "ulcerative colitis", "plaque psoriasis",
            "hidradenitis suppurativa", "uveitis"
        ]
    else:
        for canonical_name, pattern in _INDICATION_PATTERNS:
            if re.search(pattern, q_lower, re.IGNORECASE):
                if canonical_name not in detected:
                    detected.append(canonical_name)

    is_multi = bool(re.search(
        r"\b(compare|table|dosing for|dosage for|indications|versus|vs\.?|side by side|which)\b",
        q_lower
    ))

    if (is_multi and len(detected) >= 2) or len(detected) >= 3:
        drug = drug_name or "HUMIRA"
        sub_queries = [query]
        for ind in detected:
            sub_queries.append(f"{drug} recommended dosage {ind}")
        logger.info(
            "Decomposed multi-indication query into %d sub-queries for indications: %s",
            len(sub_queries), detected
        )
        return sub_queries

    return [query]


# ── Compact context resolver prompt ──────────────────────────────────────

CONTEXT_RESOLVER_PROMPT = """\
Analyze this chatbot message given the conversation state. Determine if it's a follow-up, has comparison intent, and rewrite it into a self-contained query.

State: {state_json}
Message: {message}

Rules:
1. is_followup: true if message uses pronouns/references like "it","this","that","what about" or depends on prior context.
2. intent: "comparison" if comparing drugs/indications/populations; else "standard".
3. is_ambiguous: true only if multiple valid interpretations exist and you cannot determine which.
4. resolved_query: rewrite into complete standalone query resolving all pronouns.
5. retrieval_queries: list of 1-2 search strings.
6. entities: {{drug, primary_indication, primary_population, comparison_indication, topic}}.
7. required_sections: likely section numbers (e.g. ["2.4","4"]).

Output ONLY valid JSON:
{{"is_followup":false,"is_ambiguous":false,"clarification_question":null,"intent":"standard","resolved_query":"...","entities":{{}},"retrieval_queries":["..."],"required_sections":[]}}"""


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    return text.strip()

def _parse_resolver_json(raw: str) -> dict | None:
    cleaned = _clean_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None

def _validate_resolver_schema(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    data.setdefault("is_followup", False)
    data.setdefault("is_ambiguous", False)
    data.setdefault("clarification_question", None)
    data.setdefault("intent", "standard")
    data.setdefault("entities", {})
    if not isinstance(data.get("entities"), dict):
        data["entities"] = {}
    if not isinstance(data.get("retrieval_queries"), list):
        data["retrieval_queries"] = [str(data.get("resolved_query", ""))]
    if not isinstance(data.get("required_sections"), list):
        data["required_sections"] = []
    return bool(data.get("resolved_query"))


def _is_simple_standalone_query(message: str) -> bool:
    """Detect if a message is clearly a standalone question (no follow-up resolution needed)."""
    m = message.strip().lower()
    if re.search(r"\b(it|this|that|which|former|latter|what about|how about|compare with)\b", m, re.I):
        return False
    if re.search(r"(what|how|list|describe|compare|tell me)", m, re.I) and \
       re.search(r"(humira|rinvoq|skyrizi|keytruda|opdivo|dupixent|stelara|remicade)", m, re.I):
        return True
    return False


def _detect_comparison_intent(message: str) -> bool:
    """Deterministic comparison detection."""
    return bool(re.search(
        r"\b(compare|versus|vs\.?|difference|how does .+ compare|"
        r"which (has|is|one)|higher|lower|in a table|side.?by.?side)\b",
        message, re.IGNORECASE
    ))


def _resolve_followup_deterministically(message: str, state: dict) -> ResolutionResult | None:
    m = message.strip().lower()
    drug = state.get("drug") or "HUMIRA"
    comparison_entities = state.get("comparison_entities") or []
    if not isinstance(comparison_entities, list):
        comparison_entities = []
    
    if not comparison_entities and state.get("current_indication"):
        comparison_entities = [state["current_indication"]]

    is_anaphora_which = bool(re.search(r"\b(which (one|of these|indication|has|is)|the former|the latter)\b", m, re.I))
    is_compare_phrase = bool(re.search(r"\b(how does that compare|that compare|compare with|versus|vs\.?)\b", m, re.I))
    is_what_about = bool(re.search(r"\b(what about|how about)\b", m, re.I))

    new_inds = []
    for canonical_name, pattern in _INDICATION_PATTERNS:
        if re.search(pattern, m, re.I) and canonical_name not in new_inds:
            new_inds.append(canonical_name)

    all_entities = list(comparison_entities)
    for ind in new_inds:
        if ind not in all_entities:
            all_entities.append(ind)

    if is_anaphora_which and all_entities:
        topic_phrase = "initial dose" if re.search(r"initial", m, re.I) else "recommended dosage"
        entities_str = ", ".join(all_entities)
        resolved_q = f"Which of {entities_str} has the higher {topic_phrase} for {drug}? Compare initial and recommended doses."
        sub_qs = [resolved_q] + [f"{drug} recommended dosage {e}" for e in all_entities]
        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="comparison",
            resolved_query=resolved_q,
            entities={"drug": drug, "comparison_entities": all_entities, "topic": topic_phrase},
            retrieval_queries=sub_qs,
            required_sections=["2.2", "2.4", "2.5", "2.6", "2.7"]
        )

    if is_compare_phrase and all_entities:
        entities_str = ", ".join(all_entities)
        resolved_q = f"Compare recommended {drug} dosage for {entities_str}."
        sub_qs = [resolved_q] + [f"{drug} recommended dosage {e}" for e in all_entities]
        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="comparison",
            resolved_query=resolved_q,
            entities={"drug": drug, "comparison_entities": all_entities, "topic": "dosage"},
            retrieval_queries=sub_qs,
            required_sections=["2.2", "2.4", "2.5", "2.6", "2.7"]
        )

    if is_what_about and new_inds and state.get("current_topic"):
        topic = state.get("current_topic") or "dosage and administration"
        ind = new_inds[0]
        resolved_q = f"{drug} {topic} for {ind}"
        sub_qs = [resolved_q]
        if comparison_entities:
            all_entities = list(comparison_entities)
            if ind not in all_entities:
                all_entities.append(ind)
            sub_qs.append(f"Compare {drug} dosage for {', '.join(all_entities)}")
        return ResolutionResult(
            is_followup=True,
            is_ambiguous=False,
            clarification_question=None,
            intent="standard" if len(all_entities) < 2 else "comparison",
            resolved_query=resolved_q,
            entities={"drug": drug, "primary_indication": ind, "comparison_entities": all_entities, "topic": topic},
            retrieval_queries=sub_qs,
            required_sections=[]
        )

    return None


def resolve_context(message: str, state: dict, llm: ChatGroq) -> ResolutionResult:
    # 1. Check deterministic anaphora / follow-up first
    det_res = _resolve_followup_deterministically(message, state)
    if det_res:
        logger.info("Context resolver resolved follow-up deterministically: resolved_query='%s'", det_res.resolved_query)
        return det_res

    # 2. For simple standalone queries with no state, skip LLM call
    if _is_simple_standalone_query(message) and not state.get("last_question"):
        intent = "comparison" if _detect_comparison_intent(message) else "standard"
        return ResolutionResult(
            is_followup=False,
            is_ambiguous=False,
            clarification_question=None,
            intent=intent,
            resolved_query=message,
            entities={},
            retrieval_queries=[message] if intent == "standard" else [message],
            required_sections=[]
        )

    # Compact state: only include non-null fields
    compact_state = {k: v for k, v in state.items() if v is not None and v != [] and k != "last_answer"}
    state_json = json.dumps(compact_state, indent=1) if compact_state else "{}"
    prompt = CONTEXT_RESOLVER_PROMPT.format(state_json=state_json, message=message)

    try:
        from app.core.llm_retry import retry_llm_call
        response = retry_llm_call(llm.invoke, prompt, label="context_resolver")
        data = _parse_resolver_json(response.content)
        if data and _validate_resolver_schema(data):
            return ResolutionResult(
                is_followup=bool(data["is_followup"]),
                is_ambiguous=bool(data["is_ambiguous"]),
                clarification_question=data.get("clarification_question"),
                intent=data["intent"],
                resolved_query=data["resolved_query"],
                entities=data["entities"],
                retrieval_queries=data["retrieval_queries"],
                required_sections=data["required_sections"]
            )
        logger.warning("Resolver: Invalid JSON or schema from LLM")
    except Exception as e:
        logger.warning("Resolver LLM failed: %s", e)

    # Deterministic fallback
    logger.warning("Context resolver falling back to deterministic rules")
    intent = "comparison" if _detect_comparison_intent(message) else "standard"
    return ResolutionResult(
        is_followup=False,
        is_ambiguous=False,
        clarification_question=None,
        intent=intent,
        resolved_query=message,
        entities={},
        retrieval_queries=[message],
        required_sections=[]
    )


def update_conversation_state(
    session_id: str,
    message: str,
    resolved_query: str,
    answer: str,
    chunks: list[dict],
    llm: ChatGroq
) -> dict:
    """Extract conversation state deterministically — NO LLM call.

    Saves ~1500-3000 tokens per request by parsing state from chunk metadata
    and the answer text directly, instead of calling the LLM.
    """
    from app.rag.memory import memory
    prev_state = memory.get_state(session_id)

    # Extract drug from chunks
    drug = prev_state.get("drug")
    sections = []
    for c in chunks:
        meta = c.get("metadata") or {}
        if meta.get("drug_name") and not drug:
            drug = meta["drug_name"]
        sec = meta.get("section")
        if sec and sec not in sections:
            sections.append(sec)

    # Detect topic from the query
    q = (resolved_query + " " + message).lower()
    topic = prev_state.get("current_topic")
    topic_map = [
        (r"dosag|dosing|dose|administration|initial dose", "dosage and administration"),
        (r"contraindicat", "contraindications"),
        (r"warning|precaution", "warnings and precautions"),
        (r"adverse|side effect", "adverse reactions"),
        (r"indication|approved|used for", "indications and usage"),
        (r"description|active ingredient|composition", "description"),
        (r"interaction", "drug interactions"),
        (r"mechanism|pharmacol", "clinical pharmacology"),
    ]
    for pattern, t in topic_map:
        if re.search(pattern, q, re.I):
            topic = t
            break

    # Extract ALL indications in resolved_query, message, or answer
    detected_inds = []
    full_text = (resolved_query + " " + message + " " + answer).lower()
    for canonical_name, pattern in _INDICATION_PATTERNS:
        if re.search(pattern, full_text, re.I):
            if canonical_name not in detected_inds:
                detected_inds.append(canonical_name)

    prev_comparison = prev_state.get("comparison_entities") or []
    if not isinstance(prev_comparison, list):
        prev_comparison = []

    new_comparison = list(prev_comparison)
    for ind in detected_inds:
        if ind not in new_comparison:
            new_comparison.append(ind)

    updates = {
        "last_question": message,
        "last_answer": answer[:200],  # Truncate to save memory
    }
    if drug:
        updates["drug"] = drug
    if topic:
        updates["current_topic"] = topic
    if detected_inds:
        updates["current_indication"] = detected_inds[-1]
    if new_comparison:
        updates["comparison_entities"] = new_comparison
    if sections:
        updates["current_section"] = sections[0]

    memory.update_state(session_id, updates)
    return memory.get_state(session_id)
