"""Safety classifier — categorises every user question BEFORE retrieval/generation.

Categories
----------
general_label     : factual question about what the label says / medical information — answer normally
patient_specific  : user describes their own situation — RAG + safety boundary
high_risk         : overdose / acute emergency — safety response, no diagnosis
unsupported       : medical topic not in the label — say so explicitly
out_of_scope      : non-medical / non-pharmaceutical query — gentle empathetic refusal

Uses deterministic regex patterns FIRST (zero LLM cost for obvious cases),
falling back to the LLM only when needed.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_groq import ChatGroq
from app.core.config import settings, groq_llm_kwargs

logger = logging.getLogger(__name__)

# ── Empathetic Refusal Template ───────────────────────────────────────────

EMPATHETIC_OUT_OF_SCOPE_REFUSAL = (
    "I am an AI assistant dedicated exclusively to pharmaceutical, pharmacological, "
    "and clinical medical information (such as medications, dosages, drug interactions, "
    "contraindications, and prescribing guidelines).\n\n"
    "While I am unable to assist with non-medical topics, I would be very happy to help "
    "answer questions about medications, clinical research, or official FDA prescribing documentation. "
    "Please feel free to ask a drug- or health-related question!"
)

# ── Classifier prompt (compact with domain restriction) ────────────────────

_CLASSIFIER_PROMPT = """\
Classify this chatbot question into ONE category based on strict domain guidelines.

Domain Scope:
This system ONLY processes questions related to:
- Pharmaceuticals, drugs, medications, pharmacology, drug interactions, dosages, adverse effects
- Medical conditions, diseases, symptoms, clinical treatments, healthcare procedures, pathology
- Clinical research, prescribing guidelines, patient care, medical biotechnology
- Regulatory aspects of drug approvals, labeling, warnings, and safety

Categories:
  general_label    – factual question about medications, diseases, or prescribing information
  patient_specific – user describes their own personal medical situation or symptoms
  high_risk        – overdose, acute poisoning, emergency, severe acute reaction
  unsupported      – medical/drug question not covered in available prescribing labels
  out_of_scope     – NON-MEDICAL query (e.g. general coding/technology, entertainment, movies, music, sports, politics, non-medical finance/crypto, recipes/cooking, travel, general math/trivia, personal relationships)

Output ONLY valid JSON:
{{"category": "general_label|patient_specific|high_risk|unsupported|out_of_scope", "risk_level": "low|moderate|high|critical", "requires_rag": true, "requires_safety_notice": false, "allow_direct_medical_instruction": false, "reason": "..."}}

Rules:
- high_risk: risk_level="critical", requires_rag=false
- out_of_scope: risk_level="low", requires_rag=false, requires_safety_notice=false
- patient_specific: requires_safety_notice=true, requires_rag=true
- general_label: requires_rag=true

Question: {question}

JSON:"""

# ── Deterministic patterns ────────────────────────────────────────────────

_HIGH_RISK_PATTERNS = re.compile(
    r"\b(overdose|took too much|accidental(ly)? took|difficulty breath|"
    r"can'?t breath|chest pain|severe reaction|anaphyla|emergency|"
    r"call 911|poison control|unconscious|faint(ing)?|seizure)\b",
    re.IGNORECASE,
)

_PATIENT_SPECIFIC_PATTERNS = re.compile(
    r"\b(i am taking|i('?m| am) on|i (have|had|developed|got|noticed)|"
    r"my (doctor|symptoms?|dose|medication|condition|lab|test|result)|"
    r"should i (stop|take|start|skip|double|change|switch)|"
    r"can i (take|stop|start|drink|eat)|is it safe for me|"
    r"i missed (a |my )?dose|i('?m| am) pregnant|i have (liver|kidney|renal|hepatic)|"
    r"what should i do)\b",
    re.IGNORECASE,
)

# Patterns that are almost certainly general_label — skip LLM
_GENERAL_LABEL_PATTERNS = re.compile(
    r"^(what (is|are)|how (is|are)|list|describe|explain|tell me about|"
    r"what does the label say|what does the prescribing information|"
    r"what (are|is) the (contraindication|indication|dosag|dose|warning|"
    r"precaution|adverse|side effect|interaction|mechanism|pharmacol|"
    r"supply|storage|description|active ingredient|composition))",
    re.IGNORECASE,
)

# Obvious non-medical / out-of-scope patterns — skip LLM
_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b("
    r"write (a |the )?(python|javascript|code|script|sql|java|c\+\+|html|css|program|app|function|poem|essay|story|song)|"
    r"debug this code|coding tutorial|binary search|git commit|install linux|docker container|kubernetes|npm install|"
    r"stock market|invest in (crypto|bitcoin|stocks|forex|real estate)|crypto trading|nft price|buy bitcoin|dogecoin|ethereum|"
    r"who won (the match|the game|the world cup|the super bowl|the oscar|the election|the grammy)|"
    r"movie review|box office|football score|nba finals|fifa world cup|video games?|playstation|xbox|fortnite|minecraft|gta|"
    r"who is the (president|prime minister|governor|king|queen|ceo) of|who won the election|political party|democrats vs republicans|"
    r"recipe for|how to cook|how to bake|bake a cake|make (pasta|pizza|cookies|cocktail|coffee)|"
    r"weather in [a-z]+|flight tickets to|tourist attractions in|hotels in|vacation in|"
    r"solve (the|this) (equation|math|puzzle|riddle)|calculate [0-9]+|what is [0-9]+\s*[\+\-\*\/]\s*[0-9]+|"
    r"horoscope|astrology sign|zodiac|tarot|"
    r"relationship advice|dating advice|breakup advice|tell me a joke|write a joke"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    category: str           # general_label | patient_specific | high_risk | unsupported | out_of_scope
    risk_level: str         # low | moderate | high | critical
    requires_rag: bool
    requires_safety_notice: bool
    allow_direct_medical_instruction: bool
    reason: str
    source: str = "llm"     # "llm" | "regex" | "fallback"


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def _validate(data: dict) -> bool:
    valid_cats = {"general_label", "patient_specific", "high_risk", "unsupported", "out_of_scope"}
    valid_risks = {"low", "moderate", "high", "critical"}
    return (
        isinstance(data, dict)
        and data.get("category") in valid_cats
        and data.get("risk_level") in valid_risks
        and isinstance(data.get("requires_rag"), bool)
        and isinstance(data.get("requires_safety_notice"), bool)
        and isinstance(data.get("allow_direct_medical_instruction"), bool)
    )


def _deterministic_classify(question: str) -> ClassificationResult | None:
    """Try to classify using regex patterns. Returns None if uncertain."""
    q = question.strip()

    if _HIGH_RISK_PATTERNS.search(q):
        return ClassificationResult(
            category="high_risk", risk_level="critical",
            requires_rag=False, requires_safety_notice=True,
            allow_direct_medical_instruction=False,
            reason="High-risk emergency pattern detected.",
            source="regex",
        )
    if _OUT_OF_SCOPE_PATTERNS.search(q):
        return ClassificationResult(
            category="out_of_scope", risk_level="low",
            requires_rag=False, requires_safety_notice=False,
            allow_direct_medical_instruction=False,
            reason="Non-medical topic detected via deterministic domain filter.",
            source="regex",
        )
    if _PATIENT_SPECIFIC_PATTERNS.search(q):
        return ClassificationResult(
            category="patient_specific", risk_level="moderate",
            requires_rag=True, requires_safety_notice=True,
            allow_direct_medical_instruction=False,
            reason="User appears to be describing a personal medical situation.",
            source="regex",
        )
    if _GENERAL_LABEL_PATTERNS.search(q):
        return ClassificationResult(
            category="general_label", risk_level="low",
            requires_rag=True, requires_safety_notice=False,
            allow_direct_medical_instruction=False,
            reason="Standard prescribing information question.",
            source="regex",
        )
    return None


def _from_dict(data: dict, source: str) -> ClassificationResult:
    return ClassificationResult(
        category=data["category"],
        risk_level=data["risk_level"],
        requires_rag=bool(data.get("requires_rag", True)),
        requires_safety_notice=bool(data.get("requires_safety_notice", False)),
        allow_direct_medical_instruction=bool(data.get("allow_direct_medical_instruction", False)),
        reason=data.get("reason", ""),
        source=source,
    )


def classify_question(question: str, llm: ChatGroq | None = None) -> ClassificationResult:
    """Classify: regex first (free), LLM only if ambiguous."""
    # Try deterministic classification first — saves an LLM call
    result = _deterministic_classify(question)
    if result:
        logger.info("Safety classifier resolved via regex: %s", result.category)
        return result

    # Fall back to LLM
    from app.core.llm_retry import retry_llm_call
    llm = llm or ChatGroq(**groq_llm_kwargs(temperature=0.0, max_tokens=150))
    prompt = _CLASSIFIER_PROMPT.format(question=question)

    try:
        response = retry_llm_call(llm.invoke, prompt, label="safety_classifier")
        data = _parse_json(response.content)
        if data and _validate(data):
            return _from_dict(data, "llm")
        logger.warning("Safety classifier: invalid JSON/schema from LLM: %s", response.content)
    except Exception as exc:
        logger.warning("Safety classifier LLM failed: %s", exc)

    # Fallback to general_label (safest default)
    logger.warning("Safety classifier falling back to general_label default")
    return ClassificationResult(
        category="general_label", risk_level="low",
        requires_rag=True, requires_safety_notice=False,
        allow_direct_medical_instruction=False,
        reason="No personal or high-risk indicators detected.",
        source="fallback",
    )
