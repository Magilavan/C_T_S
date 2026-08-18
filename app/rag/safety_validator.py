"""Differentiator #1: Safety Validator gate.

Decision locked in: hard refusal, not silent rewrite. If a drafted answer
fails a safety check, it is replaced entirely with a refusal message —
rewriting risks the model re-introducing the same unsafe content, and a
hard refusal is more demonstrably safe to reviewers/judges.

This runs AFTER groundedness passes and BEFORE the answer is shown, i.e.
it is a second, independent gate — an answer can be well-grounded in the
source text and still need to be blocked (e.g. if a user asks the system
to combine that grounded info into an unsafe personal instruction).
"""
from dataclasses import dataclass
from langchain_groq import ChatGroq
from app.core.config import settings, groq_llm_kwargs

SAFETY_PROMPT = """You are a clinical-safety reviewer for a drug-information \
chatbot response. The chatbot answers only from FDA prescribing information, \
but must still be checked for safety issues before being shown to a user.

Flag the answer if it does ANY of the following:
- Recommends a specific dose/action for an individual as if personalized \
medical advice, rather than stating what the label says in general terms.
- Advises changing, stopping, or combining medications without stating that \
a healthcare provider should be consulted.
- States something as fact that contradicts a contraindication or boxed \
warning implied by the source context.
- Omits a safety-critical caveat (e.g. answers a dosage question for a \
population — renal/hepatic impairment, pediatric, elderly — without \
noting when the general dose does NOT apply) when the source context \
contains that caveat.
- Provides guidance that could enable self-harm or medication misuse.

Output ONLY a JSON object, no other text:
{{"safe": <true|false>, "reason": "<short explanation, empty string if safe>"}}

Drafted answer:
{answer}

Source context available to the drafter:
{context_block}

JSON:"""

REFUSAL_MESSAGE = (
    "I can't provide that response as drafted because it didn't pass our safety "
    "check ({reason}). Please consult a healthcare provider or pharmacist for "
    "guidance specific to your situation. If you'd like, I can share the relevant "
    "general label information instead."
)


@dataclass
class SafetyResult:
    safe: bool
    reason: str


def check_safety(answer: str, chunks: list[dict], llm: ChatGroq | None = None) -> SafetyResult:
    llm = llm or ChatGroq(**groq_llm_kwargs(temperature=0.0))
    context_block = "\n\n".join(c["text"] for c in chunks) if chunks else "(no context)"
    prompt = SAFETY_PROMPT.format(answer=answer, context_block=context_block)

    import json
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return SafetyResult(safe=bool(data.get("safe", False)), reason=data.get("reason", ""))
    except Exception:
        # fail closed: an unparseable safety check result is treated as unsafe
        return SafetyResult(safe=False, reason="safety check failed to parse — blocked by default")


def apply_safety_gate(answer: str, chunks: list[dict], llm: ChatGroq | None = None) -> str:
    result = check_safety(answer, chunks, llm)
    if result.safe:
        return answer
    return REFUSAL_MESSAGE.format(reason=result.reason or "unspecified safety concern")
