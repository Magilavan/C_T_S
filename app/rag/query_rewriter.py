from langchain_groq import ChatGroq
from app.core.config import settings, groq_llm_kwargs

REWRITE_PROMPT = """You rewrite a user's follow-up question into a standalone \
question, using the conversation history for context. Do not answer the \
question. Only output the rewritten standalone question, nothing else.

If the question is already standalone, output it unchanged.
If the history mentions a specific drug and the new question doesn't name one, \
include that drug's name in the rewritten question.

Conversation history:
{history}

New message: {message}

Rewritten standalone question:"""


def rewrite_query(message: str, history_text: str, llm: ChatGroq | None = None) -> str:
    llm = llm or ChatGroq(**groq_llm_kwargs(temperature=0.0))
    prompt = REWRITE_PROMPT.format(history=history_text, message=message)
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        # If the LLM fails (rate limit, network, etc.), fall back to the
        # original user message so the pipeline can continue deterministically.
        return message.strip()
