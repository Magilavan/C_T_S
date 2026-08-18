"""Standalone Groq connectivity health check.

Run this before testing the full RAG pipeline to confirm GROQ_API_KEY is
valid and the configured model responds:

    python scripts/check_groq.py

Exits 0 and prints "GROQ_TEST_SUCCESS" on success; exits 1 with a short
error category on failure. Never prints the API key.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings, groq_llm_kwargs  # noqa: E402


def main() -> int:
    if not settings.groq_api_key:
        print("FAIL: GROQ_API_KEY is not set (check your .env file)")
        return 1

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("FAIL: langchain-groq is not installed (pip install -r requirements.txt)")
        return 1

    try:
        llm = ChatGroq(**groq_llm_kwargs(temperature=0.0))
        response = llm.invoke("Respond with exactly: GROQ_TEST_SUCCESS")
        text = (response.content or "").strip()
    except Exception as exc:
        msg = str(exc).lower()
        if any(t in msg for t in ("401", "unauthorized", "invalid api key", "authentication")):
            print("FAIL: authentication error — check GROQ_API_KEY")
        elif any(t in msg for t in ("429", "rate limit")):
            print("FAIL: rate limited — try again shortly")
        elif "timeout" in msg:
            print("FAIL: request timed out")
        elif any(t in msg for t in ("model_not_found", "does not exist", "invalid request")):
            print(f"FAIL: model error — check GROQ_MODEL ({settings.groq_model})")
        else:
            print(f"FAIL: unexpected error — {type(exc).__name__}: {exc}")
        return 1

    if "GROQ_TEST_SUCCESS" in text:
        print("GROQ_TEST_SUCCESS")
        print(f"(model: {settings.groq_model})")
        return 0

    print(f"FAIL: model responded but not with the expected text. Got: {text!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
