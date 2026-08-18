"""Smart retry with automatic Groq model fallback and TPD exhaustion handling.

Fallback strategy:
  401 / invalid API key      → fail immediately (AuthenticationError, do NOT fallback)
  429 TPD / 404 / unavailable → immediate model fallback (NO retry on exhausted model)
  500 / 502 / 503 / 504      → retry with backoff on current model
  timeout                    → retry with backoff on current model
"""
import time
import logging
import re
from langchain_groq import ChatGroq
from app.core.config import settings, groq_llm_kwargs

logger = logging.getLogger(__name__)
_token_logger = logging.getLogger("drugbot.tokens")

_LAST_USED_MODEL = settings.groq_model


def get_last_used_model() -> str:
    """Return the model name used by the most recent successful LLM call."""
    return _LAST_USED_MODEL


class DailyTokenLimitError(Exception):
    """Raised when Groq daily token quota or fallback chain is exhausted."""
    pass


class AuthenticationError(Exception):
    """Raised when Groq API key is invalid — do NOT fallback or retry."""
    pass


def _is_auth_error(err_str: str) -> bool:
    """True for invalid API key or unauthorized errors."""
    if any(t in err_str for t in ("unauthorized", "invalid api key", "invalid_api_key")):
        return True
    return bool(re.search(r"\b401\b", err_str))


def _is_fallback_trigger_error(err_str: str) -> bool:
    """True when the error indicates model exhaustion or unavailability."""
    return any(t in err_str for t in (
        "429", "rate_limit_exceeded", "tokens per day", "tpd", "daily limit",
        "404", "model_not_found", "model unavailable", "does not exist"
    ))


def _is_transient_error(err_str: str) -> bool:
    """True for genuine transient server/network errors."""
    return any(t in err_str for t in (
        "500", "502", "503", "504",
        "timeout", "unavailable", "overloaded", "connection"
    ))


def log_token_usage(response, label: str = "LLM call", model_name: str = ""):
    """Log token usage from a LangChain response if metadata is available."""
    global _LAST_USED_MODEL
    try:
        if model_name:
            _LAST_USED_MODEL = model_name

        usage = None
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage") or response.response_metadata.get("token_usage")
            res_model = response.response_metadata.get("model_name") or response.response_metadata.get("model")
            if res_model:
                _LAST_USED_MODEL = res_model
        elif hasattr(response, "usage_metadata"):
            usage = response.usage_metadata

        if usage:
            input_tok = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
            output_tok = usage.get("completion_tokens") or usage.get("output_tokens", 0)
            total_tok = usage.get("total_tokens", input_tok + output_tok)

            _token_logger.info(
                "[%s] model=%s input=%d output=%d total=%d",
                label, _LAST_USED_MODEL, input_tok, output_tok, total_tok,
            )
            return {"input_tokens": input_tok, "output_tokens": output_tok, "total_tokens": total_tok}
    except Exception:
        pass
    return None


def retry_llm_call(func, *args, label: str = "LLM call", **kwargs):
    """Call *func* with automatic model fallback and smart transient retry.

    Fallback Order: Primary model -> Fallback 1 -> Fallback 2 ...
    - 401 (Auth Error) -> Stop immediately, raise AuthenticationError.
    - 429 TPD / 404 (Quota / Model unavailable) -> Immediate fallback to next model (NO retry on exhausted model).
    - 5xx / Timeout (Transient) -> Retry on SAME model up to max_retries.
    """
    global _LAST_USED_MODEL

    # Extract target ChatGroq instance if bound
    llm_obj = getattr(func, "__self__", None) if hasattr(func, "__self__") else None
    try:
        if isinstance(func, ChatGroq):
            llm_obj = func
    except TypeError:
        if type(func).__name__ == "ChatGroq":
            llm_obj = func

    primary_model = getattr(llm_obj, "model_name", None) or getattr(llm_obj, "model", None) or settings.groq_model
    fallback_models = settings.get_fallback_models()

    # Build candidate models list: primary model first, followed by fallbacks (deduped)
    candidate_models = [primary_model]
    for fb in fallback_models:
        if fb not in candidate_models:
            candidate_models.append(fb)

    temp = getattr(llm_obj, "temperature", 0.1) if llm_obj else 0.1
    max_tok = getattr(llm_obj, "max_tokens", None) if llm_obj else settings.max_output_tokens

    error_log = []

    for model_idx, model_name in enumerate(candidate_models):
        is_primary = (model_idx == 0)

        # Instantiate ChatGroq for current model candidate
        current_llm = ChatGroq(**groq_llm_kwargs(model=model_name, temperature=temp, max_tokens=max_tok))

        max_retries = 2
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                # If args passed, pass to current_llm.invoke
                result = current_llm.invoke(*args, **kwargs)
                _LAST_USED_MODEL = model_name
                log_token_usage(result, label=label, model_name=model_name)

                if not is_primary:
                    logger.warning(
                        "Primary model '%s' failed. Fallback model '%s' result: SUCCESS",
                        primary_model, model_name
                    )
                else:
                    logger.info("Primary model '%s' result: SUCCESS", model_name)

                return result

            except Exception as e:
                err_str = str(e).lower()

                # 1. Auth error -> STOP immediately
                if _is_auth_error(err_str):
                    logger.error("Authentication error (invalid API key) on model '%s': %s", model_name, e)
                    raise AuthenticationError(f"Invalid API key or unauthorized request: {e}") from e

                # 2. Fallback trigger error (429 TPD / 404) -> DO NOT retry this model, move to next model immediately
                if _is_fallback_trigger_error(err_str):
                    status_lbl = "429 TPD" if ("429" in err_str or "tpd" in err_str or "rate_limit" in err_str) else "404 model_not_found"
                    logger.warning(
                        "Model '%s' failed with %s: %s. Moving to next fallback model.",
                        model_name, status_lbl, e
                    )
                    error_log.append(f"{model_name}: {status_lbl}")
                    break  # Exit attempt loop to advance model candidate

                # 3. Genuine transient error -> Retry on current model
                if _is_transient_error(err_str):
                    logger.warning(
                        "Model '%s' transient error (attempt %d/%d). Retrying in %.1fs: %s",
                        model_name, attempt + 1, max_retries, backoff, e
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                # 4. Unknown error -> retry once on current model then fallback
                logger.warning(
                    "Model '%s' unknown error (attempt %d/%d). Retrying in %.1fs: %s",
                    model_name, attempt + 1, max_retries, backoff, e
                )
                time.sleep(backoff)
                backoff *= 2.0

    err_summary = "; ".join(error_log) if error_log else "All models failed"
    logger.error("All configured AI models failed: %s", err_summary)
    raise DailyTokenLimitError("All configured AI models are temporarily unavailable. Please try again later.")
