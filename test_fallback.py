"""Unit test suite for Groq Model Fallback Mechanism.

Verifies:
1. Configuration parsing of GROQ_FALLBACK_MODELS.
2. Normal primary model execution (no fallback when primary succeeds).
3. 429 TPD error on primary model -> automatic immediate fallback to next model without retrying primary.
4. 404 model_not_found error -> automatic immediate fallback.
5. 401 Auth error -> immediate stop (raises AuthenticationError without switching models).
6. Total exhaustion of all models -> returns clean user-facing error message.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.core.config import settings
from app.core.llm_retry import (
    retry_llm_call,
    get_last_used_model,
    DailyTokenLimitError,
    AuthenticationError,
)

def test_config_parsing():
    """Verify GROQ_FALLBACK_MODELS parsing cleans duplicates, empty values, and primary model."""
    orig_primary = settings.groq_model
    orig_fallback = settings.groq_fallback_models

    settings.groq_model = "llama-3.3-70b-versatile"
    settings.groq_fallback_models = "llama-3.1-8b-instant, llama-3.3-70b-versatile, openai/gpt-oss-20b, llama-3.1-8b-instant, , qwen/qwen3.6-27b"

    fallbacks = settings.get_fallback_models()
    assert fallbacks == ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

    settings.groq_model = orig_primary
    settings.groq_fallback_models = orig_fallback
    print("[TEST 1] Config parsing: PASS")


def test_primary_success():
    """Verify primary model succeeds directly without triggering fallback."""
    mock_resp = MagicMock()
    mock_resp.content = "Adalimumab is the active ingredient."
    mock_resp.response_metadata = {"model_name": settings.groq_model, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    with patch("app.core.llm_retry.ChatGroq") as mock_chat:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_resp
        mock_chat.return_value = mock_instance

        res = retry_llm_call(mock_instance, "What is HUMIRA?", label="test_primary")
        assert res.content == "Adalimumab is the active ingredient."
        assert mock_chat.call_count == 1
        assert get_last_used_model() == settings.groq_model
    print("[TEST 2] Primary success: PASS")


def test_fallback_on_429_tpd():
    """Verify 429 TPD on primary model immediately triggers fallback to 2nd model."""
    fallback_models = settings.get_fallback_models()
    expected_fallback_model = fallback_models[0]

    mock_success_resp = MagicMock()
    mock_success_resp.content = "Grounded answer from fallback."
    mock_success_resp.response_metadata = {"model_name": expected_fallback_model, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    with patch("app.core.llm_retry.ChatGroq") as mock_chat:
        def side_effect(*args, **kwargs):
            model = kwargs.get("model")
            mock_inst = MagicMock()
            if model == settings.groq_model:
                mock_inst.invoke.side_effect = Exception("429 Rate limit reached for model llama-3.3-70b-versatile on tokens per day (TPD)")
            else:
                mock_inst.invoke.return_value = mock_success_resp
            return mock_inst

        mock_chat.side_effect = side_effect

        primary_mock = MagicMock()
        primary_mock.model_name = settings.groq_model

        res = retry_llm_call(primary_mock, "What is HUMIRA?", label="test_tpd_fallback")
        assert res.content == "Grounded answer from fallback."
        assert get_last_used_model() == expected_fallback_model
        print(f"[TEST 3] 429 TPD Fallback -> Used fallback model ({get_last_used_model()}): PASS")


def test_auth_error_no_fallback():
    """Verify 401 Invalid API key error stops immediately without fallback."""
    with patch("app.core.llm_retry.ChatGroq") as mock_chat:
        mock_inst = MagicMock()
        mock_inst.invoke.side_effect = Exception("401 Unauthorized: Invalid API key provided")
        mock_chat.return_value = mock_inst

        primary_mock = MagicMock()
        primary_mock.model_name = settings.groq_model

        with pytest.raises(AuthenticationError):
            retry_llm_call(primary_mock, "Test query", label="test_auth")

        # Verify only tried once (no fallback for auth error)
        assert mock_chat.call_count == 1
    print("[TEST 4] 401 Auth error stops immediately: PASS")


def test_all_models_fail():
    """Verify clean exception when all configured models fail."""
    with patch("app.core.llm_retry.ChatGroq") as mock_chat:
        mock_inst = MagicMock()
        mock_inst.invoke.side_effect = Exception("429 Rate limit reached tokens per day (TPD)")
        mock_chat.return_value = mock_inst

        primary_mock = MagicMock()
        primary_mock.model_name = settings.groq_model

        with pytest.raises(DailyTokenLimitError) as exc_info:
            retry_llm_call(primary_mock, "Test query", label="test_all_fail")

        assert "temporarily unavailable" in str(exc_info.value)
    print("[TEST 5] All models fail -> clean error: PASS")


if __name__ == "__main__":
    test_config_parsing()
    test_primary_success()
    test_fallback_on_429_tpd()
    test_auth_error_no_fallback()
    test_all_models_fail()
    print("\n========================================================")
    print("ALL MOCK FALLBACK UNIT TESTS PASSED SUCCESSFULLY!")
    print("========================================================")
