from app.core.config import settings


def test_groq_settings_present():
    assert hasattr(settings, "groq_api_key")
    assert hasattr(settings, "groq_model")
    assert hasattr(settings, "groq_vision_model")
    assert settings.groq_model
    assert settings.groq_vision_model
