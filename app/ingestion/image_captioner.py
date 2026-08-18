"""Captions extracted PDF images at ingestion time (not query time) so the
captions become normal searchable text chunks — decision locked in per the
project's Phase 2b requirements: caption-at-ingestion, not on-demand."""
import base64
import io
import logging

from langchain_groq import ChatGroq
from app.core.config import settings

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "This image is from an FDA prescribing information document. "
    "Describe factually what it shows — e.g. an injection-device diagram, "
    "a dosing table rendered as an image, a chemical structure, or a "
    "storage/handling illustration. Be literal and specific about any text, "
    "numbers, or steps visible in the image. Do not add medical interpretation "
    "beyond what is visibly depicted."
)


def _image_to_data_url(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


_VISION_DISABLED = False


def reset_vision_state():
    global _VISION_DISABLED
    _VISION_DISABLED = False


def caption_image(pil_image, client=None) -> str:
    global _VISION_DISABLED
    if _VISION_DISABLED:
        return ""

    if not settings.groq_api_key or not settings.groq_vision_model:
        if not _VISION_DISABLED:
            logger.warning("Groq vision model or API key is not configured — skipping image captioning")
            _VISION_DISABLED = True
        return ""

    try:
        client = client or ChatGroq(
            model=settings.groq_vision_model,
            temperature=0.0,
            max_tokens=300,
            groq_api_key=settings.groq_api_key,
        )
        data_url = _image_to_data_url(pil_image)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
        response = client.invoke([message])
        return response.content.strip()
    except Exception as e:
        if not _VISION_DISABLED:
            logger.warning(f"Vision model '{settings.groq_vision_model}' is unavailable or failed ({e}) — skipping image captioning")
            _VISION_DISABLED = True
        return ""

