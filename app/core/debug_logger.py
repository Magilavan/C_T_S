"""Debug logger for agent pipeline stages.

When DEBUG_AGENT_PIPELINE is enabled, this formats and logs every agent step
with standardized headers and fields (without exposing hidden chain-of-thought).
"""
import logging
import json
from app.core.config import settings

logger = logging.getLogger("agent_pipeline_debug")


def _format_val(val) -> str:
    if val is None:
        return "None"
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, indent=2)
        except Exception:
            return str(val)
    return str(val).strip()


def log_agent_stage(
    agent_name: str,
    input_data: str | dict | None = None,
    model: str | None = None,
    prompt: str | None = None,
    tools_called: list | str | None = None,
    tool_args: dict | str | None = None,
    tool_result: str | None = None,
    retrieved_context: str | list | None = None,
    raw_model_output: str | None = None,
    parsed_output: str | dict | None = None,
    output_passed_to_next: str | dict | None = None,
):
    if not getattr(settings, "debug_agent_pipeline", True):
        return

    banner = "=" * 50
    sub_banner = "=" * len(agent_name)

    lines = [
        "",
        banner,
        f"AGENT: {agent_name.upper()}",
        f"======={sub_banner}",
        "",
        "INPUT:",
        _format_val(input_data),
        "",
        "MODEL:",
        _format_val(model or settings.groq_model),
        "",
        "PROMPT/TEMPLATE:",
        _format_val(prompt),
        "",
        "TOOLS CALLED:",
        _format_val(tools_called or "None"),
        "",
        "TOOL ARGUMENTS:",
        _format_val(tool_args or "None"),
        "",
        "TOOL RESULT:",
        _format_val(tool_result or "None"),
        "",
        "RETRIEVED CONTEXT:",
        _format_val(retrieved_context or "None"),
        "",
        "RAW MODEL OUTPUT:",
        _format_val(raw_model_output),
        "",
        "PARSED OUTPUT:",
        _format_val(parsed_output),
        "",
        "OUTPUT PASSED TO NEXT AGENT:",
        _format_val(output_passed_to_next),
        "",
        banner,
        "",
    ]

    log_str = "\n".join(lines)
    logger.info(log_str)
    # Also print to stdout for terminal view
    print(log_str, flush=True)
