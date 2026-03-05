"""Photographer Agent: uses LLM tool calling to determine image adjustments."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agent.llm_client import LLMClient
from agent.tools_schema import TOOLS
from core.raw_engine import RawMetadata

# Default adjustment parameters (no-op)
DEFAULT_PARAMS: Dict[str, float] = {
    "exposure": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "white_balance_red": 1.0,
    "white_balance_blue": 1.0,
}

_SYSTEM_PROMPT = (
    "You are an expert photographer and photo editor. "
    "Given basic camera metadata about a RAW image, decide what adjustments "
    "should be made to produce a well-exposed, correctly white-balanced JPEG. "
    "Always call the `adjust_image` tool with the recommended parameters."
)


class PhotographerAgent:
    """Agent that analyses image metadata and calls the LLM to get adjustment params."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def determine_adjustments(self, metadata: RawMetadata) -> Dict[str, float]:
        """Ask the LLM which adjustments to apply given image metadata.

        Args:
            metadata: RawMetadata extracted from the source image.

        Returns:
            A dict of adjustment parameters matching ``DEFAULT_PARAMS`` keys.
            Falls back to defaults if the LLM response cannot be parsed.
        """
        user_message = _build_user_message(metadata)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = self.llm_client.chat(messages=messages, tools=TOOLS)
        return _parse_tool_call(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(metadata: RawMetadata) -> str:
    return (
        f"Camera: {metadata.camera_make} {metadata.camera_model}\n"
        f"ISO: {metadata.iso}\n"
        f"Shutter speed: {metadata.shutter_speed}s\n"
        f"Aperture: f/{metadata.aperture}\n"
        f"Focal length: {metadata.focal_length}mm\n"
        "Please suggest the best image adjustments."
    )


def _parse_tool_call(response: Dict[str, Any]) -> Dict[str, float]:
    """Parse the LLM response and extract adjustment parameters.

    Args:
        response: The message dict returned by :class:`LLMClient`.

    Returns:
        Adjustment parameters dict, falling back to defaults for missing keys.
    """
    tool_calls = response.get("tool_calls")
    if not tool_calls:
        return dict(DEFAULT_PARAMS)

    for call in tool_calls:
        if call.get("function", {}).get("name") == "adjust_image":
            try:
                args: Dict[str, Any] = json.loads(call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                return dict(DEFAULT_PARAMS)

            params = dict(DEFAULT_PARAMS)
            for key in DEFAULT_PARAMS:
                if key in args:
                    try:
                        params[key] = float(args[key])
                    except (TypeError, ValueError):
                        pass
            return params

    return dict(DEFAULT_PARAMS)
