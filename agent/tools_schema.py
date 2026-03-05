"""JSON Schema definitions for LLM tool calling (OpenAI format)."""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Individual tool schemas
# ---------------------------------------------------------------------------

ADJUST_IMAGE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "adjust_image",
        "description": (
            "Apply photographic adjustments to an image. "
            "Call this function with the desired correction values based on image metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exposure": {
                    "type": "number",
                    "description": (
                        "Exposure (brightness) multiplier. "
                        "1.0 = unchanged, >1.0 = brighter, <1.0 = darker. Typical range 0.5–2.0."
                    ),
                    "default": 1.0,
                },
                "contrast": {
                    "type": "number",
                    "description": (
                        "Contrast multiplier. "
                        "1.0 = unchanged. Typical range 0.5–2.0."
                    ),
                    "default": 1.0,
                },
                "saturation": {
                    "type": "number",
                    "description": (
                        "Color saturation multiplier. "
                        "0.0 = grayscale, 1.0 = unchanged. Typical range 0.0–2.0."
                    ),
                    "default": 1.0,
                },
                "white_balance_red": {
                    "type": "number",
                    "description": (
                        "Red channel scale for white balance correction. "
                        "1.0 = unchanged. Typical range 0.5–2.0."
                    ),
                    "default": 1.0,
                },
                "white_balance_blue": {
                    "type": "number",
                    "description": (
                        "Blue channel scale for white balance correction. "
                        "1.0 = unchanged. Typical range 0.5–2.0."
                    ),
                    "default": 1.0,
                },
            },
            "required": [],
        },
    },
}

# The list of tools exposed to the LLM
TOOLS: List[Dict[str, Any]] = [ADJUST_IMAGE_TOOL]
