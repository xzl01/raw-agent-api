"""Shared pytest fixtures for the RAW Agent API test suite."""

from __future__ import annotations

import io
import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> TestClient:
    """A synchronous TestClient wrapping the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock image helpers
# ---------------------------------------------------------------------------

def make_rgb_image(width: int = 64, height: int = 64) -> Image.Image:
    """Create a small solid-colour RGB Pillow Image for testing."""
    arr = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return JPEG bytes of a small test image."""
    buf = io.BytesIO()
    make_rgb_image(width, height).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_image() -> Image.Image:
    return make_rgb_image()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    return make_jpeg_bytes()


# ---------------------------------------------------------------------------
# Mock LLM response
# ---------------------------------------------------------------------------

def make_llm_response(
    exposure: float = 1.2,
    contrast: float = 1.1,
    saturation: float = 1.0,
    white_balance_red: float = 1.05,
    white_balance_blue: float = 0.95,
) -> Dict[str, Any]:
    """Build a fake LLM response dict that mimics the OpenAI tool-call format."""
    arguments = json.dumps(
        {
            "exposure": exposure,
            "contrast": contrast,
            "saturation": saturation,
            "white_balance_red": white_balance_red,
            "white_balance_blue": white_balance_blue,
        }
    )
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "adjust_image",
                    "arguments": arguments,
                },
            }
        ],
    }


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    return make_llm_response()


# ---------------------------------------------------------------------------
# Mock rawpy to avoid needing real .ARW files
# ---------------------------------------------------------------------------

def _make_mock_rawpy(width: int = 64, height: int = 64):
    """Return a MagicMock that behaves like a rawpy.RawPy context manager."""
    mock_raw = MagicMock()
    rgb_array = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
    mock_raw.postprocess.return_value = rgb_array
    mock_raw.other_params = {
        "iso_speed": 400,
        "shutter": 0.01,
        "aperture": 5.6,
        "focal_len": 50.0,
    }
    mock_raw.__enter__ = MagicMock(return_value=mock_raw)
    mock_raw.__exit__ = MagicMock(return_value=False)
    return mock_raw


@pytest.fixture
def mock_rawpy():
    """Patch rawpy.imread so tests never need real ARW files."""
    mock_raw = _make_mock_rawpy()
    with patch("rawpy.imread", return_value=mock_raw) as patched:
        yield patched, mock_raw
