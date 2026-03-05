"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import make_llm_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_rgb_array(width: int = 64, height: int = 64) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _make_mock_raw(rgb_array: np.ndarray):
    mock_raw = MagicMock()
    mock_raw.postprocess.return_value = rgb_array
    mock_raw.other_params = {}
    mock_raw.__enter__ = MagicMock(return_value=mock_raw)
    mock_raw.__exit__ = MagicMock(return_value=False)
    return mock_raw


def _dummy_arw_bytes() -> bytes:
    """Return some dummy bytes that represent a 'RAW file' in tests."""
    return b"FAKE_ARW_CONTENT"


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Root (frontend) route
# ---------------------------------------------------------------------------

class TestRootRoute:
    def test_root_serves_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# /process endpoint
# ---------------------------------------------------------------------------

class TestProcessEndpoint:
    def test_returns_200_with_mock(self, client: TestClient) -> None:
        rgb_array = _mock_rgb_array()
        mock_raw = _make_mock_raw(rgb_array)
        mock_response = make_llm_response(exposure=1.2)

        with (
            patch("rawpy.imread", return_value=mock_raw),
            patch("agent.photographer.PhotographerAgent.determine_adjustments") as mock_agent,
        ):
            mock_agent.return_value = {
                "exposure": 1.2,
                "contrast": 1.0,
                "saturation": 1.0,
                "white_balance_red": 1.0,
                "white_balance_blue": 1.0,
            }

            response = client.post(
                "/process",
                files={"file": ("test.arw", _dummy_arw_bytes(), "image/x-sony-arw")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Processing successful."
        assert "adjustments" in body
        assert "jpeg_b64" in body
        assert body["filename"] == "test.arw"

    def test_jpeg_b64_is_valid_jpeg(self, client: TestClient) -> None:
        rgb_array = _mock_rgb_array()
        mock_raw = _make_mock_raw(rgb_array)

        with (
            patch("rawpy.imread", return_value=mock_raw),
            patch("agent.photographer.PhotographerAgent.determine_adjustments") as mock_agent,
        ):
            mock_agent.return_value = {
                "exposure": 1.0,
                "contrast": 1.0,
                "saturation": 1.0,
                "white_balance_red": 1.0,
                "white_balance_blue": 1.0,
            }
            response = client.post(
                "/process",
                files={"file": ("test.arw", _dummy_arw_bytes(), "image/x-sony-arw")},
            )

        body = response.json()
        jpeg_bytes = base64.b64decode(body["jpeg_b64"])
        assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic

    def test_invalid_file_returns_422(self, client: TestClient) -> None:
        with patch("rawpy.imread", side_effect=Exception("Not a RAW file")):
            response = client.post(
                "/process",
                files={"file": ("bad.txt", b"not a raw file", "text/plain")},
            )
        assert response.status_code == 422
        assert "Failed to decode" in response.json()["detail"]

    def test_adjustments_shape(self, client: TestClient) -> None:
        rgb_array = _mock_rgb_array()
        mock_raw = _make_mock_raw(rgb_array)
        expected_params = {
            "exposure": 1.3,
            "contrast": 0.9,
            "saturation": 1.1,
            "white_balance_red": 1.05,
            "white_balance_blue": 0.95,
        }

        with (
            patch("rawpy.imread", return_value=mock_raw),
            patch("agent.photographer.PhotographerAgent.determine_adjustments") as mock_agent,
        ):
            mock_agent.return_value = expected_params
            response = client.post(
                "/process",
                files={"file": ("test.arw", _dummy_arw_bytes(), "image/x-sony-arw")},
            )

        body = response.json()
        assert body["adjustments"] == expected_params


# ---------------------------------------------------------------------------
# /batch-process endpoint
# ---------------------------------------------------------------------------

class TestBatchProcessEndpoint:
    def test_batch_returns_results_list(self, client: TestClient) -> None:
        rgb_array = _mock_rgb_array()
        mock_raw = _make_mock_raw(rgb_array)
        params = {
            "exposure": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "white_balance_red": 1.0,
            "white_balance_blue": 1.0,
        }

        with (
            patch("rawpy.imread", return_value=mock_raw),
            patch("agent.photographer.PhotographerAgent.determine_adjustments") as mock_agent,
        ):
            mock_agent.return_value = params
            response = client.post(
                "/batch-process",
                files=[
                    ("files", ("a.arw", _dummy_arw_bytes(), "image/x-sony-arw")),
                    ("files", ("b.arw", _dummy_arw_bytes(), "image/x-sony-arw")),
                ],
            )

        assert response.status_code == 200
        body = response.json()
        assert "results" in body
        assert len(body["results"]) == 2
        for result in body["results"]:
            assert result["message"] == "Processing successful."

    def test_batch_empty_files_returns_400(self, client: TestClient) -> None:
        response = client.post("/batch-process", files=[])
        assert response.status_code in (400, 422)

    def test_batch_partial_failure(self, client: TestClient) -> None:
        rgb_array = _mock_rgb_array()
        mock_raw = _make_mock_raw(rgb_array)
        params = {
            "exposure": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "white_balance_red": 1.0,
            "white_balance_blue": 1.0,
        }

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Bad RAW data")
            return mock_raw

        with (
            patch("rawpy.imread", side_effect=side_effect),
            patch("agent.photographer.PhotographerAgent.determine_adjustments") as mock_agent,
        ):
            mock_agent.return_value = params
            response = client.post(
                "/batch-process",
                files=[
                    ("files", ("good.arw", _dummy_arw_bytes(), "image/x-sony-arw")),
                    ("files", ("bad.arw", _dummy_arw_bytes(), "image/x-sony-arw")),
                ],
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 2
        # First result should succeed
        assert body["results"][0]["message"] == "Processing successful."
        # Second result should have an error
        assert "error" in body["results"][1]
