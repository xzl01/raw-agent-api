"""Unit tests for the Photographer Agent."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from agent.photographer import DEFAULT_PARAMS, PhotographerAgent, _parse_tool_call
from core.raw_engine import RawMetadata


def _make_tool_call_response(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Helper: build a fake LLM message dict with a single tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": "adjust_image",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# _parse_tool_call
# ---------------------------------------------------------------------------

class TestParseToolCall:
    def test_parses_all_fields(self) -> None:
        response = _make_tool_call_response(
            {
                "exposure": 1.3,
                "contrast": 0.9,
                "saturation": 1.2,
                "white_balance_red": 1.1,
                "white_balance_blue": 0.8,
            }
        )
        params = _parse_tool_call(response)
        assert params["exposure"] == pytest.approx(1.3)
        assert params["contrast"] == pytest.approx(0.9)
        assert params["saturation"] == pytest.approx(1.2)
        assert params["white_balance_red"] == pytest.approx(1.1)
        assert params["white_balance_blue"] == pytest.approx(0.8)

    def test_partial_fields_use_defaults(self) -> None:
        response = _make_tool_call_response({"exposure": 1.5})
        params = _parse_tool_call(response)
        assert params["exposure"] == pytest.approx(1.5)
        assert params["contrast"] == pytest.approx(DEFAULT_PARAMS["contrast"])
        assert params["saturation"] == pytest.approx(DEFAULT_PARAMS["saturation"])

    def test_no_tool_calls_returns_defaults(self) -> None:
        response = {"role": "assistant", "content": "No adjustments.", "tool_calls": None}
        params = _parse_tool_call(response)
        assert params == DEFAULT_PARAMS

    def test_empty_tool_calls_returns_defaults(self) -> None:
        response = {"role": "assistant", "content": None, "tool_calls": []}
        params = _parse_tool_call(response)
        assert params == DEFAULT_PARAMS

    def test_unknown_tool_returns_defaults(self) -> None:
        response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "unknown_tool", "arguments": "{}"},
                }
            ],
        }
        params = _parse_tool_call(response)
        assert params == DEFAULT_PARAMS

    def test_invalid_json_returns_defaults(self) -> None:
        response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "adjust_image", "arguments": "NOT JSON"},
                }
            ],
        }
        params = _parse_tool_call(response)
        assert params == DEFAULT_PARAMS

    def test_non_numeric_value_uses_default(self) -> None:
        response = _make_tool_call_response({"exposure": "bright"})
        params = _parse_tool_call(response)
        assert params["exposure"] == pytest.approx(DEFAULT_PARAMS["exposure"])


# ---------------------------------------------------------------------------
# PhotographerAgent
# ---------------------------------------------------------------------------

class TestPhotographerAgent:
    def _make_agent(self, llm_response: Dict[str, Any]) -> PhotographerAgent:
        mock_client = MagicMock()
        mock_client.chat.return_value = llm_response
        return PhotographerAgent(llm_client=mock_client)

    def test_determine_adjustments_returns_params(self, mock_llm_response) -> None:
        agent = self._make_agent(mock_llm_response)
        metadata = RawMetadata(iso=400, shutter_speed=0.01, aperture=5.6, focal_length=50.0)
        params = agent.determine_adjustments(metadata)

        assert "exposure" in params
        assert "contrast" in params
        assert "saturation" in params
        assert "white_balance_red" in params
        assert "white_balance_blue" in params

    def test_llm_called_with_tools(self, mock_llm_response) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_llm_response
        agent = PhotographerAgent(llm_client=mock_client)

        metadata = RawMetadata(iso=100)
        agent.determine_adjustments(metadata)

        call_kwargs = mock_client.chat.call_args
        assert call_kwargs is not None
        # tools should be passed
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "tools" in kwargs
        assert len(kwargs["tools"]) > 0

    def test_metadata_present_in_user_message(self, mock_llm_response) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_llm_response
        agent = PhotographerAgent(llm_client=mock_client)

        metadata = RawMetadata(iso=3200, camera_make="Sony", camera_model="A7 IV")
        agent.determine_adjustments(metadata)

        messages = mock_client.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "3200" in user_msg["content"]
        assert "Sony" in user_msg["content"]

    def test_agent_falls_back_to_defaults_on_no_tool_call(self) -> None:
        no_tool_response = {"role": "assistant", "content": "Just use defaults.", "tool_calls": None}
        agent = self._make_agent(no_tool_response)
        params = agent.determine_adjustments(RawMetadata())
        assert params == DEFAULT_PARAMS
