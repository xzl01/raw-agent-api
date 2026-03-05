"""Stub/Wrapper for LLM API calls (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class LLMClient:
    """Thin wrapper around the OpenAI chat completions endpoint.

    The actual ``openai`` package is imported lazily so that the class can be
    instantiated (and mocked) without an active API key in tests.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        self.model = model
        self._api_key = api_key

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: List of OpenAI-format message dicts.
            tools: Optional list of tool definitions (JSON schema).
            tool_choice: How the model should choose tools ("auto", "none", or
                a specific tool).

        Returns:
            The raw response dict from the OpenAI API (the first choice's
            message object).
        """
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = client.chat.completions.create(**kwargs)
        # Return the first choice message as a plain dict
        choice = response.choices[0].message
        return {
            "role": choice.role,
            "content": choice.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.tool_calls
                ]
                if choice.tool_calls
                else None
            ),
        }
