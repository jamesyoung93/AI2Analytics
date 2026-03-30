"""LLM client abstraction for Databricks Model Serving endpoints."""

from __future__ import annotations

import json
import re
import warnings
from typing import Any


class LLMClient:
    """Thin wrapper around Databricks OpenAI-compatible LLM endpoints.

    Usage (Databricks notebook)::

        from ai2analytics.llm import LLMClient
        client = LLMClient(endpoint="databricks-qwen3-next-80b-a3b-instruct")

    Or provide your own OpenAI-compatible client::

        client = LLMClient(endpoint="gpt-4o", openai_client=my_client)
    """

    def __init__(
        self,
        endpoint: str,
        openai_client: Any | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._make_databricks_client()

    def _make_databricks_client(self) -> Any:
        try:
            from databricks.sdk import WorkspaceClient
            return WorkspaceClient().serving_endpoints.get_open_ai_client()
        except ImportError:
            raise ImportError(
                "databricks-sdk is required for auto-client. "
                "Install with: pip install ai2analytics[databricks] "
                "Or pass openai_client= explicitly."
            )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            response = self._client.chat.completions.create(
                model=self.endpoint,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            )
        return response.choices[0].message.content

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> dict | list:
        """Call LLM and parse response as JSON, stripping markdown fences."""
        raw = self.call(system_prompt, user_prompt, **kwargs)
        cleaned = strip_markdown_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON object or array
            match = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from LLM response:\n{cleaned[:500]}")


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and <think> blocks from LLM output."""
    text = text.strip()
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
