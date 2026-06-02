"""Anthropic model adapter."""

import os
import time
from typing import Any

import httpx

from agent_bench.core.adapters import ModelAdapter, ModelResponse


class AnthropicModelAdapter(ModelAdapter):
    """Adapter for Anthropic API (Claude Sonnet, Opus, Haiku)."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
    ):
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "anthropic"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Extract system message if present
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            else:
                user_messages.append(msg)

        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            body["system"] = system_msg
        if tools:
            body["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {}),
                }
                for t in tools
            ]

        t0 = time.perf_counter()
        response = await self._client.post(
            f"{self._base_url}/v1/messages",
            json=body,
            headers=headers,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        response.raise_for_status()
        data = response.json()

        # Parse content blocks
        content_parts = []
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["input"],
                })

        content = "\n".join(content_parts)
        usage = data.get("usage", {})

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            time_to_first_token_ms=latency_ms,
            raw=data,
        )

    async def close(self) -> None:
        await self._client.aclose()
