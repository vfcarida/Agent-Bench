"""OpenAI model adapter."""

import os
import time
from typing import Any

import httpx

from agent_bench.core.adapters import ModelAdapter, ModelResponse


class OpenAIModelAdapter(ModelAdapter):
    """Adapter for OpenAI API (GPT-4o, GPT-4, etc.)."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self._model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "openai"

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
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]

        t0 = time.perf_counter()
        ttft: float = 0.0

        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        ttft = latency_ms  # non-streaming: ttft == latency

        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""

        tool_calls_raw = message.get("tool_calls") or []
        tool_calls = [
            {
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }
            for tc in tool_calls_raw
        ]

        usage = data.get("usage", {})

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft,
            raw=data,
        )

    async def close(self) -> None:
        await self._client.aclose()
