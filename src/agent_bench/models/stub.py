"""Stub model adapter for testing without API calls."""

from typing import Any

from agent_bench.core.adapters import ModelAdapter, ModelResponse


class StubModelAdapter(ModelAdapter):
    """A deterministic stub model for testing the framework."""

    def __init__(self, model_id: str = "stub-model", responses: list[str] | None = None):
        self._model_id = model_id
        self._responses = responses or ["[STUB] This is a stub response."]
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "stub"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        response_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return ModelResponse(
            content=response_text,
            tokens_in=sum(len(m.get("content", "")) // 4 for m in messages),
            tokens_out=len(response_text) // 4,
            latency_ms=50.0,
            time_to_first_token_ms=10.0,
        )
