"""PEFT model adapter: wraps AdapterManager as a standard ModelAdapter.

Each PEFTModelAdapter instance is bound to a specific LoRA adapter
but shares the same base model via the AdapterManager. This allows
Agent-Bench to treat each adapter as a separate model for evaluation
while the base model stays loaded in VRAM.
"""

from __future__ import annotations

from typing import Any

import structlog

from agent_bench.core.adapters import ModelAdapter, ModelResponse

logger = structlog.get_logger()


class PEFTModelAdapter(ModelAdapter):
    """ModelAdapter that delegates to AdapterManager for LoRA-based inference.

    Multiple PEFTModelAdapter instances can share the same AdapterManager,
    each representing a different LoRA adapter on the same base model.

    Args:
        adapter_manager: The shared AdapterManager instance.
        adapter_name: Name of the LoRA adapter to use for this instance.
        model_id: Override for the model_id property. Defaults to
            "{base_model}:{adapter_name}".
    """

    def __init__(
        self,
        adapter_manager: Any,  # AdapterManager — avoid circular import
        adapter_name: str,
        model_id: str | None = None,
    ):
        self._manager = adapter_manager
        self._adapter_name = adapter_name
        self._model_id_override = (
            model_id or f"{adapter_manager.base_model_path}:{adapter_name}"
        )

    @property
    def model_id(self) -> str:
        return self._model_id_override

    @property
    def provider(self) -> str:
        return "peft"

    @property
    def adapter_name(self) -> str:
        return self._adapter_name

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        """Generate using the bound LoRA adapter.

        Sets the active adapter on the shared AdapterManager before
        delegating to its generate method. This ensures that even if
        multiple PEFTModelAdapters are used interleaved, the correct
        adapter is always active.
        """
        self._manager.set_active_adapter(self._adapter_name)

        response = await self._manager.generate_async(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        logger.debug(
            "peft_generation_complete",
            adapter=self._adapter_name,
            tokens_out=response.tokens_out,
            latency_ms=response.latency_ms,
        )

        return response
