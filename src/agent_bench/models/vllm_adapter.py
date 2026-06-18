"""vLLM model adapter for high-throughput local inference.

Requires: pip install agent-bench[vllm]
Note: vLLM only supports Linux with CUDA GPUs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from agent_bench.core.adapters import ModelAdapter, ModelResponse

logger = structlog.get_logger()

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _check_vllm_available() -> None:
    """Raise ImportError with install instructions if vLLM is not available."""
    if not VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is not installed. Install it with: pip install agent-bench[vllm]\n"
            "Note: vLLM requires Linux with CUDA-capable GPUs."
        )


class VLLMModelAdapter(ModelAdapter):
    """Adapter for local model inference via vLLM's AsyncLLMEngine.

    Supports continuous batching for high-throughput evaluation of
    Athena's local Hugging Face models (7B, 13B parameter weights).

    Args:
        model_path: Local path to HF model directory or Hub model ID.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0–1.0).
        max_model_len: Maximum sequence length. None = auto-detect.
        dtype: Data type for model weights (auto/float16/bfloat16).
        trust_remote_code: Whether to trust remote code in HF models.
        quantization: Quantization method (None/awq/gptq/squeezellm).
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        quantization: str | None = None,
        model_id: str | None = None,
    ):
        _check_vllm_available()

        self._model_path = model_path
        self._model_id_override = model_id or model_path
        self._engine: AsyncLLMEngine | None = None
        self._request_counter = 0

        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
        )
        if max_model_len is not None:
            engine_args.max_model_len = max_model_len

        self._engine_args = engine_args
        logger.info(
            "vllm_adapter_init",
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    async def _ensure_engine(self) -> AsyncLLMEngine:
        """Lazily initialize the vLLM engine on first use."""
        if self._engine is None:
            logger.info("vllm_engine_starting", model=self._model_path)
            self._engine = AsyncLLMEngine.from_engine_args(self._engine_args)
            logger.info("vllm_engine_ready", model=self._model_path)
        return self._engine

    @property
    def model_id(self) -> str:
        return self._model_id_override

    @property
    def provider(self) -> str:
        return "vllm"

    def _format_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Format messages into a single prompt string.

        For chat-formatted models, a PromptFormatter should be used
        upstream. This is a fallback for simple concatenation.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        engine = await self._ensure_engine()

        prompt = self._format_prompt(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        self._request_counter += 1
        request_id = f"vllm-{self._request_counter}"

        t0 = time.perf_counter()
        ttft: float | None = None
        full_output = ""

        async for request_output in engine.generate(prompt, sampling_params, request_id):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            if request_output.outputs:
                full_output = request_output.outputs[0].text

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract token counts from the final output
        tokens_in = 0
        tokens_out = 0
        if request_output and request_output.outputs:  # noqa: F821 — defined in loop
            output = request_output.outputs[0]
            tokens_out = len(output.token_ids) if hasattr(output, "token_ids") else len(full_output) // 4
        if request_output and hasattr(request_output, "prompt_token_ids"):
            tokens_in = len(request_output.prompt_token_ids)
        else:
            tokens_in = len(prompt) // 4  # rough estimate

        # Detect <think> blocks
        thinking_content = None
        import re

        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        think_matches = think_pattern.findall(full_output)
        if think_matches:
            thinking_content = "\n".join(think_matches)

        return ModelResponse(
            content=full_output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft or latency_ms,
            thinking_content=thinking_content,
            raw={
                "model": self._model_path,
                "request_id": request_id,
                "provider": "vllm",
            },
        )

    async def close(self) -> None:
        """Shutdown the vLLM engine and release GPU memory."""
        if self._engine is not None:
            # vLLM engines don't have a standard close method;
            # we rely on garbage collection for cleanup
            self._engine = None
            logger.info("vllm_engine_shutdown", model=self._model_path)
