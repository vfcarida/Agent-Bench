"""Hugging Face Pipeline model adapter for CPU/basic GPU inference.

Requires: pip install agent-bench[huggingface]
Works on all platforms (Linux, macOS, Windows) with CPU or CUDA GPUs.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import structlog

from agent_bench.core.adapters import ModelAdapter, ModelResponse

logger = structlog.get_logger()

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _check_hf_available() -> None:
    """Raise ImportError with install instructions if transformers is not available."""
    if not HF_AVAILABLE:
        raise ImportError(
            "Hugging Face transformers is not installed. "
            "Install with: pip install agent-bench[huggingface]"
        )


def _resolve_torch_dtype(dtype_str: str) -> Any:
    """Convert string dtype to torch dtype."""
    if not HF_AVAILABLE:
        return None
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str, "auto")


class HuggingFacePipelineAdapter(ModelAdapter):
    """Adapter for local model inference via Hugging Face transformers pipeline.

    Suitable for hardware that doesn't support vLLM (CPU, basic GPUs,
    macOS with MPS). Inference is synchronous but wrapped in asyncio.to_thread()
    for non-blocking evaluation.

    Args:
        model_path: Local path to HF model directory or Hub model ID.
        device: Device to load model on (auto/cpu/cuda/cuda:0/mps).
        torch_dtype: Data type (auto/float16/bfloat16/float32).
        load_in_8bit: Enable 8-bit quantization via bitsandbytes.
        load_in_4bit: Enable 4-bit quantization via bitsandbytes.
        trust_remote_code: Whether to trust remote code in HF models.
        model_id: Override for the model_id property.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        model_id: str | None = None,
    ):
        _check_hf_available()

        self._model_path = model_path
        self._model_id_override = model_id or model_path
        self._device = device
        self._torch_dtype = _resolve_torch_dtype(torch_dtype)
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._trust_remote_code = trust_remote_code

        # Lazy-loaded model and tokenizer
        self._tokenizer: Any = None
        self._model: Any = None
        self._pipeline: Any = None
        self._loaded = False

        logger.info(
            "hf_adapter_init",
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            quantization="8bit" if load_in_8bit else ("4bit" if load_in_4bit else "none"),
        )

    def _ensure_loaded(self) -> None:
        """Lazily load model and tokenizer on first use."""
        if self._loaded:
            return

        logger.info("hf_model_loading", model=self._model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=self._trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
        }
        if self._torch_dtype != "auto":
            model_kwargs["torch_dtype"] = self._torch_dtype
        if self._load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self._load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        # Determine device_map
        if self._device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self._device in ("cpu",):
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            model_kwargs["device_map"] = {"": self._device}

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **model_kwargs,
        )

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._loaded = True
        logger.info("hf_model_loaded", model=self._model_path)

    @property
    def model_id(self) -> str:
        return self._model_id_override

    @property
    def provider(self) -> str:
        return "huggingface"

    def _format_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Format messages using tokenizer's chat template or fallback."""
        self._ensure_loaded()

        # Try to use the tokenizer's built-in chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: simple concatenation
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _generate_sync(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> tuple[str, int, int, float]:
        """Synchronous generation — called via asyncio.to_thread."""
        self._ensure_loaded()

        if seed is not None:
            import torch as _torch

            _torch.manual_seed(seed)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Move to model device
        if hasattr(self._model, "device"):
            input_ids = input_ids.to(self._model.device)

        tokens_in = input_ids.shape[1]

        t0 = time.perf_counter()

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only new tokens
        new_token_ids = output_ids[0][tokens_in:]
        tokens_out = len(new_token_ids)
        response_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return response_text, tokens_in, tokens_out, latency_ms

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        prompt = self._format_prompt(messages)

        response_text, tokens_in, tokens_out, latency_ms = await asyncio.to_thread(
            self._generate_sync, prompt, temperature, max_tokens, seed
        )

        # Detect <think> blocks
        thinking_content = None
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        think_matches = think_pattern.findall(response_text)
        if think_matches:
            thinking_content = "\n".join(think_matches)

        return ModelResponse(
            content=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            time_to_first_token_ms=latency_ms,  # non-streaming
            thinking_content=thinking_content,
            raw={
                "model": self._model_path,
                "provider": "huggingface",
                "device": self._device,
            },
        )

    @property
    def tokenizer(self) -> Any:
        """Expose tokenizer for prompt template detection."""
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Expose underlying model for PEFT adapter loading."""
        self._ensure_loaded()
        return self._model
