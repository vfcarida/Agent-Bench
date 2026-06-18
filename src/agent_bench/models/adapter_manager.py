"""Adapter manager for dynamic PEFT (LoRA) hot-swapping.

Loads a base model once into VRAM and seamlessly swaps LoRA adapter
weights between evaluation cases, avoiding the cost of reloading
the full 7B/13B base model for each adapter.

Requires: pip install agent-bench[peft]
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_bench.core.adapters import ModelAdapter, ModelResponse

logger = structlog.get_logger()

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _check_peft_available() -> None:
    """Raise ImportError with install instructions if PEFT is not available."""
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is not installed. Install with: pip install agent-bench[peft]\n"
            "This requires: peft, transformers, and torch."
        )


@dataclass
class AdapterInfo:
    """Metadata for a loaded LoRA adapter."""

    name: str
    path: str
    loaded_at: float = 0.0
    memory_bytes: int = 0


class AdapterManager:
    """Manages a base model with multiple hot-swappable LoRA adapters.

    The base model is loaded once into GPU memory. LoRA adapters
    (~50-100MB each) are loaded/unloaded dynamically via PEFT's
    model.load_adapter() and model.set_adapter() methods.

    Args:
        base_model_path: Local path or HF Hub ID for the base model.
        device: Target device (auto/cpu/cuda/cuda:0).
        torch_dtype: Data type (auto/float16/bfloat16/float32).
        trust_remote_code: Whether to trust remote code.
    """

    def __init__(
        self,
        base_model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        _check_peft_available()

        self._base_model_path = base_model_path
        self._device = device
        self._trust_remote_code = trust_remote_code

        # Resolve dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._torch_dtype = dtype_map.get(torch_dtype, "auto")

        self._model: Any = None
        self._tokenizer: Any = None
        self._adapters: dict[str, AdapterInfo] = {}
        self._active_adapter: str | None = None
        self._loaded = False

        logger.info(
            "adapter_manager_init",
            base_model=base_model_path,
            device=device,
        )

    def _ensure_base_loaded(self) -> None:
        """Load the base model and tokenizer if not already loaded."""
        if self._loaded:
            return

        logger.info("base_model_loading", model=self._base_model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._base_model_path,
            trust_remote_code=self._trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
        }
        if self._torch_dtype != "auto":
            model_kwargs["torch_dtype"] = self._torch_dtype

        if self._device == "auto":
            model_kwargs["device_map"] = "auto"
        elif self._device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            model_kwargs["device_map"] = {"": self._device}

        self._model = AutoModelForCausalLM.from_pretrained(
            self._base_model_path,
            **model_kwargs,
        )

        self._loaded = True
        logger.info("base_model_loaded", model=self._base_model_path)

    def load_adapter(self, adapter_name: str, adapter_path: str) -> None:
        """Load a LoRA adapter into the base model.

        Args:
            adapter_name: Logical name for the adapter (e.g., 'math-lora').
            adapter_path: Path to the adapter weights directory.
        """
        self._ensure_base_loaded()

        if adapter_name in self._adapters:
            logger.warning(
                "adapter_already_loaded",
                adapter=adapter_name,
            )
            return

        logger.info("adapter_loading", adapter=adapter_name, path=adapter_path)
        t0 = time.perf_counter()

        # Check if model already has PEFT adapters
        if isinstance(self._model, PeftModel):
            self._model.load_adapter(adapter_path, adapter_name=adapter_name)
        else:
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_path,
                adapter_name=adapter_name,
            )

        load_time = time.perf_counter() - t0
        self._adapters[adapter_name] = AdapterInfo(
            name=adapter_name,
            path=adapter_path,
            loaded_at=time.time(),
        )

        logger.info(
            "adapter_loaded",
            adapter=adapter_name,
            load_time_ms=load_time * 1000,
        )

    def set_active_adapter(self, adapter_name: str) -> None:
        """Set the active LoRA adapter for subsequent generations.

        Args:
            adapter_name: Name of a previously loaded adapter.

        Raises:
            ValueError: If adapter is not loaded.
        """
        if adapter_name not in self._adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not loaded. "
                f"Available: {list(self._adapters.keys())}"
            )

        if self._active_adapter == adapter_name:
            return

        self._model.set_adapter(adapter_name)
        self._active_adapter = adapter_name
        logger.info("adapter_activated", adapter=adapter_name)

    def unload_adapter(self, adapter_name: str) -> None:
        """Unload a LoRA adapter to free memory.

        Args:
            adapter_name: Name of the adapter to unload.
        """
        if adapter_name not in self._adapters:
            logger.warning("adapter_not_found", adapter=adapter_name)
            return

        if self._active_adapter == adapter_name:
            self._active_adapter = None

        # PEFT doesn't have a clean unload for individual adapters,
        # but we can delete from our registry
        if isinstance(self._model, PeftModel) and hasattr(self._model, "delete_adapter"):
            self._model.delete_adapter(adapter_name)

        del self._adapters[adapter_name]
        logger.info("adapter_unloaded", adapter=adapter_name)

    def list_adapters(self) -> list[str]:
        """Return names of all loaded adapters."""
        return list(self._adapters.keys())

    def get_active_adapter(self) -> str | None:
        """Return the name of the currently active adapter."""
        return self._active_adapter

    def get_adapter_info(self, adapter_name: str) -> AdapterInfo | None:
        """Return metadata for a loaded adapter."""
        return self._adapters.get(adapter_name)

    @property
    def tokenizer(self) -> Any:
        """Access the tokenizer (for prompt formatting)."""
        self._ensure_base_loaded()
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Access the underlying model."""
        self._ensure_base_loaded()
        return self._model

    @property
    def base_model_path(self) -> str:
        return self._base_model_path

    def _format_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Format messages using tokenizer's chat template or fallback."""
        self._ensure_base_loaded()

        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def generate_sync(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        """Synchronous generation with the currently active adapter."""
        self._ensure_base_loaded()

        if seed is not None:
            torch.manual_seed(seed)

        prompt = self._format_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        if hasattr(self._model, "device"):
            device = self._model.device
            if hasattr(device, "type") and device.type != "meta":
                input_ids = input_ids.to(device)

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

        new_token_ids = output_ids[0][tokens_in:]
        tokens_out = len(new_token_ids)
        response_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)

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
            time_to_first_token_ms=latency_ms,
            thinking_content=thinking_content,
            raw={
                "model": self._base_model_path,
                "active_adapter": self._active_adapter,
                "provider": "peft",
            },
        )

    async def generate_async(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        """Async wrapper around synchronous generation."""
        return await asyncio.to_thread(
            self.generate_sync, messages, temperature, max_tokens, seed
        )
