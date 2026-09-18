"""Factory for resolving system configurations to ModelAdapters and AgentRunners."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent_bench.core.protocols import AgentRunner
from agent_bench.core.settings import settings
from agent_bench.runners.case_runner import DefaultAgentRunner

if TYPE_CHECKING:
    from agent_bench.core.adapters import ModelAdapter
    from agent_bench.core.config import BenchConfig, ModelConfig, SystemConfig


class ConfigError(Exception):
    """Raised when system resolution or adapter configuration fails."""


def build_model_adapter(model_cfg: ModelConfig) -> ModelAdapter:
    """Instantiate a ModelAdapter based on provider and ModelConfig."""
    provider = model_cfg.provider.lower()

    if provider == "stub":
        from agent_bench.models.stub import StubModelAdapter

        responses = model_cfg.parameters.get("responses") if model_cfg.parameters else None
        return StubModelAdapter(model_id=model_cfg.model_id, responses=responses)

    elif provider == "openai":
        from agent_bench.models.openai_adapter import OpenAIModelAdapter

        api_key = (
            os.environ.get(model_cfg.api_key_env)
            if model_cfg.api_key_env
            else None
        ) or settings.openai_api_key or ""
        base_url = model_cfg.endpoint or "https://api.openai.com/v1"
        timeout = float(model_cfg.parameters.get("timeout", 60.0)) if model_cfg.parameters else 60.0
        return OpenAIModelAdapter(
            model_id=model_cfg.model_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    elif provider == "anthropic":
        from agent_bench.models.anthropic_adapter import AnthropicModelAdapter

        api_key = (
            os.environ.get(model_cfg.api_key_env)
            if model_cfg.api_key_env
            else None
        ) or settings.anthropic_api_key or ""
        base_url = model_cfg.endpoint or "https://api.anthropic.com"
        timeout = float(model_cfg.parameters.get("timeout", 60.0)) if model_cfg.parameters else 60.0
        return AnthropicModelAdapter(
            model_id=model_cfg.model_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    elif provider == "internal":
        from agent_bench.models.openai_adapter import OpenAIModelAdapter

        api_key = (
            os.environ.get(model_cfg.api_key_env)
            if model_cfg.api_key_env
            else None
        ) or ""
        base_url = model_cfg.endpoint or "http://localhost:8000/v1"
        timeout = float(model_cfg.parameters.get("timeout", 60.0)) if model_cfg.parameters else 60.0
        return OpenAIModelAdapter(
            model_id=model_cfg.model_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    elif provider == "huggingface":
        from agent_bench.models.huggingface_adapter import HuggingFacePipelineAdapter

        model_path = model_cfg.model_path or model_cfg.model_id
        return HuggingFacePipelineAdapter(
            model_path=model_path,
            device=model_cfg.device,
            torch_dtype=model_cfg.torch_dtype,
            load_in_8bit=model_cfg.load_in_8bit,
            load_in_4bit=model_cfg.load_in_4bit,
            model_id=model_cfg.model_id,
        )

    elif provider == "vllm":
        from agent_bench.models.vllm_adapter import VLLMModelAdapter

        model_path = model_cfg.model_path or model_cfg.model_id
        return VLLMModelAdapter(
            model_path=model_path,
            tensor_parallel_size=model_cfg.tensor_parallel_size,
            gpu_memory_utilization=model_cfg.gpu_memory_utilization,
            dtype=model_cfg.torch_dtype,
        )

    else:
        raise ConfigError(
            f"Unsupported provider '{model_cfg.provider}' for model '{model_cfg.model_id}'."
        )


def build_agent_runner(system_id: str, config: BenchConfig) -> AgentRunner:
    """Resolve a system_id to an AgentRunner using the provided benchmark configuration.

    Resolution:
    1. Looks up SystemConfig matching system_id. If missing, raises ConfigError.
    2. If architecture is scripted, returns ScriptedAgentRunner.
    3. Looks up ModelConfig matching system.model. If missing, raises ConfigError.
    4. Builds the appropriate ModelAdapter for the provider.
    5. Returns DefaultAgentRunner configured with the adapter.
    """
    sys_cfg: SystemConfig | None = next(
        (s for s in config.systems if s.system_id == system_id), None
    )
    if sys_cfg is None:
        raise ConfigError(
            f"System '{system_id}' not found in configuration. "
            f"Available systems: {[s.system_id for s in config.systems]}"
        )

    if sys_cfg.architecture in ("scripted", "scripted_policy") or sys_cfg.model == "scripted":
        from agent_bench.runners.scripted import ScriptedAgentRunner

        return ScriptedAgentRunner(system_id=system_id)

    model_cfg: ModelConfig | None = next(
        (m for m in config.models if m.model_id == sys_cfg.model), None
    )
    if model_cfg is None:
        raise ConfigError(
            f"Model '{sys_cfg.model}' specified by system '{system_id}' not found in configuration. "
            f"Available models: {[m.model_id for m in config.models]}"
        )

    adapter = build_model_adapter(model_cfg)
    return DefaultAgentRunner(system_id=system_id, model=adapter)
