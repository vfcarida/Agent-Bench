"""Model adapter implementations."""

from agent_bench.models.stub import StubModelAdapter

__all__ = ["StubModelAdapter"]
from typing import Type, Any

# Lazy imports to avoid requiring optional dependencies
def get_openai_adapter() -> Type[Any]:
    from agent_bench.models.openai_adapter import OpenAIModelAdapter
    return OpenAIModelAdapter

def get_anthropic_adapter() -> Type[Any]:
    from agent_bench.models.anthropic_adapter import AnthropicModelAdapter
    return AnthropicModelAdapter

def get_vllm_adapter() -> Type[Any]:
    """Get VLLMModelAdapter class (requires: pip install agent-bench[vllm])."""
    from agent_bench.models.vllm_adapter import VLLMModelAdapter
    return VLLMModelAdapter

def get_huggingface_adapter() -> Type[Any]:
    """Get HuggingFacePipelineAdapter class (requires: pip install agent-bench[huggingface])."""
    from agent_bench.models.huggingface_adapter import HuggingFacePipelineAdapter
    return HuggingFacePipelineAdapter

def get_peft_adapter() -> Type[Any]:
    """Get PEFTModelAdapter class (requires: pip install agent-bench[peft])."""
    from agent_bench.models.peft_adapter import PEFTModelAdapter
    return PEFTModelAdapter

def get_adapter_manager() -> Type[Any]:
    """Get AdapterManager class (requires: pip install agent-bench[peft])."""
    from agent_bench.models.adapter_manager import AdapterManager
    return AdapterManager
