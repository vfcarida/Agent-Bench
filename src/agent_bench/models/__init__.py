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
