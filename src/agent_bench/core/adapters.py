"""Abstract adapters for models, agents, tools, retrieval, and judges."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_bench.core.artifacts import TraceEvent


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    thinking_content: str | None = None  # Raw <think> block content if present
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[dict[str, Any]]
    query: str
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ToolCallResult:
    tool_name: str
    arguments: dict[str, Any]
    output: Any
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class JudgeVerdict:
    score: float
    passed: bool
    reasoning: str
    judge_id: str
    criteria: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Adapter for a single LLM provider/model."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse: ...


class AgentSystemAdapter(ABC):
    """Adapter for a complete agent system (model + tools + retrieval + orchestration)."""

    @property
    @abstractmethod
    def system_id(self) -> str: ...

    @property
    @abstractmethod
    def architecture(self) -> str: ...

    @abstractmethod
    async def run(
        self,
        task_input: dict[str, Any],
        *,
        tools: list["ToolAdapter"] | None = None,
        retrieval: "RetrievalAdapter | None" = None,
        max_steps: int = 10,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], list[TraceEvent]]: ...


class ToolAdapter(ABC):
    """Adapter for a callable tool available to the agent."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult: ...


class RetrievalAdapter(ABC):
    """Adapter for document retrieval systems."""

    @property
    @abstractmethod
    def retriever_id(self) -> str: ...

    @abstractmethod
    async def retrieve(
        self, query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> RetrievalResult: ...


class JudgeAdapter(ABC):
    """Adapter for evaluation judges (deterministic or LLM-based)."""

    @property
    @abstractmethod
    def judge_id(self) -> str: ...

    @property
    @abstractmethod
    def judge_type(self) -> str:
        """One of: deterministic, semantic, grounding, policy, composite."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        task: Any,
        result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict: ...
