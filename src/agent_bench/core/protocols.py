"""Clean Architecture Protocols for Agent-Bench.

This module defines Python Protocols establishing strict boundaries between the
benchmark execution framework, agent runners, task execution environments, and
evaluators.
"""

from typing import Any, Protocol, runtime_checkable

from agent_bench.core.adapters import JudgeVerdict, ToolCallResult
from agent_bench.core.artifacts import TraceEvent
from agent_bench.core.scenarios import Task


@runtime_checkable
class TaskEnvironment(Protocol):
    """Protocol for managing task execution state, tool calls, and sandbox lifecycle."""

    @property
    def environment_id(self) -> str:
        """Unique identifier of the environment instance."""
        ...

    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Resets environment to initial state and returns state dictionary."""
        ...

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        """Executes a tool within the environment and returns ToolCallResult."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Returns current environment state snapshot."""
        ...

    def cleanup(self) -> None:
        """Cleans up resources allocated by the environment."""
        ...


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol for executing agent tasks independently of LLM frameworks (LangChain, CrewAI, etc.)."""

    @property
    def system_id(self) -> str:
        """Unique identifier of the agent runner system."""
        ...

    @property
    def architecture(self) -> str:
        """Architecture tag (e.g., 'react', 'crewai', 'langchain', 'custom_llm')."""
        ...

    async def run_task(
        self,
        task: Task,
        environment: TaskEnvironment,
        *,
        max_steps: int = 10,
        seed: int | None = None,
        user_simulator: Any = None,
    ) -> tuple[dict[str, Any], list[TraceEvent]]:
        """Executes a task in the provided environment.

        Args:
            task: Task scenario definition.
            environment: Interactive environment for tool execution and state observation.
            max_steps: Maximum interaction steps allowed.
            seed: Random seed for deterministic generation.

        Returns:
            Tuple containing final execution result dictionary and collected TraceEvent list.
        """
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluating agent task outputs and traces."""

    @property
    def evaluator_id(self) -> str:
        """Unique identifier of the evaluator implementation."""
        ...

    async def evaluate(
        self,
        task: Task,
        execution_result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict:
        """Evaluates execution result against task requirements and returns verdict.

        Args:
            task: Task scenario containing expected final state and gold references.
            execution_result: Final result output from AgentRunner execution.
            traces: Full sequence of TraceEvents emitted during execution.

        Returns:
            JudgeVerdict containing score, pass/fail status, and explanation.
        """
        ...


# Re-export UserSimulator and UserTurn for Clean Architecture protocol access
from agent_bench.core.user_simulator import UserSimulator, UserTurn  # noqa: E402

__all__ = ["AgentRunner", "TaskEnvironment", "Evaluator", "UserSimulator", "UserTurn"]
