"""Unit tests for Clean Architecture protocols."""

from typing import Any

import pytest

from agent_bench.core.adapters import JudgeVerdict, ToolCallResult
from agent_bench.core.artifacts import TraceEvent
from agent_bench.core.protocols import AgentRunner, Evaluator, TaskEnvironment
from agent_bench.core.scenarios import Task


class SampleEnvironment:
    """Sample class implementing TaskEnvironment protocol."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    @property
    def environment_id(self) -> str:
        return "sample_env"

    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
        self._state = dict(initial_state or {})
        return self._state

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        return ToolCallResult(tool_name=tool_name, arguments=arguments, output="ok")

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def cleanup(self) -> None:
        self._state.clear()


class SampleRunner:
    """Sample class implementing AgentRunner protocol."""

    @property
    def system_id(self) -> str:
        return "sample_runner"

    @property
    def architecture(self) -> str:
        return "test_arch"

    async def run_task(
        self,
        task: Task,
        environment: TaskEnvironment,
        *,
        max_steps: int = 10,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], list[TraceEvent]]:
        environment.reset(task.initial_state)
        return {"response": "done", "final_state": environment.get_state()}, []


class SampleEvaluator:
    """Sample class implementing Evaluator protocol."""

    @property
    def evaluator_id(self) -> str:
        return "sample_evaluator"

    async def evaluate(
        self,
        task: Task,
        execution_result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict:
        return JudgeVerdict(
            score=1.0,
            passed=True,
            reasoning="Passed",
            judge_id=self.evaluator_id,
            criteria="test",
        )


def test_protocol_runtime_checks() -> None:
    env = SampleEnvironment()
    runner = SampleRunner()
    evaluator = SampleEvaluator()

    assert isinstance(env, TaskEnvironment)
    assert isinstance(runner, AgentRunner)
    assert isinstance(evaluator, Evaluator)


@pytest.mark.asyncio
async def test_sample_runner_execution() -> None:
    env = SampleEnvironment()
    runner = SampleRunner()
    task = Task(
        task_id="T001",
        domain="test",
        name="Test Task",
        description="Desc",
        input_messages=[{"role": "user", "content": "Hi"}],
        initial_state={"key": "val"},
    )

    result, _traces = await runner.run_task(task, env)
    assert result["response"] == "done"
    assert result["final_state"] == {"key": "val"}
