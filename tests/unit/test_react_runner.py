"""Unit tests for autonomous multi-turn ReAct reasoning loop in DefaultAgentRunner."""

from typing import Any

import pytest

from agent_bench.core.adapters import ModelAdapter, ModelResponse
from agent_bench.core.artifacts import TraceEventType
from agent_bench.core.scenarios import Task
from agent_bench.runners.case_runner import DefaultAgentRunner, DefaultTaskEnvironment


class StatefulReActModel(ModelAdapter):
    """Model that demonstrates sequential multi-turn reasoning across observations."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def model_id(self) -> str:
        return "stateful_react_model"

    @property
    def provider(self) -> str:
        return "mock"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        self.call_count += 1

        # Check if there is any tool observation in history
        has_tool_res = any(m.get("role") == "tool" for m in messages)

        if not has_tool_res:
            # Turn 1: Decide to check balance first
            return ModelResponse(
                content="I need to check your balance first.",
                tool_calls=[{"name": "check_balance", "arguments": {}}],
                tokens_in=30,
                tokens_out=20,
                latency_ms=15.0,
            )

        # Check which tool outputs exist
        tool_names = [m.get("name") for m in messages if m.get("role") == "tool"]

        if "check_balance" in tool_names and "execute_pix_transfer" not in tool_names:
            # Turn 2: Balance verified, proceed with transfer
            return ModelResponse(
                content="Balance is sufficient. Executing transfer.",
                tool_calls=[{"name": "execute_pix_transfer", "arguments": {"amount": 150.0}}],
                tokens_in=60,
                tokens_out=25,
                latency_ms=15.0,
            )

        # Turn 3: Transfer finished, give final summary without further tool calls
        return ModelResponse(
            content="Transfer of $150.00 has been completed successfully.",
            tool_calls=[],
            tokens_in=90,
            tokens_out=30,
            latency_ms=12.0,
        )


class InfiniteLoopModel(ModelAdapter):
    """Simulates an agent that continuously calls a tool on every turn."""

    @property
    def model_id(self) -> str:
        return "infinite_tool_model"

    @property
    def provider(self) -> str:
        return "mock"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        # Returns distinct tool arguments each time
        step = len(messages)
        return ModelResponse(
            content=f"Calling tool step {step}",
            tool_calls=[{"name": "request_user_confirmation", "arguments": {"step": step}}],
            tokens_in=10,
            tokens_out=10,
            latency_ms=5.0,
        )


@pytest.mark.asyncio
async def test_multiturn_react_reasoning_flow() -> None:
    model = StatefulReActModel()
    runner = DefaultAgentRunner(system_id="react_system", model=model)
    env = DefaultTaskEnvironment()

    task = Task(
        task_id="react_task_01",
        domain="pix_assist",
        name="Multi-turn PIX transfer",
        description="Verify multi-turn tool calling and state updates",
        input_messages=[{"role": "user", "content": "Please send 150 PIX"}],
        initial_state={"balance": 1000.0},
    )

    result, traces = await runner.run_task(task, env, max_steps=5)

    # 3 model calls: check_balance -> execute_pix_transfer -> final response
    assert model.call_count == 3
    assert result["response"] == "Transfer of $150.00 has been completed successfully."
    assert "check_balance" in result["tools_called"]
    assert "execute_pix_transfer" in result["tools_called"]

    # Verify environment state was mutated
    assert result["final_state"]["balance"] == 850.0
    assert result["final_state"]["transfer_completed"] is True

    # Verify token and latency accumulation
    assert result["tokens_in"] == 180  # 30 + 60 + 90
    assert result["tokens_out"] == 75  # 20 + 25 + 30
    assert result["latency_ms"] == pytest.approx(42.0)

    # Verify trace events
    tool_call_events = [t for t in traces if t.event_type == TraceEventType.TOOL_CALL]
    tool_output_events = [t for t in traces if t.event_type == TraceEventType.TOOL_OUTPUT]
    assert len(tool_call_events) == 2
    assert len(tool_output_events) == 2


@pytest.mark.asyncio
async def test_react_runner_respects_max_steps() -> None:
    model = InfiniteLoopModel()
    runner = DefaultAgentRunner(system_id="infinite_system", model=model)
    env = DefaultTaskEnvironment()

    task = Task(
        task_id="loop_task_01",
        domain="pix_assist",
        name="Infinite loop test",
        description="Verify max_steps bound",
        input_messages=[{"role": "user", "content": "Hello"}],
        initial_state={},
    )

    result, traces = await runner.run_task(task, env, max_steps=4)

    # Must not exceed max_steps=4
    assert len(result["tool_calls"]) <= 4
    tool_call_events = [t for t in traces if t.event_type == TraceEventType.TOOL_CALL]
    assert len(tool_call_events) <= 4
