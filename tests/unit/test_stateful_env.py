"""Unit tests for stateful tool execution in DefaultTaskEnvironment and DefaultAgentRunner."""

from typing import Any

import pytest

from agent_bench.core.adapters import ModelAdapter, ModelResponse, ToolAdapter, ToolCallResult
from agent_bench.core.protocols import TaskEnvironment
from agent_bench.core.scenarios import Task
from agent_bench.runners.case_runner import DefaultAgentRunner, DefaultTaskEnvironment


class CustomEchoTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "custom_echo"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "custom_echo",
            "description": "Echo input value",
            "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}},
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"echoed": arguments.get("msg", "")},
            success=True,
        )


class MockToolCallingModel(ModelAdapter):
    """Model adapter that simulates tool calling responses."""

    def __init__(self, tool_calls: list[dict[str, Any]], content: str = "Processing") -> None:
        self._tool_calls = tool_calls
        self._content = content

    @property
    def model_id(self) -> str:
        return "mock_tc_model"

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
        return ModelResponse(
            content=self._content,
            tool_calls=self._tool_calls,
            tokens_in=50,
            tokens_out=25,
            latency_ms=10.0,
        )


def test_task_environment_protocol_compliance() -> None:
    env = DefaultTaskEnvironment()
    assert isinstance(env, TaskEnvironment)
    assert env.environment_id == "default_env"


@pytest.mark.asyncio
async def test_tool_registration_and_execution() -> None:
    custom_tool = CustomEchoTool()
    env = DefaultTaskEnvironment(tools=[custom_tool])

    assert "custom_echo" in env.registered_tools

    res = await env.execute_tool("custom_echo", {"msg": "hello world"})
    assert res.success is True
    assert res.output == {"echoed": "hello world"}


@pytest.mark.asyncio
async def test_default_tool_lazy_execution() -> None:
    env = DefaultTaskEnvironment()
    # check_balance is not pre-registered, but should be lazily loaded from pix_tools
    res = await env.execute_tool("check_balance", {})
    assert res.success is True
    assert isinstance(res.output, dict)
    assert res.output.get("balance") == 5000.00


@pytest.mark.asyncio
async def test_fallback_unrecognized_tool() -> None:
    env = DefaultTaskEnvironment()
    res = await env.execute_tool("some_unknown_tool", {"foo": "bar"})
    assert res.success is True
    assert "Executed tool 'some_unknown_tool'" in str(res.output)


@pytest.mark.asyncio
async def test_default_pix_transfer_state_mutation() -> None:
    env = DefaultTaskEnvironment()
    env.reset({"balance": 1000.00, "account_id": "ACC123"})

    res = await env.execute_tool("execute_pix_transfer", {"amount": 250.00, "pix_key": "user@bank.com"})
    assert res.success is True

    state = env.get_state()
    assert state["balance"] == 750.00
    assert state["amount"] == 250.00
    assert state["transfer_completed"] is True
    assert "last_transaction" in state


@pytest.mark.asyncio
async def test_default_pix_confirmation_and_key_validation() -> None:
    env = DefaultTaskEnvironment()
    env.reset({})

    await env.execute_tool("request_user_confirmation", {"message": "Confirm?"})
    await env.execute_tool("validate_pix_key", {"pix_key": "valid_key_123"})

    state = env.get_state()
    assert state.get("confirmation_requested") is True
    assert state.get("key_validated") is True
    assert state.get("recipient_name") == "Joao Silva"


@pytest.mark.asyncio
async def test_custom_state_mutator_registration() -> None:
    env = DefaultTaskEnvironment()
    env.reset({"counter": 0})

    def increment_mutator(
        state: dict[str, Any], args: dict[str, Any], res: ToolCallResult
    ) -> None:
        state["counter"] += args.get("step", 1)
        state["last_step"] = args.get("step", 1)

    env.register_state_mutator("custom_step", increment_mutator)

    await env.execute_tool("custom_step", {"step": 5})
    assert env.get_state()["counter"] == 5
    assert env.get_state()["last_step"] == 5

    await env.execute_tool("custom_step", {"step": 3})
    assert env.get_state()["counter"] == 8


def test_reset_and_cleanup() -> None:
    env = DefaultTaskEnvironment()
    env.reset({"initial": 42})
    assert env.get_state() == {"initial": 42}

    env.reset({"new_initial": 99})
    assert env.get_state() == {"new_initial": 99}
    assert "initial" not in env.get_state()

    env.cleanup()
    assert env.get_state() == {}


@pytest.mark.asyncio
async def test_agent_runner_dispatches_tool_calls_to_environment() -> None:
    model = MockToolCallingModel(
        tool_calls=[
            {"name": "execute_pix_transfer", "arguments": {"amount": 100.0, "pix_key": "dest@test.com"}},
            {"name": "request_user_confirmation", "arguments": {"message": "Done"}},
        ]
    )
    runner = DefaultAgentRunner(system_id="test_system", model=model)
    env = DefaultTaskEnvironment()

    task = Task(
        task_id="task_state_01",
        domain="pix_assist",
        name="PIX state test",
        description="Verify tool dispatch",
        input_messages=[{"role": "user", "content": "Send 100 PIX"}],
        initial_state={"balance": 500.0},
    )

    result, _traces = await runner.run_task(task, env)

    assert result["response"] == "Processing"
    assert "execute_pix_transfer" in result["tools_called"]
    assert "request_user_confirmation" in result["tools_called"]

    final_state = result["final_state"]
    assert final_state["balance"] == 400.0
    assert final_state["transfer_completed"] is True
    assert final_state["confirmation_requested"] is True
