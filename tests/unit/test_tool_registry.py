"""Unit tests for DomainToolRegistry and decoupled state mutation."""

from typing import Any

import pytest

from agent_bench.core.adapters import ToolAdapter, ToolCallResult
from agent_bench.runners.case_runner import DefaultTaskEnvironment
from agent_bench.tools.registry import DomainToolRegistry, default_tool_registry


class MockCustomTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "custom_order_tool"

    @property
    def schema(self) -> dict[str, Any]:
        return {"name": "custom_order_tool", "parameters": {}}

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"order_id": "ord_999", "status": "shipped"},
            success=True,
        )


def test_builtin_tools_available() -> None:
    """Verify standard domain tools are registered and instantiable."""
    tools = default_tool_registry.registered_tools()
    assert "check_balance" in tools
    assert "execute_pix_transfer" in tools
    assert "get_portfolio_summary" in tools
    assert "check_firewall_rules" in tools

    balance_tool = default_tool_registry.get_tool("check_balance")
    assert balance_tool is not None
    assert balance_tool.name == "check_balance"


def test_builtin_mutators_registered() -> None:
    """Verify built-in state mutators are present."""
    pix_mutator = default_tool_registry.get_mutator("execute_pix_transfer")
    assert pix_mutator is not None

    state: dict[str, Any] = {"balance": 1000.0}
    result = ToolCallResult(
        tool_name="execute_pix_transfer",
        arguments={"amount": 250.0},
        output={"status": "completed"},
        success=True,
    )
    pix_mutator(state, {"amount": 250.0}, result)
    assert state["balance"] == 750.0
    assert state["transfer_completed"] is True


def test_custom_tool_and_mutator_registration() -> None:
    """Verify custom domain tools and mutators can be registered dynamically."""
    registry = DomainToolRegistry()
    registry.register_tool("custom_order_tool", MockCustomTool)

    def _mutate_order(state: dict[str, Any], arguments: dict[str, Any], res: ToolCallResult) -> None:
        if res.success and isinstance(res.output, dict):
            state["order_placed"] = True
            state["order_id"] = res.output.get("order_id")

    registry.register_mutator("custom_order_tool", _mutate_order)

    tool = registry.get_tool("custom_order_tool")
    assert tool is not None
    assert isinstance(tool, MockCustomTool)

    mutator = registry.get_mutator("custom_order_tool")
    assert mutator is not None

    test_state: dict[str, Any] = {}
    mutator(
        test_state,
        {},
        ToolCallResult(
            tool_name="custom_order_tool",
            arguments={},
            output={"order_id": "ord_123"},
            success=True,
        ),
    )
    assert test_state["order_placed"] is True
    assert test_state["order_id"] == "ord_123"


@pytest.mark.asyncio
async def test_task_environment_uses_registry() -> None:
    """Verify DefaultTaskEnvironment executes tools and applies state mutations via registry."""
    env = DefaultTaskEnvironment()
    env.reset({"balance": 500.0})

    res = await env.execute_tool("execute_pix_transfer", {"amount": 100.0})
    assert res.success is True

    current_state = env.get_state()
    assert current_state["balance"] == 400.0
    assert current_state["transfer_completed"] is True
