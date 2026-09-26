"""Unit tests for CallableAgentRunner and external framework bridging."""

import pytest

from agent_bench.core.artifacts import TraceEventType
from agent_bench.core.protocols import AgentRunner, TaskEnvironment
from agent_bench.core.scenarios import Task
from agent_bench.runners.callable_runner import CallableAgentRunner
from agent_bench.runners.case_runner import DefaultTaskEnvironment, execute_task


class TestCallableAgentRunner:
    """Verifies CallableAgentRunner adapts custom agents to Agent-Bench protocols."""

    def test_protocol_conformance(self):
        def dummy_agent(prompt: str) -> str:
            return "response"

        runner = CallableAgentRunner(dummy_agent)
        assert isinstance(runner, AgentRunner)

    @pytest.mark.asyncio
    async def test_async_string_returning_agent(self):
        async def my_agent(prompt: str) -> str:
            return f"Processed: {prompt}"

        task = Task(
            task_id="CALLABLE_001",
            domain="sme_business_advisor",
            name="Callable Test",
            description="Test basic prompt dispatch",
            input_messages=[{"role": "user", "content": "How do I optimize cash flow?"}],
        )
        env = DefaultTaskEnvironment()
        runner = CallableAgentRunner(my_agent)

        res, traces = await runner.run(task, env)

        assert res["response"] == "Processed: How do I optimize cash flow?"
        assert res["latency_ms"] >= 0.0
        assert any(t.event_type == TraceEventType.USER_MESSAGE for t in traces)
        assert any(t.event_type == TraceEventType.MODEL_RESPONSE for t in traces)

    @pytest.mark.asyncio
    async def test_agent_executing_tools_via_environment_proxy(self):
        async def tool_using_agent(task: Task, env: TaskEnvironment) -> dict:
            # Simulate external LangChain agent executing an environment tool
            tool_res = await env.execute_tool("validate_pix_key", {"pix_key": "user@bank.com"})
            return {
                "response": f"Key verified: {tool_res.success}",
                "tokens_in": 150,
                "tokens_out": 40,
                "cost_usd": 0.002,
            }

        task = Task(
            task_id="CALLABLE_002",
            domain="pix_assist",
            name="Tool Using Callable",
            description="Agent calls environment tool",
            input_messages=[{"role": "user", "content": "Validate key"}],
            allowed_tools=["validate_pix_key"],
        )
        env = DefaultTaskEnvironment()
        runner = CallableAgentRunner(tool_using_agent)

        res, traces = await runner.run(task, env)

        assert "validate_pix_key" in res["tools_called"]
        assert res["tokens_in"] == 150
        assert res["tokens_out"] == 40
        assert res["cost_usd"] == 0.002
        assert any(t.event_type == TraceEventType.TOOL_CALL for t in traces)
        assert any(t.event_type == TraceEventType.TOOL_OUTPUT for t in traces)

    @pytest.mark.asyncio
    async def test_agent_exception_gracefully_handled(self):
        def failing_agent(prompt: str):
            raise ValueError("Upstream model timeout")

        task = Task(
            task_id="CALLABLE_ERR_001",
            domain="cyber_sandbox",
            name="Failing Agent",
            description="Agent that throws exception",
            input_messages=[{"role": "user", "content": "Analyze log"}],
        )
        env = DefaultTaskEnvironment()
        runner = CallableAgentRunner(failing_agent)

        res, traces = await runner.run(task, env)

        assert "Execution failed" in res["response"]
        assert "ValueError" in res["response"]
        assert any(t.event_type == TraceEventType.ERROR for t in traces)

    @pytest.mark.asyncio
    async def test_end_to_end_execute_task_with_callable_runner(self):
        from agent_bench.core.config import BenchConfig, ModelConfig, SystemConfig

        async def simple_agent(task: Task, env: TaskEnvironment) -> str:
            if hasattr(env, "set_state"):
                env.set_state({"status": "resolved"})
            return "Task completed successfully."

        task = Task(
            task_id="E2E_CALLABLE_001",
            domain="sme_business_advisor",
            name="E2E Callable",
            description="End to end execution",
            input_messages=[{"role": "user", "content": "Resolve issue"}],
            expected_final_state={"status": "resolved"},
        )
        config = BenchConfig(
            models=[ModelConfig(model_id="stub_m", provider="stub")],
            systems=[SystemConfig(system_id="callable_sys", architecture="stub", model="stub_m")],
        )
        runner = CallableAgentRunner(simple_agent)
        result = await execute_task(task, system_id="callable_sys", config=config, agent_runner=runner)

        assert result.passed is True
        assert result.safety_violated is False
        assert len(result.traces) > 0
