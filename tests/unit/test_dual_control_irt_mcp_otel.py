"""Unit tests for Dual-Control User Simulation, 2PL IRT Anchor Selection, and OpenTelemetry/MCP Bridge:
1. Tau^2-Bench Dual-Control Shared-World User Simulation (arXiv:2506.07982)
2. 2PL Item Response Theory (IRT) Fast-Anchor Evaluation (arXiv:2402.14992)
3. OpenTelemetry GenAI Semantic Conventions & Model Context Protocol (MCP) Bridge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from agent_bench import (
    DualControlAction,
    DualControlUserSimulator,
    IRTModelFit,
    MCPClientToolAdapter,
    MCPToolServer,
    Task,
    ToolAdapter,
    TraceEventType,
    UserSimulator,
    convert_run_traces_to_otel_genai,
    discover_mcp_tools,
    estimate_full_score_from_anchor,
    fit_2pl_irt,
    select_irt_anchor_tasks,
)
from agent_bench.cli.main import cli
from agent_bench.core.adapters import ModelAdapter, ModelResponse, ToolCallResult
from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig
from agent_bench.core.scenarios import Severity
from agent_bench.metrics.irt import fisher_information_2pl, logistic_2pl
from agent_bench.runners.case_runner import DefaultTaskEnvironment, execute_task
from agent_bench.runners.suite_runner import run_suite
from agent_bench.utils.observability import SpanCollector


class _MockDeviceTool(ToolAdapter):
    """Tool that toggles or updates device network state in a shared environment."""

    def __init__(self, shared_box: dict[str, Any]) -> None:
        self._box = shared_box

    @property
    def name(self) -> str:
        return "toggle_airplane_mode"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "toggle_airplane_mode",
            "description": "Toggles device airplane mode.",
            "parameters": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        enabled = bool(arguments.get("enabled", False))
        self._box["airplane_mode"] = enabled
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            success=True,
            output={"airplane_mode": enabled, "signal": "none" if enabled else "5G"},
        )


class _DualControlDialogueModel(ModelAdapter):
    """Multi-turn model that asks user to toggle airplane mode before completing task."""

    def __init__(self) -> None:
        self._turn = 0

    @property
    def model_id(self) -> str:
        return "dual-control-test-model"

    @property
    def provider(self) -> str:
        return "test"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> ModelResponse:
        self._turn += 1
        if self._turn == 1:
            return ModelResponse(
                content="Please disable airplane mode on your handset and confirm.",
                tool_calls=[],
                tokens_in=40,
                tokens_out=15,
                latency_ms=10.0,
            )
        return ModelResponse(
            content="Great, I see your 5G connection is active and the issue is resolved.",
            tool_calls=[],
            tokens_in=60,
            tokens_out=20,
            latency_ms=12.0,
        )


# ============================================================================
# 1. Tau^2-Bench Dual-Control Shared-World User Simulation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_dual_control_user_simulator_protocol_and_actions() -> None:
    task = Task(
        task_id="telecom_dual_01",
        domain="telecom",
        name="Mobile Data Connectivity",
        description="Troubleshoot mobile data connectivity in shared environment.",
        input_messages=[{"role": "user", "content": "My phone has no data connection."}],
        initial_state={"airplane_mode": True, "roaming": False},
        severity=Severity.MEDIUM,
    )

    sim = DualControlUserSimulator(
        actions=[
            DualControlAction(
                trigger_pattern=r"disable airplane mode|airplane mode",
                response="I just turned off airplane mode on my phone.",
                tool_calls=[
                    {"name": "toggle_airplane_mode", "arguments": {"enabled": False}}
                ],
                state_mutations={"airplane_mode": False, "user_confirmed_toggle": True},
                required_state={"airplane_mode": True},
                once=True,
                finish_on_trigger=False,
            )
        ]
    )
    assert isinstance(sim, UserSimulator)
    assert sim.simulator_id == "dual_control_user_simulator"
    sim.reset(task)

    # Step 1: Agent asks user to disable airplane mode while airplane_mode is True
    turn1 = await sim.step(
        agent_message="Please disable airplane mode on your handset.",
        history=[],
        task=task,
        current_state={"airplane_mode": True},
    )
    assert turn1.metadata.get("dual_control") is True
    assert len(turn1.user_tool_calls) == 1
    assert turn1.user_tool_calls[0]["name"] == "toggle_airplane_mode"
    assert turn1.state_mutations == {"airplane_mode": False, "user_confirmed_toggle": True}

    # Step 2: Action does not re-trigger once used; agent completion message finishes dialogue
    turn2 = await sim.step(
        agent_message="Great, your issue is resolved.",
        history=[],
        task=task,
        current_state={"airplane_mode": False},
    )
    assert turn2.finished is True


@pytest.mark.asyncio
async def test_dual_control_execution_mutates_shared_environment_and_emits_traces() -> None:
    shared_box: dict[str, Any] = {"airplane_mode": True}
    tool = _MockDeviceTool(shared_box)

    task = Task(
        task_id="telecom_dual_02",
        domain="telecom",
        name="Dual Control Troubleshooting",
        description="Dual-control troubleshooting with user tool execution.",
        input_messages=[{"role": "user", "content": "My mobile data is offline."}],
        initial_state={"airplane_mode": True, "network_status": "offline"},
        expected_final_state={"airplane_mode": False, "network_status": "online"},
        severity=Severity.LOW,
        metadata={
            "dual_control_actions": [
                {
                    "trigger_pattern": "disable airplane mode",
                    "response": "Okay, I disabled airplane mode in settings.",
                    "tool_name": "toggle_airplane_mode",
                    "tool_args": {"enabled": False},
                    "state_mutations": {"airplane_mode": False, "network_status": "online"},
                    "required_state": {"airplane_mode": True},
                }
            ]
        },
    )

    env = DefaultTaskEnvironment(
        environment_id="telecom_env",
        tools={"toggle_airplane_mode": tool},
    )
    env.reset(task.initial_state)

    sim = DualControlUserSimulator()
    sim.reset(task)
    model = _DualControlDialogueModel()
    config = BenchConfig(
        systems=[
            SystemConfig(
                system_id="dual_control_sys",
                architecture="tool_calling_reactive",
                model="dual-control-test-model",
            )
        ]
    )

    case_res = await execute_task(
        task=task,
        system_id="dual_control_sys",
        config=config,
        seed=42,
        model=model,
        environment=env,
        user_simulator=sim,
    )

    # Verify the user simulator executed the tool AND mutated the shared environment state
    assert shared_box["airplane_mode"] is False
    assert env.get_state()["airplane_mode"] is False
    assert env.get_state()["network_status"] == "online"

    # Verify USER_ACTION trace events were emitted
    user_action_events = [
        t for t in case_res.traces if t.event_type == TraceEventType.USER_ACTION
    ]
    assert len(user_action_events) >= 2  # 1 state_mutation + 1 tool_call
    action_types = {e.data.get("action_type") for e in user_action_events}
    assert "state_mutation" in action_types
    assert "tool_call" in action_types


def test_default_task_environment_update_state() -> None:
    env = DefaultTaskEnvironment(environment_id="env_test")
    env.reset({"a": 1})
    updated = env.update_state({"b": 2, "a": 10})
    assert updated == {"a": 10, "b": 2}
    assert env.get_state() == {"a": 10, "b": 2}


# ============================================================================
# 2. 2PL Item Response Theory (IRT) Fast-Anchor Evaluation Tests
# ============================================================================


def test_2pl_irt_math_and_fitting() -> None:
    # At theta == b, logistic_2pl is always 0.5
    assert abs(logistic_2pl(0.0, 1.5, 0.0) - 0.5) < 1e-9
    # Fisher information is maximized at theta == b: a^2 * 0.25
    info_peak = fisher_information_2pl(0.0, 2.0, 0.0)
    info_off = fisher_information_2pl(2.5, 2.0, 0.0)
    assert abs(info_peak - 1.0) < 1e-9
    assert info_peak > info_off

    # Synthetic response matrix: 4 systems across 8 tasks of varying difficulty
    response_matrix: dict[str, dict[str, bool]] = {
        "sys_weak": {
            "t1": True,
            "t2": True,
            "t3": False,
            "t4": False,
            "t5": False,
            "t6": False,
            "t7": False,
            "t8": False,
        },
        "sys_med_low": {
            "t1": True,
            "t2": True,
            "t3": True,
            "t4": True,
            "t5": False,
            "t6": False,
            "t7": False,
            "t8": False,
        },
        "sys_med_high": {
            "t1": True,
            "t2": True,
            "t3": True,
            "t4": True,
            "t5": True,
            "t6": True,
            "t7": False,
            "t8": False,
        },
        "sys_strong": {
            "t1": True,
            "t2": True,
            "t3": True,
            "t4": True,
            "t5": True,
            "t6": True,
            "t7": True,
            "t8": False,
        },
    }

    fit = fit_2pl_irt(response_matrix, max_iter=35)
    assert isinstance(fit, IRTModelFit)
    assert fit.converged is True
    assert len(fit.item_parameters) == 8

    # Ability ordering must match empirical strength
    thetas = fit.system_abilities
    assert (
        thetas["sys_weak"]
        < thetas["sys_med_low"]
        < thetas["sys_med_high"]
        < thetas["sys_strong"]
    )

    # Difficulty ordering: t1 (easiest) < t5 < t8 (hardest)
    assert (
        fit.item_parameters["t1"].difficulty_b
        < fit.item_parameters["t5"].difficulty_b
        < fit.item_parameters["t8"].difficulty_b
    )

    # Select 50% (4) anchor tasks
    anchor_info = select_irt_anchor_tasks(fit, anchor_fraction=0.5)
    anchors = anchor_info["anchor_task_ids"]
    assert len(anchors) == 4
    assert len(set(anchors)) == 4
    assert anchor_info["information_retention_ratio"] > 0.0

    # Estimate full-suite score from anchor responses for sys_med_high (actual = 6/8 = 0.75)
    anchor_obs = {tid: response_matrix["sys_med_high"][tid] for tid in anchors}
    est = estimate_full_score_from_anchor(
        anchor_obs, fit, item_weights=anchor_info["item_weights"]
    )
    assert "theta_hat" in est
    assert 0.40 <= est["irt_predicted_full_score"] <= 0.95
    assert est["anchor_tasks_evaluated"] == 4


@pytest.mark.asyncio
async def test_suite_runner_fast_anchor_irt_mode(tmp_path: Path) -> None:
    suite_cfg = SuiteConfig(
        suite_id="irt_fast_suite",
        name="IRT Fast Suite",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["sys_scripted_a", "sys_scripted_b"],
        repeat_n=1,
        seed=42,
        split="dev",
    )
    config = BenchConfig(
        systems=[
            SystemConfig(system_id="sys_scripted_a", architecture="scripted", model="scripted"),
            SystemConfig(
                system_id="sys_scripted_b", architecture="scripted_policy", model="scripted"
            ),
        ],
        suites=[suite_cfg],
    )
    runs_dir = tmp_path / "runs"
    artifact = await run_suite(
        suite_cfg,
        config,
        output_dir=runs_dir,
        fast_anchor=True,
        anchor_fraction=0.30,
    )
    irt_diag = artifact.metrics.get("diagnostics", {}).get("irt_analysis", {})
    assert irt_diag.get("fast_anchor_mode") is True
    assert irt_diag.get("anchor_fraction") == 0.30
    assert "model_fit" in irt_diag
    assert "recommended_anchor" in irt_diag
    assert "anchor_score_estimates" in irt_diag

    # Also verify OTel GenAI file was automatically exported
    run_dir = runs_dir / artifact.run_id
    assert (run_dir / "otel_genai_spans.json").exists()


# ============================================================================
# 3. OpenTelemetry GenAI Semantic Conventions & MCP Bridge Tests
# ============================================================================


def test_otel_genai_span_collector_and_trace_conversion() -> None:
    collector = SpanCollector()
    with collector.trace("invoke_agent", **{"gen_ai.system": "openai"}) as sp:
        sp.set_attribute("gen_ai.usage.input_tokens", 120)
        sp.set_attribute("gen_ai.usage.output_tokens", 45)

    otlp_from_collector = collector.export_otlp_genai_json(service_name="agent-bench-test")
    assert "resourceSpans" in otlp_from_collector
    spans = otlp_from_collector["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == "invoke_agent"

    # Test convert_run_traces_to_otel_genai from run manifest and traces
    run_manifest = {
        "run_id": "run_otel_01",
        "suite_id": "Fintech_PIX",
        "metrics": {
            "scorecards": [
                {
                    "system_id": "react_agent",
                    "domain": "pix_assist",
                    "overall_score": 0.92,
                    "safety_violation_rate": 0.0,
                    "pass_hat_3": 0.88,
                }
            ],
            "task_results": [
                {
                    "task_id": "pix_001",
                    "system_id": "react_agent",
                    "domain": "pix_assist",
                    "passed": True,
                    "tokens_in": 100,
                    "tokens_out": 30,
                    "cost_usd": 0.002,
                    "latency_ms": 150.0,
                }
            ],
        },
    }
    raw_traces = [
        {
            "event_type": TraceEventType.TOOL_CALL.value,
            "data": {"tool_name": "lookup_pix", "arguments": {"key": "123"}},
        },
        {
            "event_type": TraceEventType.USER_ACTION.value,
            "data": {"actor": "user", "action_type": "state_mutation"},
        },
    ]

    otlp_payload = convert_run_traces_to_otel_genai(run_manifest, raw_traces)
    exported_spans = otlp_payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(exported_spans) >= 4
    op_names = [s["name"] for s in exported_spans]
    assert any("invoke_benchmark_suite" in n for n in op_names)
    assert any("invoke_agent" in n for n in op_names)
    assert any("execute_tool" in n for n in op_names)


def test_cli_export_otel_command(tmp_path: Path) -> None:
    run_id = "test_run_otel_cli"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)

    run_json = {
        "run_id": run_id,
        "suite_id": "Orchestration",
        "system_id": "react_agent",
        "metrics": {"scorecards": [], "task_results": []},
    }
    (tmp_path / f"{run_id}.json").write_text(json.dumps(run_json), encoding="utf-8")

    trace_line = json.dumps(
        {
            "event_type": TraceEventType.TOOL_CALL.value,
            "data": {"tool_name": "verify_kyc", "arguments": {}},
        }
    )
    (run_dir / "traces.jsonl").write_text(trace_line + "\n", encoding="utf-8")

    out_file = tmp_path / "custom_otel.json"
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["export-otel", run_id, "--runs-dir", str(tmp_path), "--output", str(out_file)],
    )
    assert res.exit_code == 0
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert "resourceSpans" in payload


@pytest.mark.asyncio
async def test_mcp_bridge_server_and_client_adapter_roundtrip() -> None:
    shared_state: dict[str, Any] = {"airplane_mode": True}
    device_tool = _MockDeviceTool(shared_state)
    env = DefaultTaskEnvironment(
        environment_id="mcp_test_env",
        tools={"toggle_airplane_mode": device_tool},
    )
    env.reset({"airplane_mode": True, "nightly_limit_brl": 1000.0})

    server = MCPToolServer(
        environment=env,
        tools=[device_tool],
    )

    # 1. Initialize handshake
    init_resp = await server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"

    # 2. Ping
    ping_resp = await server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping_resp["result"] == {}

    # 3. Resources list & read
    res_list = await server.handle_jsonrpc({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    assert len(res_list["result"]["resources"]) == 1
    res_read = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "bench://environment/state"},
        }
    )
    assert "1000.0" in res_read["result"]["contents"][0]["text"]

    # 4. Discover tools and wrap as MCPClientToolAdapter
    discovered = await discover_mcp_tools(server)
    assert len(discovered) == 1
    mcp_tool = discovered[0]
    assert mcp_tool.name == "toggle_airplane_mode"
    assert isinstance(mcp_tool, MCPClientToolAdapter)
    assert mcp_tool.schema["name"] == "toggle_airplane_mode"

    # 5. Execute tool through MCP JSON-RPC bridge
    call_result = await mcp_tool.execute({"enabled": False})
    assert call_result.success is True
    assert call_result.output["airplane_mode"] is False
    assert shared_state["airplane_mode"] is False

    # 6. Error handling on unknown method and missing tool name
    unknown_method = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 5, "method": "nonexistent/method"}
    )
    assert unknown_method["error"]["code"] == -32601

    missing_tool_name = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "", "arguments": {}},
        }
    )
    assert missing_tool_name["error"]["code"] == -32602
