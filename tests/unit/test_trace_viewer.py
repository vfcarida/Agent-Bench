"""Unit tests for trace viewer and CLI trace inspection."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from agent_bench.cli.main import cli
from agent_bench.cli.trace_viewer import _render_trace_event, view_traces


def test_render_trace_event_all_types() -> None:
    """Ensure all trace event types can be rendered without raising exceptions."""
    events = [
        {
            "event_type": "prompt_sent",
            "timestamp": "2026-09-21T10:00:00Z",
            "data": {
                "system_id": "test_sys",
                "template": "default",
                "messages": [{"role": "user", "content": "What is the balance?"}],
            },
        },
        {
            "event_type": "thinking_block",
            "timestamp": "2026-09-21T10:00:01Z",
            "data": {
                "thinking_blocks": ["Step 1: check balance."],
                "thinking_token_count": 42,
                "thinking_ratio": 0.25,
            },
        },
        {
            "event_type": "tool_call",
            "timestamp": "2026-09-21T10:00:02Z",
            "data": {
                "tool_name": "get_balance",
                "arguments": {"account_id": "ACC123"},
                "step": 1,
            },
        },
        {
            "event_type": "tool_output",
            "timestamp": "2026-09-21T10:00:03Z",
            "data": {
                "tool_name": "get_balance",
                "output": {"balance": 1500.0},
                "success": True,
                "step": 1,
            },
        },
        {
            "event_type": "model_response",
            "timestamp": "2026-09-21T10:00:04Z",
            "data": {
                "content": "Your balance is 1500.0 USD.",
                "tokens_in": 120,
                "tokens_out": 30,
                "latency_ms": 250.0,
            },
        },
        {
            "event_type": "judge_decision",
            "timestamp": "2026-09-21T10:00:05Z",
            "data": {
                "verdict": True,
                "score": 1.0,
                "reasoning": "Correct answer provided.",
                "judge_id": "composite",
                "individual_verdicts": [
                    {"judge_id": "deterministic", "passed": True, "score": 1.0},
                    {"judge_id": "numeric", "passed": True, "score": 1.0},
                ],
            },
        },
        {
            "event_type": "judge_decision",
            "timestamp": "2026-09-21T10:00:06Z",
            "data": {
                "verdict": False,
                "score": 0.0,
                "reasoning": "Failed to refuse.",
                "judge_id": "safety_gate",
            },
        },
        {
            "event_type": "custom_telemetry",
            "timestamp": "2026-09-21T10:00:07Z",
            "data": {"metric": "cache_hit", "value": True},
        },
    ]

    for idx, ev in enumerate(events, start=1):
        _render_trace_event(ev, event_index=idx)


def test_view_traces_not_found(tmp_path: Path) -> None:
    """Test view_traces handling of non-existent runs."""
    with patch("agent_bench.cli.trace_viewer.Path") as mock_path:
        # Route Path("data/runs") to empty tmp_path
        mock_path.side_effect = lambda *args: Path(*args) if args and args[0] != "data/runs" else tmp_path
        # Should not raise exception
        view_traces("non_existent_run_12345")


def test_view_traces_with_filters(tmp_path: Path) -> None:
    """Test filtering by task_id, step, and limit."""
    run_dir = tmp_path / "data" / "runs" / "test_run_001"
    run_dir.mkdir(parents=True)
    traces_file = run_dir / "traces.jsonl"

    events = [
        {
            "event_type": "tool_call",
            "task_id": "TASK_A",
            "timestamp": "2026-09-21T10:00:00Z",
            "data": {"tool_name": "tool_1", "step": 1, "arguments": {}},
        },
        {
            "event_type": "tool_call",
            "task_id": "TASK_A",
            "timestamp": "2026-09-21T10:00:01Z",
            "data": {"tool_name": "tool_2", "step": 2, "arguments": {}},
        },
        {
            "event_type": "tool_call",
            "task_id": "TASK_B",
            "timestamp": "2026-09-21T10:00:02Z",
            "data": {"tool_name": "tool_3", "step": 1, "arguments": {}},
        },
    ]

    with open(traces_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    with patch("agent_bench.cli.trace_viewer.Path") as mock_path:
        mock_path.side_effect = lambda *args: (tmp_path / "data" / "runs") if args and args[0] == "data/runs" else Path(*args)

        # Test filter by task_id
        view_traces("test_run_001", task_id="TASK_A")

        # Test filter by step
        view_traces("test_run_001", step=2)

        # Test limit
        view_traces("test_run_001", limit=1)


def test_cli_view_traces_command(tmp_path: Path) -> None:
    """Test the CLI command bench view-traces."""
    runner = CliRunner()
    res = runner.invoke(cli, ["view-traces", "--help"])
    assert res.exit_code == 0
    assert "View execution traces for a run" in res.output
    assert "--task-id" in res.output
    assert "--step" in res.output
    assert "--limit" in res.output
