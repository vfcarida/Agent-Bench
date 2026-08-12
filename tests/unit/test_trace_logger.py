"""Unit tests for ExecutionTraceLogger and cost analytics."""

from pathlib import Path

from agent_bench.core.artifacts import TraceEventType
from agent_bench.storage.trace_logger import ExecutionTraceLogger


def test_execution_trace_logger_metrics(tmp_path: Path) -> None:
    logger = ExecutionTraceLogger(
        run_id="run_001",
        task_id="task_001",
        system_id="gpt4_agent",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    )

    # Log events
    logger.log_event(TraceEventType.PROMPT_SENT, {"prompt": "Hello"})
    logger.log_model_call(tokens_in=1000, tokens_out=500, latency_ms=1200.0, model_id="gpt4")

    summary = logger.get_summary()

    assert summary["tokens_in"] == 1000
    assert summary["tokens_out"] == 500
    assert summary["total_tokens"] == 1500
    assert summary["latency_ms"] == 1200.0
    # cost: (1000/1000)*0.01 + (500/1000)*0.03 = 0.01 + 0.015 = 0.025
    assert abs(summary["total_cost_usd"] - 0.025) < 1e-6
    assert summary["trace_count"] == 2


def test_execution_trace_logger_exports(tmp_path: Path) -> None:
    logger = ExecutionTraceLogger(
        run_id="run_002",
        task_id="task_002",
        system_id="stub_agent",
    )

    logger.log_model_call(tokens_in=200, tokens_out=100, latency_ms=300.0)

    jsonl_path = tmp_path / "trace.jsonl"
    parquet_path = tmp_path / "trace.parquet"

    saved_jsonl = logger.export_jsonl(jsonl_path)
    saved_parquet = logger.export_parquet(parquet_path)

    assert saved_jsonl.exists()
    assert saved_parquet.exists()
    assert saved_jsonl.stat().st_size > 0
    assert saved_parquet.stat().st_size > 0
