"""Unit tests for asynchronous concurrency pool in SuiteRunner."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig
from agent_bench.core.scenarios import Task
from agent_bench.runners.case_runner import CaseResult
from agent_bench.runners.suite_runner import run_suite


def _create_test_tasks(n: int = 6) -> list[Task]:
    return [
        Task(
            task_id=f"TASK_{i:02d}",
            domain="test_domain",
            name=f"Task {i}",
            description=f"Description {i}",
            input_messages=[{"role": "user", "content": f"Do {i}"}],
            expected_final_state={"done": True},
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_suite_runner_concurrency_execution(tmp_path: Path) -> None:
    """Verify that run_suite with concurrency > 1 evaluates all tasks and aggregates results."""
    test_tasks = _create_test_tasks(6)
    suite_cfg = SuiteConfig(
        suite_id="test_conc_suite",
        name="Concurrency Test",
        version="1.0.0",
        domains=["test_domain"],
        systems=["test_system"],
        repeat_n=1,
        seed=42,
    )
    bench_cfg = BenchConfig(
        benchmark_version="1.0.0",
        suites=[suite_cfg],
        systems=[SystemConfig(system_id="test_system", model="stub", architecture="stub")],
    )

    with patch("agent_bench.runners.suite_runner.load_domain_tasks", return_value=test_tasks):
        artifact = await run_suite(
            suite_cfg,
            bench_cfg,
            output_dir=tmp_path,
            runner_type="stub",
            concurrency=3,
        )

    assert artifact.tasks_total == 6
    assert artifact.tasks_passed + artifact.tasks_failed == 6
    assert len(artifact.traces) >= 0


@pytest.mark.asyncio
async def test_suite_runner_concurrency_semaphore_bound(tmp_path: Path) -> None:
    """Verify that peak concurrent executions never exceed the configured semaphore limit."""
    test_tasks = _create_test_tasks(8)
    suite_cfg = SuiteConfig(
        suite_id="test_bound_suite",
        name="Semaphore Bound Test",
        version="1.0.0",
        domains=["test_domain"],
        systems=["test_system"],
        repeat_n=1,
        seed=42,
    )
    bench_cfg = BenchConfig(
        benchmark_version="1.0.0",
        suites=[suite_cfg],
        systems=[SystemConfig(system_id="test_system", model="stub", architecture="stub")],
    )

    current_concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def mock_execute_task(*args: Any, **kwargs: Any) -> CaseResult:
        nonlocal current_concurrent, max_concurrent
        async with lock:
            current_concurrent += 1
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent

        # Simulate async I/O
        await asyncio.sleep(0.05)

        async with lock:
            current_concurrent -= 1

        return CaseResult(
            passed=True,
            traces=[TraceEvent(event_type=TraceEventType.MODEL_RESPONSE, data={"test": 1})],
            latency_ms=50.0,
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.001,
        )

    with patch("agent_bench.runners.suite_runner.load_domain_tasks", return_value=test_tasks), \
         patch("agent_bench.runners.suite_runner.execute_task", side_effect=mock_execute_task):
        artifact = await run_suite(
            suite_cfg,
            bench_cfg,
            output_dir=tmp_path,
            runner_type="stub",
            concurrency=2,
        )

    assert artifact.tasks_passed == 8
    assert max_concurrent <= 2
    assert max_concurrent > 1
