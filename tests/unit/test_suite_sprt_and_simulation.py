"""Unit tests for SuiteRunner SPRT early stopping and user simulator integration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig
from agent_bench.core.scenarios import Task
from agent_bench.core.user_simulator import ScriptedUserSimulator
from agent_bench.runners.case_runner import CaseResult
from agent_bench.runners.suite_runner import run_suite


@pytest.fixture
def sample_config() -> BenchConfig:
    return BenchConfig(
        models=[],
        systems=[
            SystemConfig(
                system_id="mock_sys",
                model="mock_model",
                architecture="tool_calling_reactive",
            )
        ],
        suites=[
            SuiteConfig(
                suite_id="sprt_test_suite",
                name="SPRT Test Suite",
                version="1.0.0",
                domains=["pix_assist"],
                tasks=["PIX_001"],
                systems=["mock_sys"],
                repeat_n=10,
                seed=42,
            )
        ],
    )


@pytest.fixture
def sample_task() -> Task:
    return Task(
        task_id="PIX_001",
        domain="pix_assist",
        name="Test Task",
        description="Test task for SPRT early stopping",
        input_messages=[{"role": "user", "content": "Transfer 50 BRL"}],
    )


@pytest.mark.asyncio
async def test_suite_runner_sprt_early_stopping(
    tmp_path: Path, sample_config: BenchConfig, sample_task: Task
) -> None:
    suite_cfg = sample_config.suites[0]

    call_count = 0

    async def mock_execute_task(*args, **kwargs) -> CaseResult:
        nonlocal call_count
        call_count += 1
        return CaseResult(
            passed=False,
            traces=[],
            latency_ms=100.0,
            tokens_in=50,
            tokens_out=20,
            cost_usd=0.001,
            safety_violated=False,
        )

    with (
        patch("agent_bench.runners.suite_runner.load_domain_tasks", return_value=[sample_task]),
        patch("agent_bench.runners.suite_runner.execute_task", side_effect=mock_execute_task),
    ):
        artifact = await run_suite(
            suite_cfg=suite_cfg,
            config=sample_config,
            output_dir=tmp_path,
            runner_type="scripted",
            enable_sprt=True,
            sprt_p0=0.50,
            sprt_p1=0.80,
        )

        assert artifact is not None
        # With repeat_n=10, continuous failure should trigger REJECT_H1 in <= 6 repetitions
        assert call_count < 10
        task_results = artifact.metrics.get("task_results", [])
        assert len(task_results) == 1
        assert task_results[0]["sprt_decision"] == "reject_h1"
        assert task_results[0]["repetitions_executed"] == call_count


@pytest.mark.asyncio
async def test_suite_runner_safety_violation_immediate_stop(
    tmp_path: Path, sample_config: BenchConfig, sample_task: Task
) -> None:
    suite_cfg = sample_config.suites[0]
    call_count = 0

    async def mock_execute_task(*args, **kwargs) -> CaseResult:
        nonlocal call_count
        call_count += 1
        return CaseResult(
            passed=False,
            traces=[],
            latency_ms=80.0,
            tokens_in=30,
            tokens_out=15,
            cost_usd=0.0005,
            safety_violated=True,  # Zero-tolerance safety violation
        )

    with (
        patch("agent_bench.runners.suite_runner.load_domain_tasks", return_value=[sample_task]),
        patch("agent_bench.runners.suite_runner.execute_task", side_effect=mock_execute_task),
    ):
        artifact = await run_suite(
            suite_cfg=suite_cfg,
            config=sample_config,
            output_dir=tmp_path,
            runner_type="scripted",
            enable_sprt=False,  # Safety stop applies even without SPRT
        )

        # Safety violation must immediately stop subsequent repetitions
        assert call_count == 1
        task_results = artifact.metrics.get("task_results", [])
        assert task_results[0]["sprt_decision"] == "safety_violation_stop"
        assert task_results[0]["policy_violated"] is True


@pytest.mark.asyncio
async def test_suite_runner_propagates_user_simulator(
    tmp_path: Path, sample_config: BenchConfig, sample_task: Task
) -> None:
    suite_cfg = sample_config.suites[0]
    suite_cfg.repeat_n = 1
    received_simulator = None

    async def mock_execute_task(*args, **kwargs) -> CaseResult:
        nonlocal received_simulator
        received_simulator = kwargs.get("user_simulator")
        return CaseResult(
            passed=True,
            traces=[],
            latency_ms=100.0,
            tokens_in=50,
            tokens_out=20,
            cost_usd=0.001,
            safety_violated=False,
        )

    simulator = ScriptedUserSimulator()

    with (
        patch("agent_bench.runners.suite_runner.load_domain_tasks", return_value=[sample_task]),
        patch("agent_bench.runners.suite_runner.execute_task", side_effect=mock_execute_task),
    ):
        await run_suite(
            suite_cfg=suite_cfg,
            config=sample_config,
            output_dir=tmp_path,
            runner_type="scripted",
            user_simulator=simulator,
        )

        assert received_simulator is simulator
