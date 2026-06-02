"""Integration test: run a minimal suite end-to-end."""

import pytest
from pathlib import Path

from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig, ModelConfig
from agent_bench.runners.suite_runner import run_suite


@pytest.mark.asyncio
async def test_run_pix_basic_suite(tmp_path):
    suite_cfg = SuiteConfig(
        suite_id="test_pix",
        name="Test PIX",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["stub_system"],
        repeat_n=2,
        seed=42,
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="stub_system", architecture="prompt_only", model="stub")],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, tmp_path)

    assert artifact.tasks_total > 0
    assert artifact.tasks_passed >= 0
    assert artifact.finished_at is not None
    assert artifact.duration_ms is not None
    assert artifact.duration_ms > 0

    # Check artifact file was saved
    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1
