"""Integration test: run investment_advisor suite end-to-end."""

import pytest
from pathlib import Path

from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig, ModelConfig
from agent_bench.runners.suite_runner import run_suite


@pytest.mark.asyncio
async def test_run_investment_suite(tmp_path):
    suite_cfg = SuiteConfig(
        suite_id="test_investment",
        name="Test Investment",
        version="1.0.0",
        domains=["investment_advisor"],
        systems=["rag_basic_stub"],
        repeat_n=2,
        seed=42,
        weighting_profile="advisory_regulated",
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="rag_basic_stub", architecture="rag_basic", model="stub")],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, tmp_path)

    # 12 tasks × 1 system × 2 repeats = 24 total executions
    assert artifact.tasks_total == 24
    assert artifact.tasks_passed >= 0
    assert artifact.finished_at is not None

    # Check artifacts saved
    run_dir = tmp_path / artifact.run_id
    assert run_dir.exists()
    assert (run_dir / "traces.jsonl").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "metrics.parquet").exists()
    assert (run_dir / f"{artifact.run_id}_manifest.json").exists()

    # Check scorecards computed
    assert "scorecards" in artifact.metrics
    scorecards = artifact.metrics["scorecards"]
    assert len(scorecards) >= 1
    assert scorecards[0]["domain"] == "investment_advisor"
    assert 0 <= scorecards[0]["global_score"] <= 1


@pytest.mark.asyncio
async def test_multi_system_comparison(tmp_path):
    suite_cfg = SuiteConfig(
        suite_id="test_compare",
        name="Test Compare",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["sys_a", "sys_b"],
        repeat_n=1,
        seed=42,
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[
            SystemConfig(system_id="sys_a", architecture="prompt_only", model="stub"),
            SystemConfig(system_id="sys_b", architecture="tool_calling_reactive", model="stub"),
        ],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, tmp_path)

    # 10 tasks × 2 systems × 1 repeat = 20
    assert artifact.tasks_total == 20
    scorecards = artifact.metrics["scorecards"]
    assert len(scorecards) == 2  # one per system
    systems = {sc["system_id"] for sc in scorecards}
    assert "sys_a" in systems
    assert "sys_b" in systems
