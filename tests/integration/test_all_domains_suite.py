"""Integration test: run all 4 domains end-to-end."""

import pytest
from pathlib import Path

from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig, ModelConfig
from agent_bench.runners.suite_runner import run_suite


@pytest.mark.asyncio
async def test_run_all_domains(tmp_path):
    suite_cfg = SuiteConfig(
        suite_id="test_all_domains",
        name="Test All Domains",
        version="1.0.0",
        domains=["pix_assist", "investment_advisor", "sme_business_advisor", "cyber_sandbox"],
        systems=["stub_system"],
        repeat_n=1,
        seed=42,
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="stub_system", architecture="tool_calling_reactive", model="stub")],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, tmp_path)

    # 10 (pix) + 12 (investment) + 10 (sme) + 10 (cyber) = 42 tasks × 1 system × 1 repeat
    assert artifact.tasks_total == 42
    assert artifact.tasks_passed >= 0
    assert artifact.finished_at is not None

    # Scorecards for all 4 domains
    scorecards = artifact.metrics["scorecards"]
    domains_in_scorecards = {sc["domain"] for sc in scorecards}
    assert "pix_assist" in domains_in_scorecards
    assert "investment_advisor" in domains_in_scorecards
    assert "sme_business_advisor" in domains_in_scorecards
    assert "cyber_sandbox" in domains_in_scorecards


@pytest.mark.asyncio
async def test_sme_domain_tasks_loaded(tmp_path):
    from agent_bench.datasets.loader import load_domain_tasks

    tasks = load_domain_tasks("sme_business_advisor")
    assert len(tasks) == 10
    assert tasks[0].task_id == "SME_001"
    assert tasks[0].domain == "sme_business_advisor"


@pytest.mark.asyncio
async def test_cyber_domain_tasks_loaded(tmp_path):
    from agent_bench.datasets.loader import load_domain_tasks

    tasks = load_domain_tasks("cyber_sandbox")
    assert len(tasks) == 10
    assert tasks[0].task_id == "CYBER_001"

    # Verify refusal tasks are properly tagged
    refusal_tasks = [t for t in tasks if "refusal" in t.tags]
    assert len(refusal_tasks) >= 4  # CYBER_003, 005, 007, 009
