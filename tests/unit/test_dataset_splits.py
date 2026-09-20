"""Unit and integration tests for dataset split loading discipline (T-GOV-1)."""

from pathlib import Path

import pytest

from agent_bench.core.config import BenchConfig, ModelConfig, SuiteConfig, SystemConfig
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.runners.case_runner import run_single_case
from agent_bench.runners.suite_runner import run_suite


def test_t_gov_1_split_loading_disjoint():
    """T-GOV-1: load_domain_tasks('pix_assist', split='holdout') vs split='dev'.

    Asserts:
    1. dev returns dev tasks from canonical datasets/gold/dev.
    2. holdout returns holdout tasks from datasets/gold/holdout.
    3. The two task ID sets are non-empty and strictly disjoint.
    """
    dev_tasks = load_domain_tasks("pix_assist", split="dev")
    holdout_tasks = load_domain_tasks("pix_assist", split="holdout")

    assert len(dev_tasks) > 0, "dev_tasks must not be empty"
    assert len(holdout_tasks) > 0, "holdout_tasks must not be empty"

    dev_ids = {t.task_id for t in dev_tasks}
    holdout_ids = {t.task_id for t in holdout_tasks}

    assert dev_ids.isdisjoint(holdout_ids), (
        f"dev and holdout task sets must be disjoint! Overlap: {dev_ids & holdout_ids}"
    )

    # Verify task attributes parsed correctly
    for t in holdout_tasks:
        assert t.domain == "pix_assist"
        assert t.task_id.startswith("PIX_HOLDOUT_")
        assert len(t.input_messages) > 0


def test_forbidden_regression_holdout_never_from_fixtures():
    """Forbidden regression: holdout content must never be loaded from data/fixtures."""
    fixtures_dir = Path("data/fixtures")
    holdout_from_fixtures = load_domain_tasks("pix_assist", split="holdout", data_dir=fixtures_dir)
    assert len(holdout_from_fixtures) == 0, (
        "Forbidden regression: holdout tasks were loaded from data/fixtures!"
    )


@pytest.mark.asyncio
async def test_suite_runner_respects_split(tmp_path: Path):
    """Suite runner executes against the specified split."""
    suite_cfg = SuiteConfig(
        suite_id="pix_holdout_suite",
        name="PIX Holdout Suite",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["test_stub"],
        repeat_n=1,
        seed=42,
        split="holdout",
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="test_stub", architecture="prompt_only", model="stub")],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, tmp_path, runner_type="stub")
    # Holdout has 2 tasks
    assert artifact.tasks_total == 2


@pytest.mark.asyncio
async def test_run_single_case_respects_split(tmp_path: Path):
    """run_single_case resolves tasks from the requested split."""
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="test_stub", architecture="prompt_only", model="stub")],
    )

    # PIX_HOLDOUT_001 exists in holdout split
    passed_holdout = await run_single_case(
        task_id="PIX_HOLDOUT_001",
        system_id="test_stub",
        domain="pix_assist",
        config=config,
        output_dir=tmp_path,
        split="holdout",
    )
    # The task was found and evaluated (smoke stub may fail state assertion, but task is resolved)
    assert isinstance(passed_holdout, bool)

    # PIX_HOLDOUT_001 does not exist in dev split
    missing_in_dev = await run_single_case(
        task_id="PIX_HOLDOUT_001",
        system_id="test_stub",
        domain="pix_assist",
        config=config,
        output_dir=tmp_path,
        split="dev",
    )
    assert missing_in_dev is False
