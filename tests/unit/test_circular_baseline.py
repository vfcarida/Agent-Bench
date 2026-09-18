"""Tests verifying that circular stub validation is eliminated.

Covers:
- T-CIRC-1: Honest stub yields functional_score < 0.2 on pix_basic_v1.
- T-CIRC-2: _stub_execute output is strictly independent of task answer key (property check).
"""

import copy
from pathlib import Path

import pytest

from agent_bench.core.config import BenchConfig, ModelConfig, SuiteConfig, SystemConfig
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.runners.case_runner import _stub_execute
from agent_bench.runners.suite_runner import run_suite


@pytest.mark.asyncio
async def test_t_circ_1_honest_stub_score_low(tmp_path: Path):
    """T-CIRC-1: Run pix_basic_v1 with the honest stub; assert functional_score < 0.2."""
    suite_cfg = SuiteConfig(
        suite_id="pix_basic_v1",
        name="PIX Assist Basic Suite",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["test_stub_system"],
        repeat_n=1,
        seed=42,
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="test_stub_system", architecture="prompt_only", model="stub")],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, tmp_path, runner_type="stub")

    assert artifact.tasks_total == 10
    scorecards = artifact.metrics.get("scorecards", [])
    assert len(scorecards) == 1
    sc = scorecards[0]
    assert sc["domain"] == "pix_assist"
    # The honest smoke stub must not achieve high scores on state-match benchmarks
    assert sc["functional_score"] < 0.2
    assert sc["functional_score"] == 0.0


def test_t_circ_2_stub_independent_of_answer_key():
    """T-CIRC-2: Monkeypatch expected_final_state and allowed_tools; output must be byte-identical."""
    tasks = load_domain_tasks("pix_assist")
    assert len(tasks) > 0

    for task in tasks:
        # Original task execution
        task_copy = copy.deepcopy(task)
        output_original = _stub_execute(task_copy, "test_system")

        # Mutate the gold answer key fields to arbitrary sentinels
        task_tampered = copy.deepcopy(task)
        task_tampered.expected_final_state = {
            "__sentinel_fraud_key__": "injected_tampered_value_123",
            "balance": 99999999.99,
        }
        task_tampered.allowed_tools = ["__sentinel_unauthorized_tool__"]
        task_tampered.metadata["expected_numeric_values"] = {"sentinel_val": 424242}

        output_tampered = _stub_execute(task_tampered, "test_system")

        # Output must be strictly identical regardless of tampering with the answer key
        assert output_original == output_tampered, (
            f"Stub output differed after tampering with answer key on task {task.task_id}!"
        )
