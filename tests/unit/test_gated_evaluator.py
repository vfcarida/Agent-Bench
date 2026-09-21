"""Unit tests for GatedEvaluator logic."""

from unittest.mock import AsyncMock

import pytest

from agent_bench.core.adapters import JudgeVerdict
from agent_bench.core.scenarios import Task
from agent_bench.graders.gated_evaluator import GatedEvaluator


@pytest.mark.asyncio
async def test_gated_evaluator_short_circuit_on_failure() -> None:
    # Mock deterministic judge that fails
    mock_det_judge = AsyncMock()
    mock_det_judge.evaluate.return_value = JudgeVerdict(
        score=0.0,
        passed=False,
        reasoning="State mismatch: expected key 'balance' missing",
        judge_id="mock_det",
        criteria="state_match",
    )

    # Mock subjective judge that should NOT be called
    mock_subj_judge = AsyncMock()

    gated_eval = GatedEvaluator(
        deterministic_judge=mock_det_judge,
        subjective_judges=[mock_subj_judge],
    )

    task = Task(
        task_id="GATED_001",
        domain="pix_assist",
        name="Failed Task",
        description="Fails deterministic check",
        input_messages=[{"role": "user", "content": "Transfer"}],
    )

    verdict = await gated_eval.evaluate(task, {"final_state": {}}, [])

    assert not verdict.passed
    assert verdict.score == 0.0
    assert "LLM Judge evaluation skipped" in verdict.reasoning
    assert verdict.metadata["gated_status"] == "short_circuited"
    assert verdict.metadata["llm_judges_invoked"] is False
    mock_subj_judge.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_gated_evaluator_passes_to_subjective_judges() -> None:
    # Mock deterministic judge that passes
    mock_det_judge = AsyncMock()
    mock_det_judge.evaluate.return_value = JudgeVerdict(
        score=1.0,
        passed=True,
        reasoning="State match 100%",
        judge_id="mock_det",
        criteria="state_match",
    )

    # Mock subjective judge that scores 0.8
    mock_subj_judge = AsyncMock()
    mock_subj_judge.judge_id = "mock_subjective"
    mock_subj_judge.evaluate.return_value = JudgeVerdict(
        score=0.8,
        passed=True,
        reasoning="Good response quality",
        judge_id="mock_subjective",
        criteria="rubric",
    )

    gated_eval = GatedEvaluator(
        deterministic_judge=mock_det_judge,
        subjective_judges=[mock_subj_judge],
    )

    task = Task(
        task_id="GATED_002",
        domain="pix_assist",
        name="Passed Task",
        description="Passes deterministic check",
        input_messages=[{"role": "user", "content": "Transfer"}],
    )

    verdict = await gated_eval.evaluate(task, {"final_state": {"ok": True}}, [])

    assert verdict.passed
    assert verdict.score == 0.9  # Average of 1.0 and 0.8
    assert verdict.metadata["gated_status"] == "passed_full_evaluation"
    assert verdict.metadata["llm_judges_invoked"] is True
    mock_subj_judge.evaluate.assert_called_once()
