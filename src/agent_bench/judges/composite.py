"""Composite judge: orchestrates multiple judges with priority."""

from typing import Any

from agent_bench.core.adapters import JudgeVerdict
from agent_bench.core.artifacts import TraceEvent
from agent_bench.core.scenarios import Task
from agent_bench.judges.deterministic import DeterministicJudge
from agent_bench.judges.grounding import GroundingJudge
from agent_bench.judges.numeric import NumericCorrectnessJudge


class CompositeJudge:
    """Orchestrates multiple judges and produces a combined verdict.

    Priority order:
    1. Deterministic (state match / refusal) — always runs
    2. Numeric correctness — runs if expected_numeric_values exist
    3. Grounding — runs if retrieval was involved
    """

    def __init__(self, numeric_tolerance: float = 0.01):
        self._deterministic = DeterministicJudge()
        self._numeric = NumericCorrectnessJudge(default_tolerance=numeric_tolerance)
        self._grounding = GroundingJudge()

    @property
    def judge_id(self) -> str:
        return "composite"

    @property
    def judge_type(self) -> str:
        return "composite"

    async def evaluate(
        self,
        task: Task,
        result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> tuple[JudgeVerdict, list[JudgeVerdict]]:
        """Returns (final_verdict, all_individual_verdicts)."""
        verdicts: list[JudgeVerdict] = []

        # 1. Deterministic judge (always)
        det_verdict = await self._deterministic.evaluate(task, result, traces)
        verdicts.append(det_verdict)

        # 2. Numeric judge (if applicable)
        has_numeric = (
            task.metadata.get("expected_numeric_values")
            or task.gold_references
        )
        if has_numeric:
            num_verdict = await self._numeric.evaluate(task, result, traces)
            verdicts.append(num_verdict)

        # 3. Grounding judge (if retrieval was involved)
        has_retrieval = (
            result.get("retrieved_documents")
            or any(t.event_type.value == "retrieval_result" for t in traces)
        )
        if has_retrieval:
            ground_verdict = await self._grounding.evaluate(task, result, traces)
            verdicts.append(ground_verdict)

        # Compute final score: weighted average with deterministic as primary
        final_score = _compute_composite_score(verdicts)
        all_passed = all(v.passed for v in verdicts)

        reasoning_parts = [f"{v.judge_id}: {v.score:.2f}" for v in verdicts]

        final_verdict = JudgeVerdict(
            score=final_score,
            passed=all_passed,
            reasoning=f"Composite [{', '.join(reasoning_parts)}]",
            judge_id=self.judge_id,
            criteria="composite",
            metadata={
                "individual_verdicts": [
                    {"judge_id": v.judge_id, "score": v.score, "passed": v.passed}
                    for v in verdicts
                ],
            },
        )
        return final_verdict, verdicts


def _compute_composite_score(verdicts: list[JudgeVerdict]) -> float:
    """Weighted score: deterministic=0.5, others split remaining 0.5."""
    if not verdicts:
        return 0.0
    if len(verdicts) == 1:
        return verdicts[0].score

    det_score = verdicts[0].score
    other_scores = [v.score for v in verdicts[1:]]
    other_avg = sum(other_scores) / len(other_scores)

    return det_score * 0.5 + other_avg * 0.5
