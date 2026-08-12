"""Gated evaluation engine for Agent-Bench.

Implements a two-phase gated evaluation logic:
Phase 1: Deterministic verification (state diffs, assertions, tool calls, refusal checks).
If Phase 1 fails -> Short-circuit immediately (returns failure verdict without invoking LLM judges).
Phase 2: LLM / Subjective Judge evaluation (runs only if Phase 1 passes).
"""

from typing import Any, Sequence

from agent_bench.core.adapters import JudgeAdapter, JudgeVerdict
from agent_bench.core.artifacts import TraceEvent
from agent_bench.core.protocols import Evaluator
from agent_bench.core.scenarios import Task
from agent_bench.judges.deterministic import DeterministicJudge


class GatedEvaluator(Evaluator):
    """Gated evaluation pipeline combining fast deterministic checks with optional LLM judges."""

    def __init__(
        self,
        deterministic_judge: JudgeAdapter | None = None,
        subjective_judges: Sequence[JudgeAdapter] | None = None,
        evaluator_id: str = "gated_evaluator_v1",
    ) -> None:
        """Initialize GatedEvaluator.

        Args:
            deterministic_judge: Primary deterministic evaluator. Defaults to DeterministicJudge.
            subjective_judges: Optional list of LLM-based or semantic judges for quality scoring.
            evaluator_id: Unique string identifier for this evaluator instance.
        """
        self._deterministic_judge = deterministic_judge or DeterministicJudge()
        self._subjective_judges = list(subjective_judges) if subjective_judges else []
        self._evaluator_id = evaluator_id

    @property
    def evaluator_id(self) -> str:
        """Returns evaluator identifier."""
        return self._evaluator_id

    async def evaluate(
        self,
        task: Task,
        execution_result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict:
        """Evaluates task execution with gated short-circuit logic.

        Args:
            task: Task definition.
            execution_result: Output dictionary from agent execution.
            traces: List of execution TraceEvents.

        Returns:
            JudgeVerdict reflecting combined or short-circuited evaluation score.
        """
        # Phase 1: Deterministic verification
        det_verdict = await self._deterministic_judge.evaluate(task, execution_result, traces)

        # Gate Check: Short-circuit on deterministic failure
        if not det_verdict.passed:
            return JudgeVerdict(
                score=det_verdict.score,
                passed=False,
                reasoning=f"[GATED FAIL] Deterministic check failed: {det_verdict.reasoning}. "
                f"LLM Judge evaluation skipped.",
                judge_id=f"{self.evaluator_id}:gated_short_circuit",
                criteria=det_verdict.criteria,
                metadata={
                    "gated_status": "short_circuited",
                    "deterministic_verdict": {
                        "score": det_verdict.score,
                        "reasoning": det_verdict.reasoning,
                    },
                    "llm_judges_invoked": False,
                },
            )

        # Phase 2: Subjective / LLM Judge evaluation if Phase 1 passed and subjective judges exist
        if not self._subjective_judges:
            return JudgeVerdict(
                score=det_verdict.score,
                passed=True,
                reasoning=f"[GATED PASS] Deterministic check passed: {det_verdict.reasoning}",
                judge_id=self.evaluator_id,
                criteria=det_verdict.criteria,
                metadata={
                    "gated_status": "passed_deterministic_only",
                    "llm_judges_invoked": False,
                },
            )

        subjective_verdicts: list[JudgeVerdict] = []
        for judge in self._subjective_judges:
            sub_verdict = await judge.evaluate(task, execution_result, traces)
            subjective_verdicts.append(sub_verdict)

        # Compute combined score (average of deterministic and subjective judges)
        all_scores = [det_verdict.score] + [v.score for v in subjective_verdicts]
        combined_score = sum(all_scores) / len(all_scores)
        all_passed = det_verdict.passed and all(v.passed for v in subjective_verdicts)

        combined_reasoning = (
            f"[GATED PASS] Deterministic pass. LLM Judges: "
            + "; ".join([f"{v.judge_id} (score={v.score:.2f})" for v in subjective_verdicts])
        )

        return JudgeVerdict(
            score=combined_score,
            passed=all_passed,
            reasoning=combined_reasoning,
            judge_id=self.evaluator_id,
            criteria="gated_composite",
            metadata={
                "gated_status": "passed_full_evaluation",
                "deterministic_verdict": {
                    "score": det_verdict.score,
                    "reasoning": det_verdict.reasoning,
                },
                "subjective_verdicts": [
                    {"judge_id": v.judge_id, "score": v.score, "reasoning": v.reasoning}
                    for v in subjective_verdicts
                ],
                "llm_judges_invoked": True,
            },
        )
