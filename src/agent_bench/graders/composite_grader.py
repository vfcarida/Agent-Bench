"""Composite grader: selects and combines graders based on strategy."""
from collections.abc import Callable
from typing import Any

from agent_bench.graders.rubric_grader import RubricGrader
from agent_bench.graders.state_grader import GradeResult, StateGrader
from agent_bench.graders.tool_call_grader import ToolCallGrader


class CompositeGrader:
    """Selects and runs the appropriate grader(s) based on case grading_strategy."""

    def __init__(self) -> None:
        self.state_grader = StateGrader()
        self.tool_call_grader = ToolCallGrader()
        self.rubric_grader = RubricGrader()

    def grade(
        self,
        case: Any,
        actual_state: dict[str, Any] | None = None,
        actual_tool_calls: list[dict[str, Any]] | None = None,
        actual_response: str = "",
        judge_fn: Callable[..., Any] | None = None,
    ) -> GradeResult:
        if isinstance(case, dict):
            strategy = case.get("grading_strategy", "composite")
        else:
            strategy = getattr(case, "grading_strategy", None)
            if not strategy and hasattr(case, "metadata") and isinstance(case.metadata, dict):
                strategy = case.metadata.get("grading_strategy", "composite")
            strategy = getattr(strategy, "value", str(strategy)) if strategy else "composite"

        if strategy in ("state_based", "state_change_based"):
            return self.state_grader.grade(case, actual_state or {}, actual_response)
        elif strategy in ("tool_call", "tool_call_based"):
            return self.tool_call_grader.grade(case, actual_tool_calls or [])
        elif strategy in ("rubric", "rubric_based"):
            return self.rubric_grader.grade(case, actual_response, judge_fn)
        else:
            # composite: run all applicable, combine with weights
            return self._composite_grade(case, actual_state, actual_tool_calls, actual_response, judge_fn)

    def _composite_grade(
        self,
        case: Any,
        actual_state: dict[str, Any] | None,
        actual_tool_calls: list[dict[str, Any]] | None,
        actual_response: str,
        judge_fn: Callable[..., Any] | None,
    ) -> GradeResult:
        from agent_bench.graders.state_grader import _extract_case_dict

        case_dict = _extract_case_dict(case)
        weights = {"state": 0.4, "tool_call": 0.3, "rubric": 0.3}
        scores: dict[str, float] = {}
        sub_results: dict[str, GradeResult] = {}
        active_weight = 0.0

        # State grading
        if actual_state is not None and (case_dict.get("expected_state_changes") or case_dict.get("expected_outcome")):
            result = self.state_grader.grade(case, actual_state, actual_response)
            scores["state"] = result.score
            sub_results["state"] = result
            active_weight += weights["state"]

        # Tool call grading
        if actual_tool_calls is not None and (case_dict.get("required_tool_patterns") or case_dict.get("forbidden_tool_patterns")):
            result = self.tool_call_grader.grade(case, actual_tool_calls)
            scores["tool_call"] = result.score
            sub_results["tool_call"] = result
            active_weight += weights["tool_call"]

        # Rubric grading
        if actual_response and case_dict.get("rubric"):
            result = self.rubric_grader.grade(case, actual_response, judge_fn)
            scores["rubric"] = result.score
            sub_results["rubric"] = result
            active_weight += weights["rubric"]

        if active_weight == 0.0:
            return GradeResult(score=1.0, passed=True, strategy="composite", details={"reason": "no_applicable_graders"})

        # Weighted average normalized by active weights
        final_score = sum(
            scores[k] * weights[k] for k in scores
        ) / active_weight

        passed = final_score >= 0.8

        # Determine failure category from worst sub-result
        failure_category = ""
        if not passed:
            worst = min(sub_results.values(), key=lambda r: r.score)
            failure_category = worst.failure_category or "composite_below_threshold"

        return GradeResult(
            score=final_score,
            passed=passed,
            strategy="composite",
            failure_category=failure_category,
            details={"sub_scores": scores, "active_weight": active_weight},
        )
