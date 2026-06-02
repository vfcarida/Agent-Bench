"""Rubric-based grader: scores response against weighted dimensions."""
import re
from typing import Callable, Any

from agent_bench.graders.state_grader import GradeResult


class RubricGrader:
    """Grades by scoring response against rubric dimensions."""

    def grade(
        self,
        case: dict[str, Any],
        actual_response: str,
        judge_fn: Callable[..., Any] | None = None,
    ) -> GradeResult:
        rubric = case.get("rubric", {})
        dimensions = rubric.get("dimensions", [])

        if not dimensions:
            return GradeResult(score=1.0, passed=True, strategy="rubric", details={"reason": "no_rubric"})

        if judge_fn is not None:
            return self._grade_with_judge(dimensions, actual_response, judge_fn)

        return self._grade_with_keywords(dimensions, actual_response)

    def _grade_with_keywords(self, dimensions: list[dict[str, Any]], response: str) -> GradeResult:
        total_weight = sum(d.get("weight", 1.0) for d in dimensions)
        weighted_score = 0.0
        dim_scores: dict[str, float] = {}
        missing_dimensions: list[str] = []

        response_lower = response.lower()

        for dim in dimensions:
            name = dim.get("name", "unnamed")
            weight = dim.get("weight", 1.0)
            keywords = dim.get("keywords", [])

            if not keywords:
                # No keywords to check, full score for this dimension
                dim_score = 1.0
            else:
                matched = sum(1 for kw in keywords if re.search(re.escape(kw.lower()), response_lower))
                dim_score = matched / len(keywords)

            dim_scores[name] = dim_score
            weighted_score += dim_score * weight

            if dim_score == 0.0:
                missing_dimensions.append(name)

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        passed = final_score >= 0.6

        if not passed:
            if missing_dimensions:
                failure_category = "missing_dimension"
            elif final_score < 0.4:
                failure_category = "low_coverage"
            else:
                failure_category = "insufficient_evidence"
        else:
            failure_category = ""

        return GradeResult(
            score=final_score,
            passed=passed,
            strategy="rubric",
            failure_category=failure_category,
            details={"dimension_scores": dim_scores, "missing_dimensions": missing_dimensions},
        )

    def _grade_with_judge(self, dimensions: list[dict[str, Any]], response: str, judge_fn: Callable[..., Any]) -> GradeResult:
        total_weight = sum(d.get("weight", 1.0) for d in dimensions)
        weighted_score = 0.0
        dim_scores: dict[str, float] = {}

        for dim in dimensions:
            name = dim.get("name", "unnamed")
            weight = dim.get("weight", 1.0)
            criteria = dim.get("criteria", "")

            score = judge_fn(response, criteria)
            score = max(0.0, min(1.0, float(score)))
            dim_scores[name] = score
            weighted_score += score * weight

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        passed = final_score >= 0.6
        failure_category = "" if passed else "insufficient_evidence"

        return GradeResult(
            score=final_score,
            passed=passed,
            strategy="rubric",
            failure_category=failure_category,
            details={"dimension_scores": dim_scores},
        )
