"""Rubric-based grader: scores response against weighted dimensions.

Inspired by BankerToolBench (arXiv:2604.11304) "Gandalf the Grader":
- Criteria have importance levels (critical/high/medium/low) with weights (10/5/3/1)
- Critical criteria act as hard gates: if ANY critical criterion fails,
  the overall score is capped at 0.5 regardless of other dimensions
- Failure analysis breaks down by importance level
"""
import re
from typing import Callable, Any

from agent_bench.core.schema_v2 import IMPORTANCE_WEIGHTS
from agent_bench.graders.state_grader import GradeResult


class RubricGrader:
    """Grades by scoring response against rubric dimensions with importance hierarchy."""

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
        response_lower = response.lower()
        dim_scores: dict[str, float] = {}
        missing_dimensions: list[str] = []
        critical_failed: list[str] = []
        failure_by_importance: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        # Compute per-dimension scores using importance weights
        total_weight = 0.0
        weighted_score = 0.0

        for dim in dimensions:
            name = dim.get("name", "unnamed")
            importance = dim.get("importance", "medium")
            weight = IMPORTANCE_WEIGHTS.get(importance, dim.get("weight", 3))
            keywords = dim.get("keywords", [])

            if not keywords:
                dim_score = 1.0
            else:
                matched = sum(1 for kw in keywords if re.search(re.escape(kw.lower()), response_lower))
                dim_score = matched / len(keywords)

            dim_scores[name] = dim_score
            total_weight += weight
            weighted_score += dim_score * weight

            if dim_score == 0.0:
                missing_dimensions.append(name)

            # Track failures by importance level
            if dim_score < 0.6:
                failure_by_importance[importance] = failure_by_importance.get(importance, 0) + 1
                if importance == "critical":
                    critical_failed.append(name)

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # BTB hard gate: if any critical criterion fails, cap score at 0.5
        if critical_failed:
            final_score = min(final_score, 0.5)

        passed = final_score >= 0.6

        if not passed:
            if critical_failed:
                failure_category = "critical_gate_failed"
            elif missing_dimensions:
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
            details={
                "dimension_scores": dim_scores,
                "missing_dimensions": missing_dimensions,
                "critical_failed": critical_failed,
                "failure_by_importance": failure_by_importance,
            },
        )

    def _grade_with_judge(self, dimensions: list[dict[str, Any]], response: str, judge_fn: Callable[..., Any]) -> GradeResult:
        dim_scores: dict[str, float] = {}
        critical_failed: list[str] = []
        failure_by_importance: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        total_weight = 0.0
        weighted_score = 0.0

        for dim in dimensions:
            name = dim.get("name", "unnamed")
            importance = dim.get("importance", "medium")
            weight = IMPORTANCE_WEIGHTS.get(importance, dim.get("weight", 3))
            criteria = dim.get("criteria", "")

            score = judge_fn(response, criteria)
            score = max(0.0, min(1.0, float(score)))
            dim_scores[name] = score
            total_weight += weight
            weighted_score += score * weight

            # Track failures by importance level
            if score < 0.6:
                failure_by_importance[importance] = failure_by_importance.get(importance, 0) + 1
                if importance == "critical":
                    critical_failed.append(name)

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # BTB hard gate: critical failures cap the score
        if critical_failed:
            final_score = min(final_score, 0.5)

        passed = final_score >= 0.6
        failure_category = ""
        if not passed:
            failure_category = "critical_gate_failed" if critical_failed else "insufficient_evidence"

        return GradeResult(
            score=final_score,
            passed=passed,
            strategy="rubric",
            failure_category=failure_category,
            details={
                "dimension_scores": dim_scores,
                "critical_failed": critical_failed,
                "failure_by_importance": failure_by_importance,
            },
        )
