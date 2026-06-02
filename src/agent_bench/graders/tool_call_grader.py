"""Tool-call grader: verifies expected tool calls were made correctly."""
import re

from typing import Any
from agent_bench.graders.state_grader import GradeResult


class ToolCallGrader:
    """Grades by comparing actual tool calls against required/forbidden patterns."""

    @staticmethod
    def _pattern_to_str(pattern: Any) -> str:
        """Extract tool name string from a pattern (could be dict or str)."""
        if isinstance(pattern, dict):
            return str(pattern.get("tool", pattern.get("name", "")))
        return str(pattern)

    def grade(self, case: dict[str, Any], actual_tool_calls: list[dict[str, Any]]) -> GradeResult:
        required_patterns = [self._pattern_to_str(p) for p in (case.get("required_tool_patterns") or [])]
        forbidden_patterns = [self._pattern_to_str(p) for p in (case.get("forbidden_tool_patterns") or [])]
        # Filter out empty patterns
        required_patterns = [p for p in required_patterns if p]
        forbidden_patterns = [p for p in forbidden_patterns if p]

        if not required_patterns and not forbidden_patterns:
            return GradeResult(score=1.0, passed=True, strategy="tool_call", details={"reason": "no_expectations"})

        actual_names = [tc.get("name", tc.get("tool", "")) for tc in actual_tool_calls]

        # Check forbidden tools
        for pattern in forbidden_patterns:
            for name in actual_names:
                if re.search(pattern, name):
                    return GradeResult(
                        score=0.0,
                        passed=False,
                        strategy="tool_call",
                        failure_category="forbidden_tool_used",
                        details={"forbidden_pattern": pattern, "matched": name},
                    )

        # Check required tools (in order)
        if not required_patterns:
            return GradeResult(score=1.0, passed=True, strategy="tool_call", details={"no_forbidden_used": True})

        matched_required = 0
        search_start = 0
        order_correct = True

        for pattern in required_patterns:
            found = False
            for i in range(search_start, len(actual_names)):
                if re.search(pattern, actual_names[i]):
                    matched_required += 1
                    search_start = i + 1
                    found = True
                    break
            if not found:
                # Check if it exists out of order
                for name in actual_names:
                    if re.search(pattern, name):
                        matched_required += 1
                        order_correct = False
                        break

        recall = matched_required / len(required_patterns) if required_patterns else 1.0
        precision = matched_required / len(actual_tool_calls) if actual_tool_calls else (1.0 if not required_patterns else 0.0)

        if not order_correct and recall == 1.0:
            score = 0.7  # penalty for wrong order
            failure_category = "wrong_order"
        elif recall < 1.0:
            score = recall * 0.8
            failure_category = "required_tool_missing"
        elif len(actual_tool_calls) > len(required_patterns) * 2:
            score = 0.8
            failure_category = "extra_calls"
        else:
            score = 1.0
            failure_category = ""

        passed = score >= 0.8
        if passed:
            failure_category = ""

        return GradeResult(
            score=score,
            passed=passed,
            strategy="tool_call",
            failure_category=failure_category,
            details={
                "precision": precision,
                "recall": recall,
                "order_correct": order_correct,
                "matched": matched_required,
                "required": len(required_patterns),
                "actual_count": len(actual_tool_calls),
            },
        )
