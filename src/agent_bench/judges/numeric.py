"""Numeric correctness judge: verifies numerical values in responses."""

import re
from typing import Any

from agent_bench.core.adapters import JudgeVerdict
from agent_bench.core.artifacts import TraceEvent
from agent_bench.core.scenarios import Task


class NumericCorrectnessJudge:
    """Evaluates numeric accuracy of responses against gold references.

    Supports:
    - Exact match (integers, IDs)
    - Approximate match with tolerance (percentages, financial values)
    - Range checks (within expected bounds)
    """

    def __init__(self, default_tolerance: float = 0.01):
        self._default_tolerance = default_tolerance

    @property
    def judge_id(self) -> str:
        return "numeric_correctness"

    @property
    def judge_type(self) -> str:
        return "deterministic"

    async def evaluate(
        self,
        task: Task,
        result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict:
        response = result.get("response", "")
        expected_values = task.metadata.get("expected_numeric_values", {})
        tolerance_overrides = task.metadata.get("numeric_tolerances", {})

        if not expected_values:
            # Try gold_references for numeric expectations
            expected_values = _extract_expected_from_gold(task.gold_references)

        if not expected_values:
            return JudgeVerdict(
                score=1.0,
                passed=True,
                reasoning="No numeric expectations defined.",
                judge_id=self.judge_id,
                criteria="numeric_correctness",
            )

        # Extract numbers from response
        response_numbers = _extract_numbers(response)

        # Check each expected value
        checks = []
        for key, expected in expected_values.items():
            tolerance = tolerance_overrides.get(key, self._default_tolerance)
            found = _find_matching_number(expected, response_numbers, tolerance)
            checks.append({
                "key": key,
                "expected": expected,
                "found": found,
                "tolerance": tolerance,
                "passed": found is not None,
            })

        passed_count = sum(1 for c in checks if c["passed"])
        total = len(checks)
        score = passed_count / total if total > 0 else 1.0
        passed = score >= 0.8

        failed_checks = [c for c in checks if not c["passed"]]

        return JudgeVerdict(
            score=score,
            passed=passed,
            reasoning=(
                f"Numeric correctness: {passed_count}/{total} values matched. "
                f"Failed: {[c['key'] for c in failed_checks]}"
            ),
            judge_id=self.judge_id,
            criteria="numeric_correctness",
            metadata={
                "checks": checks,
                "response_numbers": response_numbers[:20],
            },
        )


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    # Match patterns like: 100, 1.5, 1,5, R$1.000,00, 15%, 100.000,50
    patterns = [
        r'R\$\s*([\d.]+,\d{2})',  # BRL format: R$ 1.000,00
        r'([\d.]+,\d+)\s*%',      # Percentage with comma decimal
        r'(\d+,\d+)\s*%',         # Simple percentage
        r'(\d+[\.,]\d+)',          # Decimal numbers
        r'(\d+)\s*%',             # Integer percentage
        r'(?<!\.)(\d{1,3}(?:\.\d{3})*,\d{2})(?!\.)',  # BR format 1.000,00
        r'(\d+)',                  # Plain integers
    ]

    numbers = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                value_str = match.group(1)
                numbers.append(_parse_br_number(value_str))
            except (ValueError, IndexError):
                continue

    return list(set(numbers))


def _parse_br_number(s: str) -> float:
    """Parse a Brazilian-format number (1.000,50 -> 1000.50)."""
    # If has both . and , : BR format
    if '.' in s and ',' in s:
        return float(s.replace('.', '').replace(',', '.'))
    # If only comma: decimal separator
    if ',' in s:
        return float(s.replace(',', '.'))
    return float(s)


def _find_matching_number(
    expected: float, candidates: list[float], tolerance: float
) -> float | None:
    """Find a candidate number within tolerance of expected."""
    if expected == 0:
        for c in candidates:
            if abs(c) <= tolerance:
                return c
        return None

    for c in candidates:
        relative_diff = abs(c - expected) / abs(expected)
        if relative_diff <= tolerance:
            return c
    return None


def _extract_expected_from_gold(gold_references: list[str]) -> dict[str, float]:
    """Try to extract numeric expectations from gold reference strings."""
    values = {}
    for i, ref in enumerate(gold_references):
        numbers = _extract_numbers(ref)
        for j, n in enumerate(numbers):
            values[f"gold_{i}_num_{j}"] = n
    return values
