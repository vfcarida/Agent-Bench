"""Expanded metric computations for agent bench."""
import math

from typing import Any
from agent_bench.graders.state_grader import GradeResult


def compute_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) using normal approximation."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (values[0], values[0], values[0])

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    # Z-scores for common confidence levels (normal approximation)
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    margin = z * std_err
    return (mean, mean - margin, mean + margin)


def compute_tool_call_precision(expected_calls: list[Any], actual_calls: list[Any]) -> float:
    """Fraction of actual calls that match an expected call."""
    if not actual_calls:
        return 1.0 if not expected_calls else 0.0
    matched = sum(1 for ac in actual_calls if ac in expected_calls)
    return matched / len(actual_calls)


def compute_tool_call_recall(expected_calls: list[Any], actual_calls: list[Any]) -> float:
    """Fraction of expected calls that appear in actual calls."""
    if not expected_calls:
        return 1.0
    matched = sum(1 for ec in expected_calls if ec in actual_calls)
    return matched / len(expected_calls)


def compute_policy_compliance_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of results where policy was not violated."""
    if not results:
        return 1.0
    compliant = sum(1 for r in results if not r.get("policy_violated", False))
    return compliant / len(results)


def compute_cost_per_success(total_cost: float, successes: int) -> float:
    """Cost divided by number of successes."""
    if successes == 0:
        return float("inf")
    return total_cost / successes


def categorize_failures(results: list[GradeResult]) -> dict[str, int]:
    """Counts failures by failure_category."""
    counts: dict[str, int] = {}
    for r in results:
        if not r.passed and r.failure_category:
            counts[r.failure_category] = counts.get(r.failure_category, 0) + 1
    return counts


def compute_state_accuracy(results: list[GradeResult]) -> float:
    """Average state match score across all graded cases."""
    state_results = [r for r in results if r.strategy == "state_based" or r.details.get("match_ratio") is not None]
    if not state_results:
        return 1.0
    scores = []
    for r in state_results:
        if "match_ratio" in r.details:
            scores.append(r.details["match_ratio"])
        else:
            scores.append(r.score)
    return sum(scores) / len(scores) if scores else 1.0


def compute_groundedness(results: list[dict[str, Any]]) -> float:
    """Fraction of claims with evidence (claims_with_evidence / total_claims)."""
    if not results:
        return 1.0
    total_claims = 0
    supported_claims = 0
    for r in results:
        claims = r.get("total_claims", 0)
        supported = r.get("supported_claims", 0)
        total_claims += claims
        supported_claims += supported
    if total_claims == 0:
        return 1.0
    return supported_claims / total_claims
