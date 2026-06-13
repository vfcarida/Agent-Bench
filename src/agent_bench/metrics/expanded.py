"""Expanded metric computations for agent bench."""
import math

from typing import Any
from agent_bench.graders.state_grader import GradeResult


def compute_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) using normal approximation.

    NOTE: For more accurate intervals, especially with small samples,
    use compute_bootstrap_ci() instead (FINESSE-Bench style).
    """
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


def compute_bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) using bootstrap resampling.

    Inspired by FINESSE-Bench (arXiv:2605.15482, Section 5.4):
    95% confidence intervals computed using bootstrap. For aggregated
    benchmark groups, stratified bootstrap with weights proportional
    to dataset size is used.

    This is more accurate than normal approximation for:
    - Small sample sizes
    - Non-normal distributions
    - Binary outcomes (pass/fail)
    """
    import random

    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (values[0], values[0], values[0])

    rng = random.Random(seed)
    mean = sum(values) / n

    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(values, k=n)
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = max(0, int(alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

    return (mean, bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def compute_stratified_bootstrap_ci(
    group_values: dict[str, list[float]],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stratified bootstrap CI with weights proportional to group size.

    FINESSE-Bench Section 5.4: for aggregated benchmark groups,
    use stratified bootstrap with weights proportional to dataset size.

    Args:
        group_values: Mapping from group/dataset name to scores
    """
    import random

    all_values = []
    group_weights: dict[str, float] = {}
    total = sum(len(v) for v in group_values.values())

    if total == 0:
        return (0.0, 0.0, 0.0)

    for group, vals in group_values.items():
        all_values.extend(vals)
        group_weights[group] = len(vals) / total

    mean = sum(all_values) / total
    rng = random.Random(seed)

    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        boot_mean = 0.0
        for group, vals in group_values.items():
            if vals:
                sample = rng.choices(vals, k=len(vals))
                boot_mean += (sum(sample) / len(sample)) * group_weights[group]
        bootstrap_means.append(boot_mean)

    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = max(0, int(alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

    return (mean, bootstrap_means[lower_idx], bootstrap_means[upper_idx])


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
