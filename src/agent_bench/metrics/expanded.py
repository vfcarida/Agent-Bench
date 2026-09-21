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

    Vectorized using NumPy for high-throughput batch evaluation with
    fallback to Python standard library when NumPy is unavailable.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (values[0], values[0], values[0])

    try:
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
        mean_val = float(np.mean(arr))
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, n, size=(n_bootstrap, n))
        bootstrap_means = np.mean(arr[indices], axis=1)
        alpha = 1.0 - confidence
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0
        lower = float(np.percentile(bootstrap_means, lower_pct))
        upper = float(np.percentile(bootstrap_means, upper_pct))
        return (mean_val, lower, upper)
    except ImportError:
        import random

        rng_py = random.Random(seed)
        mean = sum(values) / n

        bootstrap_means_list: list[float] = []
        for _ in range(n_bootstrap):
            sample = rng_py.choices(values, k=n)
            bootstrap_means_list.append(sum(sample) / n)

        bootstrap_means_list.sort()
        alpha = 1 - confidence
        lower_idx = max(0, int(alpha / 2 * n_bootstrap))
        upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

        return (mean, bootstrap_means_list[lower_idx], bootstrap_means_list[upper_idx])


def compute_stratified_bootstrap_ci(
    group_values: dict[str, list[float]],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stratified bootstrap CI with weights proportional to group size.

    FINESSE-Bench Section 5.4: for aggregated benchmark groups,
    use stratified bootstrap with weights proportional to dataset size.

    Vectorized using NumPy for high-throughput batch evaluation with
    fallback to Python standard library when NumPy is unavailable.

    Args:
        group_values: Mapping from group/dataset name to scores
    """
    all_values: list[float] = []
    total = sum(len(v) for v in group_values.values())

    if total == 0:
        return (0.0, 0.0, 0.0)

    for vals in group_values.values():
        all_values.extend(vals)

    mean = sum(all_values) / total

    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        boot_means = np.zeros(n_bootstrap, dtype=np.float64)

        for _group, vals in group_values.items():
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                weight = len(vals) / total
                indices = rng.integers(0, len(vals), size=(n_bootstrap, len(vals)))
                group_boot_means = np.mean(arr[indices], axis=1)
                boot_means += group_boot_means * weight

        alpha = 1.0 - confidence
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0
        lower = float(np.percentile(boot_means, lower_pct))
        upper = float(np.percentile(boot_means, upper_pct))
        return (mean, lower, upper)
    except ImportError:
        import random

        group_weights: dict[str, float] = {
            group: len(vals) / total for group, vals in group_values.items()
        }
        rng_py = random.Random(seed)

        bootstrap_means_list = []
        for _ in range(n_bootstrap):
            boot_mean = 0.0
            for group, vals in group_values.items():
                if vals:
                    sample = rng_py.choices(vals, k=len(vals))
                    boot_mean += (sum(sample) / len(sample)) * group_weights[group]
            bootstrap_means_list.append(boot_mean)

        bootstrap_means_list.sort()
        alpha = 1 - confidence
        lower_idx = max(0, int(alpha / 2 * n_bootstrap))
        upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

        return (mean, bootstrap_means_list[lower_idx], bootstrap_means_list[upper_idx])


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
