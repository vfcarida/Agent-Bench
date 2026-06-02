"""Metrics report aggregation for benchmark runs."""
import math
from dataclasses import dataclass, field

from agent_bench.graders.state_grader import GradeResult
from agent_bench.metrics.expanded import (
    categorize_failures,
    compute_confidence_interval,
    compute_state_accuracy,
)


@dataclass
class MetricsReport:
    """Aggregated metrics for a benchmark run."""

    success_rate: float = 0.0
    pass_at_1: float = 0.0
    pass_at_3: float = 0.0
    pass_at_5: float = 0.0
    state_accuracy: float = 0.0
    tool_call_precision: float = 0.0
    tool_call_recall: float = 0.0
    policy_compliance_rate: float = 0.0
    groundedness: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    cost_per_success: float = 0.0
    failure_taxonomy: dict[str, int] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile from sorted list."""
    if not sorted_values:
        return 0.0
    idx = (p / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_values[lower]
    frac = idx - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def _compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k: probability at least one of k samples passes given c/n pass."""
    if n == 0 or k == 0:
        return 0.0
    if c >= n:
        return 1.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    # 1 - C(n-c, k) / C(n, k)
    numerator = 1.0
    denominator = 1.0
    for i in range(k):
        numerator *= (n - c - i)
        denominator *= (n - i)
    return 1.0 - numerator / denominator


from typing import Any

def build_metrics_report(grade_results: list[GradeResult], run_metadata: dict[str, Any]) -> MetricsReport:
    """Build a MetricsReport from grade results and run metadata."""
    report = MetricsReport()

    if not grade_results:
        return report

    n = len(grade_results)
    passed = sum(1 for r in grade_results if r.passed)

    report.success_rate = passed / n
    report.pass_at_1 = report.success_rate
    report.pass_at_3 = _compute_pass_at_k(n, passed, 3)
    report.pass_at_5 = _compute_pass_at_k(n, passed, 5)

    # State accuracy
    report.state_accuracy = compute_state_accuracy(grade_results)

    # Tool call metrics from details
    precisions = [float(str(r.details.get("precision"))) for r in grade_results if r.details.get("precision") is not None]
    recalls = [float(str(r.details.get("recall"))) for r in grade_results if r.details.get("recall") is not None]
    report.tool_call_precision = sum(precisions) / len(precisions) if precisions else 0.0
    report.tool_call_recall = sum(recalls) / len(recalls) if recalls else 0.0

    # Policy compliance and groundedness from metadata
    report.policy_compliance_rate = run_metadata.get("policy_compliance_rate", 1.0)
    report.groundedness = run_metadata.get("groundedness", 1.0)

    # Latencies
    latencies = sorted(run_metadata.get("latencies", []))
    if latencies:
        report.latency_p50 = _percentile(latencies, 50)
        report.latency_p95 = _percentile(latencies, 95)
        report.latency_p99 = _percentile(latencies, 99)

    # Cost / tokens
    report.total_tokens = run_metadata.get("total_tokens", 0)
    report.total_cost = run_metadata.get("total_cost", 0.0)
    report.cost_per_success = report.total_cost / passed if passed > 0 else float("inf")

    # Failure taxonomy
    report.failure_taxonomy = categorize_failures(grade_results)

    # Confidence intervals on scores
    scores = [r.score for r in grade_results]
    report.confidence_intervals["score"] = compute_confidence_interval(scores)
    if latencies:
        report.confidence_intervals["latency"] = compute_confidence_interval(latencies)

    return report
