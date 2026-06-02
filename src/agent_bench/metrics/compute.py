"""Metric computation functions."""

import math
from typing import Any

from agent_bench.core.metrics import MetricCategory, MetricResult


def compute_pass_k(results: list[bool], k: int = 1) -> float:
    """Compute pass@k: probability that at least one of k samples passes.

    Uses the unbiased estimator: 1 - C(n-c, k) / C(n, k)
    where n = total samples, c = correct samples.
    """
    n = len(results)
    c = sum(results)
    if n < k:
        return float(c > 0)
    if c == n:
        return 1.0
    if c == 0:
        return 0.0
    # Unbiased estimator
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_task_metrics(
    results: list[bool],
    latencies_ms: list[float] | None = None,
    tokens_in: list[int] | None = None,
    tokens_out: list[int] | None = None,
    cost_per_1k_in: float = 0.0,
    cost_per_1k_out: float = 0.0,
) -> list[MetricResult]:
    """Compute standard metrics for a set of task repetitions."""
    metrics = []
    n = len(results)
    success_count = sum(results)

    metrics.append(MetricResult(
        name="task_success",
        value=success_count / n if n > 0 else 0.0,
        category=MetricCategory.FUNCTIONAL,
        metadata={"threshold": 0.8, "direction": "higher_is_better"},
    ))

    metrics.append(MetricResult(
        name="pass_1",
        value=compute_pass_k(results, 1),
        category=MetricCategory.RELIABILITY,
    ))
    metrics.append(MetricResult(
        name="pass_3",
        value=compute_pass_k(results, 3),
        category=MetricCategory.RELIABILITY,
    ))
    metrics.append(MetricResult(
        name="pass_5",
        value=compute_pass_k(results, min(5, n)),
        category=MetricCategory.RELIABILITY,
    ))

    if latencies_ms:
        avg_latency = sum(latencies_ms) / len(latencies_ms)
        metrics.append(MetricResult(
            name="latency_ms",
            value=avg_latency,
            category=MetricCategory.LATENCY,
            unit="ms",
            metadata={"threshold": 5000, "direction": "lower_is_better"},
        ))

    if tokens_in and tokens_out:
        total_in = sum(tokens_in)
        total_out = sum(tokens_out)
        cost = (total_in / 1000 * cost_per_1k_in) + (total_out / 1000 * cost_per_1k_out)
        metrics.append(MetricResult(
            name="total_tokens_in",
            value=float(total_in),
            category=MetricCategory.COST,
            unit="tokens",
        ))
        metrics.append(MetricResult(
            name="total_tokens_out",
            value=float(total_out),
            category=MetricCategory.COST,
            unit="tokens",
        ))
        metrics.append(MetricResult(
            name="estimated_cost_usd",
            value=cost,
            category=MetricCategory.COST,
            unit="USD",
        ))

    return metrics
