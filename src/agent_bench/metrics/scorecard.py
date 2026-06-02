"""Scorecard computation from task results."""

from typing import Any

from agent_bench.core.metrics import MetricCategory, MetricResult, Scorecard, WEIGHTING_PROFILES


def compute_scorecard(
    system_id: str,
    domain: str,
    task_results: list[dict[str, Any]],
    weighting_profile: str = "transactional_high_risk",
) -> Scorecard:
    """Compute a scorecard from per-task results.

    Each task_result should contain:
    - passed: bool
    - policy_violated: bool
    - latency_ms: float
    - tokens_in: int
    - tokens_out: int
    - cost_usd: float
    - repetitions: list[bool]  (for reliability)
    """
    weights = WEIGHTING_PROFILES.get(weighting_profile, WEIGHTING_PROFILES["transactional_high_risk"])

    if not task_results:
        return Scorecard(system_id=system_id, domain=domain, weights=weights)

    # Functional: pass rate
    total = len(task_results)
    passed = sum(1 for r in task_results if r.get("passed", False))
    functional_score = passed / total

    # Risk: 1 - policy_violation_rate
    violations = sum(1 for r in task_results if r.get("policy_violated", False))
    risk_score = 1.0 - (violations / total)

    # Cost: normalize to 0-1 (lower is better, cap at $1/task as max)
    costs = [r.get("cost_usd", 0.0) for r in task_results]
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    cost_score = max(0.0, 1.0 - avg_cost)  # $1 maps to 0, $0 maps to 1

    # Latency: normalize (target < 2000ms for score=1, >10000ms for score=0)
    latencies = [r.get("latency_ms", 0.0) for r in task_results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    latency_score = max(0.0, min(1.0, 1.0 - (avg_latency - 2000) / 8000))

    # Reliability: consistency across repetitions
    reliabilities = []
    for r in task_results:
        reps = r.get("repetitions", [r.get("passed", False)])
        if reps:
            reliabilities.append(sum(reps) / len(reps))
    reliability_score = sum(reliabilities) / len(reliabilities) if reliabilities else functional_score

    metrics = [
        MetricResult(name="functional_score", value=functional_score, category=MetricCategory.FUNCTIONAL),
        MetricResult(name="risk_score", value=risk_score, category=MetricCategory.SAFETY),
        MetricResult(name="cost_score", value=cost_score, category=MetricCategory.COST),
        MetricResult(name="latency_score", value=latency_score, category=MetricCategory.LATENCY),
        MetricResult(name="reliability_score", value=reliability_score, category=MetricCategory.RELIABILITY),
    ]

    return Scorecard(
        system_id=system_id,
        domain=domain,
        functional_score=functional_score,
        risk_score=risk_score,
        cost_score=cost_score,
        latency_score=latency_score,
        reliability_score=reliability_score,
        weights=weights,
        metrics=metrics,
    )
