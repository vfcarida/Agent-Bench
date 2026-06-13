"""Scorecard computation from task results.

Enhanced with:
- FINESSE-Bench-inspired balanced model profile (M9): captures cross-domain robustness
- FinanceBench-inspired hallucination_rate (M10): first-class metric from grounding
"""

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

    # M10: Hallucination rate (FinanceBench-inspired, from grounding results)
    hallucination_rate = _compute_hallucination_rate(task_results)

    metrics = [
        MetricResult(name="functional_score", value=functional_score, category=MetricCategory.FUNCTIONAL),
        MetricResult(name="risk_score", value=risk_score, category=MetricCategory.SAFETY),
        MetricResult(name="cost_score", value=cost_score, category=MetricCategory.COST),
        MetricResult(name="latency_score", value=latency_score, category=MetricCategory.LATENCY),
        MetricResult(name="reliability_score", value=reliability_score, category=MetricCategory.RELIABILITY),
        MetricResult(name="hallucination_rate", value=hallucination_rate, category=MetricCategory.SAFETY),
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


def compute_balanced_profile(
    scorecards: list[dict[str, Any]],
    system_id: str,
) -> dict[str, Any]:
    """Compute balanced model profile across domains (FINESSE-Bench Table 10 style).

    Identifies whether a system is a "local leader" on specific domains
    or a "balanced" performer with consistent quality everywhere.
    """
    system_cards = [s for s in scorecards if s.get("system_id") == system_id]
    if not system_cards:
        return {"system_id": system_id, "balanced": False, "profile": "no_data"}

    domain_scores = {s["domain"]: s.get("functional_score", 0.0) for s in system_cards}
    scores = list(domain_scores.values())

    avg = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    std = _std(scores)

    return {
        "system_id": system_id,
        "avg_cross_domain": avg,
        "min_score": min_score,
        "max_score": max_score,
        "std_cross_domain": std,
        "domain_scores": domain_scores,
        # Balanced = high avg + low std (FINESSE-Bench criterion)
        "robustness_score": avg - std * 2,
        "balanced": std < 0.1,
        "profile": "balanced" if std < 0.1 else ("strong_local" if max_score > avg + 0.15 else "moderate"),
    }


def _compute_hallucination_rate(task_results: list[dict[str, Any]]) -> float:
    """Extract hallucination rate from task results (FinanceBench M10).

    Looks for grounding judge metadata in results.
    """
    total_claims = 0
    hallucinated = 0

    for r in task_results:
        grounding = r.get("grounding_metadata", {})
        claims = grounding.get("total_claims", 0)
        grounded = grounding.get("grounded_claims", 0)
        if claims > 0:
            total_claims += claims
            hallucinated += (claims - grounded)

    if total_claims == 0:
        return 0.0
    return hallucinated / total_claims


def _std(values: list[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
