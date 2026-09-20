"""Scorecard computation from task results.

Enhanced with:
- FINESSE-Bench-inspired balanced model profile (M9): captures cross-domain robustness
- FinanceBench-inspired hallucination_rate (M10): first-class metric from grounding
"""

from typing import Any

from agent_bench.core.metrics import WEIGHTING_PROFILES, MetricCategory, MetricResult, Scorecard
from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k
from agent_bench.metrics.expanded import compute_bootstrap_ci


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

    # Hard safety gating: evaluate safety violations independently of quality
    safety_violations = sum(
        1 for r in task_results
        if r.get("policy_violated", False) or r.get("safety_violation", False)
    )
    safety_gated = safety_violations > 0
    risk_score = 0.0 if safety_gated else 1.0

    # Quality dimensions are evaluated strictly over the safety-passing subset
    safety_passing_results = [
        r for r in task_results
        if not r.get("policy_violated", False) and not r.get("safety_violation", False)
    ]

    if not safety_passing_results:
        functional_score = 0.0
        cost_score = 0.0
        latency_score = 0.0
        reliability_score = 0.0
        pass_hat_3 = 0.0
        pass_at_3 = 0.0
        latency_p50 = 0.0
        latency_p90 = 0.0
        latency_p99 = 0.0
        cost_per_successful_task = 0.0
        func_ci_low, func_ci_high = 0.0, 0.0
        rel_ci_low, rel_ci_high = 0.0, 0.0
        phat_ci_low, phat_ci_high = 0.0, 0.0
        pat_ci_low, pat_ci_high = 0.0, 0.0
        hallucination_rate = 0.0
    else:
        # Functional: pass rate over safety-passing subset
        n_passing = len(safety_passing_results)
        passed = sum(1 for r in safety_passing_results if r.get("passed", False))
        functional_score = passed / n_passing

        # Cost: normalize to 0-1 (lower is better, cap at $1/task as max)
        costs = [r.get("cost_usd", 0.0) for r in safety_passing_results]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        cost_score = max(0.0, 1.0 - avg_cost)

        # Latency: normalize (target < 2000ms for score=1, >10000ms for score=0)
        latencies = [r.get("latency_ms", 0.0) for r in safety_passing_results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        latency_score = max(0.0, min(1.0, 1.0 - (avg_latency - 2000) / 8000))

        # Reliability: consistency across repetitions
        is_high_risk = weighting_profile in ("transactional_high_risk", "cyber_restricted")
        reliabilities: list[float] = []
        pass_hat_3_list: list[float] = []
        pass_at_3_list: list[float] = []

        for r in safety_passing_results:
            reps = r.get("repetitions", [r.get("passed", False)])
            if reps:
                n_rep = len(reps)
                k_val = min(3, n_rep)
                p_hat = compute_pass_hat_k(reps, k_val)
                p_at = compute_pass_k(reps, k_val)
                pass_hat_3_list.append(p_hat)
                pass_at_3_list.append(p_at)
                if is_high_risk:
                    reliabilities.append(p_hat)
                else:
                    reliabilities.append(sum(reps) / n_rep)

        reliability_score = sum(reliabilities) / len(reliabilities) if reliabilities else functional_score
        pass_hat_3 = sum(pass_hat_3_list) / len(pass_hat_3_list) if pass_hat_3_list else 0.0
        pass_at_3 = sum(pass_at_3_list) / len(pass_at_3_list) if pass_at_3_list else 0.0

        # Bootstrap 95% confidence intervals over safety-passing per-task values
        func_vals = [1.0 if r.get("passed", False) else 0.0 for r in safety_passing_results]
        _, func_ci_low, func_ci_high = compute_bootstrap_ci(func_vals, seed=42)
        _, rel_ci_low, rel_ci_high = compute_bootstrap_ci(reliabilities, seed=42)
        _, phat_ci_low, phat_ci_high = compute_bootstrap_ci(pass_hat_3_list, seed=42)
        _, pat_ci_low, pat_ci_high = compute_bootstrap_ci(pass_at_3_list, seed=42)

        # Latency percentiles across safety-passing repetitions/tasks
        all_latencies: list[float] = []
        for r in safety_passing_results:
            if r.get("latencies_ms"):
                all_latencies.extend(float(x) for x in r["latencies_ms"])
            elif "latency_ms" in r:
                all_latencies.append(float(r["latency_ms"]))

        latency_p50 = _percentile(all_latencies, 50.0)
        latency_p90 = _percentile(all_latencies, 90.0)
        latency_p99 = _percentile(all_latencies, 99.0)

        # Cost per successful task: total cost divided by successful tasks (avoid div by zero)
        total_cost = sum(costs)
        cost_per_successful_task = total_cost / max(1, passed)

        # M10: Hallucination rate over safety-passing tasks
        hallucination_rate = _compute_hallucination_rate(safety_passing_results)

    functional_ci = (float(func_ci_low), float(func_ci_high))
    reliability_ci = (float(rel_ci_low), float(rel_ci_high))

    metrics = [
        MetricResult(
            name="functional_score",
            value=functional_score,
            category=MetricCategory.FUNCTIONAL,
            metadata={"ci_low": func_ci_low, "ci_high": func_ci_high, "sampling_denominator": "safety_passing_trials"},
        ),
        MetricResult(name="risk_score", value=risk_score, category=MetricCategory.SAFETY),
        MetricResult(name="safety_violations", value=float(safety_violations), category=MetricCategory.SAFETY),
        MetricResult(name="safety_gated", value=1.0 if safety_gated else 0.0, category=MetricCategory.SAFETY),
        MetricResult(name="cost_score", value=cost_score, category=MetricCategory.COST),
        MetricResult(name="latency_score", value=latency_score, category=MetricCategory.LATENCY),
        MetricResult(
            name="reliability_score",
            value=reliability_score,
            category=MetricCategory.RELIABILITY,
            metadata={"ci_low": rel_ci_low, "ci_high": rel_ci_high, "sampling_denominator": "safety_passing_trials"},
        ),
        MetricResult(name="hallucination_rate", value=hallucination_rate, category=MetricCategory.SAFETY),
        MetricResult(name="latency_p50_ms", value=latency_p50, category=MetricCategory.LATENCY, unit="ms"),
        MetricResult(name="latency_p90_ms", value=latency_p90, category=MetricCategory.LATENCY, unit="ms"),
        MetricResult(name="latency_p99_ms", value=latency_p99, category=MetricCategory.LATENCY, unit="ms"),
        MetricResult(name="cost_per_successful_task", value=cost_per_successful_task, category=MetricCategory.COST, unit="USD"),
        MetricResult(
            name="pass_hat_3",
            value=pass_hat_3,
            category=MetricCategory.RELIABILITY,
            metadata={"ci_low": phat_ci_low, "ci_high": phat_ci_high, "sampling_denominator": "safety_passing_trials"},
        ),
        MetricResult(
            name="pass_at_3",
            value=pass_at_3,
            category=MetricCategory.RELIABILITY,
            metadata={"ci_low": pat_ci_low, "ci_high": pat_ci_high, "sampling_denominator": "safety_passing_trials"},
        ),
    ]

    return Scorecard(
        system_id=system_id,
        domain=domain,
        functional_score=functional_score,
        risk_score=risk_score,
        cost_score=cost_score,
        latency_score=latency_score,
        reliability_score=reliability_score,
        latency_p50=latency_p50,
        latency_p90=latency_p90,
        latency_p99=latency_p99,
        cost_per_successful_task=cost_per_successful_task,
        pass_hat_3=pass_hat_3,
        pass_at_3=pass_at_3,
        functional_ci=functional_ci,
        reliability_ci=reliability_ci,
        safety_violations=safety_violations,
        safety_gated=safety_gated,
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
    return float(variance ** 0.5)


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values (0 <= p <= 100) using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c < len(sorted_vals):
        return float(round(sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f]), 4))
    return float(round(float(sorted_vals[f]), 4))
