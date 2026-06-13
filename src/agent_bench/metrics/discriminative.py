"""Discriminative power analysis for benchmark tasks.

Inspired by FINESSE-Bench (arXiv:2605.15482) Section 7.4:
- Measures how well tasks/questions separate models
- Computes unanimous fail rate, unanimous success rate, and mid-band 10-90
- Detects benchmark saturation (too many unanimous successes)
- Enables benchmark quality assessment beyond raw accuracy

Key insight from FINESSE-Bench: many open benchmarks have low discriminative
power because too many questions are solved by all models (saturation).
The most informative questions are those in the "mid-band" where 10-90%
of models answer correctly. CFA-like Level 2 achieved 81.23% mid-band,
while TAT-QA had only 12.35%.
"""

from typing import Any


def compute_discriminative_power(
    task_results: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    """Compute discriminative power metrics for a set of tasks.

    FINESSE-Bench Table 8 style analysis.

    Args:
        task_results: Mapping from system_id -> task_id -> passed (bool)

    Returns:
        Dict with discriminative power metrics
    """
    if not task_results:
        return _empty_result()

    # Collect all task IDs
    all_tasks: set[str] = set()
    for sys_results in task_results.values():
        all_tasks.update(sys_results.keys())

    n_systems = len(task_results)
    if n_systems == 0 or not all_tasks:
        return _empty_result()

    # Compute pass rate per task
    task_pass_rates: dict[str, float] = {}
    unanimous_fail = 0
    unanimous_success = 0
    mid_band = 0
    near_unanimous_fail = 0  # 0 < rate < 0.1
    near_unanimous_success = 0  # 0.9 < rate < 1.0

    for task_id in all_tasks:
        passes = sum(
            1 for sys_results in task_results.values()
            if sys_results.get(task_id, False)
        )
        pass_rate = passes / n_systems
        task_pass_rates[task_id] = pass_rate

        if pass_rate == 0.0:
            unanimous_fail += 1
        elif pass_rate == 1.0:
            unanimous_success += 1
        elif 0.1 <= pass_rate <= 0.9:
            mid_band += 1
        elif pass_rate < 0.1:
            near_unanimous_fail += 1
        else:  # pass_rate > 0.9
            near_unanimous_success += 1

    total = len(all_tasks)

    # Compute distribution of pass rates (histogram buckets)
    buckets = _compute_pass_rate_distribution(task_pass_rates)

    # Gini coefficient of pass rates (0 = all same, 1 = max inequality)
    gini = _gini_coefficient(list(task_pass_rates.values()))

    # Compute item discrimination index (point-biserial correlation proxy)
    avg_discrimination = _avg_item_discrimination(task_results, task_pass_rates)

    result = {
        "total_tasks": total,
        "total_systems": n_systems,
        "unanimous_fail_count": unanimous_fail,
        "unanimous_fail_rate": unanimous_fail / total,
        "unanimous_success_count": unanimous_success,
        "unanimous_success_rate": unanimous_success / total,
        "mid_band_10_90_count": mid_band,
        "mid_band_10_90_rate": mid_band / total,
        "near_unanimous_fail_rate": near_unanimous_fail / total,
        "near_unanimous_success_rate": near_unanimous_success / total,
        "pass_rate_distribution": buckets,
        "gini_coefficient": gini,
        "avg_item_discrimination": avg_discrimination,
    }

    # Saturation and quality alerts
    result["alerts"] = _generate_alerts(result)

    return result


def compute_per_domain_discriminative_power(
    domain_task_results: dict[str, dict[str, dict[str, bool]]],
) -> dict[str, dict[str, Any]]:
    """Compute discriminative power per domain.

    Args:
        domain_task_results: domain -> system_id -> task_id -> passed
    """
    return {
        domain: compute_discriminative_power(task_results)
        for domain, task_results in domain_task_results.items()
    }


def _empty_result() -> dict[str, Any]:
    return {
        "total_tasks": 0,
        "total_systems": 0,
        "unanimous_fail_rate": 0.0,
        "unanimous_success_rate": 0.0,
        "mid_band_10_90_rate": 0.0,
        "alerts": [],
    }


def _compute_pass_rate_distribution(
    task_pass_rates: dict[str, float],
) -> dict[str, int]:
    """Histogram of pass rates in 10% buckets."""
    buckets = {f"{i*10}-{(i+1)*10}%": 0 for i in range(10)}
    buckets["100%"] = 0

    for rate in task_pass_rates.values():
        if rate == 1.0:
            buckets["100%"] += 1
        else:
            bucket_idx = min(int(rate * 10), 9)
            key = f"{bucket_idx*10}-{(bucket_idx+1)*10}%"
            buckets[key] += 1

    return buckets


def _gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient of a distribution."""
    if not values:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sorted_vals))
    denominator = n * sum(sorted_vals)
    return numerator / denominator if denominator > 0 else 0.0


def _avg_item_discrimination(
    task_results: dict[str, dict[str, bool]],
    task_pass_rates: dict[str, float],
) -> float:
    """Average item discrimination (simplified point-biserial proxy).

    Higher values mean individual tasks better separate strong from weak systems.
    """
    if not task_pass_rates:
        return 0.0

    # Compute system total scores
    system_totals: dict[str, float] = {}
    for sys_id, results in task_results.items():
        system_totals[sys_id] = sum(1.0 for v in results.values() if v) / max(1, len(results))

    discriminations: list[float] = []
    for task_id, pass_rate in task_pass_rates.items():
        if pass_rate == 0.0 or pass_rate == 1.0:
            discriminations.append(0.0)
            continue

        # Systems that passed vs failed this task
        passed_scores = []
        failed_scores = []
        for sys_id, results in task_results.items():
            total = system_totals[sys_id]
            if results.get(task_id, False):
                passed_scores.append(total)
            else:
                failed_scores.append(total)

        if passed_scores and failed_scores:
            mean_passed = sum(passed_scores) / len(passed_scores)
            mean_failed = sum(failed_scores) / len(failed_scores)
            discriminations.append(mean_passed - mean_failed)
        else:
            discriminations.append(0.0)

    return sum(discriminations) / len(discriminations) if discriminations else 0.0


def _generate_alerts(metrics: dict[str, Any]) -> list[str]:
    """Generate quality alerts based on discriminative power metrics."""
    alerts: list[str] = []

    # FINESSE-Bench insight: saturation detection
    if metrics.get("unanimous_success_rate", 0) > 0.5:
        alerts.append(
            "SATURATION: >50% of tasks solved by all systems. "
            "Benchmark may need harder tasks for meaningful differentiation."
        )

    if metrics.get("mid_band_10_90_rate", 0) < 0.3:
        alerts.append(
            "LOW_DISCRIMINABILITY: <30% of tasks in the informative mid-band (10-90%). "
            "Benchmark provides limited separation between systems."
        )

    if metrics.get("unanimous_fail_rate", 0) > 0.3:
        alerts.append(
            "TOO_HARD: >30% of tasks unsolved by any system. "
            "Consider adding easier tasks for baseline measurement."
        )

    if metrics.get("gini_coefficient", 0) < 0.1:
        alerts.append(
            "UNIFORM_DIFFICULTY: Tasks are too homogeneous in difficulty. "
            "Consider adding varied difficulty levels."
        )

    return alerts
