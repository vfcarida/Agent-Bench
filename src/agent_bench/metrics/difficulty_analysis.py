"""Difficulty degradation analysis metrics.

Inspired by FINESSE-Bench (arXiv:2605.15482) Section 7.2:
- Computes Δ between adjacent difficulty levels to measure performance degradation
- Identifies models that maintain quality vs. those that degrade severely
- Generates "difficulty profiles" per model/system
- Supports hierarchical analysis like CFA Level 1 → Level 2 → Level 3

Key insight from FINESSE-Bench: performance on basic-level questions may
underestimate a model's actual limitations in more advanced scenarios.
The difficulty hierarchy measures not only average financial literacy but
also the robustness of model quality when moving to harder tasks.
"""

from typing import Any


# Canonical difficulty ordering (easy → expert)
DIFFICULTY_LEVELS = ["easy", "medium", "hard", "expert"]


def compute_difficulty_degradation(
    results: list[dict[str, Any]],
    difficulty_field: str = "difficulty",
    score_field: str = "score",
) -> dict[str, Any]:
    """Compute Δ between difficulty levels (FINESSE-Bench style).

    Args:
        results: List of task results, each with difficulty and score fields
        difficulty_field: Key for difficulty level in each result
        score_field: Key for score value in each result (or 'passed' for binary)

    Returns:
        Dict with per-level accuracy, deltas between levels, and profile metadata
    """
    # Group results by difficulty
    by_difficulty: dict[str, list[float]] = {}
    for r in results:
        diff = r.get(difficulty_field, "medium")
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        score = r.get(score_field, r.get("passed", 0))
        by_difficulty[diff].append(float(score) if isinstance(score, (int, float)) else (1.0 if score else 0.0))

    # Compute per-level accuracy
    level_accuracy: dict[str, float] = {}
    level_counts: dict[str, int] = {}
    for level in DIFFICULTY_LEVELS:
        scores = by_difficulty.get(level, [])
        if scores:
            level_accuracy[level] = sum(scores) / len(scores)
            level_counts[level] = len(scores)
        else:
            level_accuracy[level] = float("nan")
            level_counts[level] = 0

    # Compute deltas between adjacent levels
    deltas: dict[str, float | None] = {}
    for i in range(len(DIFFICULTY_LEVELS) - 1):
        curr_level = DIFFICULTY_LEVELS[i]
        next_level = DIFFICULTY_LEVELS[i + 1]
        key = f"delta_{curr_level}_to_{next_level}"

        curr_acc = level_accuracy.get(curr_level)
        next_acc = level_accuracy.get(next_level)

        if curr_acc is not None and next_acc is not None and curr_acc == curr_acc and next_acc == next_acc:
            # Positive Δ means degradation (curr is better than next)
            deltas[key] = curr_acc - next_acc
        else:
            deltas[key] = None

    # Compute end-to-end delta (easy → expert)
    easy_acc = level_accuracy.get("easy")
    expert_acc = level_accuracy.get("expert")
    if easy_acc is not None and expert_acc is not None and easy_acc == easy_acc and expert_acc == expert_acc:
        deltas["delta_easy_to_expert"] = easy_acc - expert_acc
    else:
        deltas["delta_easy_to_expert"] = None

    # Classify degradation pattern
    pattern = _classify_degradation_pattern(level_accuracy, deltas)

    return {
        "level_accuracy": level_accuracy,
        "level_counts": level_counts,
        "deltas": deltas,
        "degradation_pattern": pattern,
        "total_tasks": sum(level_counts.values()),
    }


def compute_difficulty_profile(
    system_results: dict[str, list[dict[str, Any]]],
    difficulty_field: str = "difficulty",
) -> dict[str, dict[str, Any]]:
    """Compute difficulty profiles for multiple systems.

    Args:
        system_results: Mapping from system_id to list of task results

    Returns:
        Mapping from system_id to difficulty degradation analysis
    """
    profiles: dict[str, dict[str, Any]] = {}
    for system_id, results in system_results.items():
        profiles[system_id] = compute_difficulty_degradation(results, difficulty_field)
    return profiles


def compare_difficulty_robustness(
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank systems by difficulty robustness (smallest end-to-end degradation).

    Inspired by FINESSE-Bench Table 10: balanced models have high average
    and small std across difficulty levels.

    Returns:
        Sorted list of system rankings with robustness metrics
    """
    rankings: list[dict[str, Any]] = []

    for system_id, profile in profiles.items():
        level_acc = profile.get("level_accuracy", {})
        valid_accs = [v for v in level_acc.values() if v == v]  # filter NaN

        if not valid_accs:
            continue

        avg_accuracy = sum(valid_accs) / len(valid_accs)
        min_accuracy = min(valid_accs)
        max_accuracy = max(valid_accs)
        std_accuracy = _std(valid_accs)

        # End-to-end delta (positive = degradation)
        e2e_delta = profile.get("deltas", {}).get("delta_easy_to_expert")

        rankings.append({
            "system_id": system_id,
            "avg_accuracy": avg_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "std_accuracy": std_accuracy,
            "end_to_end_delta": e2e_delta,
            "degradation_pattern": profile.get("degradation_pattern", "unknown"),
            # Robustness score: high avg + low std + small degradation
            "robustness_score": avg_accuracy - std_accuracy * 2,
        })

    # Sort by robustness score descending
    rankings.sort(key=lambda x: x["robustness_score"], reverse=True)
    return rankings


def _classify_degradation_pattern(
    level_accuracy: dict[str, float],
    deltas: dict[str, float | None],
) -> str:
    """Classify the degradation pattern observed across difficulty levels.

    Patterns (from FINESSE-Bench Section 7.2):
    - "monotonic_degradation": each level is worse than the previous
    - "robust": minimal degradation across levels (Δ < 5%)
    - "cliff": sudden drop at a specific level
    - "non_monotonic": some levels are harder than expected
    - "insufficient_data": not enough levels to classify
    """
    valid_deltas = [(k, v) for k, v in deltas.items() if v is not None and k != "delta_easy_to_expert"]

    if len(valid_deltas) < 2:
        return "insufficient_data"

    delta_values = [v for _, v in valid_deltas]

    # All deltas positive and significant → monotonic degradation
    if all(d > 0.02 for d in delta_values):
        return "monotonic_degradation"

    # All deltas small → robust
    if all(abs(d) < 0.05 for d in delta_values):
        return "robust"

    # One delta much larger than others → cliff
    if max(delta_values) > 3 * _mean_positive(delta_values):
        return "cliff"

    # Mixed positive and negative → non-monotonic
    if any(d < -0.02 for d in delta_values):
        return "non_monotonic"

    return "gradual_degradation"


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _mean_positive(values: list[float]) -> float:
    """Mean of positive values, or 0.01 if none."""
    positives = [v for v in values if v > 0]
    return sum(positives) / len(positives) if positives else 0.01
