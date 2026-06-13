"""Transfer gap analysis between evaluation families.

Inspired by FINESSE-Bench (arXiv:2605.15482) Section 7.1:
- Computes Δ_transfer between benchmark families/domains for each system
- Measures how well performance on one family transfers to another
- Identifies models with asymmetric transfer profiles

Key insight: strong results on standard public benchmarks do NOT guarantee
equally strong results on professionally-oriented tasks. Transfer gaps of
0.27-0.36 have been observed between public and exam-like benchmarks.
"""

from typing import Any


def compute_transfer_gaps(
    results_by_family: dict[str, list[dict[str, Any]]],
    reference_family: str | None = None,
    score_field: str = "score",
) -> dict[str, Any]:
    """Compute transfer gaps between benchmark families.

    Like FINESSE-Bench's Δ_public→exam and Δ_public→ta.

    Args:
        results_by_family: Mapping from family name to list of task results
        reference_family: The "source" family to compare against.
            If None, uses the family with the highest accuracy.
        score_field: Key for score value in each result

    Returns:
        Dict with per-family accuracy and transfer gaps
    """
    # Compute per-family accuracy
    family_accuracy: dict[str, float] = {}
    family_counts: dict[str, int] = {}

    for family, results in results_by_family.items():
        if not results:
            continue
        scores = [
            float(r.get(score_field, r.get("passed", 0)))
            if isinstance(r.get(score_field, r.get("passed", 0)), (int, float))
            else (1.0 if r.get(score_field, r.get("passed", False)) else 0.0)
            for r in results
        ]
        family_accuracy[family] = sum(scores) / len(scores) if scores else 0.0
        family_counts[family] = len(scores)

    if not family_accuracy:
        return {"family_accuracy": {}, "transfer_gaps": {}, "reference_family": None}

    # Determine reference family
    if reference_family is None or reference_family not in family_accuracy:
        reference_family = max(family_accuracy, key=family_accuracy.get)  # type: ignore[arg-type]

    ref_accuracy = family_accuracy[reference_family]

    # Compute gaps: positive Δ means degradation from reference
    transfer_gaps: dict[str, float] = {}
    for family, acc in family_accuracy.items():
        if family != reference_family:
            transfer_gaps[f"delta_{reference_family}_to_{family}"] = ref_accuracy - acc

    # Classify transfer pattern
    gap_values = list(transfer_gaps.values())
    if gap_values:
        avg_gap = sum(gap_values) / len(gap_values)
        max_gap = max(gap_values)
        min_gap = min(gap_values)
        asymmetry = max_gap - min_gap
    else:
        avg_gap = max_gap = min_gap = asymmetry = 0.0

    return {
        "reference_family": reference_family,
        "family_accuracy": family_accuracy,
        "family_counts": family_counts,
        "transfer_gaps": transfer_gaps,
        "avg_gap": avg_gap,
        "max_gap": max_gap,
        "min_gap": min_gap,
        "asymmetry": asymmetry,
        "transfer_pattern": _classify_transfer(avg_gap, asymmetry),
    }


def compute_system_transfer_profiles(
    system_family_results: dict[str, dict[str, list[dict[str, Any]]]],
    reference_family: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute transfer profiles for multiple systems.

    Args:
        system_family_results: system_id -> family -> results

    Returns:
        Mapping from system_id to transfer gap analysis
    """
    profiles: dict[str, dict[str, Any]] = {}
    for system_id, family_results in system_family_results.items():
        profiles[system_id] = compute_transfer_gaps(family_results, reference_family)
    return profiles


def rank_by_transfer_robustness(
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank systems by how well their performance transfers across families.

    Inspired by FINESSE-Bench Table 6: models with smallest transfer gaps
    are considered more robust and better suited for professional deployment.
    """
    rankings: list[dict[str, Any]] = []

    for system_id, profile in profiles.items():
        family_acc = profile.get("family_accuracy", {})
        valid_accs = list(family_acc.values())

        if not valid_accs:
            continue

        avg = sum(valid_accs) / len(valid_accs)
        min_acc = min(valid_accs)
        std = _std(valid_accs) if len(valid_accs) > 1 else 0.0

        rankings.append({
            "system_id": system_id,
            "avg_accuracy": avg,
            "min_accuracy": min_acc,
            "std_accuracy": std,
            "avg_transfer_gap": profile.get("avg_gap", 0.0),
            "max_transfer_gap": profile.get("max_gap", 0.0),
            "transfer_pattern": profile.get("transfer_pattern", "unknown"),
            # Balanced robustness: high avg, low std, small gaps
            "transfer_robustness": avg - std * 2 - profile.get("avg_gap", 0.0),
        })

    rankings.sort(key=lambda x: x["transfer_robustness"], reverse=True)
    return rankings


def _classify_transfer(avg_gap: float, asymmetry: float) -> str:
    """Classify transfer pattern."""
    if avg_gap < 0.03:
        return "robust_transfer"
    elif avg_gap < 0.10:
        return "moderate_gap"
    elif asymmetry < 0.05:
        return "uniform_degradation"
    else:
        return "asymmetric_degradation"


def _std(values: list[float]) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
