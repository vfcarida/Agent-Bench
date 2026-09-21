"""Inter-annotator agreement and judge calibration statistics.

Provides implementations for:
- Cohen's Kappa (pair of raters, nominal scale)
- Krippendorff's Alpha (multiple raters, handles missing values, nominal scale)
- Multi-annotator dataset evaluation and Landis & Koch interpretation
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def compute_cohens_kappa(rater1: Sequence[Any], rater2: Sequence[Any]) -> float:
    """Compute unweighted Cohen's Kappa coefficient between two raters.

    Formula:
        kappa = (P_o - P_e) / (1 - P_e)
    where:
        P_o = observed proportional agreement
        P_e = hypothetical probability of chance agreement

    Args:
        rater1: Sequence of categorical ratings from rater 1.
        rater2: Sequence of categorical ratings from rater 2.

    Returns:
        float: Kappa score between -1.0 and 1.0 (1.0 = perfect agreement).
    """
    if len(rater1) != len(rater2):
        raise ValueError(f"Rater sequence lengths must match: {len(rater1)} != {len(rater2)}")
    if len(rater1) == 0:
        return 1.0

    n = len(rater1)
    # Convert to string for consistent categorical matching
    r1_str = [str(x) for x in rater1]
    r2_str = [str(x) for x in rater2]

    # Observed agreement
    agreements = sum(1 for a, b in zip(r1_str, r2_str, strict=False) if a == b)
    p_o = agreements / n

    # Marginal probabilities
    counts1 = Counter(r1_str)
    counts2 = Counter(r2_str)
    all_categories = set(counts1.keys()) | set(counts2.keys())

    p_e = sum((counts1.get(c, 0) / n) * (counts2.get(c, 0) / n) for c in all_categories)

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(float(kappa), 4)


def compute_krippendorffs_alpha(
    matrix: Sequence[Sequence[Any | None]],
    level_of_measurement: str = "nominal",
) -> float:
    """Compute Krippendorff's Alpha reliability coefficient.

    Supports arbitrary numbers of raters and missing data (None values).

    Args:
        matrix: 2D matrix where rows represent units of analysis (test items)
                and columns represent raters. Missing ratings are indicated by None.
        level_of_measurement: Metric scale. Defaults to "nominal".

    Returns:
        float: Alpha score between -1.0 and 1.0.
    """
    if not matrix:
        return 1.0

    # Filter out units with fewer than 2 valid ratings
    valid_units: list[list[str]] = []
    for row in matrix:
        valid_ratings = [str(x) for x in row if x is not None]
        if len(valid_ratings) >= 2:
            valid_units.append(valid_ratings)

    if not valid_units:
        return 1.0

    # Collect all categories
    categories = sorted(set(val for unit in valid_units for val in unit))
    if len(categories) <= 1:
        return 1.0

    # Number of pairable values n and marginal counts
    total_pairable = 0
    marginal_counts: Counter[str] = Counter()
    d_o = 0.0

    for unit in valid_units:
        m_u = len(unit)
        total_pairable += m_u
        marginal_counts.update(unit)

        unit_counts = Counter(unit)
        # Observed disagreement within unit: sum_{v != c} n_{uv} * n_{uc} / (m_u - 1)
        # Equivalent to (m_u^2 - sum_v n_{uv}^2) / (m_u - 1) for nominal distance
        sum_sq = sum(cnt * cnt for cnt in unit_counts.values())
        disagree_u = (m_u * m_u - sum_sq) / (m_u - 1)
        d_o += disagree_u

    n = total_pairable
    if n <= 1:
        return 1.0

    # Expected disagreement
    sum_marginal_sq = sum(cnt * cnt for cnt in marginal_counts.values())
    d_e = (n * n - sum_marginal_sq) / (n - 1)

    if abs(d_e) < 1e-12:
        return 1.0 if abs(d_o) < 1e-12 else 0.0

    alpha = 1.0 - (d_o / d_e)
    return round(float(alpha), 4)


def interpret_agreement(score: float) -> str:
    """Return qualitative interpretation of agreement based on Landis & Koch (1977)."""
    if score < 0.0:
        return "Poor Agreement"
    elif score <= 0.20:
        return "Slight Agreement"
    elif score <= 0.40:
        return "Fair Agreement"
    elif score <= 0.60:
        return "Moderate Agreement"
    elif score <= 0.80:
        return "Substantial Agreement"
    else:
        return "Almost Perfect Agreement"


def evaluate_annotation_dataset(path: Path) -> dict[str, Any]:
    """Load an annotations file and compute comprehensive inter-annotator metrics.

    Expected file structure (YAML or JSON):
    items:
      - id: "item_1"
        ratings:
          annotator_a: "pass"
          annotator_b: "pass"
          annotator_c: "fail"
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "items" not in data:
        raise ValueError("Annotations dataset must contain a top-level 'items' list.")

    items = data["items"]
    # Collect all unique annotators
    all_raters = sorted(
        set(
            rater
            for item in items
            if isinstance(item, dict) and "ratings" in item
            for rater in item["ratings"].keys()
        )
    )

    if len(all_raters) < 2:
        raise ValueError(f"At least 2 distinct raters required, found {len(all_raters)}")

    # Build matrix for Krippendorff's alpha (units x raters)
    matrix: list[list[Any | None]] = []
    # Build per-rater parallel lists for pairwise Cohen's kappa
    rater_columns: dict[str, list[Any | None]] = {r: [] for r in all_raters}

    for item in items:
        ratings = item.get("ratings", {})
        row = [ratings.get(r) for r in all_raters]
        matrix.append(row)
        for r in all_raters:
            rater_columns[r].append(ratings.get(r))

    # Pairwise Cohen's Kappa
    pairwise_kappas: dict[str, float] = {}
    kappa_values: list[float] = []

    for i in range(len(all_raters)):
        for j in range(i + 1, len(all_raters)):
            r1 = all_raters[i]
            r2 = all_raters[j]
            # Extract common valid pairs
            p1: list[Any] = []
            p2: list[Any] = []
            for v1, v2 in zip(rater_columns[r1], rater_columns[r2], strict=False):
                if v1 is not None and v2 is not None:
                    p1.append(v1)
                    p2.append(v2)
            if p1:
                k = compute_cohens_kappa(p1, p2)
                pairwise_kappas[f"{r1}_vs_{r2}"] = k
                kappa_values.append(k)

    alpha = compute_krippendorffs_alpha(matrix)
    mean_kappa = round(sum(kappa_values) / len(kappa_values), 4) if kappa_values else 0.0

    return {
        "dataset_path": str(path),
        "total_items": len(items),
        "raters": all_raters,
        "krippendorffs_alpha": alpha,
        "mean_cohens_kappa": mean_kappa,
        "pairwise_cohens_kappa": pairwise_kappas,
        "alpha_interpretation": interpret_agreement(alpha),
        "kappa_interpretation": interpret_agreement(mean_kappa),
    }
