"""Unit tests for inter-annotator agreement metrics (Cohen's Kappa & Krippendorff's Alpha)."""

from pathlib import Path

import pytest

from agent_bench.metrics.inter_annotator import (
    compute_cohens_kappa,
    compute_krippendorffs_alpha,
    evaluate_annotation_dataset,
    interpret_agreement,
)


def test_cohens_kappa_perfect_agreement() -> None:
    rater1 = ["pass", "pass", "fail", "fail", "pass"]
    rater2 = ["pass", "pass", "fail", "fail", "pass"]
    kappa = compute_cohens_kappa(rater1, rater2)
    assert kappa == 1.0
    assert interpret_agreement(kappa) == "Almost Perfect Agreement"


def test_cohens_kappa_partial_agreement() -> None:
    # 20 observations, 15 agree, 5 disagree
    rater1 = ["yes"] * 10 + ["no"] * 10
    rater2 = ["yes"] * 8 + ["no"] * 2 + ["yes"] * 3 + ["no"] * 7
    kappa = compute_cohens_kappa(rater1, rater2)
    # Expected: P_o = 15/20 = 0.75
    # P_e = (10/20)*(11/20) + (10/20)*(9/20) = 0.50
    # kappa = (0.75 - 0.50) / (1 - 0.50) = 0.50
    assert kappa == pytest.approx(0.50, abs=1e-3)
    assert interpret_agreement(kappa) == "Moderate Agreement"


def test_cohens_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Rater sequence lengths must match"):
        compute_cohens_kappa(["pass"], ["pass", "fail"])


def test_cohens_kappa_empty_returns_one() -> None:
    assert compute_cohens_kappa([], []) == 1.0


def test_krippendorffs_alpha_perfect() -> None:
    matrix = [
        ["pass", "pass", "pass"],
        ["fail", "fail", "fail"],
        ["pass", "pass", "pass"],
    ]
    alpha = compute_krippendorffs_alpha(matrix)
    assert alpha == 1.0


def test_krippendorffs_alpha_with_missing_data() -> None:
    matrix: list[list[str | None]] = [
        ["pass", "pass", None],
        [None, "fail", "fail"],
        ["pass", None, "pass"],
        ["fail", "fail", "fail"],
    ]
    alpha = compute_krippendorffs_alpha(matrix)
    assert alpha == 1.0


def test_krippendorffs_alpha_disagreement() -> None:
    matrix = [
        ["a", "b"],
        ["b", "a"],
        ["a", "b"],
        ["b", "a"],
    ]
    alpha = compute_krippendorffs_alpha(matrix)
    assert alpha < 0.0
    assert interpret_agreement(alpha) == "Poor Agreement"


def test_evaluate_annotation_dataset(tmp_path: Path) -> None:
    yaml_content = """
version: "1.0.0"
items:
  - id: "item_1"
    ratings:
      rater_1: "pass"
      rater_2: "pass"
  - id: "item_2"
    ratings:
      rater_1: "fail"
      rater_2: "fail"
  - id: "item_3"
    ratings:
      rater_1: "pass"
      rater_2: "fail"
"""
    file_path = tmp_path / "annotations.yaml"
    file_path.write_text(yaml_content)

    res = evaluate_annotation_dataset(file_path)
    assert res["total_items"] == 3
    assert "rater_1" in res["raters"]
    assert "rater_2" in res["raters"]
    assert "rater_1_vs_rater_2" in res["pairwise_cohens_kappa"]
    assert -1.0 <= res["krippendorffs_alpha"] <= 1.0
