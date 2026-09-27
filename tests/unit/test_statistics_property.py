"""Property-based and combinatorial edge-case tests for statistical estimators.

Verifies mathematical invariants, monotonicity, bounds preservation, and extreme
edge-case behavior (n=1, k=100, c=0, c=n) for Pass@k, Pass^k, Wilson score intervals,
and Wald's Sequential Probability Ratio Test (SPRT).
"""

import pytest

from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k
from agent_bench.metrics.expanded import (
    SPRTDecision,
    compute_bootstrap_ci,
    compute_wilson_score_interval,
    evaluate_wald_sprt,
)


class TestPassKPropertyInvariants:
    """Mathematical invariant tests for Pass@k (at least one success in k trials)."""

    def test_pass_at_k_bounds(self) -> None:
        """Pass@k must strictly lie within [0.0, 1.0] across all parameter spaces."""
        for n in range(1, 15):
            for c in range(0, n + 1):
                results = [True] * c + [False] * (n - c)
                for k in range(1, n + 2):
                    score = compute_pass_k(results, k=k)
                    assert 0.0 <= score <= 1.0, f"Violated for n={n}, c={c}, k={k}: {score}"

    def test_pass_at_k_boundary_zero_and_full(self) -> None:
        """When c=0, Pass@k must be 0.0; when c=n, Pass@k must be 1.0."""
        for n in [1, 2, 5, 10, 20]:
            zeros = [False] * n
            ones = [True] * n
            for k in range(1, n + 1):
                assert compute_pass_k(zeros, k=k) == 0.0
                assert compute_pass_k(ones, k=k) == 1.0

    def test_pass_at_k_monotonicity_in_c(self) -> None:
        """For fixed n and k, Pass@k must be monotonically non-decreasing in c."""
        n = 10
        for k in [1, 2, 3, 5]:
            previous_score = -1.0
            for c in range(0, n + 1):
                results = [True] * c + [False] * (n - c)
                score = compute_pass_k(results, k=k)
                assert score >= previous_score - 1e-9, f"Monotonicity failed at c={c}, k={k}"
                previous_score = score

    def test_pass_at_k_empty_or_invalid_inputs(self) -> None:
        """Edge cases: empty results list, k=0, or negative k."""
        assert compute_pass_k([], k=1) == 0.0
        assert compute_pass_k([True, True], k=0) == 0.0
        assert compute_pass_k([True, True], k=-5) == 0.0


class TestPassHatKPropertyInvariants:
    """Mathematical invariant tests for Pass^k (all k trials must succeed)."""

    def test_pass_hat_k_bounds(self) -> None:
        """Pass^k must strictly lie within [0.0, 1.0]."""
        for n in range(1, 15):
            for c in range(0, n + 1):
                results = [True] * c + [False] * (n - c)
                for k in range(1, n + 2):
                    score = compute_pass_hat_k(results, k=k)
                    assert 0.0 <= score <= 1.0, f"Violated for n={n}, c={c}, k={k}: {score}"

    def test_pass_hat_k_requires_at_least_k_successes(self) -> None:
        """If c < k, Pass^k must be exactly 0.0."""
        for n in [5, 10]:
            for k in [3, 5]:
                for c in range(0, k):
                    results = [True] * c + [False] * (n - c)
                    assert compute_pass_hat_k(results, k=k) == 0.0

    def test_pass_hat_k_stricter_than_pass_at_k(self) -> None:
        """Pass^k <= Pass@k for all k >= 1 when 0 < c < n."""
        for n in range(2, 10):
            for c in range(1, n):
                results = [True] * c + [False] * (n - c)
                for k in range(1, n + 1):
                    pass_all = compute_pass_hat_k(results, k=k)
                    pass_any = compute_pass_k(results, k=k)
                    assert pass_all <= pass_any + 1e-9, f"Pass^k > Pass@k for n={n}, c={c}, k={k}"


class TestWilsonScoreIntervalProperties:
    """Mathematical invariant tests for asymmetric Wilson score confidence intervals."""

    def test_wilson_interval_bounds_and_containment(self) -> None:
        """Lower and upper bounds must be in [0.0, 1.0], with lower <= upper."""
        for total in [1, 2, 5, 10, 50, 100]:
            for succ in range(0, total + 1):
                p_hat, lower, upper = compute_wilson_score_interval(succ, total, confidence=0.95)
                assert 0.0 <= lower <= upper <= 1.0, f"Bound failure for {succ}/{total}: ({lower}, {upper})"
                if 0 < succ < total:
                    assert lower <= p_hat <= upper, f"p_hat not contained for {succ}/{total}: ({lower}, {upper})"

    def test_wilson_asymmetry_at_extremes(self) -> None:
        """At 0% successes, lower=0 and upper>0. At 100% successes, lower<1 and upper=1."""
        _, lower_0, upper_0 = compute_wilson_score_interval(0, 20, confidence=0.95)
        assert lower_0 == 0.0
        assert upper_0 > 0.0

        _, lower_1, upper_1 = compute_wilson_score_interval(20, 20, confidence=0.95)
        assert lower_1 < 1.0
        assert upper_1 == 1.0

    def test_wilson_confidence_level_monotonicity(self) -> None:
        """Higher confidence level must produce wider intervals."""
        total, succ = 20, 15
        _, low_90, high_90 = compute_wilson_score_interval(succ, total, confidence=0.90)
        _, low_99, high_99 = compute_wilson_score_interval(succ, total, confidence=0.99)

        width_90 = high_90 - low_90
        width_99 = high_99 - low_99
        assert width_99 >= width_90, f"Width 99% ({width_99}) < Width 90% ({width_90})"

    def test_wilson_boundary_inputs(self) -> None:
        p_hat, low, high = compute_wilson_score_interval(0, 0)
        assert (p_hat, low, high) == (0.0, 0.0, 0.0)


class TestWaldSPRTProperties:
    """Mathematical invariant tests for Wald's Sequential Probability Ratio Test."""

    def test_sprt_conclusive_sequences(self) -> None:
        """Unbroken success sequences must trigger ACCEPT_H1; failures must trigger REJECT_H1."""
        # Consecutive successes
        consecutive_wins = 15
        decision_win, _ = evaluate_wald_sprt(consecutive_wins, consecutive_wins, p0=0.5, p1=0.85)
        assert decision_win == SPRTDecision.ACCEPT_H1

        # Consecutive failures
        consecutive_fails = 15
        decision_fail, _ = evaluate_wald_sprt(0, consecutive_fails, p0=0.5, p1=0.85)
        assert decision_fail == SPRTDecision.REJECT_H1

    def test_sprt_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            evaluate_wald_sprt(5, 10, p0=0.9, p1=0.8)  # p0 >= p1
        with pytest.raises(ValueError):
            evaluate_wald_sprt(5, 10, alpha=0.0)
        with pytest.raises(ValueError):
            evaluate_wald_sprt(5, 10, beta=1.0)


class TestBootstrapCIProperties:
    """Mathematical invariant tests for non-parametric bootstrap confidence intervals."""

    def test_bootstrap_identical_values(self) -> None:
        """A sample of identical values must yield a zero-width CI matching that value."""
        values = [42.0] * 20
        mean, lower, upper = compute_bootstrap_ci(values, n_bootstrap=200)
        assert lower == pytest.approx(42.0)
        assert upper == pytest.approx(42.0)
        assert mean == pytest.approx(42.0)

    def test_bootstrap_bounds_containment(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        mean, lower, upper = compute_bootstrap_ci(values, n_bootstrap=500, seed=42)
        assert min(values) <= lower <= mean <= upper <= max(values)

