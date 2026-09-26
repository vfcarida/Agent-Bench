"""Unit tests for Wilson Score intervals and Wald's SPRT sequential analysis."""

import pytest

from agent_bench.metrics.expanded import (
    SPRTDecision,
    compute_wilson_score_interval,
    evaluate_wald_sprt,
)


class TestWilsonScoreInterval:
    def test_empty_sample(self) -> None:
        p, low, high = compute_wilson_score_interval(0, 0)
        assert (p, low, high) == (0.0, 0.0, 0.0)

    def test_zero_successes_bounded(self) -> None:
        p, low, high = compute_wilson_score_interval(0, 10, confidence=0.95)
        assert p == 0.0
        assert low == 0.0
        assert 0.0 < high < 0.35  # Upper bound is strictly positive and bounded

    def test_all_successes_bounded(self) -> None:
        p, low, high = compute_wilson_score_interval(10, 10, confidence=0.95)
        assert p == 1.0
        assert 0.65 < low < 1.0
        assert high == 1.0

    def test_nominal_interval_contains_point_estimate(self) -> None:
        p, low, high = compute_wilson_score_interval(7, 10, confidence=0.95)
        assert p == 0.70
        assert low < p < high
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0

    def test_confidence_level_monotonicity(self) -> None:
        # Higher confidence yields wider interval
        _, low_90, high_90 = compute_wilson_score_interval(15, 20, confidence=0.90)
        _, low_99, high_99 = compute_wilson_score_interval(15, 20, confidence=0.99)
        assert (high_99 - low_99) > (high_90 - low_90)


class TestWaldSPRT:
    def test_early_rejection_low_performance(self) -> None:
        # Agent has 0 successes out of 6 trials: should decisively reject target p1=0.80
        decision, llr = evaluate_wald_sprt(
            successes=0, trials=6, p0=0.50, p1=0.80, alpha=0.05, beta=0.10
        )
        assert decision == SPRTDecision.REJECT_H1
        assert llr < 0

    def test_early_acceptance_high_performance(self) -> None:
        # Agent has 10 successes out of 10 trials: should decisively accept target p1=0.80
        decision, llr = evaluate_wald_sprt(
            successes=10, trials=10, p0=0.50, p1=0.80, alpha=0.05, beta=0.10
        )
        assert decision == SPRTDecision.ACCEPT_H1
        assert llr > 0

    def test_continue_sampling_indeterminate(self) -> None:
        # Borderline performance (3 of 5): requires more data
        decision, _ = evaluate_wald_sprt(
            successes=3, trials=5, p0=0.50, p1=0.80, alpha=0.05, beta=0.10
        )
        assert decision == SPRTDecision.CONTINUE

    def test_zero_trials_continues(self) -> None:
        decision, llr = evaluate_wald_sprt(0, 0)
        assert decision == SPRTDecision.CONTINUE
        assert llr == 0.0

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="Invalid hypothesis boundaries"):
            evaluate_wald_sprt(5, 10, p0=0.80, p1=0.50)  # p0 >= p1
        with pytest.raises(ValueError, match="Error bounds alpha and beta"):
            evaluate_wald_sprt(5, 10, p0=0.40, p1=0.70, alpha=0.6)
