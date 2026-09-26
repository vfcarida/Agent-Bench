"""Property-based and boundary invariance tests for statistical estimators.

Verifies mathematical invariants:
1. Monotonicity: Pass^k <= Pass@1 <= Pass@k for all valid (n, c, k).
2. Monotonicity across k:
   - For k1 < k2: Pass@k1 <= Pass@k2 (increasing chance of at least 1 success).
   - For k1 < k2: Pass^k1 >= Pass^k2 (decreasing chance of all k successes).
3. Boundary conditions:
   - Empty results return 0.0 without ZeroDivisionError.
   - k <= 0 returns 0.0 without ValueError.
   - k > n returns proper exact values.
4. Bootstrap confidence intervals:
   - Invariant: lower <= mean <= upper.
   - Higher confidence levels yield wider intervals.
"""

import random

import pytest

from agent_bench.metrics.compute import (
    compute_pass_hat_k,
    compute_pass_k,
    compute_task_metrics,
)
from agent_bench.metrics.expanded import compute_bootstrap_ci, compute_confidence_interval


class TestPassKInvariants:
    """Mathematical invariants of combinatorial Pass@k and Pass^k estimators."""

    @pytest.mark.parametrize("k", [-2, -1, 0])
    def test_pass_k_non_positive_k(self, k: int):
        assert compute_pass_k([True, False, True], k=k) == 0.0
        assert compute_pass_hat_k([True, False, True], k=k) == 0.0

    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_empty_results_safety(self, k: int):
        assert compute_pass_k([], k=k) == 0.0
        assert compute_pass_hat_k([], k=k) == 0.0

    def test_exhaustive_monotonicity_small_n(self):
        """For all n in [1, 8], all c in [0, n], and all k in [1, n]:
        Pass^k <= Pass@1 <= Pass@k
        """
        for n in range(1, 9):
            for c in range(n + 1):
                results = [True] * c + [False] * (n - c)
                pass_1 = compute_pass_k(results, k=1)
                for k in range(1, n + 1):
                    pass_at_k = compute_pass_k(results, k=k)
                    pass_hat_k = compute_pass_hat_k(results, k=k)

                    assert pass_hat_k <= pass_1 + 1e-9, f"Failed Pass^{k} <= Pass@1 for n={n}, c={c}"
                    assert pass_1 <= pass_at_k + 1e-9, f"Failed Pass@1 <= Pass@{k} for n={n}, c={c}"

    def test_k_ordering_monotonicity(self):
        """Pass@k increases with k; Pass^k decreases with k."""
        rng = random.Random(1337)
        for _ in range(50):
            n = rng.randint(5, 20)
            c = rng.randint(0, n)
            results = [True] * c + [False] * (n - c)
            rng.shuffle(results)

            for k in range(1, n):
                p_at_k1 = compute_pass_k(results, k=k)
                p_at_k2 = compute_pass_k(results, k=k + 1)
                assert p_at_k1 <= p_at_k2 + 1e-9, f"Pass@k should be non-decreasing: {p_at_k1} > {p_at_k2}"

                p_hat_k1 = compute_pass_hat_k(results, k=k)
                p_hat_k2 = compute_pass_hat_k(results, k=k + 1)
                assert p_hat_k1 >= p_hat_k2 - 1e-9, f"Pass^k should be non-increasing: {p_hat_k1} < {p_hat_k2}"

    def test_extreme_sample_sizes(self):
        """Test large n with zero and total success without overflow."""
        large_n = 500
        # All passed
        assert compute_pass_k([True] * large_n, k=10) == 1.0
        assert compute_pass_hat_k([True] * large_n, k=10) == 1.0

        # None passed
        assert compute_pass_k([False] * large_n, k=10) == 0.0
        assert compute_pass_hat_k([False] * large_n, k=10) == 0.0


class TestBootstrapConfidenceIntervalInvariants:
    """Mathematical invariants of bootstrap and normal confidence intervals."""

    def test_empty_and_singleton(self):
        assert compute_bootstrap_ci([]) == (0.0, 0.0, 0.0)
        assert compute_confidence_interval([]) == (0.0, 0.0, 0.0)

        assert compute_bootstrap_ci([10.0]) == (10.0, 10.0, 10.0)
        assert compute_confidence_interval([10.0]) == (10.0, 10.0, 10.0)

    def test_constant_array(self):
        constant = [7.5] * 20
        mean, lower, upper = compute_bootstrap_ci(constant)
        assert abs(mean - 7.5) < 1e-9
        assert abs(lower - 7.5) < 1e-9
        assert abs(upper - 7.5) < 1e-9

    def test_lower_mean_upper_ordering(self):
        rng = random.Random(42)
        values = [rng.gauss(0.75, 0.15) for _ in range(30)]

        mean, lower, upper = compute_bootstrap_ci(values, n_bootstrap=1000, confidence=0.95)
        assert lower <= mean <= upper

        n_mean, n_lower, n_upper = compute_confidence_interval(values, confidence=0.95)
        assert n_lower <= n_mean <= n_upper

    def test_confidence_level_monotonicity(self):
        """Higher confidence level creates wider interval."""
        rng = random.Random(999)
        values = [rng.gauss(10.0, 2.0) for _ in range(50)]

        _, low_90, up_90 = compute_bootstrap_ci(values, n_bootstrap=2000, confidence=0.90, seed=123)
        _, low_99, up_99 = compute_bootstrap_ci(values, n_bootstrap=2000, confidence=0.99, seed=123)

        width_90 = up_90 - low_90
        width_99 = up_99 - low_99
        assert width_99 >= width_90 - 1e-6


class TestComputeTaskMetrics:
    """Verifies end-to-end task metrics aggregation."""

    def test_empty_results_safe(self):
        metrics = compute_task_metrics([])
        assert len(metrics) >= 7
        success_metric = next(m for m in metrics if m.name == "task_success")
        assert success_metric.value == 0.0

    def test_standard_metrics_with_costs_and_latency(self):
        results = [True, True, False, True]
        latencies = [120.0, 150.0, 90.0, 110.0]
        tokens_in = [1000, 1200, 800, 1100]
        tokens_out = [200, 250, 150, 220]

        metrics = compute_task_metrics(
            results=results,
            latencies_ms=latencies,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_per_1k_in=0.003,
            cost_per_1k_out=0.015,
        )

        metric_dict = {m.name: m.value for m in metrics}
        assert metric_dict["task_success"] == 0.75
        assert metric_dict["latency_ms"] == sum(latencies) / len(latencies)
        assert metric_dict["total_tokens_in"] == sum(tokens_in)
        assert metric_dict["total_tokens_out"] == sum(tokens_out)
        assert metric_dict["estimated_cost_usd"] > 0.0
