"""Unit and statistical simulation tests for AB-T05.

Verifies:
- T-PASSK-2: Monotonicity (pass^k <= pass@1 <= pass@k) and edge cases.
- T-PASSK-1: Closed-form expectation and >=90/100 coverage of 95% bootstrap CI over Bernoulli(p).
- Anti-pooling: Per-task aggregation vs pooled flattening.
- High-risk reliability weighting: pass^k primary for transactional_high_risk & cyber_restricted.
"""

import math
import random

import pytest

from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k, compute_task_metrics
from agent_bench.metrics.expanded import compute_bootstrap_ci, compute_stratified_bootstrap_ci
from agent_bench.metrics.scorecard import compute_scorecard


class TestPassHatK:
    """Unit tests for compute_pass_hat_k."""

    def test_all_pass(self):
        assert compute_pass_hat_k([True, True, True], k=1) == 1.0
        assert compute_pass_hat_k([True, True, True], k=2) == 1.0
        assert compute_pass_hat_k([True, True, True], k=3) == 1.0

    def test_none_pass(self):
        assert compute_pass_hat_k([False, False, False], k=1) == 0.0
        assert compute_pass_hat_k([False, False, False], k=2) == 0.0
        assert compute_pass_hat_k([False, False, False], k=3) == 0.0

    def test_partial_pass(self):
        # 2 correct out of 3
        # pass^1 = 2/3
        assert abs(compute_pass_hat_k([True, True, False], k=1) - 2 / 3) < 1e-6
        # pass^2 = C(2, 2) / C(3, 2) = 1/3
        assert abs(compute_pass_hat_k([True, True, False], k=2) - 1 / 3) < 1e-6
        # pass^3 = C(2, 3) / C(3, 3) = 0.0 (c < k)
        assert compute_pass_hat_k([True, True, False], k=3) == 0.0

    def test_empty_results(self):
        assert compute_pass_hat_k([], k=1) == 0.0

    def test_k_greater_than_n(self):
        # When n < k:
        # All passed:
        assert compute_pass_hat_k([True, True], k=3) == 1.0
        # Not all passed:
        assert compute_pass_hat_k([True, False], k=3) == 0.0
        assert compute_pass_hat_k([False, False], k=3) == 0.0


class TestTPassK2MonotonicityAndEdgeCases:
    """T-PASSK-2: Monotonicity pass^k <= pass@1 <= pass@k and edge cases."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_monotonicity_exhaustive(self, n: int):
        """For all combinations of c <= n and k <= n: pass^k <= pass@1 <= pass@k."""
        for c in range(n + 1):
            results = [True] * c + [False] * (n - c)
            pass_at_1 = compute_pass_k(results, k=1)

            for k in range(1, n + 1):
                pass_hat_k = compute_pass_hat_k(results, k=k)
                pass_at_k = compute_pass_k(results, k=k)

                assert pass_hat_k <= pass_at_1 + 1e-9, (
                    f"Violation pass^k ({pass_hat_k}) <= pass@1 ({pass_at_1}) for n={n}, c={c}, k={k}"
                )
                assert pass_at_1 <= pass_at_k + 1e-9, (
                    f"Violation pass@1 ({pass_at_1}) <= pass@k ({pass_at_k}) for n={n}, c={c}, k={k}"
                )

    def test_k1_equality(self):
        """When k=1, pass^1 == pass@1 == c/n."""
        for n in [1, 2, 4, 8]:
            for c in range(n + 1):
                results = [True] * c + [False] * (n - c)
                p_hat_1 = compute_pass_hat_k(results, k=1)
                p_at_1 = compute_pass_k(results, k=1)
                expected = c / n
                assert abs(p_hat_1 - expected) < 1e-9
                assert abs(p_at_1 - expected) < 1e-9

    def test_edge_cases_c_zero_and_c_n(self):
        """c=0 matches 0.0 and c=n matches 1.0 for all k."""
        for n in [1, 3, 5]:
            for k in range(1, n + 1):
                assert compute_pass_k([False] * n, k=k) == 0.0
                assert compute_pass_hat_k([False] * n, k=k) == 0.0
                assert compute_pass_k([True] * n, k=k) == 1.0
                assert compute_pass_hat_k([True] * n, k=k) == 1.0


class TestTPassK1StatisticalSimulation:
    """T-PASSK-1: Seeded Bernoulli(p) per-task simulation and bootstrap CI coverage."""

    def test_unbiased_estimator_exact_expectation(self):
        """Closed form: The binomial expectation of per-task pass@k and pass^k match theory exactly."""
        p = 0.65
        n = 5
        k = 3

        expected_pass_k = 0.0
        expected_pass_hat_k = 0.0

        for c in range(n + 1):
            prob_c = math.comb(n, c) * (p ** c) * ((1 - p) ** (n - c))
            res = [True] * c + [False] * (n - c)
            expected_pass_k += prob_c * compute_pass_k(res, k=k)
            expected_pass_hat_k += prob_c * compute_pass_hat_k(res, k=k)

        theoretical_pass_k = 1.0 - (1.0 - p) ** k
        theoretical_pass_hat_k = p ** k

        assert abs(expected_pass_k - theoretical_pass_k) < 1e-9
        assert abs(expected_pass_hat_k - theoretical_pass_hat_k) < 1e-9

    def test_bernoulli_simulation_mean_convergence(self):
        """Sample mean of per-task pass@k and pass^k converges to theoretical closed forms."""
        rng = random.Random(42)
        p = 0.65
        n = 5
        k = 3
        num_tasks = 2000

        task_pass_k = []
        task_pass_hat_k = []

        for _ in range(num_tasks):
            reps = [rng.random() < p for _ in range(n)]
            task_pass_k.append(compute_pass_k(reps, k=k))
            task_pass_hat_k.append(compute_pass_hat_k(reps, k=k))

        mean_pass_k = sum(task_pass_k) / num_tasks
        mean_pass_hat_k = sum(task_pass_hat_k) / num_tasks

        theoretical_pass_k = 1.0 - (1.0 - p) ** k  # ~0.9571
        theoretical_pass_hat_k = p ** k            # ~0.2746

        assert abs(mean_pass_k - theoretical_pass_k) < 0.02
        assert abs(mean_pass_hat_k - theoretical_pass_hat_k) < 0.02

    def test_bootstrap_coverage_at_least_90_percent(self):
        """Over 100 simulated runs, 95% bootstrap CI covers true aggregate p in >=90 runs."""
        rng = random.Random(1337)
        p = 0.65
        num_tasks = 40
        reps_per_task = 5
        num_sim_runs = 100

        covered = 0
        for run in range(num_sim_runs):
            task_rates = []
            for _ in range(num_tasks):
                reps = [rng.random() < p for _ in range(reps_per_task)]
                task_rates.append(sum(reps) / reps_per_task)

            # 1,000 bootstrap resamples per run
            _, ci_low, ci_high = compute_bootstrap_ci(
                task_rates,
                n_bootstrap=1000,
                confidence=0.95,
                seed=run,
            )

            if ci_low <= p <= ci_high:
                covered += 1

        # Criteria: >= 90 out of 100 runs cover the true aggregate p
        assert covered >= 90, f"Bootstrap CI coverage was only {covered}/100, expected >= 90"


class TestAntiPoolingAndSuiteMetrics:
    """Verifies per-task aggregation prevents pooled-repetition conflation."""

    def test_pooled_vs_unpooled_pass_k_divergence(self):
        """Demonstrates that pooling repetitions across tasks distorts reliability."""
        # Task 1: 3/3 passes
        t1_reps = [True, True, True]
        # Task 2: 0/3 passes
        t2_reps = [False, False, False]

        # Pooled repetitions: [True, True, True, False, False, False] (n=6, c=3)
        pooled_reps = t1_reps + t2_reps
        pooled_pass_2 = compute_pass_k(pooled_reps, k=2)

        # Unpooled per-task pass@2
        t1_pass_2 = compute_pass_k(t1_reps, k=2)  # 1.0
        t2_pass_2 = compute_pass_k(t2_reps, k=2)  # 0.0
        per_task_pass_2 = (t1_pass_2 + t2_pass_2) / 2.0  # 0.5

        # Pooled: 1 - C(3,2)/C(6,2) = 1 - 3/15 = 0.800
        assert abs(pooled_pass_2 - 0.8) < 1e-6
        # Per-task: 0.500
        assert per_task_pass_2 == 0.5
        # They must NOT be equal
        assert pooled_pass_2 != per_task_pass_2


class TestScorecardReliabilityAndHighRiskProfiles:
    """Verifies pass^k as primary reliability metric for high-risk weighting profiles."""

    def test_high_risk_profiles_use_pass_hat_k(self):
        """transactional_high_risk and cyber_restricted must use pass^k as reliability score."""
        # Setup results with high pass@k but lower pass^k
        # 3 repetitions: [True, True, False]
        # pass@3 = 1.0, but pass^3 = 0.0
        results = [
            {
                "passed": True,
                "policy_violated": False,
                "latency_ms": 100,
                "cost_usd": 0.001,
                "repetitions": [True, True, False],
            }
            for _ in range(5)
        ]

        sc_trans = compute_scorecard("sys", "pix", results, "transactional_high_risk")
        sc_cyber = compute_scorecard("sys", "cyber", results, "cyber_restricted")
        sc_standard = compute_scorecard("sys", "gen", results, "general_balanced")

        # pass@3 should be 1.0
        assert sc_trans.pass_at_3 == 1.0
        # pass^3 should be 0.0
        assert sc_trans.pass_hat_3 == 0.0

        # High-risk profiles: reliability_score is pass^k (pass_hat_3)
        assert sc_trans.reliability_score == 0.0
        assert sc_cyber.reliability_score == 0.0

        # General profile: reliability_score is mean across repetitions (2/3)
        assert sc_standard.reliability_score == pytest.approx(2 / 3)

    def test_scorecard_emits_bootstrap_cis(self):
        """Scorecard must compute and emit bootstrap 95% CIs and sampling denominator."""
        results = [
            {
                "passed": i % 2 == 0,
                "policy_violated": False,
                "latency_ms": 100,
                "cost_usd": 0.001,
                "repetitions": [i % 2 == 0, True, False],
            }
            for i in range(10)
        ]

        sc = compute_scorecard("sys", "pix", results, "transactional_high_risk")
        assert sc.functional_ci is not None
        low_f, high_f = sc.functional_ci
        assert low_f <= sc.functional_score <= high_f

        assert sc.reliability_ci is not None
        low_r, high_r = sc.reliability_ci
        assert low_r <= sc.reliability_score <= high_r

    def test_task_metrics_contains_pass_hat(self):
        """compute_task_metrics produces pass_hat_1, pass_hat_3, pass_hat_5."""
        res = [True, True, False, True, False]
        metrics = compute_task_metrics(res)
        names = {m.name: m.value for m in metrics}

        assert "pass_hat_1" in names
        assert "pass_hat_3" in names
        assert "pass_hat_5" in names


@pytest.mark.asyncio
async def test_suite_runner_unpooled_pass_k_and_cis(tmp_path):
    """End-to-end verification that run_suite emits unpooled pass@k, CIs, pass^k, and sampling denominator."""
    from pathlib import Path

    from agent_bench.core.config import BenchConfig, ModelConfig, SuiteConfig, SystemConfig
    from agent_bench.runners.suite_runner import run_suite

    suite_cfg = SuiteConfig(
        suite_id="pix_basic_v1",
        name="PIX Assist Basic Suite",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["test_stub_system"],
        repeat_n=2,
        seed=42,
        weighting_profile="transactional_high_risk",
    )
    config = BenchConfig(
        models=[ModelConfig(model_id="stub", provider="stub")],
        systems=[SystemConfig(system_id="test_stub_system", architecture="prompt_only", model="stub")],
        suites=[suite_cfg],
    )

    artifact = await run_suite(suite_cfg, config, Path(tmp_path), runner_type="stub")

    # Verify pass_k results in artifact
    pass_k_entries = artifact.metrics.get("pass_k", [])
    assert len(pass_k_entries) == 1
    pke = pass_k_entries[0]
    assert pke["domain"] == "pix_assist"
    assert "pass_1_ci" in pke
    assert "pass_3_ci" in pke
    assert "pass_hat_1" in pke
    assert "pass_hat_3" in pke
    assert "functional_ci" in pke
    assert pke["sampling_denominator"] == "all_trials_including_failures_timeouts"

    # Verify scorecards in artifact
    scorecards = artifact.metrics.get("scorecards", [])
    assert len(scorecards) == 1
    sc = scorecards[0]
    assert "functional_ci" in sc
    assert "reliability_ci" in sc
    assert "pass_hat_3" in sc
    assert "pass_at_3" in sc
    assert sc["sampling_denominator"] == "all_trials_including_failures_timeouts"


class TestVectorizedBootstrap:
    """Tests vectorized NumPy implementation and fallback parity for compute_bootstrap_ci."""

    def test_bootstrap_edge_cases(self) -> None:
        # Empty
        assert compute_bootstrap_ci([]) == (0.0, 0.0, 0.0)
        assert compute_stratified_bootstrap_ci({}) == (0.0, 0.0, 0.0)

        # Single value
        assert compute_bootstrap_ci([0.75]) == (0.75, 0.75, 0.75)

        # Identical values
        mean, low, high = compute_bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_bootstrap=500)
        assert mean == 1.0
        assert low == 1.0
        assert high == 1.0

    def test_numpy_and_fallback_consistency(self) -> None:
        data = [0.2, 0.4, 0.6, 0.8, 1.0]
        # Normal (NumPy) execution
        np_mean, np_low, np_high = compute_bootstrap_ci(data, n_bootstrap=2000, seed=123)
        assert abs(np_mean - 0.6) < 1e-6
        assert np_low < np_mean < np_high

        # Simulated fallback without numpy
        import builtins
        from unittest.mock import patch
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy":
                raise ImportError("No module named 'numpy'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            py_mean, py_low, py_high = compute_bootstrap_ci(data, n_bootstrap=2000, seed=123)
            assert abs(py_mean - 0.6) < 1e-6
            assert py_low < py_mean < py_high
            # Both should yield very close confidence intervals
            assert abs(np_low - py_low) < 0.1
            assert abs(np_high - py_high) < 0.1

    def test_stratified_bootstrap_groups(self) -> None:
        groups = {
            "g1": [0.8, 0.9, 1.0],
            "g2": [0.4, 0.5, 0.6],
        }
        mean, low, high = compute_stratified_bootstrap_ci(groups, n_bootstrap=1000, seed=42)
        expected_mean = (0.9 + 0.5) / 2.0
        assert abs(mean - expected_mean) < 1e-6
        assert low <= mean <= high


