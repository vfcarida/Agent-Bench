"""Unit tests for metric computation."""

import pytest

from agent_bench.metrics.compute import compute_pass_k, compute_task_metrics


class TestPassK:
    def test_all_pass(self):
        assert compute_pass_k([True, True, True], k=1) == 1.0

    def test_none_pass(self):
        assert compute_pass_k([False, False, False], k=1) == 0.0

    def test_one_of_three_pass_at_1(self):
        result = compute_pass_k([True, False, False], k=1)
        assert 0.3 <= result <= 0.4  # ~1/3

    def test_one_of_three_pass_at_3(self):
        result = compute_pass_k([True, False, False], k=3)
        assert result == 1.0  # guaranteed to find it in 3 tries

    def test_two_of_five_pass_at_1(self):
        result = compute_pass_k([True, True, False, False, False], k=1)
        assert 0.35 <= result <= 0.45

    def test_pass_k_with_k_greater_than_n(self):
        result = compute_pass_k([True, False], k=5)
        assert result > 0  # falls back gracefully

    def test_empty_results(self):
        assert compute_pass_k([], k=1) == 0.0


class TestTaskMetrics:
    def test_basic_metrics(self):
        results = [True, True, False, True, False]
        metrics = compute_task_metrics(results)
        names = {m.name: m.value for m in metrics}
        assert names["task_success"] == 0.6
        assert "pass_1" in names
        assert "pass_3" in names

    def test_with_latencies(self):
        results = [True, True]
        metrics = compute_task_metrics(results, latencies_ms=[100.0, 200.0])
        names = {m.name: m.value for m in metrics}
        assert names["latency_ms"] == 150.0

    def test_with_tokens_and_cost(self):
        results = [True]
        metrics = compute_task_metrics(
            results,
            tokens_in=[1000],
            tokens_out=[500],
            cost_per_1k_in=0.01,
            cost_per_1k_out=0.03,
        )
        names = {m.name: m.value for m in metrics}
        assert names["total_tokens_in"] == 1000.0
        assert names["total_tokens_out"] == 500.0
        assert abs(names["estimated_cost_usd"] - 0.025) < 0.001
