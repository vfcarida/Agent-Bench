"""Unit tests for safety-constrained Pareto frontier, Graduated Autonomy, and psychometric diagnostics."""

from __future__ import annotations

import math

from agent_bench.graders.state_grader import GradeResult
from agent_bench.metrics.difficulty_analysis import (
    compare_difficulty_robustness,
    compute_difficulty_degradation,
    compute_difficulty_profile,
)
from agent_bench.metrics.discriminative import (
    compute_discriminative_power,
    compute_per_domain_discriminative_power,
)
from agent_bench.metrics.pareto import (
    AutonomyTier,
    classify_autonomy_tier,
    compute_hypervolume_2d,
    compute_pareto_frontier,
)
from agent_bench.metrics.report import MetricsReport, build_metrics_report
from agent_bench.metrics.scorecard import compute_balanced_profile
from agent_bench.metrics.transfer_analysis import (
    compute_system_transfer_profiles,
    compute_transfer_gaps,
    rank_by_transfer_robustness,
)


class TestGraduatedAutonomyTiers:
    def test_blocked_unsafe_when_safety_gated(self) -> None:
        sc = {
            "system_id": "unsafe_agent",
            "functional_score": 0.99,
            "pass_hat_3": 0.99,
            "safety_violations": 1,
            "safety_gated": True,
        }
        res = classify_autonomy_tier(sc)
        assert res["tier"] == AutonomyTier.BLOCKED_UNSAFE.value
        assert res["deployable"] is False

    def test_tier_4_autonomous(self) -> None:
        sc = {
            "system_id": "frontier_agent",
            "functional_score": 0.94,
            "pass_hat_3": 0.90,
            "hallucination_rate": 0.01,
            "safety_violations": 0,
            "safety_gated": False,
        }
        res = classify_autonomy_tier(sc)
        assert res["tier"] == AutonomyTier.TIER_4_AUTONOMOUS.value
        assert res["deployable"] is True

    def test_tier_3_guided_autonomy(self) -> None:
        sc = {
            "system_id": "guided_agent",
            "functional_score": 0.85,
            "pass_hat_3": 0.78,
            "safety_violations": 0,
            "safety_gated": False,
        }
        res = classify_autonomy_tier(sc)
        assert res["tier"] == AutonomyTier.TIER_3_GUIDED_AUTONOMY.value
        assert res["deployable"] is True

    def test_tier_2_supervised(self) -> None:
        sc = {
            "system_id": "copilot_agent",
            "functional_score": 0.68,
            "pass_hat_3": 0.55,
            "safety_violations": 0,
            "safety_gated": False,
        }
        res = classify_autonomy_tier(sc)
        assert res["tier"] == AutonomyTier.TIER_2_SUPERVISED.value
        assert res["deployable"] is True

    def test_tier_1_shadow(self) -> None:
        sc = {
            "system_id": "weak_agent",
            "functional_score": 0.40,
            "pass_hat_3": 0.25,
            "safety_violations": 0,
            "safety_gated": False,
        }
        res = classify_autonomy_tier(sc)
        assert res["tier"] == AutonomyTier.TIER_1_SHADOW.value
        assert res["deployable"] is False


class TestSafetyConstrainedParetoFrontier:
    def test_empty_scorecards(self) -> None:
        res = compute_pareto_frontier([])
        assert res["frontier"] == []
        assert res["dominated"] == []
        assert res["safety_disqualified"] == []
        assert res["hypervolume_2d"] == 0.0

    def test_pareto_dominance_and_safety_disqualification(self) -> None:
        scorecards = [
            {
                "system_id": "fast_cheap",
                "domain": "pix_assist",
                "functional_score": 0.75,
                "pass_hat_3": 0.70,
                "cost_per_successful_task": 0.01,
                "latency_p90": 500.0,
                "safety_violations": 0,
                "safety_gated": False,
            },
            {
                "system_id": "accurate_moderate_cost",
                "domain": "pix_assist",
                "functional_score": 0.92,
                "pass_hat_3": 0.88,
                "cost_per_successful_task": 0.05,
                "latency_p90": 1200.0,
                "safety_violations": 0,
                "safety_gated": False,
            },
            {
                "system_id": "dominated_expensive",
                "domain": "pix_assist",
                "functional_score": 0.80,
                "pass_hat_3": 0.72,
                "cost_per_successful_task": 0.12,
                "latency_p90": 2500.0,
                "safety_violations": 0,
                "safety_gated": False,
            },
            {
                "system_id": "unsafe_high_accuracy",
                "domain": "pix_assist",
                "functional_score": 0.99,
                "pass_hat_3": 0.99,
                "cost_per_successful_task": 0.005,
                "latency_p90": 300.0,
                "safety_violations": 2,
                "safety_gated": True,
            },
        ]
        res = compute_pareto_frontier(scorecards, max_cost_reference=1.0)
        frontier_ids = [s["system_id"] for s in res["frontier"]]
        dominated_ids = [s["system_id"] for s in res["dominated"]]
        disqualified_ids = [s["system_id"] for s in res["safety_disqualified"]]

        assert frontier_ids == ["fast_cheap", "accurate_moderate_cost"]
        assert dominated_ids == ["dominated_expensive"]
        assert disqualified_ids == ["unsafe_high_accuracy"]
        assert res["dominated"][0]["dominated_by"] == ["accurate_moderate_cost"]
        assert res["dominated"][0]["quality_deficit"] > 0.0
        assert res["dominated"][0]["excess_cost_usd"] > 0.0
        assert 0.85 < res["hypervolume_2d"] <= 1.0

    def test_hypervolume_2d_boundary_cases(self) -> None:
        assert compute_hypervolume_2d([], max_cost_reference=1.0) == 0.0
        assert compute_hypervolume_2d([(0.0, 1.0)], max_cost_reference=1.0) == 1.0
        assert math.isclose(compute_hypervolume_2d([(0.5, 0.8)], max_cost_reference=1.0), 0.4, rel_tol=1e-6)


class TestDifficultyDegradationAnalysis:
    def test_monotonic_degradation_and_rankings(self) -> None:
        results_sys_a = [
            {"difficulty": "easy", "passed": True},
            {"difficulty": "easy", "passed": True},
            {"difficulty": "medium", "passed": True},
            {"difficulty": "medium", "passed": False},
            {"difficulty": "hard", "passed": True},
            {"difficulty": "hard", "passed": False},
            {"difficulty": "hard", "passed": False},
            {"difficulty": "expert", "passed": False},
        ]
        results_sys_b = [
            {"difficulty": "easy", "passed": True},
            {"difficulty": "medium", "passed": True},
            {"difficulty": "hard", "passed": True},
            {"difficulty": "expert", "passed": True},
        ]
        deg_a = compute_difficulty_degradation(results_sys_a, score_field="passed")
        assert deg_a["degradation_pattern"] == "monotonic_degradation"
        assert deg_a["deltas"]["delta_easy_to_expert"] == 1.0

        deg_b = compute_difficulty_degradation(results_sys_b, score_field="passed")
        assert deg_b["degradation_pattern"] == "robust"
        assert deg_b["deltas"]["delta_easy_to_expert"] == 0.0

        profiles = compute_difficulty_profile({"sys_a": results_sys_a, "sys_b": results_sys_b})
        rankings = compare_difficulty_robustness(profiles)
        assert rankings[0]["system_id"] == "sys_b"
        assert rankings[1]["system_id"] == "sys_a"

    def test_cliff_and_non_monotonic_patterns(self) -> None:
        cliff_results = [
            {"difficulty": "easy", "score": 0.95},
            {"difficulty": "medium", "score": 0.94},
            {"difficulty": "hard", "score": 0.93},
            {"difficulty": "expert", "score": 0.10},
        ]
        assert compute_difficulty_degradation(cliff_results)["degradation_pattern"] == "cliff"

        non_mono = [
            {"difficulty": "easy", "score": 0.60},
            {"difficulty": "medium", "score": 0.90},
            {"difficulty": "hard", "score": 0.50},
            {"difficulty": "expert", "score": 0.40},
        ]
        assert compute_difficulty_degradation(non_mono)["degradation_pattern"] == "non_monotonic"

        insufficient = [{"difficulty": "easy", "score": 1.0}]
        assert compute_difficulty_degradation(insufficient)["degradation_pattern"] == "insufficient_data"


class TestDiscriminativePowerAnalysis:
    def test_discriminative_power_and_alerts(self) -> None:
        assert compute_discriminative_power({})["total_tasks"] == 0

        task_results = {
            "sys_strong": {"t1": True, "t2": True, "t3": True, "t4": True, "t5": False},
            "sys_mid": {"t1": True, "t2": True, "t3": False, "t4": False, "t5": False},
            "sys_weak": {"t1": True, "t2": False, "t3": False, "t4": False, "t5": False},
        }
        dp = compute_discriminative_power(task_results)
        assert dp["total_tasks"] == 5
        assert dp["total_systems"] == 3
        assert dp["unanimous_success_count"] == 1
        assert dp["unanimous_fail_count"] == 1
        assert dp["mid_band_10_90_count"] == 3
        assert dp["avg_item_discrimination"] > 0.0
        assert dp["gini_coefficient"] > 0.0

        per_domain = compute_per_domain_discriminative_power({"pix_assist": task_results})
        assert "pix_assist" in per_domain
        assert per_domain["pix_assist"]["total_tasks"] == 5

    def test_saturation_and_hard_alerts(self) -> None:
        saturated = {
            "s1": {"t1": True, "t2": True, "t3": True},
            "s2": {"t1": True, "t2": True, "t3": True},
        }
        dp_sat = compute_discriminative_power(saturated)
        assert any("SATURATION" in a for a in dp_sat["alerts"])
        assert any("LOW_DISCRIMINABILITY" in a for a in dp_sat["alerts"])

        too_hard = {
            "s1": {"t1": False, "t2": False, "t3": False},
            "s2": {"t1": False, "t2": False, "t3": False},
        }
        dp_hard = compute_discriminative_power(too_hard)
        assert any("TOO_HARD" in a for a in dp_hard["alerts"])


class TestTransferAndBalancedProfiles:
    def test_transfer_gaps_and_ranking(self) -> None:
        assert compute_transfer_gaps({})["reference_family"] is None

        sys_results = {
            "robust_sys": {
                "transactional": [{"passed": True}, {"passed": True}],
                "knowledge_rag": [{"passed": True}, {"passed": True}],
                "security": [{"passed": True}, {"passed": True}],
            },
            "fragile_sys": {
                "transactional": [{"passed": True}, {"passed": True}],
                "knowledge_rag": [{"passed": True}, {"passed": False}],
                "security": [{"passed": False}, {"passed": False}],
            },
        }
        profiles = compute_system_transfer_profiles(sys_results)
        assert profiles["robust_sys"]["transfer_pattern"] == "robust_transfer"
        assert profiles["fragile_sys"]["transfer_pattern"] == "asymmetric_degradation"

        ranked = rank_by_transfer_robustness(profiles)
        assert ranked[0]["system_id"] == "robust_sys"
        assert ranked[1]["system_id"] == "fragile_sys"

    def test_balanced_profile_computation(self) -> None:
        assert compute_balanced_profile([], "missing")["profile"] == "no_data"

        scorecards = [
            {"system_id": "balanced_sys", "domain": "pix", "functional_score": 0.90},
            {"system_id": "balanced_sys", "domain": "inv", "functional_score": 0.88},
            {"system_id": "balanced_sys", "domain": "sme", "functional_score": 0.86},
            {"system_id": "spiky_sys", "domain": "pix", "functional_score": 0.98},
            {"system_id": "spiky_sys", "domain": "inv", "functional_score": 0.45},
            {"system_id": "spiky_sys", "domain": "sme", "functional_score": 0.50},
        ]
        p_bal = compute_balanced_profile(scorecards, "balanced_sys")
        p_spiky = compute_balanced_profile(scorecards, "spiky_sys")
        assert p_bal["balanced"] is True
        assert p_bal["profile"] == "balanced"
        assert p_spiky["balanced"] is False
        assert p_spiky["profile"] == "strong_local"


class TestMetricsReportBuilder:
    def test_build_metrics_report(self) -> None:
        empty = build_metrics_report([], {})
        assert isinstance(empty, MetricsReport)
        assert empty.success_rate == 0.0

        grades = [
            GradeResult(
                passed=True,
                score=1.0,
                strategy="deterministic",
                details={"precision": 1.0, "recall": 1.0},
            ),
            GradeResult(
                passed=True,
                score=0.8,
                strategy="deterministic",
                details={"precision": 0.8, "recall": 0.9},
            ),
            GradeResult(
                passed=False,
                score=0.2,
                strategy="deterministic",
                failure_category="tool_mismatch",
                details={"precision": 0.4, "recall": 0.5},
            ),
            GradeResult(
                passed=True,
                score=0.9,
                strategy="deterministic",
                details={"precision": 0.9, "recall": 1.0},
            ),
        ]
        meta = {
            "latencies": [120.0, 250.0, 400.0, 900.0],
            "total_tokens": 2400,
            "total_cost": 0.06,
            "policy_compliance_rate": 1.0,
            "groundedness": 0.95,
        }
        rep = build_metrics_report(grades, meta)
        assert rep.success_rate == 0.75
        assert rep.pass_at_3 == 1.0
        assert rep.latency_p50 > 0.0
        assert rep.latency_p95 >= rep.latency_p50
        assert math.isclose(rep.cost_per_success, 0.02, rel_tol=1e-6)
        assert "score" in rep.confidence_intervals
        assert "latency" in rep.confidence_intervals
