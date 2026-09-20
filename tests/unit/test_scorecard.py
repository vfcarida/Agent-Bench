"""Unit tests for scorecard computation."""


from agent_bench.metrics.scorecard import compute_scorecard


def test_all_pass_scorecard():
    results = [
        {"passed": True, "policy_violated": False, "latency_ms": 1000, "cost_usd": 0.01, "repetitions": [True, True, True]}
        for _ in range(10)
    ]
    sc = compute_scorecard("sys_a", "pix_assist", results, "transactional_high_risk")
    assert sc.functional_score == 1.0
    assert sc.risk_score == 1.0
    assert sc.reliability_score == 1.0
    assert sc.global_score > 0.9


def test_mixed_results_scorecard():
    results = [
        {"passed": True, "policy_violated": False, "latency_ms": 500, "cost_usd": 0.005, "repetitions": [True, True, False]},
        {"passed": False, "policy_violated": True, "latency_ms": 8000, "cost_usd": 0.05, "repetitions": [False, False, False]},
        {"passed": True, "policy_violated": False, "latency_ms": 2000, "cost_usd": 0.01, "repetitions": [True, False, True]},
        {"passed": False, "policy_violated": False, "latency_ms": 1000, "cost_usd": 0.01, "repetitions": [False, False, False]},
    ]
    sc = compute_scorecard("sys_b", "investment_advisor", results, "advisory_regulated")
    assert 0 < sc.functional_score < 1
    assert sc.safety_violations == 1
    assert sc.safety_gated is True
    assert sc.global_score > 0


def test_empty_results():
    sc = compute_scorecard("sys_x", "domain", [], "high_volume_low_risk")
    assert sc.functional_score == 0.0
    assert sc.global_score == 0.0


def test_weighting_profile_affects_score():
    results = [
        {"passed": True, "policy_violated": False, "latency_ms": 6000, "cost_usd": 0.001, "repetitions": [True, False, False]},
    ]
    sc_risk = compute_scorecard("sys", "d", results, "transactional_high_risk")
    sc_cost = compute_scorecard("sys", "d", results, "high_volume_low_risk")
    # High risk profile penalizes low pass^k reliability more heavily
    assert sc_risk.global_score < sc_cost.global_score


def test_safety_violation_gates_scorecard():
    # With policy violation, system is gated and excluded from global_score
    results = [
        {"passed": True, "policy_violated": True, "latency_ms": 100, "cost_usd": 0.001, "repetitions": [True]},
    ]
    sc = compute_scorecard("sys", "d", results, "transactional_high_risk")
    assert sc.safety_gated is True
    assert sc.safety_violations == 1
    assert sc.global_score == 0.0

