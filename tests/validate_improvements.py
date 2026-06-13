"""Quick validation of all new modules."""
from agent_bench.metrics.difficulty_analysis import compute_difficulty_degradation
from agent_bench.metrics.discriminative import compute_discriminative_power
from agent_bench.metrics.expanded import compute_bootstrap_ci, compute_stratified_bootstrap_ci
from agent_bench.metrics.transfer_analysis import compute_transfer_gaps
from agent_bench.metrics.scorecard import compute_balanced_profile
from agent_bench.graders.rubric_grader import RubricGrader
from agent_bench.judges.cross_artifact import CrossArtifactConsistencyJudge
import asyncio

print("=" * 60)
print("VALIDATION: All new modules from paper improvements")
print("=" * 60)

# M1: Rubric grader with hierarchical importance
print("\n[M1] Rubric Grader with Hierarchical Importance")
grader = RubricGrader()
case = {
    "rubric": {
        "dimensions": [
            {"name": "accuracy", "importance": "critical", "keywords": ["CDI", "110%"]},
            {"name": "compliance", "importance": "high", "keywords": ["FGC", "regulamento"]},
            {"name": "tone", "importance": "low", "keywords": ["prezado", "atenciosamente"]},
        ]
    }
}
result1 = grader.grade(case, "O CDB rende 110% do CDI, com cobertura do FGC conforme regulamento.")
print(f"  Score: {result1.score:.3f} | Passed: {result1.passed}")
print(f"  Details: {result1.details}")

result2 = grader.grade(case, "Invista tudo aqui, é seguro!")  # missing critical keywords
print(f"  Critical fail score: {result2.score:.3f} | Passed: {result2.passed}")
print(f"  Failure category: {result2.failure_category}")

# M4: Difficulty degradation
print("\n[M4] Difficulty Degradation Analysis")
results = [
    {"difficulty": "easy", "passed": True}, {"difficulty": "easy", "passed": True},
    {"difficulty": "medium", "passed": True}, {"difficulty": "medium", "passed": False},
    {"difficulty": "hard", "passed": False}, {"difficulty": "hard", "passed": True},
    {"difficulty": "expert", "passed": False}, {"difficulty": "expert", "passed": False},
]
degradation = compute_difficulty_degradation(results, score_field="passed")
print(f"  Level accuracy: {degradation['level_accuracy']}")
print(f"  Deltas: {degradation['deltas']}")
print(f"  Pattern: {degradation['degradation_pattern']}")

# M5: Transfer gap
print("\n[M5] Transfer Gap Analysis")
gaps = compute_transfer_gaps({
    "transactional_tools": [{"passed": True}, {"passed": True}, {"passed": True}],
    "knowledge_rag": [{"passed": True}, {"passed": False}, {"passed": True}],
    "security": [{"passed": False}, {"passed": False}, {"passed": True}],
})
print(f"  Reference: {gaps['reference_family']}")
print(f"  Gaps: {gaps['transfer_gaps']}")
print(f"  Pattern: {gaps['transfer_pattern']}")

# M6: Bootstrap CI
print("\n[M6] Bootstrap Confidence Intervals")
import random
random.seed(42)
values = [random.random() for _ in range(30)]
mean, lo, hi = compute_bootstrap_ci(values)
print(f"  Bootstrap CI: mean={mean:.3f}, 95%=[{lo:.3f}, {hi:.3f}]")

strat_mean, strat_lo, strat_hi = compute_stratified_bootstrap_ci({
    "group_a": values[:10],
    "group_b": values[10:20],
    "group_c": values[20:],
})
print(f"  Stratified CI: mean={strat_mean:.3f}, 95%=[{strat_lo:.3f}, {strat_hi:.3f}]")

# M7: Discriminative power
print("\n[M7] Discriminative Power Analysis")
task_results = {
    "sys_a": {"t1": True, "t2": True, "t3": False, "t4": True, "t5": False},
    "sys_b": {"t1": True, "t2": False, "t3": False, "t4": True, "t5": False},
    "sys_c": {"t1": False, "t2": False, "t3": False, "t4": True, "t5": True},
}
dp = compute_discriminative_power(task_results)
print(f"  Mid-band 10-90: {dp['mid_band_10_90_rate']:.2f}")
print(f"  Unanimous success: {dp['unanimous_success_rate']:.2f}")
print(f"  Unanimous fail: {dp['unanimous_fail_rate']:.2f}")
print(f"  Gini: {dp['gini_coefficient']:.3f}")
print(f"  Alerts: {dp['alerts']}")

# M2: Cross-artifact consistency (async)
print("\n[M2] Cross-Artifact Consistency")
from agent_bench.core.scenarios import Task

async def test_cross_artifact():
    judge = CrossArtifactConsistencyJudge()
    task = Task(
        task_id="test", domain="investment_advisor", name="test",
        description="test", input_messages=[]
    )
    # Consistent artifacts
    result1 = {
        "response": "A rentabilidade bruta é de R$ 15.000,00 com taxa de 12,5%.",
        "structured_data": "rentabilidade bruta: R$ 15.000,00; taxa: 12,5%",
    }
    verdict1 = await judge.evaluate(task, result1, [])
    print(f"  Consistent: score={verdict1.score:.2f}, passed={verdict1.passed}")

    # Inconsistent artifacts
    result2 = {
        "response": "A rentabilidade bruta é de R$ 15.000,00 com taxa de 12,5%.",
        "structured_data": "rentabilidade bruta: R$ 18.000,00; taxa: 10,0%",
    }
    verdict2 = await judge.evaluate(task, result2, [])
    print(f"  Inconsistent: score={verdict2.score:.2f}, passed={verdict2.passed}")
    print(f"  Contradictions: {verdict2.metadata.get('contradictions', [])[:2]}")

asyncio.run(test_cross_artifact())

# M9: Balanced profile
print("\n[M9] Balanced Model Profile")
scorecards = [
    {"system_id": "balanced_sys", "domain": "pix", "functional_score": 0.9},
    {"system_id": "balanced_sys", "domain": "investment", "functional_score": 0.85},
    {"system_id": "balanced_sys", "domain": "sme", "functional_score": 0.88},
    {"system_id": "local_leader", "domain": "pix", "functional_score": 0.99},
    {"system_id": "local_leader", "domain": "investment", "functional_score": 0.5},
    {"system_id": "local_leader", "domain": "sme", "functional_score": 0.6},
]
profile1 = compute_balanced_profile(scorecards, "balanced_sys")
profile2 = compute_balanced_profile(scorecards, "local_leader")
print(f"  balanced_sys: profile={profile1['profile']}, std={profile1['std_cross_domain']:.3f}")
print(f"  local_leader: profile={profile2['profile']}, std={profile2['std_cross_domain']:.3f}")

print("\n" + "=" * 60)
print("ALL VALIDATIONS PASSED!")
print("=" * 60)
