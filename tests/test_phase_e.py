"""Tests for Phase E: schema v2, validators, generators, graders, metrics."""
import json
from pathlib import Path

import pytest
import yaml

from agent_bench.core.schema_v2 import EvalCase, Family, SourceType, Split, Difficulty, RiskLevel, GradingStrategy
from agent_bench.core.schema_migration import migrate_task_to_v2, migrate_fixture_file
from agent_bench.validators.schema_validator import validate_eval_case, ValidationResult
from agent_bench.validators.consistency_checker import check_state_consistency, check_tool_consistency, check_numeric_consistency
from agent_bench.validators.dedup_checker import compute_case_hash, find_duplicates
from agent_bench.validators.pipeline import run_validation_pipeline, normalize_case
from agent_bench.generators.transactional import TransactionalGenerator
from agent_bench.generators.knowledge_rag import KnowledgeRagGenerator
from agent_bench.generators.business_horizon import BusinessHorizonGenerator
from agent_bench.generators.security import SecurityGenerator
from agent_bench.generators.perturbations import apply_noise, add_urgency, add_ambiguity, inject_distraction
from agent_bench.graders.state_grader import StateGrader, GradeResult
from agent_bench.graders.tool_call_grader import ToolCallGrader
from agent_bench.graders.rubric_grader import RubricGrader
from agent_bench.graders.composite_grader import CompositeGrader
from agent_bench.metrics.expanded import (
    compute_confidence_interval,
    compute_tool_call_precision,
    compute_tool_call_recall,
    compute_policy_compliance_rate,
    compute_cost_per_success,
    categorize_failures,
    compute_state_accuracy,
)


# === Schema V2 ===

class TestSchemaV2:
    def test_eval_case_creation(self):
        case = EvalCase(
            id="T001", family=Family.TRANSACTIONAL_TOOLS, domain="pix",
            prompt_or_user_goal="Transfer money",
            input_messages=[{"role": "user", "content": "Transfer money"}],
        )
        assert case.id == "T001"
        assert case.family == Family.TRANSACTIONAL_TOOLS
        assert case.locale == "pt-BR"
        assert case.difficulty == Difficulty.MEDIUM

    def test_enums(self):
        assert Family.SECURITY_GUARDRAILS.value == "security_guardrails"
        assert SourceType.HUMAN_GOLD.value == "human_gold"
        assert Split.HOLDOUT.value == "holdout"
        assert GradingStrategy.TOOL_CALL_BASED.value == "tool_call_based"


# === Schema Migration ===

class TestSchemaMigration:
    def test_migrate_basic_task(self):
        old_task = {
            "task_id": "PIX_001",
            "name": "Simple PIX transfer",
            "description": "Test",
            "input_messages": [{"role": "user", "content": "Faz PIX"}],
            "initial_state": {"balance": 5000},
            "expected_final_state": {"balance": 4900},
            "allowed_tools": ["check_balance"],
            "severity": "medium",
            "business_criticality": "financial",
            "tags": ["happy_path"],
        }
        v2 = migrate_task_to_v2(old_task, "pix_assist")
        assert v2["id"] == "PIX_001"
        assert v2["domain"] == "pix_assist"
        assert v2["family"] == "transactional_tools"
        assert v2["expected_state_changes"] == {"balance": 4900}
        assert v2["source_type"] == "human_gold"

    def test_migrate_fixture_file(self):
        cases = migrate_fixture_file(Path("data/fixtures/pix_assist.yaml"))
        assert len(cases) == 10
        assert all(c["domain"] == "pix_assist" for c in cases)
        assert all("id" in c for c in cases)

    def test_migrated_cases_pass_validation(self):
        cases = migrate_fixture_file(Path("data/fixtures/investment_advisor.yaml"))
        report = run_validation_pipeline(cases)
        assert report.valid == report.total


# === Validators ===

class TestSchemaValidator:
    def test_valid_case(self):
        case = {
            "id": "T001", "family": "transactional_tools", "domain": "pix",
            "source_type": "human_gold", "split": "dev", "difficulty": "medium",
            "risk_level": "high", "grading_strategy": "state_based",
            "prompt_or_user_goal": "Transfer money",
            "initial_state": {"balance": 1000},
            "expected_state_changes": {"balance": 900},
        }
        result = validate_eval_case(case)
        assert result.is_valid

    def test_missing_required_field(self):
        case = {"family": "transactional_tools", "domain": "pix"}
        result = validate_eval_case(case)
        assert not result.is_valid
        assert any("Missing required field" in e for e in result.errors)

    def test_invalid_enum_value(self):
        case = {
            "id": "T001", "family": "invalid_family", "domain": "pix",
            "source_type": "human_gold", "split": "dev", "difficulty": "medium",
            "risk_level": "high", "grading_strategy": "state_based",
            "prompt_or_user_goal": "test",
        }
        result = validate_eval_case(case)
        assert not result.is_valid
        assert any("Invalid value for 'family'" in e for e in result.errors)

    def test_tool_overlap_detection(self):
        case = {
            "id": "T001", "family": "transactional_tools", "domain": "pix",
            "source_type": "human_gold", "split": "dev", "difficulty": "medium",
            "risk_level": "high", "grading_strategy": "state_based",
            "prompt_or_user_goal": "test",
            "allowed_tools": ["check_balance", "transfer"],
            "forbidden_tools": ["transfer"],
        }
        result = validate_eval_case(case)
        assert not result.is_valid
        assert any("overlap" in e for e in result.errors)


class TestConsistencyChecker:
    def test_state_consistency_no_initial(self):
        case = {"expected_state_changes": {"done": True}, "initial_state": {}}
        errors = check_state_consistency(case)
        assert len(errors) == 1

    def test_tool_consistency_ok(self):
        case = {"allowed_tools": ["a", "b"], "required_tool_patterns": [{"tool": "a"}]}
        errors = check_tool_consistency(case)
        assert errors == []

    def test_tool_consistency_violation(self):
        case = {"allowed_tools": ["a"], "required_tool_patterns": [{"tool": "c"}]}
        errors = check_tool_consistency(case)
        assert len(errors) == 1


class TestDedupChecker:
    def test_no_duplicates(self):
        cases = [
            {"id": "1", "prompt_or_user_goal": "Transfer R$100 to Maria"},
            {"id": "2", "prompt_or_user_goal": "Check my balance please"},
        ]
        dups = find_duplicates(cases)
        assert dups == []

    def test_finds_duplicates(self):
        cases = [
            {"id": "1", "prompt_or_user_goal": "Transfer R$100 to Maria via PIX"},
            {"id": "2", "prompt_or_user_goal": "Transfer R$100 to Maria via PIX please"},
        ]
        dups = find_duplicates(cases, threshold=0.7)
        assert len(dups) >= 1

    def test_hash_deterministic(self):
        case = {"prompt_or_user_goal": "Test prompt", "initial_state": {"x": 1}}
        h1 = compute_case_hash(case)
        h2 = compute_case_hash(case)
        assert h1 == h2


class TestValidationPipeline:
    def test_pipeline_all_valid(self):
        cases = [
            {"id": f"T{i}", "family": "transactional_tools", "domain": "pix",
             "source_type": "human_gold", "split": "dev", "difficulty": "medium",
             "risk_level": "medium", "grading_strategy": "state_based",
             "prompt_or_user_goal": f"Unique prompt number {i}",
             "initial_state": {"balance": 1000 + i},
             "expected_state_changes": {"done": True}}
            for i in range(5)
        ]
        report = run_validation_pipeline(cases)
        assert report.valid == 5
        assert report.invalid == 0

    def test_normalize_case(self):
        old = {"task_id": "X", "expected_final_state": {"a": 1}, "input_messages": [{"role": "user", "content": "hi"}]}
        n = normalize_case(old)
        assert n["id"] == "X"
        assert n["expected_state_changes"] == {"a": 1}
        assert n["grading_strategy"] == "state_based"


# === Generators ===

class TestTransactionalGenerator:
    def test_generate_single(self):
        gen = TransactionalGenerator()
        r = gen.generate(difficulty="medium", seed=42)
        assert not r.rejected
        assert r.confidence_score > 0

    def test_generate_batch(self):
        gen = TransactionalGenerator()
        results = gen.generate_batch(20, seed=42)
        assert len(results) == 20
        assert all(not r.rejected for r in results)

    def test_deterministic(self):
        gen = TransactionalGenerator()
        r1 = gen.generate(difficulty="easy", seed=99)
        r2 = gen.generate(difficulty="easy", seed=99)
        # Same seed should produce same prompt content
        prompt_field = "prompt_or_user_goal" if "prompt_or_user_goal" in r1.case else "input_messages"
        if prompt_field == "input_messages":
            assert r1.case["input_messages"][0]["content"] == r2.case["input_messages"][0]["content"]
        else:
            assert r1.case[prompt_field] == r2.case[prompt_field]


class TestKnowledgeRagGenerator:
    def test_generate_single(self):
        gen = KnowledgeRagGenerator()
        r = gen.generate(difficulty="hard", seed=42)
        assert not r.rejected
        assert r.case.get("family") or r.case.get("domain")

    def test_generate_batch(self):
        gen = KnowledgeRagGenerator()
        results = gen.generate_batch(15, seed=42)
        assert len(results) == 15


class TestBusinessHorizonGenerator:
    def test_generate_single(self):
        gen = BusinessHorizonGenerator()
        r = gen.generate(difficulty="medium", seed=42)
        assert not r.rejected
        assert r.confidence_score > 0

    def test_generate_batch(self):
        gen = BusinessHorizonGenerator()
        results = gen.generate_batch(10, seed=42)
        assert len(results) == 10


class TestSecurityGenerator:
    def test_generate_single(self):
        gen = SecurityGenerator()
        r = gen.generate(difficulty="hard", seed=42)
        assert not r.rejected
        assert r.case["source_type"] == "adversarial"
        assert r.case["expected_refusal_mode"] == "block_and_log"

    def test_generates_all_categories(self):
        gen = SecurityGenerator()
        results = gen.generate_batch(40, seed=42)
        domains = set(r.case["domain"] for r in results)
        assert len(domains) >= 3

    def test_expert_difficulty(self):
        gen = SecurityGenerator()
        r = gen.generate(difficulty="expert", seed=42)
        assert not r.rejected


class TestPerturbations:
    def test_apply_noise(self):
        text = "Quero fazer um PIX para minha mae"
        noisy = apply_noise(text, noise_level=0.5, seed=42)
        # Should be modified
        assert isinstance(noisy, str)

    def test_add_urgency(self):
        text = "Faz um PIX de R$100"
        urgent = add_urgency(text)
        assert len(urgent) > len(text)

    def test_inject_distraction(self):
        text = "Transfere R$500"
        distracted = inject_distraction(text)
        assert len(distracted) > len(text)


# === Graders ===

class TestStateGrader:
    def test_pass(self):
        grader = StateGrader()
        case = {"expected_state_changes": {"balance": 900}, "expected_outcome": {}}
        r = grader.grade(case, {"balance": 900})
        assert r.passed
        assert r.score == 1.0

    def test_fail(self):
        grader = StateGrader()
        case = {"expected_state_changes": {"balance": 900, "done": True}, "expected_outcome": {}}
        r = grader.grade(case, {"balance": 1000, "done": False})
        assert not r.passed
        assert r.failure_category != ""

    def test_refusal_detection(self):
        grader = StateGrader()
        case = {"expected_state_changes": {}, "expected_outcome": {"refusal_expected": True}}
        r = grader.grade(case, {"refusal": True}, "Não posso fazer isso")
        assert r.passed


class TestToolCallGrader:
    def test_pass_all_required(self):
        grader = ToolCallGrader()
        case = {"required_tool_patterns": [{"tool": "check_balance"}, {"tool": "transfer"}], "forbidden_tool_patterns": []}
        r = grader.grade(case, [{"tool": "check_balance"}, {"tool": "transfer"}])
        assert r.passed
        assert r.score == 1.0

    def test_forbidden_tool_fails(self):
        grader = ToolCallGrader()
        case = {"required_tool_patterns": [], "forbidden_tool_patterns": ["hack"]}
        r = grader.grade(case, [{"tool": "hack"}])
        assert not r.passed
        assert r.failure_category == "forbidden_tool_used"

    def test_missing_required(self):
        grader = ToolCallGrader()
        case = {"required_tool_patterns": [{"tool": "a"}, {"tool": "b"}], "forbidden_tool_patterns": []}
        r = grader.grade(case, [{"tool": "a"}])
        assert r.details["recall"] == 0.5


class TestRubricGrader:
    def test_keyword_matching(self):
        grader = RubricGrader()
        case = {"rubric": {"dimensions": [
            {"name": "coverage", "weight": 1.0, "criteria": "test", "keywords": ["investimento", "renda"]},
        ]}}
        r = grader.grade(case, "O investimento em renda fixa é seguro")
        assert r.score > 0

    def test_no_match(self):
        grader = RubricGrader()
        case = {"rubric": {"dimensions": [
            {"name": "coverage", "weight": 1.0, "criteria": "test", "keywords": ["blockchain", "crypto"]},
        ]}}
        r = grader.grade(case, "O PIX é um sistema de pagamentos")
        assert r.score == 0


class TestCompositeGrader:
    def test_state_strategy(self):
        grader = CompositeGrader()
        case = {"grading_strategy": "state_based", "expected_state_changes": {"x": 1}, "expected_outcome": {}}
        r = grader.grade(case, actual_state={"x": 1}, actual_response="", actual_tool_calls=[])
        assert r.passed

    def test_tool_call_strategy(self):
        grader = CompositeGrader()
        case = {"grading_strategy": "tool_call_based", "required_tool_patterns": [{"tool": "a"}], "forbidden_tool_patterns": [], "expected_state_changes": {}, "expected_outcome": {}}
        r = grader.grade(case, actual_state={}, actual_response="", actual_tool_calls=[{"tool": "a"}])
        assert r.passed


# === Expanded Metrics ===

class TestExpandedMetrics:
    def test_confidence_interval(self):
        mean, lower, upper = compute_confidence_interval([0.8, 0.85, 0.9, 0.75, 0.88])
        assert lower < mean < upper
        assert 0.7 < mean < 0.95

    def test_confidence_interval_single(self):
        mean, lower, upper = compute_confidence_interval([0.5])
        assert mean == 0.5

    def test_tool_call_precision(self):
        expected = ["a", "b"]
        actual = ["a", "b", "c"]
        p = compute_tool_call_precision(expected, actual)
        assert abs(p - 2/3) < 0.01

    def test_tool_call_recall(self):
        expected = ["a", "b", "c"]
        actual = ["a", "b"]
        r = compute_tool_call_recall(expected, actual)
        assert abs(r - 2/3) < 0.01

    def test_policy_compliance(self):
        results = [
            {"policy_violated": False},
            {"policy_violated": False},
            {"policy_violated": True},
        ]
        rate = compute_policy_compliance_rate(results)
        assert abs(rate - 2/3) < 0.01

    def test_cost_per_success(self):
        assert compute_cost_per_success(10.0, 5) == 2.0
        assert compute_cost_per_success(10.0, 0) == float("inf")

    def test_categorize_failures(self):
        results = [
            GradeResult(score=0.0, passed=False, strategy="s", failure_category="timeout"),
            GradeResult(score=0.0, passed=False, strategy="s", failure_category="timeout"),
            GradeResult(score=0.0, passed=False, strategy="s", failure_category="wrong_answer"),
            GradeResult(score=1.0, passed=True, strategy="s"),
        ]
        tax = categorize_failures(results)
        assert tax["timeout"] == 2
        assert tax["wrong_answer"] == 1
        assert "passed" not in tax

    def test_state_accuracy(self):
        results = [
            GradeResult(score=1.0, passed=True, strategy="state_based"),
            GradeResult(score=0.5, passed=False, strategy="state_based"),
        ]
        acc = compute_state_accuracy(results)
        assert acc == 0.75


# === Integration: Generated datasets pass validation ===

class TestGeneratedDatasetsIntegrity:
    def test_synthetic_shadow_exists(self):
        shadow_dir = Path("datasets/synthetic/shadow")
        assert shadow_dir.exists()
        yamls = list(shadow_dir.rglob("*.yaml"))
        assert len(yamls) >= 4

    def test_gold_dev_exists(self):
        gold_dir = Path("datasets/gold/dev")
        assert gold_dir.exists()
        yamls = list(gold_dir.glob("*.yaml"))
        assert len(yamls) >= 4

    def test_adversarial_exists(self):
        adv_dir = Path("datasets/adversarial")
        assert adv_dir.exists()
        yamls = list(adv_dir.glob("*.yaml"))
        assert len(yamls) >= 3

    def test_gold_cases_validate(self):
        gold_dir = Path("datasets/gold/dev")
        for f in gold_dir.glob("*.yaml"):
            with open(f) as fh:
                data = yaml.safe_load(fh)
            cases = data.get("cases", [])
            report = run_validation_pipeline(cases)
            assert report.invalid == 0, f"{f.name}: {report.errors[:3]}"

    def test_synthetic_cases_validate(self):
        shadow_dir = Path("datasets/synthetic/shadow")
        for f in shadow_dir.rglob("*.yaml"):
            with open(f) as fh:
                data = yaml.safe_load(fh)
            cases = data.get("cases", [])
            report = run_validation_pipeline(cases[:20])  # sample for speed
            assert report.invalid == 0, f"{f.name}: {report.errors[:3]}"
