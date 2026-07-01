import os
import pytest
from pathlib import Path
from datetime import datetime, timezone

from agent_bench.validators.schema_validator import validate_eval_case
from agent_bench.validators.dedup_checker import find_duplicates
from agent_bench.storage.jsonl import save_traces_jsonl, save_run_manifest, save_metrics_jsonl
from agent_bench.storage.parquet import save_metrics_parquet, save_comparison_parquet
from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType


# =====================================================================
# Strict Schema Validation Tests
# =====================================================================

def test_validate_eval_case_strict_input_messages():
    # Base valid minimal case
    base_case = {
        "id": "T_TEST",
        "family": "transactional_tools",
        "domain": "pix",
        "source_type": "human_gold",
        "split": "dev",
        "difficulty": "medium",
        "risk_level": "high",
        "grading_strategy": "state_based",
        "prompt_or_user_goal": "Transfer money",
    }

    # Case 1: input_messages is not a list
    case = dict(base_case, input_messages="not_a_list")
    res = validate_eval_case(case)
    assert not res.is_valid
    assert any("input_messages must be a list" in err for err in res.errors)

    # Case 2: input_messages element is not a dict
    case = dict(base_case, input_messages=["not_a_dict"])
    res = validate_eval_case(case)
    assert not res.is_valid
    assert any("must be a dictionary" in err for err in res.errors)

    # Case 3: input_messages dict missing role/content
    case = dict(base_case, input_messages=[{"role": "user"}])
    res = validate_eval_case(case)
    assert not res.is_valid
    assert any("must contain 'role' and 'content' keys" in err for err in res.errors)

    # Case 4: role or content not strings
    case = dict(base_case, input_messages=[{"role": 123, "content": "hello"}])
    res = validate_eval_case(case)
    assert not res.is_valid
    assert any("['role'] must be a string" in err for err in res.errors)


def test_validate_eval_case_strict_required_tool_patterns():
    base_case = {
        "id": "T_TEST",
        "family": "transactional_tools",
        "domain": "pix",
        "source_type": "human_gold",
        "split": "dev",
        "difficulty": "medium",
        "risk_level": "high",
        "grading_strategy": "tool_call_based",
        "prompt_or_user_goal": "Transfer money",
        "required_tool_patterns": "not_a_list",
    }
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("required_tool_patterns must be a list" in err for err in res.errors)

    # Dictionary pattern missing 'tool' key
    base_case["required_tool_patterns"] = [{"not_tool": "check_balance"}]
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("dictionary must contain a 'tool' key" in err for err in res.errors)

    # Valid list of strings/dicts
    base_case["required_tool_patterns"] = ["check_balance", {"tool": "do_transfer"}]
    base_case["allowed_tools"] = ["check_balance", "do_transfer"]
    res = validate_eval_case(base_case)
    assert res.is_valid


def test_validate_eval_case_strict_evidence_strings():
    base_case = {
        "id": "T_TEST",
        "family": "knowledge_rag_reasoning",
        "domain": "rag",
        "source_type": "human_gold",
        "split": "dev",
        "difficulty": "medium",
        "risk_level": "high",
        "grading_strategy": "composite",
        "prompt_or_user_goal": "Query report",
        "evidence_strings": "not_a_list",
    }
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("evidence_strings must be a list" in err for err in res.errors)

    # Missing keys in dict
    base_case["evidence_strings"] = [{"claim": "Revenue is 5M"}]
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("missing required key: 'evidence'" in err for err in res.errors)

    # Correct formats
    base_case["evidence_strings"] = [{"claim": "c", "evidence": "e", "source_doc_id": "d"}]
    res = validate_eval_case(base_case)
    assert res.is_valid


def test_validate_eval_case_strict_rubric():
    base_case = {
        "id": "T_TEST",
        "family": "security_guardrails",
        "domain": "guardrails",
        "source_type": "human_gold",
        "split": "dev",
        "difficulty": "medium",
        "risk_level": "high",
        "grading_strategy": "rubric_based",
        "prompt_or_user_goal": "Verify policy",
        "rubric": "not_a_dict",
    }
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("rubric must be a dictionary" in err for err in res.errors)

    # dimensions is not a list
    base_case["rubric"] = {"dimensions": "not_a_list"}
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("rubric['dimensions'] must be a list" in err for err in res.errors)

    # dimension missing name
    base_case["rubric"] = {"dimensions": [{"importance": "critical"}]}
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("missing required key: 'name'" in err for err in res.errors)

    # dimension importance invalid
    base_case["rubric"] = {"dimensions": [{"name": "safety", "importance": "super_critical"}]}
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("importance'] must be one of" in err for err in res.errors)

    # dimension keywords not list
    base_case["rubric"] = {"dimensions": [{"name": "safety", "importance": "critical", "keywords": "not_list"}]}
    res = validate_eval_case(base_case)
    assert not res.is_valid
    assert any("keywords'] must be a list" in err for err in res.errors)

    # Valid rubric format
    base_case["rubric"] = {
        "dimensions": [
            {"name": "safety", "importance": "critical", "keywords": ["refuse", "cannot"]}
        ]
    }
    res = validate_eval_case(base_case)
    assert res.is_valid


# =====================================================================
# Duplicate Checker Pruning & Correctness Tests
# =====================================================================

def test_find_duplicates_optimizations():
    # Setup test cases of varying lengths
    cases = [
        {"id": "c1", "prompt": "verify simple pix transaction with balance"},
        {"id": "c2", "prompt": "verify simple pix transaction with balance"}, # exact duplicate
        {"id": "c3", "prompt": "retrieve documentation on regulatory compliance and standard protocols"}, # completely different length/tokens
        {"id": "c4", "prompt": ""}, # empty prompt edge case
        {"id": "c5", "prompt": ""}, # another empty
    ]

    # Verify duplicate pairs detected
    dupes = find_duplicates(cases, threshold=0.8)
    # c1 <-> c2 should be identical (similarity = 1.0)
    # c4 <-> c5 should be identical (similarity = 1.0)
    # c1/c2 <-> c3 should be pruned instantly
    dupe_ids = {(d[0], d[1]) for d in dupes}
    assert ("c1", "c2") in dupe_ids
    assert ("c4", "c5") in dupe_ids
    assert len(dupes) == 2


# =====================================================================
# Atomic Resiliency Writes Tests
# =====================================================================

def test_jsonl_resilience_atomic_writes(tmp_path):
    output_path = tmp_path / "traces.jsonl"
    
    traces = [
        TraceEvent(
            event_id="evt_1",
            event_type=TraceEventType.TOOL_CALL,
            timestamp=datetime.now(timezone.utc),
            data={"input": "hello"},
        )
    ]
    
    # Save standard traces
    save_traces_jsonl(traces, output_path)
    assert output_path.exists()
    assert not Path(str(output_path) + ".tmp").exists()

    # Verify save_run_manifest
    artifact = RunArtifact(
        run_id="run_123",
        suite_id="suite_abc",
        system_id="sys_x",
        model_id="model_y",
        started_at=datetime.now(timezone.utc),
        config_hash="abcde12345",
        benchmark_version="1.0.0",
        tasks_total=10,
        tasks_passed=8,
        tasks_failed=2,
    )
    manifest_path = save_run_manifest(artifact, tmp_path)
    assert manifest_path.exists()
    assert not Path(str(manifest_path) + ".tmp").exists()


def test_parquet_resilience_atomic_writes(tmp_path):
    output_path = tmp_path / "metrics.parquet"
    records = [
        {
            "run_id": "r1",
            "system_id": "s1",
            "task_id": "t1",
            "domain": "pix",
            "metric_name": "accuracy",
            "metric_value": 1.0,
            "metric_category": "functional",
            "passed": True,
            "timestamp": "2026-07-01T00:00:00Z",
        }
    ]

    # Save standard metrics parquet
    save_metrics_parquet(records, output_path)
    assert output_path.exists()
    assert not Path(str(output_path) + ".tmp").exists()

    # Save comparison parquet
    comp_path = tmp_path / "comparison.parquet"
    scorecards = [
        {
            "system_id": "s1",
            "domain": "pix",
            "functional_score": 1.0,
            "risk_score": 0.9,
            "cost_score": 0.5,
            "latency_score": 0.7,
            "reliability_score": 0.9,
            "global_score": 0.8,
            "weighting_profile": "default",
        }
    ]
    save_comparison_parquet(["s1"], scorecards, comp_path)
    assert comp_path.exists()
    assert not Path(str(comp_path) + ".tmp").exists()
