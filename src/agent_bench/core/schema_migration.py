"""Migration utilities from legacy Task format to EvalCase v2."""

from pathlib import Path
from typing import Any

import yaml


def migrate_task_to_v2(task_dict: dict[str, Any], domain: str) -> dict[str, Any]:
    expected_final_state = task_dict.get("expected_final_state", {})
    input_messages = task_dict.get("input_messages", [])
    prompt = input_messages[0]["content"] if input_messages else ""

    return {
        "id": task_dict.get("task_id", ""),
        "version": task_dict.get("task_version", "1.0.0"),
        "family": "transactional_tools",
        "domain": domain,
        "locale": "pt-BR",
        "difficulty": "medium",
        "risk_level": task_dict.get("severity", "medium"),
        "source_type": "human_gold",
        "split": "dev",
        "prompt_or_user_goal": prompt,
        "input_messages": input_messages,
        "initial_state": task_dict.get("initial_state", {}),
        "allowed_tools": task_dict.get("allowed_tools", []),
        "forbidden_tools": [],
        "policy_refs": [],
        "knowledge_refs": task_dict.get("gold_references", []),
        "expected_outcome": {
            "state_changes": expected_final_state,
            "refusal_expected": task_dict.get("expected_refusal_mode", "none") != "none",
        },
        "expected_state_changes": expected_final_state,
        "required_tool_patterns": [],
        "forbidden_tool_patterns": [],
        "evidence_requirements": task_dict.get("required_capabilities", []),
        "grading_strategy": "state_based",
        "rubric": {},
        "metadata": {
            "created_by": "migration",
            "generator_model": "",
            "judge_model": "",
            "reviewer": "",
            "review_status": "draft",
            "promoted_to_gold_at": None,
            "parent_seed_id": None,
            "generation_recipe_id": None,
        },
        "tags": task_dict.get("tags", []),
        "severity": task_dict.get("severity", "medium"),
        "business_criticality": task_dict.get("business_criticality", "operational"),
        "expected_refusal_mode": task_dict.get("expected_refusal_mode", "none"),
    }


def migrate_fixture_file(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = yaml.safe_load(f)

    domain = data.get("domain", "unknown")
    tasks = data.get("tasks", [])
    return [migrate_task_to_v2(task, domain) for task in tasks]
