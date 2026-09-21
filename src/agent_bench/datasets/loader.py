"""Dataset loading utilities with split discipline and tree consolidation."""

from pathlib import Path
from typing import Any

import yaml

from agent_bench.core.scenarios import (
    BusinessCriticality,
    RefusalMode,
    Severity,
    Task,
)

# Canonical dataset roots
_REPO_ROOT = Path(__file__).parents[3]
_DATASETS_ROOT = _REPO_ROOT / "datasets"
_FIXTURES_DIR = _REPO_ROOT / "data" / "fixtures"


def _parse_task_or_case(item: dict[str, Any], domain_id: str) -> Task:
    """Parse a task dict or EvalCase dict into a Task object."""
    t_id = str(item.get("task_id") or item.get("id", ""))
    name = str(item.get("name") or item.get("prompt_or_user_goal") or item.get("prompt") or t_id)
    description = str(item.get("description") or item.get("prompt_or_user_goal") or item.get("prompt", ""))

    input_messages = item.get("input_messages")
    if not input_messages:
        prompt_text = item.get("prompt_or_user_goal") or item.get("prompt", "")
        if prompt_text:
            input_messages = [{"role": "user", "content": prompt_text}]
        else:
            input_messages = []

    exp_state = (
        item.get("expected_final_state")
        or item.get("expected_state_changes")
        or (item.get("expected_outcome", {}).get("state_changes", {}))
        or {}
    )

    req_cap = item.get("required_capabilities") or item.get("evidence_requirements", [])

    refusal_raw = item.get("expected_refusal_mode", "none")
    try:
        refusal_mode = RefusalMode(refusal_raw)
    except ValueError:
        refusal_mode = RefusalMode.NONE

    sev_raw = item.get("severity") or item.get("risk_level", "medium")
    try:
        severity = Severity(sev_raw)
    except ValueError:
        severity = Severity.MEDIUM

    crit_raw = item.get("business_criticality", "operational")
    try:
        criticality = BusinessCriticality(crit_raw)
    except ValueError:
        criticality = BusinessCriticality.OPERATIONAL

    return Task(
        task_id=t_id,
        domain=str(item.get("domain") or domain_id),
        name=name,
        description=description,
        input_messages=input_messages,
        initial_state=item.get("initial_state", {}),
        expected_final_state=exp_state,
        allowed_tools=item.get("allowed_tools", []),
        required_capabilities=req_cap,
        expected_refusal_mode=refusal_mode,
        gold_references=item.get("gold_references") or item.get("knowledge_refs", []),
        evidence_strings=item.get("evidence_strings", []),
        answer_format=item.get("answer_format", "free_form"),
        expected_deliverables=item.get("expected_deliverables", []),
        severity=severity,
        business_criticality=criticality,
        tags=item.get("tags", []),
        task_version=str(item.get("task_version") or item.get("version", "1.0.0")),
        metadata=item.get("metadata", {}),
    )


def _resolve_dataset_file(domain_id: str, split: str, data_dir: Path | None) -> Path | None:
    """Resolve the authoritative dataset file path for a domain and split."""
    if data_dir is not None:
        # Check explicit data_dir with split subfolder
        split_path = data_dir / split / f"{domain_id}.yaml"
        if split_path.exists():
            return split_path

        # Check flat data_dir
        flat_path = data_dir / f"{domain_id}.yaml"
        if flat_path.exists():
            # FORBIDDEN REGRESSION: holdout content must never be loaded from data/fixtures
            try:
                is_fixtures = (
                    data_dir.resolve() == _FIXTURES_DIR.resolve()
                    or "data/fixtures" in str(data_dir).replace("\\", "/")
                )
            except (OSError, RuntimeError):
                is_fixtures = "fixtures" in str(data_dir).lower()

            if split == "holdout" and is_fixtures:
                return None
            if split != "dev" and is_fixtures:
                return None
            return flat_path

        return None

    # Canonical Gold tree: datasets/gold/<split>/<domain>.yaml
    gold_path = _DATASETS_ROOT / "gold" / split / f"{domain_id}.yaml"
    if gold_path.exists():
        return gold_path

    # Synthetic Shadow / Candidates tree
    for syn_match in _DATASETS_ROOT.glob(f"synthetic/**/{domain_id}.yaml"):
        if syn_match.exists():
            return syn_match

    # Adversarial tree
    adv_path = _DATASETS_ROOT / "adversarial" / f"{domain_id}.yaml"
    if adv_path.exists():
        return adv_path

    return None


def load_domain_tasks(
    domain_id: str,
    split: str = "dev",
    data_dir: Path | None = None,
) -> list[Task]:
    """Load all tasks for a domain from the authoritative dataset tree.

    Resolves:
    1. datasets/gold/<split>/<domain>.yaml (Canonical Gold)
    2. datasets/synthetic/**/<domain>.yaml
    3. datasets/adversarial/<domain>.yaml

    Enforces that 'holdout' split is never loaded from data/fixtures.
    """
    domain_file = _resolve_dataset_file(domain_id, split=split, data_dir=data_dir)
    if not domain_file or not domain_file.exists():
        return []

    with open(domain_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    # Extract raw items from either EvalCase format or Task format
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        # If the file declares a top-level split that contradicts the requested split
        file_split = data.get("split")
        if file_split and file_split != split:
            return []
        raw_items = data.get("cases") or data.get("tasks") or []
    else:
        raw_items = []

    tasks: list[Task] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        item_split = item.get("split")
        if item_split and item_split != split:
            continue
        tasks.append(_parse_task_or_case(item, domain_id=domain_id))

    return tasks
