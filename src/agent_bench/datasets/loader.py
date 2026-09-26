"""Dataset loading utilities with split discipline and tree consolidation."""

from pathlib import Path
from typing import Any

import yaml

from agent_bench.core.scenarios import Task

# Canonical dataset roots
_REPO_ROOT = Path(__file__).parents[3]
_DATASETS_ROOT = _REPO_ROOT / "datasets"
_FIXTURES_DIR = _REPO_ROOT / "data" / "fixtures"


def _parse_task_or_case(item: dict[str, Any], domain_id: str) -> Task:
    """Parse a task dict or EvalCase dict into a Task object."""
    return Task.from_dict(item, domain_id=domain_id)



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
