"""Dataset loading utilities."""

from pathlib import Path

import yaml

from agent_bench.core.scenarios import (
    BusinessCriticality,
    RefusalMode,
    Severity,
    Task,
)

# Default data directory
_DATA_DIR = Path(__file__).parents[3] / "data" / "fixtures"


def load_domain_tasks(domain_id: str, data_dir: Path | None = None) -> list[Task]:
    """Load all tasks for a domain from YAML fixtures."""
    base = data_dir or _DATA_DIR
    domain_file = base / f"{domain_id}.yaml"
    if not domain_file.exists():
        return []

    with open(domain_file) as f:
        data = yaml.safe_load(f)

    tasks = []
    for t in data.get("tasks", []):
        tasks.append(Task(
            task_id=t["task_id"],
            domain=domain_id,
            name=t["name"],
            description=t.get("description", ""),
            input_messages=t.get("input_messages", []),
            initial_state=t.get("initial_state", {}),
            expected_final_state=t.get("expected_final_state", {}),
            allowed_tools=t.get("allowed_tools", []),
            required_capabilities=t.get("required_capabilities", []),
            expected_refusal_mode=RefusalMode(t.get("expected_refusal_mode", "none")),
            gold_references=t.get("gold_references", []),
            severity=Severity(t.get("severity", "medium")),
            business_criticality=BusinessCriticality(
                t.get("business_criticality", "operational")
            ),
            tags=t.get("tags", []),
            task_version=t.get("task_version", "1.0.0"),
            metadata=t.get("metadata", {}),
        ))
    return tasks
