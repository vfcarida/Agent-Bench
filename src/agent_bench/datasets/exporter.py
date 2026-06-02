"""Export dataset templates for new domains."""

from pathlib import Path

import yaml


_TEMPLATE = {
    "domain": "DOMAIN_ID",
    "version": "1.0.0",
    "description": "Description of the domain scenario",
    "policy": {
        "max_transaction_value": 0,
        "requires_confirmation": True,
        "allowed_hours": "00:00-23:59",
    },
    "available_tools": ["tool_a", "tool_b"],
    "success_criteria": ["criterion_1", "criterion_2"],
    "tasks": [
        {
            "task_id": "DOMAIN_001",
            "name": "Example task",
            "description": "Describe the task",
            "input_messages": [{"role": "user", "content": "User message here"}],
            "initial_state": {"balance": 1000.0},
            "expected_final_state": {"balance": 900.0},
            "allowed_tools": ["tool_a"],
            "required_capabilities": ["tool_calling"],
            "expected_refusal_mode": "none",
            "gold_references": [],
            "severity": "medium",
            "business_criticality": "operational",
            "tags": ["happy_path"],
            "task_version": "1.0.0",
            "metadata": {},
        }
    ],
}


def export_template(domain: str, output_path: Path) -> None:
    """Export a YAML template for a domain's task dataset."""
    template = _TEMPLATE.copy()
    template["domain"] = domain
    template["tasks"][0]["task_id"] = f"{domain.upper()}_001"  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
