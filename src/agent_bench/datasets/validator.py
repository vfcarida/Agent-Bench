"""Dataset validation: schema checks and warnings for task definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent_bench.core.scenarios import (
    BusinessCriticality,
    RefusalMode,
    Severity,
)


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    task_id: str
    field: str
    message: str


@dataclass
class ValidationResult:
    domain: str
    path: Path
    valid: bool
    task_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


REQUIRED_TASK_FIELDS = {"task_id", "name", "input_messages"}
VALID_SEVERITIES = {s.value for s in Severity}
VALID_CRITICALITIES = {bc.value for bc in BusinessCriticality}
VALID_REFUSAL_MODES = {rm.value for rm in RefusalMode}
VALID_TAGS = {
    "happy_path", "edge_case", "refusal", "policy_violation", "policy_conflict",
    "multi_step", "calculation", "temporal", "anomaly", "comparison",
    "suitability", "rebalancing", "projection", "benchmarking",
    "revenue_analysis", "expense_analysis", "diagnosis", "planning",
    "seasonality", "optimization", "payroll", "privacy",
    "competitor_data", "guarantee_prohibition", "concentration_violation",
    "insider_trading", "suitability_mismatch", "emergency_fund",
    "financial_planning", "basic_transfer", "insufficient_funds",
    "limit_exceeded", "anti_fraud", "escalation", "validation_error",
    "reversal", "reversal_denied", "social_engineering",
    "firewall", "defensive", "siem", "anomaly_detection",
    "offensive_request", "certificates", "proactive", "exploit",
    "incident_response", "isolation", "data_exfiltration",
    "credential_harvesting", "patch_management", "compliance",
    "dns", "threat_detection", "short_term", "etf", "previdencia",
    "allocation", "multi_product", "conservative", "regulatory",
}


def validate_dataset(path: Path) -> ValidationResult:
    """Validate a domain dataset YAML file."""
    result = ValidationResult(domain="", path=path, valid=True)

    if not path.exists():
        result.valid = False
        result.issues.append(ValidationIssue("error", "", "file", f"File not found: {path}"))
        return result

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.valid = False
        result.issues.append(ValidationIssue("error", "", "yaml", f"YAML parse error: {e}"))
        return result

    if not isinstance(data, dict):
        result.valid = False
        result.issues.append(ValidationIssue("error", "", "structure", "Root must be a mapping"))
        return result

    result.domain = data.get("domain", "unknown")

    # Check top-level fields
    if "domain" not in data:
        result.issues.append(ValidationIssue("warning", "", "domain", "Missing 'domain' field"))
    if "version" not in data:
        result.issues.append(ValidationIssue("warning", "", "version", "Missing 'version' field"))
    if "tasks" not in data:
        result.valid = False
        result.issues.append(ValidationIssue("error", "", "tasks", "Missing 'tasks' field"))
        return result

    tasks = data["tasks"]
    if not isinstance(tasks, list):
        result.valid = False
        result.issues.append(ValidationIssue("error", "", "tasks", "'tasks' must be a list"))
        return result

    result.task_count = len(tasks)
    task_ids = set()

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{i}")

        # Check required fields
        for field_name in REQUIRED_TASK_FIELDS:
            if field_name not in task:
                result.valid = False
                result.issues.append(
                    ValidationIssue("error", task_id, field_name, f"Missing required field '{field_name}'")
                )

        # Check duplicate IDs
        if task_id in task_ids:
            result.valid = False
            result.issues.append(ValidationIssue("error", task_id, "task_id", "Duplicate task_id"))
        task_ids.add(task_id)

        # Validate enums
        severity = task.get("severity", "medium")
        if severity not in VALID_SEVERITIES:
            result.issues.append(
                ValidationIssue("error", task_id, "severity", f"Invalid severity: {severity}")
            )
            result.valid = False

        criticality = task.get("business_criticality", "operational")
        if criticality not in VALID_CRITICALITIES:
            result.issues.append(
                ValidationIssue("error", task_id, "business_criticality", f"Invalid criticality: {criticality}")
            )
            result.valid = False

        refusal = task.get("expected_refusal_mode", "none")
        if refusal not in VALID_REFUSAL_MODES:
            result.issues.append(
                ValidationIssue("error", task_id, "expected_refusal_mode", f"Invalid refusal mode: {refusal}")
            )
            result.valid = False

        # Warnings
        if not task.get("tags"):
            result.issues.append(ValidationIssue("warning", task_id, "tags", "No tags defined"))

        if not task.get("description"):
            result.issues.append(ValidationIssue("warning", task_id, "description", "Missing description"))

        if not task.get("allowed_tools") and refusal == "none":
            result.issues.append(
                ValidationIssue("warning", task_id, "allowed_tools", "No tools defined for non-refusal task")
            )

        # Check for unknown tags
        tags = task.get("tags", [])
        unknown_tags = set(tags) - VALID_TAGS
        if unknown_tags:
            result.issues.append(
                ValidationIssue("warning", task_id, "tags", f"Unknown tags: {unknown_tags}")
            )

        # Refusal tasks should have expected_refusal_mode
        if "refusal" in tags and refusal == "none":
            result.issues.append(
                ValidationIssue("warning", task_id, "expected_refusal_mode",
                                "Task tagged 'refusal' but expected_refusal_mode is 'none'")
            )

        # Happy path should have expected_final_state
        if "happy_path" in tags and not task.get("expected_final_state"):
            result.issues.append(
                ValidationIssue("warning", task_id, "expected_final_state",
                                "Happy path task missing expected_final_state")
            )

    return result


def validate_all_datasets(fixtures_dir: Path) -> list[ValidationResult]:
    """Validate all dataset files in a directory."""
    results = []
    for path in sorted(fixtures_dir.glob("*.yaml")):
        if path.name.endswith("_template.yaml"):
            continue
        results.append(validate_dataset(path))
    return results
