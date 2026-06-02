"""Validation pipeline orchestrating all validators."""

from dataclasses import dataclass, field
from typing import Any

from .schema_validator import validate_eval_case, ValidationResult
from .consistency_checker import (
    check_state_consistency,
    check_tool_consistency,
    check_numeric_consistency,
)
from .dedup_checker import find_duplicates


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    """Normalize field names from legacy/variant formats to v2 canonical names."""
    out = dict(case)
    # Map old field names to v2
    if "task_id" in out and "id" not in out:
        out["id"] = out["task_id"]
    if "expected_final_state" in out and "expected_state_changes" not in out:
        out["expected_state_changes"] = out["expected_final_state"]
    if "prompt_or_user_goal" in out and "prompt" not in out:
        out["prompt"] = out["prompt_or_user_goal"]
    if "input_messages" in out and "prompt" not in out and "prompt_or_user_goal" not in out:
        msgs = out["input_messages"]
        if msgs:
            out["prompt_or_user_goal"] = msgs[0].get("content", "")
    # Infer missing fields with safe defaults
    if "split" not in out:
        out["split"] = "smoke"
    if "risk_level" not in out:
        sev = out.get("severity", "medium")
        out["risk_level"] = sev if sev in ("low", "medium", "high", "critical") else "medium"
    if "grading_strategy" not in out:
        if out.get("expected_state_changes") or out.get("expected_final_state"):
            out["grading_strategy"] = "state_based"
        elif out.get("required_tool_patterns"):
            out["grading_strategy"] = "tool_call_based"
        else:
            out["grading_strategy"] = "composite"
    if "difficulty" not in out:
        out["difficulty"] = "medium"
    return out


@dataclass
class ValidationReport:
    """Aggregated validation report for a collection of cases."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    summary_text: str = ""


def run_validation_pipeline(cases: list[dict[str, Any]], *, auto_normalize: bool = True) -> ValidationReport:
    """Run schema validation + consistency + dedup for all cases."""
    report = ValidationReport(total=len(cases))

    if auto_normalize:
        cases = [normalize_case(c) for c in cases]

    for case in cases:
        case_id = case.get("id", "unknown")

        # Schema validation
        result: ValidationResult = validate_eval_case(case)

        # Consistency checks (add to result)
        result.errors.extend(check_state_consistency(case))
        result.errors.extend(check_tool_consistency(case))
        result.warnings.extend(check_numeric_consistency(case))

        # Aggregate
        if result.is_valid:
            report.valid += 1
        else:
            report.invalid += 1

        for err in result.errors:
            report.errors.append(f"[{case_id}] {err}")
        for warn in result.warnings:
            report.warnings.append(f"[{case_id}] {warn}")

    # Deduplication
    report.duplicates = find_duplicates(cases)
    for id1, id2, sim in report.duplicates:
        report.warnings.append(f"Potential duplicate: {id1} <-> {id2} (similarity={sim:.2f})")

    # Summary
    lines = [
        f"Validation Report: {report.total} cases",
        f"  Valid: {report.valid}",
        f"  Invalid: {report.invalid}",
        f"  Errors: {len(report.errors)}",
        f"  Warnings: {len(report.warnings)}",
        f"  Duplicates: {len(report.duplicates)}",
    ]
    report.summary_text = "\n".join(lines)

    return report
