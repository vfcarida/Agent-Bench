"""Consistency checks for EvalCase dictionaries."""

from typing import Any


def check_state_consistency(case: dict[str, Any]) -> list[str]:
    """Return errors for state inconsistencies.

    Note: expected_state_changes MAY introduce new keys (representing state that
    gets created during execution), so missing keys are NOT errors.
    Errors are only raised for logical contradictions.
    """
    errors: list[str] = []
    initial_state = case.get("initial_state") or {}
    expected_changes = case.get("expected_state_changes") or {}

    # Check for contradictions: if a key exists in both and the expected value
    # is identical to initial (indicating no actual change was expected)
    # This is not necessarily an error, so we keep this conservative.

    if expected_changes and not initial_state:
        errors.append("expected_state_changes defined but initial_state is empty")

    return errors


def check_tool_consistency(case: dict[str, Any]) -> list[str]:
    """Return errors if required_tool_patterns use tools not in allowed_tools."""
    errors: list[str] = []
    allowed_tools = case.get("allowed_tools")
    required_patterns = case.get("required_tool_patterns") or []

    # Only check if allowed_tools is explicitly defined
    if allowed_tools is None:
        return errors

    allowed_set = set(allowed_tools)
    for pattern in required_patterns:
        tool_name = pattern.get("tool") if isinstance(pattern, dict) else pattern
        if tool_name and tool_name not in allowed_set:
            errors.append(
                f"required_tool_patterns references tool '{tool_name}' not in allowed_tools"
            )

    return errors


def check_numeric_consistency(case: dict[str, Any]) -> list[str]:
    """Return warnings if expected_outcome has numeric values but grading_strategy isn't compatible."""
    warnings: list[str] = []
    expected_outcome = case.get("expected_outcome") or {}
    grading_strategy = case.get("grading_strategy", "")

    has_numeric = False
    if isinstance(expected_outcome, dict):
        for value in expected_outcome.values():
            if isinstance(value, (int, float)):
                has_numeric = True
                break

    if has_numeric and grading_strategy not in ("state_based", "composite"):
        warnings.append(
            f"expected_outcome contains numeric values but grading_strategy is "
            f"'{grading_strategy}' (consider 'state_based' or 'composite')"
        )

    return warnings
