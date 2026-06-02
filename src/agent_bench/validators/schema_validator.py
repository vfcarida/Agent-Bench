"""Schema validation for EvalCase v2 dictionaries."""

from dataclasses import dataclass, field
from typing import Any


VALID_FAMILIES = {"transactional_tools", "knowledge_rag_reasoning", "business_long_horizon", "security_guardrails"}
VALID_SOURCE_TYPES = {"human_gold", "synthetic_shadow", "synthetic_candidate", "adversarial", "calibration"}
VALID_SPLITS = {"dev", "holdout", "calibration", "regression", "smoke"}
VALID_DIFFICULTIES = {"easy", "medium", "hard", "expert"}
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}
VALID_GRADING_STRATEGIES = {"state_based", "tool_call_based", "rubric_based", "composite"}

REQUIRED_FIELDS = [
    "id", "family", "domain", "source_type", "split", "difficulty",
    "risk_level", "grading_strategy",
]

# At least one of these must be present
PROMPT_FIELDS = ["prompt_or_user_goal", "prompt", "input_messages"]

ENUM_FIELDS: dict[str, set[str]] = {
    "family": VALID_FAMILIES,
    "source_type": VALID_SOURCE_TYPES,
    "split": VALID_SPLITS,
    "difficulty": VALID_DIFFICULTIES,
    "risk_level": VALID_RISK_LEVELS,
    "grading_strategy": VALID_GRADING_STRATEGIES,
}


@dataclass
class ValidationResult:
    """Result of validating a single eval case."""

    case_id: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_eval_case(case: dict[str, Any]) -> ValidationResult:
    """Validate a single case dict against the v2 schema."""
    result = ValidationResult(case_id=case.get("id"))

    # Check required fields
    for f in REQUIRED_FIELDS:
        if f not in case:
            result.errors.append(f"Missing required field: {f}")

    # Check at least one prompt field exists
    if not any(case.get(f) for f in PROMPT_FIELDS):
        result.errors.append(f"Missing prompt: need at least one of {PROMPT_FIELDS}")

    # Check enum values
    for field_name, valid_values in ENUM_FIELDS.items():
        if field_name in case and case[field_name] not in valid_values:
            result.errors.append(
                f"Invalid value for '{field_name}': '{case[field_name]}'. "
                f"Must be one of: {sorted(valid_values)}"
            )

    # Consistency: tool_call_based requires required_tool_patterns
    if case.get("grading_strategy") == "tool_call_based":
        patterns = case.get("required_tool_patterns")
        if not patterns:
            result.errors.append(
                "grading_strategy is 'tool_call_based' but required_tool_patterns is empty or missing"
            )

    # Consistency: expected_state_changes requires initial_state
    if case.get("expected_state_changes"):
        if not case.get("initial_state"):
            result.errors.append(
                "expected_state_changes is non-empty but initial_state is missing"
            )

    # Consistency: allowed_tools and forbidden_tools must not overlap
    allowed = set(case.get("allowed_tools") or [])
    forbidden = set(case.get("forbidden_tools") or [])
    overlap = allowed & forbidden
    if overlap:
        result.errors.append(
            f"allowed_tools and forbidden_tools overlap: {sorted(overlap)}"
        )

    return result
