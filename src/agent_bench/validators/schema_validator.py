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

    # Strict nested validation: input_messages
    if "input_messages" in case:
        messages = case["input_messages"]
        if not isinstance(messages, list):
            result.errors.append("input_messages must be a list")
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    result.errors.append(f"input_messages[{i}] must be a dictionary")
                else:
                    if "role" not in msg or "content" not in msg:
                        result.errors.append(f"input_messages[{i}] must contain 'role' and 'content' keys")
                    else:
                        if not isinstance(msg["role"], str):
                            result.errors.append(f"input_messages[{i}]['role'] must be a string")
                        if not isinstance(msg["content"], str):
                            result.errors.append(f"input_messages[{i}]['content'] must be a string")

    # Strict nested validation: required_tool_patterns
    if "required_tool_patterns" in case:
        patterns = case["required_tool_patterns"]
        if not isinstance(patterns, list):
            result.errors.append("required_tool_patterns must be a list")
        else:
            for i, pattern in enumerate(patterns):
                if isinstance(pattern, dict):
                    if "tool" not in pattern:
                        result.errors.append(f"required_tool_patterns[{i}] dictionary must contain a 'tool' key")
                    elif not isinstance(pattern["tool"], str):
                        result.errors.append(f"required_tool_patterns[{i}]['tool'] must be a string")
                elif not isinstance(pattern, str):
                    result.errors.append(f"required_tool_patterns[{i}] must be a string or a dictionary")

    # Strict nested validation: evidence_strings
    if "evidence_strings" in case:
        ev_strings = case["evidence_strings"]
        if not isinstance(ev_strings, list):
            result.errors.append("evidence_strings must be a list")
        else:
            for i, ev in enumerate(ev_strings):
                if not isinstance(ev, dict):
                    result.errors.append(f"evidence_strings[{i}] must be a dictionary")
                else:
                    required_keys = ["claim", "evidence", "source_doc_id"]
                    for k in required_keys:
                        if k not in ev:
                            result.errors.append(f"evidence_strings[{i}] missing required key: '{k}'")
                        elif not isinstance(ev[k], str):
                            result.errors.append(f"evidence_strings[{i}]['{k}'] must be a string")

    # Strict nested validation: rubric
    if "rubric" in case:
        rubric = case["rubric"]
        if not isinstance(rubric, dict):
            result.errors.append("rubric must be a dictionary")
        else:
            if "dimensions" in rubric:
                dims = rubric["dimensions"]
                if not isinstance(dims, list):
                    result.errors.append("rubric['dimensions'] must be a list")
                else:
                    for i, dim in enumerate(dims):
                        if not isinstance(dim, dict):
                            result.errors.append(f"rubric['dimensions'][{i}] must be a dictionary")
                        else:
                            if "name" not in dim:
                                result.errors.append(f"rubric['dimensions'][{i}] missing required key: 'name'")
                            elif not isinstance(dim["name"], str):
                                result.errors.append(f"rubric['dimensions'][{i}]['name'] must be a string")
                            
                            if "importance" in dim:
                                importance = dim["importance"]
                                if not isinstance(importance, str):
                                    result.errors.append(f"rubric['dimensions'][{i}]['importance'] must be a string")
                                elif importance not in ("critical", "high", "medium", "low"):
                                    result.errors.append(
                                        f"rubric['dimensions'][{i}]['importance'] must be one of: "
                                        "['critical', 'high', 'medium', 'low']"
                                    )
                            
                            if "keywords" in dim:
                                keywords = dim["keywords"]
                                if not isinstance(keywords, list):
                                    result.errors.append(f"rubric['dimensions'][{i}]['keywords'] must be a list")
                                else:
                                    for j, kw in enumerate(keywords):
                                        if not isinstance(kw, str):
                                            result.errors.append(f"rubric['dimensions'][{i}]['keywords'][{j}] must be a string")

    return result
