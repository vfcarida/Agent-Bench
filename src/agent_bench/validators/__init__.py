"""Dataset schema validation, consistency checking, deduplication, and contamination gating."""

from agent_bench.validators.consistency_checker import (
    check_numeric_consistency,
    check_state_consistency,
    check_tool_consistency,
)
from agent_bench.validators.contamination import (
    ContaminationLeak,
    ContaminationReport,
    check_contamination,
    compute_case_content_hash,
    detect_contamination,
)
from agent_bench.validators.dedup_checker import find_duplicates
from agent_bench.validators.pipeline import (
    ValidationReport,
    normalize_case,
    run_validation_pipeline,
)
from agent_bench.validators.schema_validator import ValidationResult, validate_eval_case

__all__ = [
    "ContaminationLeak",
    "ContaminationReport",
    "ValidationReport",
    "ValidationResult",
    "check_contamination",
    "check_numeric_consistency",
    "check_state_consistency",
    "check_tool_consistency",
    "compute_case_content_hash",
    "detect_contamination",
    "find_duplicates",
    "normalize_case",
    "run_validation_pipeline",
    "validate_eval_case",
]
