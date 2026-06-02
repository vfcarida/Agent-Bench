"""Unified EvalCase schema v2 for agent-bench."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Family(str, Enum):
    TRANSACTIONAL_TOOLS = "transactional_tools"
    KNOWLEDGE_RAG_REASONING = "knowledge_rag_reasoning"
    BUSINESS_LONG_HORIZON = "business_long_horizon"
    SECURITY_GUARDRAILS = "security_guardrails"


class SourceType(str, Enum):
    HUMAN_GOLD = "human_gold"
    SYNTHETIC_SHADOW = "synthetic_shadow"
    SYNTHETIC_CANDIDATE = "synthetic_candidate"
    ADVERSARIAL = "adversarial"
    CALIBRATION = "calibration"


class Split(str, Enum):
    DEV = "dev"
    HOLDOUT = "holdout"
    CALIBRATION = "calibration"
    REGRESSION = "regression"
    SMOKE = "smoke"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GradingStrategy(str, Enum):
    STATE_BASED = "state_based"
    TOOL_CALL_BASED = "tool_call_based"
    RUBRIC_BASED = "rubric_based"
    COMPOSITE = "composite"


@dataclass
class EvalCase:
    id: str
    family: str
    domain: str
    prompt_or_user_goal: str
    input_messages: list[dict[str, str]]
    version: str = "1.0.0"
    locale: str = "pt-BR"
    difficulty: str = "medium"
    risk_level: str = "low"
    source_type: str = "human_gold"
    split: str = "dev"
    initial_state: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    policy_refs: list[str] = field(default_factory=list)
    knowledge_refs: list[str] = field(default_factory=list)
    expected_outcome: dict[str, Any] = field(default_factory=dict)
    expected_state_changes: dict[str, Any] = field(default_factory=dict)
    required_tool_patterns: list[dict[str, Any]] = field(default_factory=list)
    forbidden_tool_patterns: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)
    grading_strategy: str = "state_based"
    rubric: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=lambda: {
        "created_by": "",
        "generator_model": "",
        "judge_model": "",
        "reviewer": "",
        "review_status": "draft",
        "promoted_to_gold_at": None,
        "parent_seed_id": None,
        "generation_recipe_id": None,
    })
    tags: list[str] = field(default_factory=list)
    severity: str = "medium"
    business_criticality: str = "operational"
    expected_refusal_mode: str = "none"
