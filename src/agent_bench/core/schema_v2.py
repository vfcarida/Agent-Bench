"""Unified EvalCase schema v2 for agent-bench."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------- BTB-inspired: Rubric importance levels ----------
# From BankerToolBench (arXiv:2604.11304): criteria are scored binary
# with importance weights (1, 3, 5, 10). Critical criteria act as
# hard gates — if any critical criterion fails, the overall score
# is capped at 0.5 regardless of other dimensions.
IMPORTANCE_WEIGHTS: dict[str, int] = {
    "critical": 10,
    "high": 5,
    "medium": 3,
    "low": 1,
}


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


class RubricImportance(str, Enum):
    """BTB-inspired importance levels for rubric dimensions."""
    CRITICAL = "critical"   # weight=10, hard gate
    HIGH = "high"           # weight=5
    MEDIUM = "medium"       # weight=3
    LOW = "low"             # weight=1


class AnswerFormat(str, Enum):
    """FINESSE-Bench-inspired answer format types for unified evaluation."""
    MCQ = "mcq"             # Multiple-choice question
    NAQ = "naq"             # Numerical-answer question
    SAQ = "saq"             # Short-answer question
    CASE = "case"           # Case-linked question
    FREE_FORM = "free_form" # Open-ended response
    TOOL_USE = "tool_use"   # Tool-use evaluation


class DeliverableType(str, Enum):
    """BTB-inspired deliverable types for multi-output evaluation."""
    TEXT = "text"
    STRUCTURED_DATA = "structured_data"    # JSON, tables, spreadsheets
    CALCULATION = "calculation"            # Numeric results with workings
    RECOMMENDATION = "recommendation"      # Advisory output
    REPORT = "report"                      # Multi-section document


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
    # FinanceBench-inspired: explicit evidence strings mapping claims to source docs
    evidence_strings: list[dict[str, str]] = field(default_factory=list)
    grading_strategy: str = "state_based"
    rubric: dict[str, Any] = field(default_factory=dict)
    # FINESSE-Bench-inspired: answer format for unified evaluation templates
    answer_format: str = "free_form"
    # BTB-inspired: expected deliverable types for multi-output tasks
    expected_deliverables: list[str] = field(default_factory=list)
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

    # ---------- Bidirectional Aliases for Schema Unification ----------

    @property
    def task_id(self) -> str:
        """Alias for id matching Task schema."""
        return self.id

    @task_id.setter
    def task_id(self, value: str) -> None:
        self.id = value

    @property
    def prompt(self) -> str:
        """Alias for prompt_or_user_goal matching Task schema."""
        return self.prompt_or_user_goal

    @property
    def expected_final_state(self) -> dict[str, Any]:
        """Alias for expected_state_changes matching Task schema."""
        if self.expected_state_changes:
            return self.expected_state_changes
        if isinstance(self.expected_outcome, dict):
            res = self.expected_outcome.get("state_changes", {})
            if isinstance(res, dict):
                return res
        return {}

    @property
    def task_version(self) -> str:
        """Alias for version matching Task schema."""
        return self.version

    @task_version.setter
    def task_version(self, value: str) -> None:
        self.version = value

    # ---------- Canonical Factory & Serialization Methods ----------

    def to_task(self) -> Any:
        """Convert this EvalCase to a canonical Task instance."""
        from agent_bench.core.scenarios import Task
        return Task.from_eval_case(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert this EvalCase to a dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_task(cls, task: Any) -> "EvalCase":
        """Instantiate an EvalCase from a Task object or dictionary."""
        from agent_bench.core.scenarios import Task

        if isinstance(task, dict):
            task_obj = Task.from_dict(task)
        elif isinstance(task, Task):
            task_obj = task
        else:
            task_obj = Task.from_eval_case(task)

        return cls(
            id=task_obj.task_id,
            family="transactional_tools",
            domain=task_obj.domain,
            prompt_or_user_goal=task_obj.prompt,
            input_messages=task_obj.input_messages,
            version=task_obj.task_version,
            locale="pt-BR",
            difficulty="medium",
            risk_level=task_obj.risk_level,
            source_type="human_gold",
            split="dev",
            initial_state=task_obj.initial_state,
            allowed_tools=task_obj.allowed_tools,
            forbidden_tools=[],
            policy_refs=[],
            knowledge_refs=task_obj.gold_references,
            expected_outcome={
                "state_changes": task_obj.expected_final_state,
                "refusal_expected": task_obj.expected_refusal_mode.value != "none",
            },
            expected_state_changes=task_obj.expected_final_state,
            required_tool_patterns=[{"tool": t} for t in task_obj.allowed_tools],
            forbidden_tool_patterns=[],
            evidence_requirements=task_obj.required_capabilities,
            evidence_strings=task_obj.evidence_strings,
            grading_strategy="state_based",
            rubric={},
            answer_format=task_obj.answer_format,
            expected_deliverables=task_obj.expected_deliverables,
            tags=task_obj.tags,
            severity=task_obj.severity.value if hasattr(task_obj.severity, "value") else str(task_obj.severity),
            business_criticality=(
                task_obj.business_criticality.value
                if hasattr(task_obj.business_criticality, "value")
                else str(task_obj.business_criticality)
            ),
            expected_refusal_mode=(
                task_obj.expected_refusal_mode.value
                if hasattr(task_obj.expected_refusal_mode, "value")
                else str(task_obj.expected_refusal_mode)
            ),
            metadata=dict(task_obj.metadata),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalCase":
        """Instantiate an EvalCase from a dictionary with fallback handling."""
        case_id = str(d.get("id") or d.get("task_id", ""))
        input_msgs = d.get("input_messages", [])
        prompt = d.get("prompt_or_user_goal") or d.get("prompt", "")
        if not input_msgs and prompt:
            input_msgs = [{"role": "user", "content": prompt}]

        return cls(
            id=case_id,
            family=d.get("family", "transactional_tools"),
            domain=d.get("domain", "default"),
            prompt_or_user_goal=prompt,
            input_messages=input_msgs,
            version=d.get("version") or d.get("task_version", "1.0.0"),
            locale=d.get("locale", "pt-BR"),
            difficulty=d.get("difficulty", "medium"),
            risk_level=d.get("risk_level") or d.get("severity", "medium"),
            source_type=d.get("source_type", "human_gold"),
            split=d.get("split", "dev"),
            initial_state=d.get("initial_state", {}),
            allowed_tools=d.get("allowed_tools", []),
            forbidden_tools=d.get("forbidden_tools", []),
            policy_refs=d.get("policy_refs", []),
            knowledge_refs=d.get("knowledge_refs") or d.get("gold_references", []),
            expected_outcome=d.get("expected_outcome", {}),
            expected_state_changes=d.get("expected_state_changes") or d.get("expected_final_state", {}),
            required_tool_patterns=d.get("required_tool_patterns", []),
            forbidden_tool_patterns=d.get("forbidden_tool_patterns", []),
            evidence_requirements=d.get("evidence_requirements") or d.get("required_capabilities", []),
            evidence_strings=d.get("evidence_strings", []),
            grading_strategy=d.get("grading_strategy", "state_based"),
            rubric=d.get("rubric", {}),
            answer_format=d.get("answer_format", "free_form"),
            expected_deliverables=d.get("expected_deliverables", []),
            metadata=d.get("metadata", {}),
            tags=d.get("tags", []),
            severity=d.get("severity") or d.get("risk_level", "medium"),
            business_criticality=d.get("business_criticality", "operational"),
            expected_refusal_mode=d.get("expected_refusal_mode", "none"),
        )

