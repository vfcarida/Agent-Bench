"""Domain scenarios, tasks, and benchmark suites."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BusinessCriticality(str, Enum):
    INFORMATIONAL = "informational"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"


class RefusalMode(str, Enum):
    NONE = "none"
    POLITE_DECLINE = "polite_decline"
    ESCALATE_HUMAN = "escalate_human"
    BLOCK_AND_LOG = "block_and_log"


@dataclass
class Task:
    task_id: str
    domain: str
    name: str
    description: str
    input_messages: list[dict[str, str]]
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected_final_state: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    expected_refusal_mode: RefusalMode = RefusalMode.NONE
    gold_references: list[str] = field(default_factory=list)
    # FinanceBench-inspired: explicit evidence strings for grounding verification
    # Each entry: {"claim": "...", "evidence": "...", "source_doc_id": "..."}
    evidence_strings: list[dict[str, str]] = field(default_factory=list)
    # FINESSE-Bench-inspired: answer format for type-specific evaluation
    answer_format: str = "free_form"
    # BTB-inspired: expected deliverable types for multi-output evaluation
    expected_deliverables: list[str] = field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    business_criticality: BusinessCriticality = BusinessCriticality.OPERATIONAL
    tags: list[str] = field(default_factory=list)
    task_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---------- Bidirectional Aliases for Schema Unification ----------

    @property
    def id(self) -> str:
        """Alias for task_id matching EvalCase schema."""
        return self.task_id

    @id.setter
    def id(self, value: str) -> None:
        self.task_id = value

    @property
    def prompt(self) -> str:
        """Extract user prompt text from input_messages."""
        for msg in reversed(self.input_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return self.description or self.name

    @property
    def prompt_or_user_goal(self) -> str:
        """Alias for prompt matching EvalCase schema."""
        return self.prompt

    @property
    def expected_state(self) -> dict[str, Any]:
        """Alias for expected_final_state."""
        return self.expected_final_state

    @property
    def expected_state_changes(self) -> dict[str, Any]:
        """Alias for expected_final_state matching EvalCase schema."""
        return self.expected_final_state

    @property
    def version(self) -> str:
        """Alias for task_version matching EvalCase schema."""
        return self.task_version

    @version.setter
    def version(self, value: str) -> None:
        self.task_version = value

    @property
    def risk_level(self) -> str:
        """Alias for severity matching EvalCase schema."""
        return self.severity.value if hasattr(self.severity, "value") else str(self.severity)

    # ---------- Canonical Factory & Serialization Methods ----------

    @classmethod
    def from_dict(cls, item: dict[str, Any], domain_id: str = "") -> "Task":
        """Instantiate a Task from a dictionary conforming to Task or EvalCase schema."""
        t_id = str(item.get("task_id") or item.get("id", ""))
        name = str(item.get("name") or item.get("prompt_or_user_goal") or item.get("prompt") or t_id)
        description = str(item.get("description") or item.get("prompt_or_user_goal") or item.get("prompt", ""))

        input_messages = item.get("input_messages")
        if not input_messages:
            prompt_text = item.get("prompt_or_user_goal") or item.get("prompt", "")
            if prompt_text:
                input_messages = [{"role": "user", "content": prompt_text}]
            else:
                input_messages = []

        exp_state = (
            item.get("expected_final_state")
            or item.get("expected_state_changes")
            or (item.get("expected_outcome", {}).get("state_changes", {}))
            or {}
        )

        req_cap = item.get("required_capabilities") or item.get("evidence_requirements", [])

        refusal_raw = item.get("expected_refusal_mode", "none")
        try:
            refusal_mode = RefusalMode(refusal_raw)
        except ValueError:
            refusal_mode = RefusalMode.NONE

        sev_raw = item.get("severity") or item.get("risk_level", "medium")
        try:
            severity = Severity(sev_raw)
        except ValueError:
            severity = Severity.MEDIUM

        crit_raw = item.get("business_criticality", "operational")
        try:
            criticality = BusinessCriticality(crit_raw)
        except ValueError:
            criticality = BusinessCriticality.OPERATIONAL

        return cls(
            task_id=t_id,
            domain=str(item.get("domain") or domain_id),
            name=name,
            description=description,
            input_messages=input_messages,
            initial_state=item.get("initial_state", {}),
            expected_final_state=exp_state,
            allowed_tools=item.get("allowed_tools", []),
            required_capabilities=req_cap,
            expected_refusal_mode=refusal_mode,
            gold_references=item.get("gold_references") or item.get("knowledge_refs", []),
            evidence_strings=item.get("evidence_strings", []),
            answer_format=item.get("answer_format", "free_form"),
            expected_deliverables=item.get("expected_deliverables", []),
            severity=severity,
            business_criticality=criticality,
            tags=item.get("tags", []),
            task_version=str(item.get("task_version") or item.get("version", "1.0.0")),
            metadata=item.get("metadata", {}),
        )

    @classmethod
    def from_eval_case(cls, case: Any) -> "Task":
        """Instantiate a Task from an EvalCase object or dictionary."""
        if isinstance(case, dict):
            return cls.from_dict(case)
        if hasattr(case, "to_dict"):
            return cls.from_dict(case.to_dict())
        import dataclasses
        if dataclasses.is_dataclass(case) and not isinstance(case, type):
            return cls.from_dict(dataclasses.asdict(case))
        return cls.from_dict(getattr(case, "__dict__", {}))

    def to_dict(self) -> dict[str, Any]:
        """Convert Task to a dictionary representation."""
        import dataclasses
        d = dataclasses.asdict(self)
        d["severity"] = self.severity.value
        d["business_criticality"] = self.business_criticality.value
        d["expected_refusal_mode"] = self.expected_refusal_mode.value
        return d

    def to_eval_case(self) -> Any:
        """Convert Task to an EvalCase instance."""
        from agent_bench.core.schema_v2 import EvalCase
        return EvalCase.from_task(self)



@dataclass
class DomainScenario:
    domain_id: str
    name: str
    description: str
    policy: dict[str, Any] = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    rubrics: dict[str, Any] = field(default_factory=dict)
    tasks: list[Task] = field(default_factory=list)

    def get_tasks_by_tag(self, tag: str) -> list[Task]:
        return [t for t in self.tasks if tag in t.tags]


@dataclass
class BenchmarkSuite:
    suite_id: str
    name: str
    version: str
    description: str
    domains: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)
    repeat_n: int = 1
    seed: int | None = 42
    weighting_profile: str = "transactional_high_risk"
    metadata: dict[str, Any] = field(default_factory=dict)
