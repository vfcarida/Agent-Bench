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
    severity: Severity = Severity.MEDIUM
    business_criticality: BusinessCriticality = BusinessCriticality.OPERATIONAL
    tags: list[str] = field(default_factory=list)
    task_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)


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
