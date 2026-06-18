"""Run artifacts and trace events."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class TraceEventType(str, Enum):
    PROMPT_SENT = "prompt_sent"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_RESULT = "retrieval_result"
    JUDGE_DECISION = "judge_decision"
    METRIC_COMPUTED = "metric_computed"
    ERROR = "error"
    SYSTEM_EVENT = "system_event"
    THINKING_BLOCK = "thinking_block"
    ADAPTER_SWAP = "adapter_swap"


@dataclass
class TraceEvent:
    event_type: TraceEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None


@dataclass
class RunArtifact:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    suite_id: str = ""
    system_id: str = ""
    model_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    config_hash: str = ""
    benchmark_version: str = ""
    seed: int | None = None
    tasks_total: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0
    traces: list[TraceEvent] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None

    def finalize(self) -> None:
        self.finished_at = datetime.now(timezone.utc)
