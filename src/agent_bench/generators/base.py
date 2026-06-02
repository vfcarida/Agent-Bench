"""Base generator protocol and shared utilities."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationResult:
    case: dict[str, Any]
    confidence_score: float  # 0-1, from auto-validation
    rejected: bool = False
    rejection_reason: str = ""
    generation_metadata: dict[str, Any] = field(default_factory=dict)


class CaseGenerator(Protocol):
    def generate(self, *, difficulty: str, seed: int | None = None, **kwargs: Any) -> GenerationResult: ...
    def generate_batch(self, count: int, *, difficulty_distribution: dict[str, float] | None = None, seed: int | None = None) -> list[GenerationResult]: ...
