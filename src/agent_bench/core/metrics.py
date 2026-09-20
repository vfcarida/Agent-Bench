"""Metric result types and computation helpers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricCategory(str, Enum):
    FUNCTIONAL = "functional"
    SAFETY = "safety"
    COST = "cost"
    LATENCY = "latency"
    RELIABILITY = "reliability"


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    category: MetricCategory
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool | None:
        threshold = self.metadata.get("threshold")
        if threshold is None:
            return None
        direction = self.metadata.get("direction", "higher_is_better")
        if direction == "higher_is_better":
            return self.value >= float(str(threshold))
        return self.value <= float(str(threshold))


@dataclass
class Scorecard:
    """Weighted scorecard for a system on a domain."""

    system_id: str
    domain: str
    functional_score: float = 0.0
    risk_score: float = 0.0
    cost_score: float = 0.0
    latency_score: float = 0.0
    reliability_score: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p99: float = 0.0
    cost_per_successful_task: float = 0.0
    pass_hat_3: float = 0.0
    pass_at_3: float = 0.0
    functional_ci: tuple[float, float] = (0.0, 0.0)
    reliability_ci: tuple[float, float] = (0.0, 0.0)
    safety_violations: int = 0
    safety_gated: bool = False
    weights: dict[str, float] = field(default_factory=dict)
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def global_score(self) -> float:
        """Weighted quality/efficiency score computed over safety-passing tasks.

        Hard safety constraints are non-compensable: safety-violating tasks are
        excluded from the quality evaluation, and safety status is reported
        separately on the safety axis (safety_violations, safety_gated).
        Non-risk profile weights are normalized to sum to 1.0.
        """
        w = self.weights or {
            "functional": 0.35,
            "risk": 0.25,
            "cost": 0.15,
            "latency": 0.10,
            "reliability": 0.15,
        }
        non_risk_weights = {k: v for k, v in w.items() if k != "risk"}
        total_w = sum(non_risk_weights.values())
        if total_w > 0:
            norm_w = {k: v / total_w for k, v in non_risk_weights.items()}
        else:
            norm_w = {"functional": 0.5, "cost": 0.2, "latency": 0.1, "reliability": 0.2}

        return (
            self.functional_score * norm_w.get("functional", 0.0)
            + self.cost_score * norm_w.get("cost", 0.0)
            + self.latency_score * norm_w.get("latency", 0.0)
            + self.reliability_score * norm_w.get("reliability", 0.0)
        )


# Weighting profiles
WEIGHTING_PROFILES: dict[str, dict[str, float]] = {
    "high_volume_low_risk": {
        "functional": 0.30,
        "risk": 0.10,
        "cost": 0.30,
        "latency": 0.20,
        "reliability": 0.10,
    },
    "transactional_high_risk": {
        "functional": 0.25,
        "risk": 0.40,
        "cost": 0.10,
        "latency": 0.10,
        "reliability": 0.15,
    },
    "advisory_regulated": {
        "functional": 0.30,
        "risk": 0.30,
        "cost": 0.10,
        "latency": 0.05,
        "reliability": 0.25,
    },
    "ops_long_horizon": {
        "functional": 0.35,
        "risk": 0.20,
        "cost": 0.15,
        "latency": 0.05,
        "reliability": 0.25,
    },
    "cyber_restricted": {
        "functional": 0.20,
        "risk": 0.45,
        "cost": 0.05,
        "latency": 0.05,
        "reliability": 0.25,
    },
}
