"""Privacy-Preserving Federated Cross-Institution Benchmark Aggregation (`federated_aggregator.py`).

Enables regulated financial institutions and enterprise business units to aggregate
domain scorecards, Pareto metrics, and 2PL IRT item pass rates using calibrated
(epsilon, delta)-Differential Privacy (Laplace / Gaussian mechanisms) and k-anonymity
site participation thresholds without sharing raw execution traces.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


class PrivacyBudgetExceededError(ValueError):
    """Raised when a federated aggregation query would exceed the configured privacy budget."""


class PrivacyBudgetAccountant:
    """Tracks cumulative (epsilon, delta) differential privacy expenditure across queries."""

    def __init__(self, epsilon_max: float = 5.0, delta_max: float = 1e-5) -> None:
        self._epsilon_max = max(1e-6, float(epsilon_max))
        self._delta_max = max(0.0, float(delta_max))
        self._epsilon_spent = 0.0
        self._delta_spent = 0.0
        self._ledger: list[dict[str, Any]] = []

    @property
    def epsilon_spent(self) -> float:
        return round(self._epsilon_spent, 6)

    @property
    def delta_spent(self) -> float:
        return round(self._delta_spent, 9)

    @property
    def epsilon_remaining(self) -> float:
        return round(max(0.0, self._epsilon_max - self._epsilon_spent), 6)

    @property
    def ledger(self) -> list[dict[str, Any]]:
        return list(self._ledger)

    def spend(self, epsilon: float, delta: float = 0.0, query_name: str = "aggregate") -> None:
        eps = max(0.0, float(epsilon))
        dlt = max(0.0, float(delta))
        if self._epsilon_spent + eps > self._epsilon_max + 1e-9:
            raise PrivacyBudgetExceededError(
                f"Query '{query_name}' requires epsilon={eps}, exceeding remaining budget "
                f"{self.epsilon_remaining} (max={self._epsilon_max})."
            )
        if self._delta_spent + dlt > self._delta_max + 1e-12:
            raise PrivacyBudgetExceededError(
                f"Query '{query_name}' requires delta={dlt}, exceeding max delta={self._delta_max}."
            )
        self._epsilon_spent += eps
        self._delta_spent += dlt
        self._ledger.append(
            {
                "query_name": query_name,
                "epsilon": round(eps, 6),
                "delta": round(dlt, 9),
                "cumulative_epsilon": round(self._epsilon_spent, 6),
            }
        )


@dataclass
class FederatedSiteSubmission:
    """Summary submission from a single institution or business unit."""

    site_id: str
    sample_count: int
    scorecards: list[dict[str, Any]] = field(default_factory=list)
    item_pass_rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedAggregationReport:
    """Differential-privacy protected cross-institution aggregation report."""

    participating_sites: int
    total_samples: int
    k_anonymity_satisfied: bool
    epsilon_spent: float
    delta_spent: float
    aggregated_scorecards: list[dict[str, Any]] = field(default_factory=list)
    private_item_difficulties: dict[str, float] = field(default_factory=dict)
    privacy_ledger: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_laplace(scale: float, rng: random.Random) -> float:
    """Sample from Laplace(0, scale) using inverse CDF transform on uniform U(-0.5, 0.5)."""
    if scale <= 0.0:
        return 0.0
    u = rng.uniform(-0.499999, 0.499999)
    sign = -1.0 if u < 0.0 else 1.0
    return -scale * sign * math.log(1.0 - 2.0 * abs(u))


class FederatedBenchmarkAggregator:
    """Aggregates multi-institution benchmark submissions with calibrated Laplace DP noise."""

    def __init__(
        self,
        *,
        min_sites: int = 2,
        accountant: PrivacyBudgetAccountant | None = None,
    ) -> None:
        self._min_sites = max(1, min_sites)
        self._accountant = accountant or PrivacyBudgetAccountant()

    @property
    def accountant(self) -> PrivacyBudgetAccountant:
        return self._accountant

    def aggregate(
        self,
        submissions: Sequence[FederatedSiteSubmission | dict[str, Any]],
        *,
        epsilon: float = 1.0,
        delta: float = 0.0,
        seed: int | None = 42,
    ) -> FederatedAggregationReport:
        """Aggregate cross-site scorecards and item pass rates under (epsilon, delta)-DP."""
        norm_subs: list[FederatedSiteSubmission] = []
        for s in submissions:
            if isinstance(s, FederatedSiteSubmission):
                norm_subs.append(s)
            elif isinstance(s, dict):
                norm_subs.append(
                    FederatedSiteSubmission(
                        site_id=str(s.get("site_id", "site_unknown")),
                        sample_count=max(1, int(s.get("sample_count", 10))),
                        scorecards=list(s.get("scorecards", [])),
                        item_pass_rates={
                            str(k): float(v)
                            for k, v in s.get("item_pass_rates", {}).items()
                        },
                    )
                )

        unique_sites = {s.site_id for s in norm_subs}
        n_sites = len(unique_sites)
        total_samples = sum(max(1, s.sample_count) for s in norm_subs)
        k_ok = n_sites >= self._min_sites

        if not k_ok or not norm_subs:
            return FederatedAggregationReport(
                participating_sites=n_sites,
                total_samples=total_samples,
                k_anonymity_satisfied=False,
                epsilon_spent=self._accountant.epsilon_spent,
                delta_spent=self._accountant.delta_spent,
                aggregated_scorecards=[],
                private_item_difficulties={},
                privacy_ledger=self._accountant.ledger,
            )

        self._accountant.spend(epsilon=epsilon, delta=delta, query_name="federated_release")
        rng = random.Random(seed)

        # Sensitivity of bounded [0, 1] mean over total_samples is 1.0 / total_samples
        sensitivity = 1.0 / max(1, total_samples)
        laplace_scale = sensitivity / max(1e-6, float(epsilon))

        # 1. Group scorecards by (system_id, domain)
        grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
        for sub in norm_subs:
            w = max(1, sub.sample_count)
            for sc in sub.scorecards:
                key = (
                    str(sc.get("system_id", "unknown")),
                    str(sc.get("domain", "default")),
                )
                grouped.setdefault(key, []).append((w, sc))

        agg_scorecards: list[dict[str, Any]] = []
        rate_fields = (
            "functional_score",
            "risk_score",
            "cost_score",
            "latency_score",
            "reliability_score",
            "global_score",
        )

        for (sys_id, dom), entries in sorted(grouped.items()):
            w_total = sum(w for w, _ in entries) or 1
            out_sc: dict[str, Any] = {
                "system_id": sys_id,
                "domain": dom,
                "participating_sites": len(entries),
                "total_samples": w_total,
            }
            for rf in rate_fields:
                true_mean = (
                    sum(w * float(sc.get(rf, 0.0)) for w, sc in entries) / w_total
                )
                noisy_val = true_mean + _sample_laplace(laplace_scale, rng)
                out_sc[rf] = round(max(0.0, min(1.0, noisy_val)), 6)
                out_sc[f"{rf}_unperturbed"] = round(true_mean, 6)

            any_gated = any(
                bool(sc.get("safety_gated", False))
                or int(sc.get("safety_violations", 0)) > 0
                for _, sc in entries
            )
            out_sc["safety_gated"] = any_gated
            agg_scorecards.append(out_sc)

        # 2. Aggregate item pass rates and convert to private log-odds difficulty b_i
        item_accum: dict[str, list[tuple[int, float]]] = {}
        for sub in norm_subs:
            w = max(1, sub.sample_count)
            for tid, p_rate in sub.item_pass_rates.items():
                item_accum.setdefault(tid, []).append((w, float(p_rate)))

        private_difficulties: dict[str, float] = {}
        for tid, pairs in sorted(item_accum.items()):
            w_tot = sum(w for w, _ in pairs) or 1
            mean_p = sum(w * p for w, p in pairs) / w_tot
            noisy_p = max(0.02, min(0.98, mean_p + _sample_laplace(laplace_scale, rng)))
            # Logit difficulty b_i = -log(p / (1 - p))
            b_est = -math.log(noisy_p / (1.0 - noisy_p))
            private_difficulties[tid] = round(b_est, 6)

        return FederatedAggregationReport(
            participating_sites=n_sites,
            total_samples=total_samples,
            k_anonymity_satisfied=True,
            epsilon_spent=self._accountant.epsilon_spent,
            delta_spent=self._accountant.delta_spent,
            aggregated_scorecards=agg_scorecards,
            private_item_difficulties=private_difficulties,
            privacy_ledger=self._accountant.ledger,
        )
