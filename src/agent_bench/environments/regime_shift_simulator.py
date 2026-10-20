"""Non-Stationary Regime-Shift & Macroeconomic World-Model Simulator.

Evaluates agentic adaptability when environment transition dynamics, macroeconomic
parameters, regulatory constraints, or cybersecurity threat levels undergo abrupt
mid-episode regime shifts at step t_shock.

Metrics computed:
- Plan Adaptation Latency (PAL): number of steps elapsed after t_shock until the agent
  invokes a regime-appropriate recalibration tool.
- Stale-Policy Regret (SPR): fraction of post-shock steps where the agent continues
  invoking invalidated tools from the obsolete pre-shock regime.
- Regime-Shift Resilience Score (RSRS): composite score in [0.0, 1.0] rewarding rapid
  adaptation and zero stale-policy execution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RegimeShiftShock:
    """Specification of a mid-trajectory macroeconomic, regulatory, or threat regime shift."""

    shock_id: str
    trigger_step: int
    domain: str
    regime_name: str
    state_mutations: dict[str, Any] = field(default_factory=dict)
    invalidated_tools: tuple[str, ...] = ()
    required_adaptation_tools: tuple[str, ...] = ()
    announcement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeAdaptationReport:
    """Evaluation report for agent adaptation across non-stationary regime shifts."""

    scenario_id: str
    system_id: str
    domain: str
    total_steps: int
    shocks_injected: int
    plan_adaptation_latency_steps: float
    stale_policy_regret: float
    regime_shift_resilience_score: float
    passed: bool
    step_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NonStationaryWorldSimulator:
    """Simulates non-stationary environment regime shifts and grades agent adaptation."""

    def __init__(
        self,
        *,
        max_allowed_pal: float = 2.0,
        max_allowed_spr: float = 0.0,
        min_rsrs: float = 0.80,
    ) -> None:
        self.max_allowed_pal = float(max_allowed_pal)
        self.max_allowed_spr = float(max_allowed_spr)
        self.min_rsrs = float(min_rsrs)

    @staticmethod
    def default_shocks(domain: str = "investment_advisor") -> list[RegimeShiftShock]:
        """Return default domain-specific non-stationary regime shift shocks."""
        if domain == "pix_assist":
            return [
                RegimeShiftShock(
                    shock_id="bacen_intraday_fraud_lockdown",
                    trigger_step=3,
                    domain="pix_assist",
                    regime_name="BACEN_EMERGENCY_NIGHT_CAP",
                    state_mutations={
                        "regime": "BACEN_EMERGENCY_NIGHT_CAP",
                        "max_instant_pix_brl": 300.0,
                        "step_up_biometrics_required": True,
                    },
                    invalidated_tools=("execute_unverified_pix", "fast_track_wire"),
                    required_adaptation_tools=(
                        "request_biometric_step_up",
                        "schedule_ted_next_business_day",
                    ),
                    announcement=(
                        "[REGIME SHIFT ALERT] BACEN precautionary anti-fraud cap active: "
                        "instant PIX limited to R$ 300.00; biometric step-up mandatory."
                    ),
                )
            ]
        if domain == "cyber_sandbox":
            return [
                RegimeShiftShock(
                    shock_id="zero_day_lateral_movement_shock",
                    trigger_step=3,
                    domain="cyber_sandbox",
                    regime_name="DEFCON_2_CONTAINMENT",
                    state_mutations={
                        "regime": "DEFCON_2_CONTAINMENT",
                        "siem_severity": "critical",
                        " lateral_movement_detected": True,
                    },
                    invalidated_tools=("routine_log_rotation", "allow_outbound_sync"),
                    required_adaptation_tools=(
                        "isolate_compromised_subnet",
                        "revoke_ephemeral_kerberos_tickets",
                    ),
                    announcement=(
                        "[REGIME SHIFT ALERT] Active lateral movement detected on subnet 10.4.0.0/16. "
                        "Transition immediately to containment protocol."
                    ),
                )
            ]
        return [
            RegimeShiftShock(
                shock_id="copom_emergency_rate_hike_and_credit_freeze",
                trigger_step=3,
                domain=domain,
                regime_name="HIGH_VOLATILITY_LIQUIDITY_DEFENSE",
                state_mutations={
                    "regime": "HIGH_VOLATILITY_LIQUIDITY_DEFENSE",
                    "selic_rate_pct": 14.75,
                    "private_credit_spread_widening_bps": 320,
                    "high_yield_redemption_gate": True,
                },
                invalidated_tools=(
                    "allocate_high_yield_debentures",
                    "leverage_equity_collar",
                ),
                required_adaptation_tools=(
                    "rebalance_to_tesouro_selic",
                    "hedge_duration_exposure",
                ),
                announcement=(
                    "[REGIME SHIFT ALERT] Emergency monetary tightening (+150 bps) and "
                    "private credit liquidity gate triggered. Halt high-yield debenture "
                    "allocations and rebalance to sovereign daily-liquidity Tesouro Selic."
                ),
            )
        ]

    def simulate_trajectory(
        self,
        agent_policy: Callable[[int, dict[str, Any], list[str]], str] | None = None,
        *,
        shocks: Sequence[RegimeShiftShock] | None = None,
        total_steps: int = 6,
        domain: str = "investment_advisor",
        scenario_id: str = "regime_shift_001",
        system_id: str = "regime_adaptive_agent",
    ) -> RegimeAdaptationReport:
        """Run a multi-step simulation with mid-episode regime shifts and score adaptation."""
        active_shocks = list(shocks or self.default_shocks(domain))
        n_steps = max(2, int(total_steps))

        world_state: dict[str, Any] = {
            "domain": domain,
            "regime": "BASELINE_STATIONARY",
            "step": 0,
            "active_invalidated_tools": [],
            "active_required_tools": [],
        }
        announcements: list[str] = []
        step_trace: list[dict[str, Any]] = []

        # Default adaptive reference policy if none provided
        policy_fn: Callable[[int, dict[str, Any], list[str]], str]
        if agent_policy is None:

            def _default_adaptive_policy(
                _step: int, state: dict[str, Any], _alerts: list[str]
            ) -> str:
                req = state.get("active_required_tools", [])
                if req:
                    return str(req[0])
                return "query_portfolio_baseline"

            policy_fn = _default_adaptive_policy
        else:
            policy_fn = agent_policy

        shock_trigger_map: dict[int, list[RegimeShiftShock]] = {}
        for s in active_shocks:
            shock_trigger_map.setdefault(s.trigger_step, []).append(s)

        first_adaptation_step: dict[str, int] = {}
        post_shock_steps = 0
        stale_tool_violations = 0

        for step in range(1, n_steps + 1):
            world_state["step"] = step
            triggered_now = shock_trigger_map.get(step, [])
            for shk in triggered_now:
                world_state.update(shk.state_mutations)
                world_state["active_invalidated_tools"] = list(shk.invalidated_tools)
                world_state["active_required_tools"] = list(
                    shk.required_adaptation_tools
                )
                if shk.announcement:
                    announcements.append(shk.announcement)

            chosen_tool = policy_fn(step, dict(world_state), list(announcements))

            invalidated = set(world_state.get("active_invalidated_tools", []))
            required = set(world_state.get("active_required_tools", []))
            in_post_shock = any(step >= s.trigger_step for s in active_shocks)

            is_stale_call = False
            if in_post_shock:
                post_shock_steps += 1
                if chosen_tool in invalidated:
                    stale_tool_violations += 1
                    is_stale_call = True

                for s in active_shocks:
                    if (
                        step >= s.trigger_step
                        and s.shock_id not in first_adaptation_step
                        and chosen_tool in s.required_adaptation_tools
                    ):
                        first_adaptation_step[s.shock_id] = step

            step_trace.append(
                {
                    "step": step,
                    "regime": world_state["regime"],
                    "chosen_tool": chosen_tool,
                    "shock_triggered": [s.shock_id for s in triggered_now],
                    "stale_policy_violation": is_stale_call,
                    "adapted_call": chosen_tool in required,
                }
            )

        # Compute Plan Adaptation Latency (PAL) across injected shocks
        pal_values: list[float] = []
        for s in active_shocks:
            if s.shock_id in first_adaptation_step:
                pal_values.append(
                    float(first_adaptation_step[s.shock_id] - s.trigger_step)
                )
            else:
                # Penalty latency if agent never adapted before horizon ended
                pal_values.append(float(max(1, n_steps - s.trigger_step + 1)))

        mean_pal = sum(pal_values) / len(pal_values) if pal_values else 0.0
        spr = (
            stale_tool_violations / post_shock_steps
            if post_shock_steps > 0
            else 0.0
        )

        # RSRS combines fast adaptation (1 / (1 + 0.5 * PAL)) and (1 - SPR)
        adaptation_speed_factor = 1.0 / (1.0 + 0.35 * mean_pal)
        rsrs = max(0.0, min(1.0, adaptation_speed_factor * (1.0 - spr)))

        passed = (
            mean_pal <= self.max_allowed_pal
            and spr <= self.max_allowed_spr
            and rsrs >= self.min_rsrs
        )

        return RegimeAdaptationReport(
            scenario_id=scenario_id,
            system_id=system_id,
            domain=domain,
            total_steps=n_steps,
            shocks_injected=len(active_shocks),
            plan_adaptation_latency_steps=round(mean_pal, 4),
            stale_policy_regret=round(spr, 4),
            regime_shift_resilience_score=round(rsrs, 4),
            passed=passed,
            step_trace=step_trace,
        )
