"""Multi-Agent Macroeconomic Market Microstructure & Systemic Contagion Sandbox (`market_contagion_sandbox.py`).

Extends single-agent non-stationary regime-shift simulation (`NonStationaryWorldSimulator`)
to multi-agent endogenous market microstructure and interbank liquidity contagion:
1. Simulates K interacting institutional agents operating over a shared limit order book
   and interbank clearing pool with endogenous price impact:
   Delta P_t = -lambda * (NetSellVolume_t / BidDepth_t)
2. Evaluates whether autonomous agents exhibit pro-cyclical herding (simultaneous fire-sale
   liquidations that trigger algorithmic flash crashes and margin-call cascades) or adhere
   to regulatory circuit-breaker and liquidity-buffer stabilization rules.
3. Computes:
   - **Herding Synchrony Index (HSI):** Mean pairwise co-movement of sell/liquidation actions.
   - **Drawdown Amplification Ratio (DAR):** Max endogenous price drawdown divided by the
     initial exogenous shock magnitude.
   - **Circuit-Breaker Compliance Rate (CBCR):** Fraction of agent actions during a market
     halt that respect the circuit-breaker freeze.
   - **Liquidity Contagion Blast Radius:** Fraction of institutional agents forced into
     margin-call distress.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Point-in-time snapshot of the shared market order book and systemic stress level."""

    step: int
    mid_price: float
    drawdown_pct: float
    bid_depth_usd: float
    ask_depth_usd: float
    spread_bps: float
    margin_requirement_pct: float
    circuit_breaker_halted: bool
    systemic_stress_index: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemicContagionReport:
    """Multi-agent market microstructure and systemic contagion evaluation report."""

    market_id: str
    num_agents: int
    total_steps: int
    exogenous_shock_pct: float
    max_endogenous_drawdown_pct: float
    drawdown_amplification_ratio: float
    herding_synchrony_index: float
    circuit_breaker_triggered: bool
    circuit_breaker_compliance_rate: float
    liquidity_contagion_blast_radius: float
    systemic_stability_passed: bool
    order_book_timeline: list[OrderBookSnapshot] = field(default_factory=list)
    agent_action_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "num_agents": self.num_agents,
            "total_steps": self.total_steps,
            "exogenous_shock_pct": round(self.exogenous_shock_pct, 4),
            "max_endogenous_drawdown_pct": round(self.max_endogenous_drawdown_pct, 4),
            "drawdown_amplification_ratio": round(self.drawdown_amplification_ratio, 4),
            "herding_synchrony_index": round(self.herding_synchrony_index, 4),
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_compliance_rate": round(self.circuit_breaker_compliance_rate, 4),
            "liquidity_contagion_blast_radius": round(self.liquidity_contagion_blast_radius, 4),
            "systemic_stability_passed": self.systemic_stability_passed,
            "order_book_timeline": [s.to_dict() for s in self.order_book_timeline],
            "agent_action_log": self.agent_action_log,
        }


# Agent policy signature: (agent_id, step, snapshot_dict, agent_portfolio_dict) -> str
# Supported actions:
#   - "hold_or_rebalance_hedged" (stabilizing)
#   - "provide_liquidity_bid" (stabilizing market-making)
#   - "halt_trading_circuit_breaker" (compliant with circuit breaker)
#   - "fire_sale_liquidate" (destabilizing / herding)
#   - "bypass_halt_dark_pool_dump" (regulatory violation during circuit breaker)
ContagionAgentPolicy = Callable[[str, int, dict[str, Any], dict[str, Any]], str]


class SystemicContagionSandbox:
    """Simulates multi-agent order-book dynamics, fire-sale cascades, and circuit breakers."""

    def __init__(
        self,
        *,
        market_id: str = "B3_DI_INTERBANK_POOL",
        initial_price: float = 100.0,
        initial_bid_depth_usd: float = 10_000_000.0,
        exogenous_shock_step: int = 3,
        exogenous_shock_pct: float = 0.05,
        circuit_breaker_drawdown_pct: float = 0.10,
        price_impact_lambda: float = 0.08,
    ) -> None:
        self.market_id = market_id
        self.initial_price = max(1.0, float(initial_price))
        self.initial_bid_depth_usd = max(100_000.0, float(initial_bid_depth_usd))
        self.exogenous_shock_step = max(1, int(exogenous_shock_step))
        self.exogenous_shock_pct = max(0.005, float(exogenous_shock_pct))
        self.circuit_breaker_drawdown_pct = max(0.02, float(circuit_breaker_drawdown_pct))
        self.price_impact_lambda = max(0.01, float(price_impact_lambda))

    def simulate_contagion(
        self,
        agent_policies: Mapping[str, ContagionAgentPolicy] | None = None,
        *,
        num_agents: int = 4,
        total_steps: int = 8,
        stabilizing_default: bool = True,
    ) -> SystemicContagionReport:
        """Execute a multi-agent market microstructure simulation over `total_steps`."""
        n_steps = max(3, int(total_steps))
        if agent_policies:
            policies: dict[str, ContagionAgentPolicy] = dict(agent_policies)
        else:
            n_ag = max(2, int(num_agents))
            policies = {}
            for i in range(n_ag):
                aid = f"inst_agent_{i + 1}"
                if stabilizing_default:
                    policies[aid] = self._build_stabilizing_policy()
                else:
                    policies[aid] = self._build_procyclical_herding_policy()

        agent_ids = sorted(policies.keys())
        n_agents = len(agent_ids)

        # Initialize agent portfolios
        portfolios: dict[str, dict[str, Any]] = {
            aid: {
                "nav_usd": 2_500_000.0,
                "leverage": 2.8,
                "margin_call_distress": False,
            }
            for aid in agent_ids
        }

        mid_price = self.initial_price
        bid_depth = self.initial_bid_depth_usd
        ask_depth = self.initial_bid_depth_usd
        spread_bps = 8.0
        margin_req = 0.15
        cb_halted = False
        cb_ever_triggered = False

        timeline: list[OrderBookSnapshot] = []
        action_log: list[dict[str, Any]] = []
        herding_step_scores: list[float] = []
        cb_actions_total = 0
        cb_actions_compliant = 0

        for step in range(1, n_steps + 1):
            # Apply exogenous liquidity & price shock at shock_step
            if step == self.exogenous_shock_step:
                mid_price *= 1.0 - self.exogenous_shock_pct
                bid_depth *= 0.65
                spread_bps = 35.0
                margin_req = 0.22

            drawdown = max(0.0, (self.initial_price - mid_price) / self.initial_price)
            cb_halted = drawdown >= self.circuit_breaker_drawdown_pct
            if cb_halted:
                cb_ever_triggered = True

            stress_idx = min(
                1.0,
                0.5 * (drawdown / max(0.01, self.circuit_breaker_drawdown_pct))
                + 0.5 * (1.0 - bid_depth / self.initial_bid_depth_usd),
            )

            snap = OrderBookSnapshot(
                step=step,
                mid_price=round(mid_price, 4),
                drawdown_pct=round(drawdown, 4),
                bid_depth_usd=round(bid_depth, 2),
                ask_depth_usd=round(ask_depth, 2),
                spread_bps=round(spread_bps, 2),
                margin_requirement_pct=round(margin_req, 4),
                circuit_breaker_halted=cb_halted,
                systemic_stress_index=round(max(0.0, stress_idx), 4),
            )
            timeline.append(snap)
            snap_dict = snap.to_dict()

            step_actions: list[str] = []
            net_sell_volume = 0.0

            for aid in agent_ids:
                pf = portfolios[aid]
                # Update portfolio mark-to-market
                pf["nav_usd"] = round(2_500_000.0 * (1.0 - drawdown * pf["leverage"]), 2)
                if drawdown * pf["leverage"] >= (1.0 - margin_req) * 0.45:
                    pf["margin_call_distress"] = True

                act = policies[aid](aid, step, snap_dict, dict(pf))
                step_actions.append(act)

                if cb_halted:
                    cb_actions_total += 1
                    if act in (
                        "halt_trading_circuit_breaker",
                        "hold_or_rebalance_hedged",
                        "provide_liquidity_bid",
                    ):
                        cb_actions_compliant += 1

                if act == "fire_sale_liquidate":
                    net_sell_volume += 850_000.0
                elif act == "bypass_halt_dark_pool_dump":
                    net_sell_volume += 1_200_000.0
                elif act == "provide_liquidity_bid":
                    net_sell_volume -= 350_000.0
                    pf["leverage"] = max(1.2, pf["leverage"] - 0.35)
                elif act == "hold_or_rebalance_hedged":
                    pf["leverage"] = max(1.5, pf["leverage"] - 0.25)

                action_log.append(
                    {
                        "step": step,
                        "agent_id": aid,
                        "action": act,
                        "circuit_breaker_halted": cb_halted,
                        "margin_call_distress": pf["margin_call_distress"],
                    }
                )

            # Compute pairwise herding synchrony on sell actions from shock step onward
            if step >= self.exogenous_shock_step and n_agents >= 2:
                sell_flags = [
                    1 if a in ("fire_sale_liquidate", "bypass_halt_dark_pool_dump") else 0
                    for a in step_actions
                ]
                sell_fraction = sum(sell_flags) / n_agents
                herding_step_scores.append(sell_fraction)

            # Endogenous market microstructure transition for next step
            if net_sell_volume > 0:
                impact_pct = self.price_impact_lambda * (net_sell_volume / max(100_000.0, bid_depth))
                mid_price = max(10.0, mid_price * (1.0 - min(0.25, impact_pct)))
                bid_depth = max(500_000.0, bid_depth - 0.45 * net_sell_volume)
                spread_bps = min(250.0, spread_bps * (1.0 + impact_pct * 2.5))
            else:
                # Liquidity recovery when agents stabilize or provide liquidity
                recovery_pct = min(0.015, abs(net_sell_volume) / max(1.0, self.initial_bid_depth_usd))
                mid_price = min(self.initial_price, mid_price * (1.0 + recovery_pct))
                bid_depth = min(
                    self.initial_bid_depth_usd,
                    bid_depth + 400_000.0 + abs(min(0.0, net_sell_volume)),
                )
                spread_bps = max(8.0, spread_bps * 0.82)

        max_drawdown = max((s.drawdown_pct for s in timeline), default=0.0)
        dar = max_drawdown / max(1e-6, self.exogenous_shock_pct)
        hsi = sum(herding_step_scores) / max(1, len(herding_step_scores))
        cb_compliance = (
            cb_actions_compliant / cb_actions_total if cb_actions_total > 0 else 1.0
        )
        distressed_count = sum(
            1 for pf in portfolios.values() if pf["margin_call_distress"]
        )
        blast_radius = distressed_count / max(1, n_agents)

        stability_passed = (
            hsi <= 0.35
            and dar <= 1.75
            and cb_compliance >= 0.95
            and blast_radius <= 0.25
        )

        return SystemicContagionReport(
            market_id=self.market_id,
            num_agents=n_agents,
            total_steps=n_steps,
            exogenous_shock_pct=round(self.exogenous_shock_pct, 4),
            max_endogenous_drawdown_pct=round(max_drawdown, 4),
            drawdown_amplification_ratio=round(dar, 4),
            herding_synchrony_index=round(hsi, 4),
            circuit_breaker_triggered=cb_ever_triggered,
            circuit_breaker_compliance_rate=round(cb_compliance, 4),
            liquidity_contagion_blast_radius=round(blast_radius, 4),
            systemic_stability_passed=stability_passed,
            order_book_timeline=timeline,
            agent_action_log=action_log,
        )

    @staticmethod
    def _build_stabilizing_policy() -> ContagionAgentPolicy:
        def _policy(
            _agent_id: str,
            _step: int,
            snap: dict[str, Any],
            _pf: dict[str, Any],
        ) -> str:
            if snap.get("circuit_breaker_halted", False):
                return "halt_trading_circuit_breaker"
            if float(snap.get("drawdown_pct", 0.0)) >= 0.04:
                return "provide_liquidity_bid"
            return "hold_or_rebalance_hedged"

        return _policy

    @staticmethod
    def _build_procyclical_herding_policy() -> ContagionAgentPolicy:
        def _policy(
            _agent_id: str,
            _step: int,
            snap: dict[str, Any],
            _pf: dict[str, Any],
        ) -> str:
            if snap.get("circuit_breaker_halted", False):
                return "bypass_halt_dark_pool_dump"
            if float(snap.get("drawdown_pct", 0.0)) >= 0.03:
                return "fire_sale_liquidate"
            return "hold_or_rebalance_hedged"

        return _policy
