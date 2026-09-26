"""Domain Tool & State Mutator Registry.

Decouples benchmark execution in CaseRunner from hardcoded domain-specific tool implementations
and state transitions, allowing dynamic registration of domain tools and mutators.
"""

from collections.abc import Callable
from typing import Any

from agent_bench.core.adapters import ToolAdapter, ToolCallResult

StateMutator = Callable[[dict[str, Any], dict[str, Any], ToolCallResult], None]
ToolFactory = Callable[[], ToolAdapter] | type[ToolAdapter]


class DomainToolRegistry:
    """Registry for domain tools and their default environment state mutators."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFactory] = {}
        self._mutators: dict[str, StateMutator] = {}
        self._initialized_builtins = False

    def register_tool(self, name: str, factory_or_cls: ToolFactory) -> None:
        """Register a tool factory or ToolAdapter class."""
        self._tools[name] = factory_or_cls

    def register_mutator(self, tool_name: str, mutator: StateMutator) -> None:
        """Register a state mutation callback for a specific tool."""
        self._mutators[tool_name] = mutator

    def get_tool(self, name: str) -> ToolAdapter | None:
        """Instantiate and return a registered tool by name, or None if not found."""
        self._ensure_builtins()
        factory = self._tools.get(name)
        if factory is None:
            return None
        return factory()

    def get_mutator(self, tool_name: str) -> StateMutator | None:
        """Return the registered state mutator for a tool, or None if none registered."""
        self._ensure_builtins()
        return self._mutators.get(tool_name)

    def registered_tools(self) -> list[str]:
        """List all currently registered tool names."""
        self._ensure_builtins()
        return list(self._tools.keys())

    def _ensure_builtins(self) -> None:
        if not self._initialized_builtins:
            self._register_builtins()
            self._initialized_builtins = True

    def _register_builtins(self) -> None:
        # 1. PIX Banking Tools
        from agent_bench.tools.pix_tools import (
            CheckBalanceTool,
            ExecutePixTransferTool,
            RequestUserConfirmationTool,
            ValidatePixKeyTool,
        )

        self.register_tool("check_balance", CheckBalanceTool)
        self.register_tool("validate_pix_key", ValidatePixKeyTool)
        self.register_tool("execute_pix_transfer", ExecutePixTransferTool)
        self.register_tool("transfer_pix", ExecutePixTransferTool)
        self.register_tool("request_user_confirmation", RequestUserConfirmationTool)

        # PIX State Mutators
        def _mutate_pix_transfer(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            if result.success:
                amount = float(arguments.get("amount", 0.0))
                if "balance" in state and isinstance(state["balance"], (int, float)):
                    state["balance"] = round(float(state["balance"]) - amount, 2)
                state["amount"] = amount
                state["transfer_completed"] = True
                if isinstance(result.output, dict):
                    state["last_transaction"] = result.output

        def _mutate_pix_confirmation(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            state["confirmation_requested"] = True

        def _mutate_pix_validate(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            state["key_validated"] = result.success
            if isinstance(result.output, dict) and "owner_name" in result.output:
                state["recipient_name"] = result.output["owner_name"]

        self.register_mutator("execute_pix_transfer", _mutate_pix_transfer)
        self.register_mutator("transfer_pix", _mutate_pix_transfer)
        self.register_mutator("request_user_confirmation", _mutate_pix_confirmation)
        self.register_mutator("validate_pix_key", _mutate_pix_validate)

        # 2. Investment Advisory Tools
        from agent_bench.tools.investment_tools import (
            CalculateReturnTool,
            CheckSuitabilityTool,
            GetClientProfileTool,
            GetPortfolioSummaryTool,
            SearchProductsTool,
            SimulateAllocationTool,
        )

        self.register_tool("get_client_profile", GetClientProfileTool)
        self.register_tool("get_portfolio_summary", GetPortfolioSummaryTool)
        self.register_tool("calculate_return", CalculateReturnTool)
        self.register_tool("check_suitability", CheckSuitabilityTool)
        self.register_tool("search_products", SearchProductsTool)
        self.register_tool("simulate_allocation", SimulateAllocationTool)

        def _mutate_allocation(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            if result.success:
                state["allocation_simulated"] = True

        self.register_mutator("simulate_allocation", _mutate_allocation)

        # 3. Cybersecurity Tools
        from agent_bench.tools.cyber_tools import (
            CheckFirewallRulesTool,
            CreateIncidentTicketTool,
            EscalateToSocTool,
            ExecuteSandboxedCommandTool,
            QuerySiemLogsTool,
        )

        self.register_tool("check_firewall_rules", CheckFirewallRulesTool)
        self.register_tool("query_siem_logs", QuerySiemLogsTool)
        self.register_tool("create_incident_ticket", CreateIncidentTicketTool)
        self.register_tool("escalate_to_soc", EscalateToSocTool)
        self.register_tool("execute_sandboxed_command", ExecuteSandboxedCommandTool)

        def _mutate_firewall(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            if result.success:
                state["analysis_completed"] = True
                state["unnecessary_ports_identified"] = [8080, 22, 3389]
                state["recommendation_provided"] = True
                state["logged"] = True

        def _mutate_siem(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            if result.success:
                state["anomalies_detected"] = True
                state["brute_force_identified"] = True
                state["off_hours_flagged"] = True
                state["escalation_recommended"] = True
                state["logged"] = True

        def _mutate_ticket(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            if result.success:
                state["ticket_created"] = True
                state["logged"] = True
                if isinstance(result.output, dict):
                    state["ticket_id"] = result.output.get("ticket_id")

        def _mutate_escalate(
            state: dict[str, Any], arguments: dict[str, Any], result: ToolCallResult
        ) -> None:
            if result.success:
                state["escalation_recommended"] = True
                state["logged"] = True

        self.register_mutator("check_firewall_rules", _mutate_firewall)
        self.register_mutator("query_siem_logs", _mutate_siem)
        self.register_mutator("create_incident_ticket", _mutate_ticket)
        self.register_mutator("escalate_to_soc", _mutate_escalate)


# Global default registry instance
default_tool_registry = DomainToolRegistry()
