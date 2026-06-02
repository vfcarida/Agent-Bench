"""Stub investment advisory tools for benchmark simulation."""

from typing import Any

from agent_bench.core.adapters import ToolAdapter, ToolCallResult


class GetClientProfileTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "get_client_profile"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "get_client_profile",
            "description": "Get client investment profile and suitability classification",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "profile": "moderado",
                "risk_tolerance": "medium",
                "investment_horizon": "3-5 years",
                "max_rv_pct": 40,
                "last_suitability_date": "2025-06-01",
            },
        )


class GetPortfolioSummaryTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "get_portfolio_summary"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "get_portfolio_summary",
            "description": "Get current portfolio allocation summary",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "total_value": 150000.00,
                "allocation": {
                    "renda_fixa": 0.55,
                    "renda_variavel": 0.35,
                    "multimercado": 0.10,
                },
                "largest_position": {"asset": "CDB Premium 110% CDI", "pct": 0.25},
            },
        )


class CalculateReturnTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "calculate_return"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "calculate_return",
            "description": "Calculate investment return given parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number"},
                    "rate_annual": {"type": "number"},
                    "period_days": {"type": "integer"},
                    "tax_bracket": {"type": "string"},
                },
                "required": ["principal", "rate_annual", "period_days"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        principal = arguments.get("principal", 10000)
        rate = arguments.get("rate_annual", 0.12)
        days = arguments.get("period_days", 365)

        gross = principal * ((1 + rate) ** (days / 365) - 1)
        # IR regressivo simplificado
        if days <= 180:
            ir_rate = 0.225
        elif days <= 360:
            ir_rate = 0.20
        elif days <= 720:
            ir_rate = 0.175
        else:
            ir_rate = 0.15

        net = gross * (1 - ir_rate)
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "gross_return": round(gross, 2),
                "ir_rate": ir_rate,
                "ir_value": round(gross * ir_rate, 2),
                "net_return": round(net, 2),
                "final_value": round(principal + net, 2),
            },
        )


class CheckSuitabilityTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "check_suitability"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "check_suitability",
            "description": "Check if an investment is suitable for client profile",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_type": {"type": "string"},
                    "risk_level": {"type": "string"},
                },
                "required": ["product_type", "risk_level"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        risk_level = arguments.get("risk_level", "medium")
        suitable = risk_level in ("low", "medium")
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"suitable": suitable, "reason": "Within profile limits" if suitable else "Exceeds risk tolerance"},
        )


class SearchProductsTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "search_products"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "search_products",
            "description": "Search available investment products",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "min_return": {"type": "number"},
                },
                "required": [],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "products": [
                    {"name": "CDB Premium 110% CDI", "category": "renda_fixa", "risk": "low"},
                    {"name": "Tesouro Selic 2029", "category": "renda_fixa", "risk": "low"},
                    {"name": "Fundo Multi Macro", "category": "multimercado", "risk": "medium"},
                ]
            },
        )


class SimulateAllocationTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "simulate_allocation"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "simulate_allocation",
            "description": "Simulate a portfolio allocation given constraints",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_amount": {"type": "number"},
                    "profile": {"type": "string"},
                },
                "required": ["total_amount", "profile"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        amount = arguments.get("total_amount", 100000)
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "allocation": {
                    "renda_fixa": {"pct": 0.60, "value": amount * 0.6},
                    "multimercado": {"pct": 0.25, "value": amount * 0.25},
                    "renda_variavel": {"pct": 0.15, "value": amount * 0.15},
                },
                "expected_return_annual": 0.11,
                "max_drawdown_estimate": -0.05,
            },
        )
