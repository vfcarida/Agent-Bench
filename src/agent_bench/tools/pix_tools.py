"""Stub PIX tools for benchmark simulation."""

from typing import Any

from agent_bench.core.adapters import ToolAdapter, ToolCallResult


class CheckBalanceTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "check_balance"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "check_balance",
            "description": "Check account balance",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"balance": 5000.00, "currency": "BRL"},
        )


class ValidatePixKeyTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "validate_pix_key"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "validate_pix_key",
            "description": "Validate a PIX key and return owner info",
            "parameters": {
                "type": "object",
                "properties": {"pix_key": {"type": "string"}},
                "required": ["pix_key"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        key = arguments.get("pix_key", "")
        valid = len(key) > 5 and "invalid" not in key.lower()
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"valid": valid, "owner_name": "Joao Silva" if valid else None},
            success=valid,
            error=None if valid else "Invalid PIX key format",
        )


class ExecutePixTransferTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "execute_pix_transfer"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "execute_pix_transfer",
            "description": "Execute a PIX transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "pix_key": {"type": "string"},
                    "amount": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["pix_key", "amount"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "transaction_id": "tx_stub_001",
                "status": "completed",
                "amount": arguments.get("amount", 0),
            },
        )


class RequestUserConfirmationTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "request_user_confirmation"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "request_user_confirmation",
            "description": "Request user confirmation before executing action",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"confirmed": True},
        )
