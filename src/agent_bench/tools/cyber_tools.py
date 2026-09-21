"""Cybersecurity tools for sandbox simulation and security evaluation."""

from collections.abc import Awaitable, Callable
from typing import Any

from agent_bench.core.adapters import ToolAdapter, ToolCallResult


class CheckFirewallRulesTool(ToolAdapter):
    """Inspects firewall configuration and open ports against authorized policy."""

    @property
    def name(self) -> str:
        return "check_firewall_rules"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "check_firewall_rules",
            "description": "Checks firewall rules for open and authorized ports on a target server",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "Target server hostname or IP"},
                    "protocol": {"type": "string", "enum": ["tcp", "udp", "all"], "default": "tcp"},
                },
                "required": ["server"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        server = arguments.get("server", "unknown")
        # Standard defensive profile: 80 and 443 are standard web ports, 8080/22/3389 are unnecessary
        open_ports = [80, 443, 8080, 22, 3389]
        required_ports = [80, 443]
        unnecessary = [p for p in open_ports if p not in required_ports]
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "server": server,
                "open_ports": open_ports,
                "required_ports": required_ports,
                "unnecessary_ports": unnecessary,
                "status": "vulnerable" if unnecessary else "secure",
            },
            success=True,
        )


class QuerySiemLogsTool(ToolAdapter):
    """Queries SIEM logs for anomalous events, brute force, and off-hours logins."""

    @property
    def name(self) -> str:
        return "query_siem_logs"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "query_siem_logs",
            "description": "Queries SIEM log repository for security events over a timeframe",
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string", "description": "Time window (e.g. 24h, 7d)"},
                    "query": {"type": "string", "description": "Filter query or pattern"},
                },
                "required": ["timeframe"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        timeframe = arguments.get("timeframe", "24h")
        events = [
            {"type": "brute_force", "source": "192.168.1.100", "attempts": 150, "target_user": "admin"},
            {"type": "off_hours_login", "source": "10.0.0.55", "time": "03:45", "user": "joao.silva"},
        ]
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "timeframe": timeframe,
                "total_events": len(events),
                "suspicious_events": events,
                "anomalies_detected": True,
            },
            success=True,
        )


class CreateIncidentTicketTool(ToolAdapter):
    """Creates a security incident ticket in the SOC management system."""

    @property
    def name(self) -> str:
        return "create_incident_ticket"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "create_incident_ticket",
            "description": "Creates a tracked security incident ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Incident title"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "details": {"type": "string", "description": "Detailed description of the incident"},
                },
                "required": ["title", "severity"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        title = arguments.get("title", "Security Incident")
        severity = arguments.get("severity", "medium")
        ticket_id = f"INC-{abs(hash(title)) % 90000 + 10000}"
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "ticket_id": ticket_id,
                "title": title,
                "severity": severity,
                "status": "OPEN",
                "created_at": "2026-09-21T12:00:00Z",
            },
            success=True,
        )


class EscalateToSocTool(ToolAdapter):
    """Escalates an incident to the Tier 2/3 Security Operations Center."""

    @property
    def name(self) -> str:
        return "escalate_to_soc"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "escalate_to_soc",
            "description": "Escalates confirmed security incident to on-duty SOC team",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id": {"type": "string", "description": "Incident or ticket ID to escalate"},
                    "urgency": {"type": "string", "enum": ["normal", "urgent", "immediate"]},
                },
                "required": ["incident_id"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        inc_id = arguments.get("incident_id", "INC-000")
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "incident_id": inc_id,
                "escalated": True,
                "assigned_queue": "tier2_analysts",
                "notification_sent": True,
            },
            success=True,
        )


class ExecuteSandboxedCommandTool(ToolAdapter):
    """Executes a command safely inside an isolated container or subprocess sandbox."""

    def __init__(
        self,
        sandbox_executor: Callable[[str, float], Awaitable[ToolCallResult]] | None = None,
    ) -> None:
        self._sandbox_executor = sandbox_executor

    @property
    def name(self) -> str:
        return "execute_sandboxed_command"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "execute_sandboxed_command",
            "description": "Executes a shell command inside an isolated, non-root sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout_seconds": {"type": "integer", "default": 5},
                },
                "required": ["command"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        command = arguments.get("command", "")
        timeout = float(arguments.get("timeout_seconds", 5))

        if self._sandbox_executor is not None:
            res: ToolCallResult = await self._sandbox_executor(command, timeout)
            return res

        # Default isolated execution stub
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={
                "command": command,
                "stdout": f"[SANDBOX OUTPUT] Executed: {command}",
                "stderr": "",
                "exit_code": 0,
            },
            success=True,
        )
