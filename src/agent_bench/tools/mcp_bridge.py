"""Model Context Protocol (MCP) bidirectional bridge for Agent-Bench.

Implements the JSON-RPC 2.0 Model Context Protocol (version 2024-11-05) so that:
1. `MCPToolServer` exposes any Agent-Bench `TaskEnvironment` and `ToolAdapter` registry
   as an MCP-compliant server (`initialize`, `tools/list`, `tools/call`, `resources/list`,
   `resources/read`).
2. `MCPClientToolAdapter` and `discover_mcp_tools` allow external MCP tool servers to be
   consumed directly as Agent-Bench `ToolAdapter` instances with zero custom glue code.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agent_bench.core.adapters import ToolAdapter, ToolCallResult
from agent_bench.core.protocols import TaskEnvironment

MCP_PROTOCOL_VERSION = "2024-11-05"

MCPTransport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class MCPToolServer:
    """Exposes an Agent-Bench TaskEnvironment and ToolAdapters as an MCP JSON-RPC 2.0 server."""

    def __init__(
        self,
        environment: TaskEnvironment,
        tools: Sequence[ToolAdapter] | None = None,
        *,
        server_name: str = "agent-bench-mcp-server",
        server_version: str = "1.0.0",
    ) -> None:
        self._environment = environment
        self._server_name = server_name
        self._server_version = server_version
        self._tools: dict[str, ToolAdapter] = {}

        if tools is not None:
            for t in tools:
                self._tools[t.name] = t
        else:
            from agent_bench.tools.registry import default_tool_registry

            for t_name in default_tool_registry.registered_tools():
                tool_obj = default_tool_registry.get_tool(t_name)
                if tool_obj is not None:
                    self._tools[t_name] = tool_obj

    @property
    def environment(self) -> TaskEnvironment:
        return self._environment

    def register_tool(self, tool: ToolAdapter) -> None:
        """Register an additional ToolAdapter on the MCP server."""
        self._tools[tool.name] = tool

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a single JSON-RPC 2.0 MCP request and return the JSON-RPC 2.0 response."""
        req_id = request.get("id")
        method = str(request.get("method", ""))
        params = request.get("params", {})
        if not isinstance(params, dict):
            params = {}

        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "serverInfo": {
                        "name": self._server_name,
                        "version": self._server_version,
                    },
                },
            )

        if method == "ping":
            return self._ok(req_id, {})

        if method == "tools/list":
            mcp_tools: list[dict[str, Any]] = []
            for t_name, tool in sorted(self._tools.items()):
                schema = tool.schema
                description = str(schema.get("description", f"Agent-Bench tool {t_name}"))
                input_schema = schema.get(
                    "parameters",
                    schema.get("inputSchema", {"type": "object", "properties": {}}),
                )
                mcp_tools.append({
                    "name": t_name,
                    "description": description,
                    "inputSchema": input_schema,
                })
            return self._ok(req_id, {"tools": mcp_tools})

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            if not tool_name:
                return self._error(req_id, -32602, "Invalid params: missing tool 'name'")

            try:
                call_res = await self._environment.execute_tool(tool_name, arguments)
            except Exception as exc:
                return self._ok(
                    req_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )

            text_payload = (
                json.dumps(call_res.output)
                if isinstance(call_res.output, (dict, list))
                else str(call_res.output)
            )
            return self._ok(
                req_id,
                {
                    "content": [{"type": "text", "text": text_payload}],
                    "isError": not call_res.success,
                    "structuredContent": (
                        call_res.output if isinstance(call_res.output, dict) else {"value": call_res.output}
                    ),
                },
            )

        if method == "resources/list":
            return self._ok(
                req_id,
                {
                    "resources": [
                        {
                            "uri": "bench://environment/state",
                            "name": "Sandbox Environment State",
                            "description": "Live JSON snapshot of the TaskEnvironment state machine",
                            "mimeType": "application/json",
                        }
                    ]
                },
            )

        if method == "resources/read":
            uri = str(params.get("uri", ""))
            if uri != "bench://environment/state":
                return self._error(req_id, -32602, f"Resource not found: {uri}")
            state_json = json.dumps(self._environment.get_state(), sort_keys=True)
            return self._ok(
                req_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": state_json,
                        }
                    ]
                },
            )

        return self._error(req_id, -32601, f"Method not found: {method}")

    @staticmethod
    def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }


class MCPClientToolAdapter(ToolAdapter):
    """Wraps an MCP server tool as a native Agent-Bench ToolAdapter."""

    def __init__(
        self,
        tool_name: str,
        transport: MCPTransport | MCPToolServer,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self._name = tool_name
        self._transport: MCPTransport = (
            transport.handle_jsonrpc if isinstance(transport, MCPToolServer) else transport
        )
        self._description = description or f"MCP tool '{tool_name}'"
        self._input_schema = input_schema or {"type": "object", "properties": {}}
        self._req_counter = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "description": self._description,
            "parameters": self._input_schema,
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        self._req_counter += 1
        rpc_req = {
            "jsonrpc": "2.0",
            "id": f"mcp_call_{self._req_counter}",
            "method": "tools/call",
            "params": {"name": self._name, "arguments": arguments},
        }
        rpc_resp = await self._transport(rpc_req)

        if "error" in rpc_resp:
            err_msg = str(rpc_resp["error"].get("message", "MCP JSON-RPC error"))
            return ToolCallResult(
                tool_name=self._name,
                arguments=arguments,
                output=err_msg,
                success=False,
                error=err_msg,
            )

        res_body = rpc_resp.get("result", {})
        is_error = bool(res_body.get("isError", False))

        if "structuredContent" in res_body:
            output_val: Any = res_body["structuredContent"]
        else:
            content_list = res_body.get("content", [])
            text_parts = [
                str(block.get("text", ""))
                for block in content_list
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            combined = "\n".join(text_parts)
            try:
                output_val = json.loads(combined)
            except (json.JSONDecodeError, ValueError):
                output_val = combined

        return ToolCallResult(
            tool_name=self._name,
            arguments=arguments,
            output=output_val,
            success=not is_error,
            error=str(output_val) if is_error else None,
        )


async def discover_mcp_tools(
    transport: MCPTransport | MCPToolServer,
) -> list[MCPClientToolAdapter]:
    """Query `tools/list` on an MCP server and construct Agent-Bench ToolAdapters."""
    dispatch: MCPTransport = (
        transport.handle_jsonrpc if isinstance(transport, MCPToolServer) else transport
    )
    resp = await dispatch({"jsonrpc": "2.0", "id": "mcp_discover_1", "method": "tools/list"})
    tools_raw = resp.get("result", {}).get("tools", [])
    adapters: list[MCPClientToolAdapter] = []
    for t_spec in tools_raw:
        if not isinstance(t_spec, dict) or not t_spec.get("name"):
            continue
        adapters.append(
            MCPClientToolAdapter(
                tool_name=str(t_spec["name"]),
                transport=dispatch,
                description=str(t_spec.get("description", "")),
                input_schema=t_spec.get("inputSchema", {"type": "object", "properties": {}}),
            )
        )
    return adapters
