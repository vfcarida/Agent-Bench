"""Tool adapters and tool registry for Agent-Bench domains."""

from agent_bench.tools.mcp_bridge import (
    MCPClientToolAdapter,
    MCPToolServer,
    discover_mcp_tools,
)
from agent_bench.tools.registry import DomainToolRegistry, default_tool_registry

__all__ = [
    "DomainToolRegistry",
    "MCPClientToolAdapter",
    "MCPToolServer",
    "default_tool_registry",
    "discover_mcp_tools",
]
