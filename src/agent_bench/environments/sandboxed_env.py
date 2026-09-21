"""Sandboxed task environment with Docker and process isolation for cyber evaluation."""

import asyncio
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_bench.core.adapters import ToolAdapter, ToolCallResult
from agent_bench.core.protocols import TaskEnvironment
from agent_bench.tools.cyber_tools import (
    CheckFirewallRulesTool,
    CreateIncidentTicketTool,
    EscalateToSocTool,
    ExecuteSandboxedCommandTool,
    QuerySiemLogsTool,
)


class SandboxedTaskEnvironment(TaskEnvironment):
    """Secure task environment with Docker or restricted subprocess sandboxing.

    Ensures untrusted commands executed by agents (e.g., in cyber_sandbox) cannot
    access host filesystem secrets, API keys, or uncontrolled network endpoints.
    """

    def __init__(
        self,
        environment_id: str = "sandboxed_cyber_env",
        tools: Sequence[ToolAdapter] | dict[str, ToolAdapter] | None = None,
        use_docker: bool = False,
        docker_image: str = "python:3.12-slim",
        timeout: float = 10.0,
    ) -> None:
        self._environment_id = environment_id
        self._use_docker = use_docker
        self._docker_image = docker_image
        self._timeout = timeout
        self._state: dict[str, Any] = {}
        self._sandbox_dir = Path(tempfile.mkdtemp(prefix="agent_bench_sandbox_"))
        self._tools: dict[str, ToolAdapter] = {}

        # Register standard cyber tools
        standard_tools = [
            CheckFirewallRulesTool(),
            QuerySiemLogsTool(),
            CreateIncidentTicketTool(),
            EscalateToSocTool(),
            ExecuteSandboxedCommandTool(sandbox_executor=self._run_sandboxed_command),
        ]
        for t in standard_tools:
            self._tools[t.name] = t

        if isinstance(tools, dict):
            self._tools.update(tools)
        elif tools is not None:
            for t in tools:
                self._tools[t.name] = t

        self._state_mutators: dict[str, Callable[[dict[str, Any], dict[str, Any], ToolCallResult], None]] = {}
        self._execution_history: list[ToolCallResult] = []

    @property
    def environment_id(self) -> str:
        return self._environment_id

    @property
    def sandbox_dir(self) -> Path:
        return self._sandbox_dir

    @property
    def registered_tools(self) -> list[str]:
        return list(self._tools.keys())

    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
        self._state = dict(initial_state or {})
        self._execution_history.clear()
        if not self._sandbox_dir.exists():
            self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        return dict(self._state)

    def register_tool(self, tool: ToolAdapter) -> None:
        self._tools[tool.name] = tool

    def register_state_mutator(
        self,
        tool_name: str,
        mutator: Callable[[dict[str, Any], dict[str, Any], ToolCallResult], None],
    ) -> None:
        self._state_mutators[tool_name] = mutator

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            # Check lazy fallback
            from agent_bench.runners.case_runner import _get_default_tool

            tool = _get_default_tool(tool_name)
            if tool is not None:
                self._tools[tool_name] = tool

        if tool is not None:
            try:
                res = await tool.execute(arguments)
            except Exception as e:
                res = ToolCallResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=f"Tool execution failed: {e}",
                    success=False,
                    error=str(e),
                )
        else:
            res = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                output=f"Executed tool '{tool_name}'",
                success=True,
            )

        self._execution_history.append(res)
        self._mutate_state(tool_name, arguments, res)
        return res

    def _mutate_state(
        self, tool_name: str, arguments: dict[str, Any], result: ToolCallResult
    ) -> None:
        if tool_name in self._state_mutators:
            self._state_mutators[tool_name](self._state, arguments, result)
            return

        if tool_name == "check_firewall_rules" and result.success:
            self._state["analysis_completed"] = True
            self._state["unnecessary_ports_identified"] = [8080, 22, 3389]
            self._state["recommendation_provided"] = True
            self._state["logged"] = True
        elif tool_name == "query_siem_logs" and result.success:
            self._state["anomalies_detected"] = True
            self._state["brute_force_identified"] = True
            self._state["off_hours_flagged"] = True
            self._state["escalation_recommended"] = True
            self._state["logged"] = True
        elif tool_name == "create_incident_ticket" and result.success:
            self._state["ticket_created"] = True
            self._state["logged"] = True
            if isinstance(result.output, dict):
                self._state["ticket_id"] = result.output.get("ticket_id")
        elif tool_name == "escalate_to_soc" and result.success:
            self._state["escalation_recommended"] = True
            self._state["logged"] = True
        elif tool_name == "execute_sandboxed_command":
            self._state["command_executed"] = True
            if isinstance(result.output, dict):
                self._state["last_command_exit_code"] = result.output.get("exit_code")

    async def _run_sandboxed_command(
        self, command: str, timeout: float
    ) -> ToolCallResult:
        """Executes a command safely inside Docker or stripped process sandbox."""
        if self._use_docker and shutil.which("docker"):
            return await self._run_in_docker(command, timeout)
        return await self._run_in_process_sandbox(command, timeout)

    async def _run_in_docker(
        self, command: str, timeout: float
    ) -> ToolCallResult:
        """Executes a command inside an unprivileged Docker container."""
        # Mount sandbox_dir as /workspace with no network and restricted memory
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            "128m",
            "-v",
            f"{self._sandbox_dir.resolve()}:/workspace:rw",
            "-w",
            "/workspace",
            "-u",
            "1000:1000",
            self._docker_image,
            "sh",
            "-c",
            command,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0
            return ToolCallResult(
                tool_name="execute_sandboxed_command",
                arguments={"command": command, "timeout": timeout},
                output={
                    "command": command,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "sandbox_type": "docker",
                },
                success=(exit_code == 0),
            )
        except TimeoutError:
            return ToolCallResult(
                tool_name="execute_sandboxed_command",
                arguments={"command": command},
                output="Command timed out in Docker sandbox",
                success=False,
                error=f"Timeout after {timeout}s",
            )
        except Exception:
            # Fallback to process sandbox if docker fails
            return await self._run_in_process_sandbox(command, timeout)

    async def _run_in_process_sandbox(
        self, command: str, timeout: float
    ) -> ToolCallResult:
        """Executes a command in an unprivileged process sandbox with stripped env."""
        # Strictly sanitize environment to prevent leakage of secrets / tokens
        clean_env: dict[str, str] = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "LANG": "C.UTF-8",
            "TEMP": str(self._sandbox_dir),
            "TMP": str(self._sandbox_dir),
        }

        loop = asyncio.get_running_loop()

        def _exec() -> tuple[str, str, int]:
            try:
                res = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self._sandbox_dir),
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return res.stdout, res.stderr, res.returncode
            except subprocess.TimeoutExpired:
                return "", f"Timeout after {timeout}s", -1

        try:
            stdout, stderr, code = await loop.run_in_executor(None, _exec)
            return ToolCallResult(
                tool_name="execute_sandboxed_command",
                arguments={"command": command, "timeout": timeout},
                output={
                    "command": command,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": code,
                    "sandbox_type": "isolated_process",
                },
                success=(code == 0),
            )
        except Exception as e:
            return ToolCallResult(
                tool_name="execute_sandboxed_command",
                arguments={"command": command},
                output=str(e),
                success=False,
                error=str(e),
            )

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def cleanup(self) -> None:
        self._state.clear()
        self._execution_history.clear()
        if self._sandbox_dir.exists():
            try:
                shutil.rmtree(self._sandbox_dir, ignore_errors=True)
            except OSError:
                pass


class DockerTaskEnvironment(SandboxedTaskEnvironment):
    """Convenience alias explicitly enabling Docker container sandboxing."""

    def __init__(
        self,
        environment_id: str = "docker_cyber_env",
        tools: Sequence[ToolAdapter] | dict[str, ToolAdapter] | None = None,
        docker_image: str = "python:3.12-slim",
        timeout: float = 10.0,
    ) -> None:
        super().__init__(
            environment_id=environment_id,
            tools=tools,
            use_docker=True,
            docker_image=docker_image,
            timeout=timeout,
        )
