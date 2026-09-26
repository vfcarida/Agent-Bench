"""Unit tests for SandboxedTaskEnvironment, DockerTaskEnvironment, and Cyber Tools."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from agent_bench.environments.sandboxed_env import DockerTaskEnvironment, SandboxedTaskEnvironment
from agent_bench.tools.cyber_tools import (
    CheckFirewallRulesTool,
    CreateIncidentTicketTool,
    EscalateToSocTool,
    QuerySiemLogsTool,
)


@pytest.mark.asyncio
async def test_cyber_tools_direct_execution() -> None:
    """Test direct execution of cyber tool adapters."""
    # 1. Firewall rules tool
    fw_tool = CheckFirewallRulesTool()
    assert fw_tool.name == "check_firewall_rules"
    fw_res = await fw_tool.execute({"server": "web-prod-01"})
    assert fw_res.success
    assert 8080 in fw_res.output["unnecessary_ports"]
    assert 22 in fw_res.output["unnecessary_ports"]

    # 2. SIEM logs tool
    siem_tool = QuerySiemLogsTool()
    assert siem_tool.name == "query_siem_logs"
    siem_res = await siem_tool.execute({"timeframe": "24h"})
    assert siem_res.success
    assert siem_res.output["anomalies_detected"] is True
    assert len(siem_res.output["suspicious_events"]) >= 2

    # 3. Incident ticket tool
    ticket_tool = CreateIncidentTicketTool()
    assert ticket_tool.name == "create_incident_ticket"
    ticket_res = await ticket_tool.execute({"title": "Unauthorized SSH", "severity": "high"})
    assert ticket_res.success
    assert ticket_res.output["status"] == "OPEN"
    assert "ticket_id" in ticket_res.output

    # 4. Escalate tool
    esc_tool = EscalateToSocTool()
    assert esc_tool.name == "escalate_to_soc"
    esc_res = await esc_tool.execute({"incident_id": "INC-1234", "urgency": "urgent"})
    assert esc_res.success
    assert esc_res.output["escalated"] is True


@pytest.mark.asyncio
async def test_sandboxed_task_environment_lifecycle() -> None:
    """Test SandboxedTaskEnvironment reset, tool execution, state mutation, and cleanup."""
    env = SandboxedTaskEnvironment()
    sandbox_dir = env.sandbox_dir
    assert sandbox_dir.exists()

    # Initial reset
    init_state = {"target": "web-prod-01", "scope": "sandbox"}
    state = env.reset(init_state)
    assert state["target"] == "web-prod-01"

    # Execute check_firewall_rules
    fw_res = await env.execute_tool("check_firewall_rules", {"server": "web-prod-01"})
    assert fw_res.success
    updated_state = env.get_state()
    assert updated_state["analysis_completed"] is True
    assert 8080 in updated_state["unnecessary_ports_identified"]
    assert updated_state["logged"] is True

    # Execute query_siem_logs
    siem_res = await env.execute_tool("query_siem_logs", {"timeframe": "24h"})
    assert siem_res.success
    updated_state = env.get_state()
    assert updated_state["anomalies_detected"] is True
    assert updated_state["brute_force_identified"] is True
    assert updated_state["escalation_recommended"] is True

    # Execute create_incident_ticket
    ticket_res = await env.execute_tool(
        "create_incident_ticket", {"title": "Firewall Anomaly", "severity": "high"}
    )
    assert ticket_res.success
    updated_state = env.get_state()
    assert updated_state["ticket_created"] is True
    assert "ticket_id" in updated_state

    # Execute escalate_to_soc
    esc_res = await env.execute_tool(
        "escalate_to_soc", {"incident_id": updated_state["ticket_id"]}
    )
    assert esc_res.success
    assert env.get_state()["escalation_recommended"] is True

    # Cleanup
    env.cleanup()
    assert not env.get_state()
    assert not sandbox_dir.exists()


@pytest.mark.asyncio
async def test_sandboxed_command_isolation() -> None:
    """Verify that command execution in the process sandbox strictly isolates secrets and cwd."""
    # Set a sensitive host secret
    os.environ["SENSITIVE_HOST_API_KEY"] = "super_secret_token_12345"
    env = SandboxedTaskEnvironment(timeout=5.0)

    try:
        # 1. Verify working directory is within sandbox_dir
        cmd_pwd = "cd" if os.name == "nt" else "pwd"
        res_pwd = await env.execute_tool("execute_sandboxed_command", {"command": cmd_pwd})
        assert res_pwd.success
        assert str(env.sandbox_dir).lower() in str(res_pwd.output["stdout"]).lower()

        # 2. Verify sensitive host environment variable is NOT leaked into sandbox
        check_env_cmd = "set SENSITIVE_HOST_API_KEY" if os.name == "nt" else "echo $SENSITIVE_HOST_API_KEY"
        res_env = await env.execute_tool("execute_sandboxed_command", {"command": check_env_cmd})
        # Under sanitized clean_env, the secret must not appear in stdout
        assert "super_secret_token_12345" not in res_env.output["stdout"]
    finally:
        os.environ.pop("SENSITIVE_HOST_API_KEY", None)
        env.cleanup()


@pytest.mark.asyncio
async def test_sandboxed_command_timeout() -> None:
    """Verify that commands exceeding timeout are killed gracefully."""
    env = SandboxedTaskEnvironment(timeout=0.5)
    try:
        # Sleep command that exceeds 0.5s timeout
        sleep_cmd = "powershell -Command Start-Sleep -Seconds 2" if os.name == "nt" else "sleep 2"
        res = await env.execute_tool("execute_sandboxed_command", {"command": sleep_cmd, "timeout_seconds": 0.5})
        assert not res.success or res.output.get("exit_code") == -1
    finally:
        env.cleanup()


@pytest.mark.asyncio
async def test_docker_task_environment_delegation() -> None:
    """Test DockerTaskEnvironment configuration and docker fallback."""
    docker_env = DockerTaskEnvironment(docker_image="python:3.12-slim", timeout=10.0)
    assert docker_env._use_docker is True
    assert docker_env._docker_image == "python:3.12-slim"

    with patch("shutil.which", return_value="/usr/bin/docker"), patch.object(
        docker_env, "_run_in_docker", new_callable=AsyncMock
    ) as mock_docker:
        mock_docker.return_value.success = True
        mock_docker.return_value.output = {"stdout": "docker execution", "exit_code": 0}

        res = await docker_env._run_sandboxed_command("echo hello", timeout=5.0)
        assert mock_docker.called
        assert res.success

    docker_env.cleanup()


@pytest.mark.asyncio
async def test_sandboxed_command_security_restrictions() -> None:
    """Verify that dangerous shell operators, command chaining, and path traversal are rejected."""
    env = SandboxedTaskEnvironment(timeout=5.0)
    try:
        # 1. Shell chaining with &&
        res_chain = await env.execute_tool(
            "execute_sandboxed_command", {"command": "echo harmless && whoami"}
        )
        assert not res_chain.success
        assert "Security violation" in str(res_chain.output.get("stderr"))

        # 2. Shell injection with semicolon
        res_semi = await env.execute_tool(
            "execute_sandboxed_command", {"command": "echo 1; echo 2"}
        )
        assert not res_semi.success
        assert "Security violation" in str(res_semi.output.get("stderr"))

        # 3. Piping operator
        res_pipe = await env.execute_tool(
            "execute_sandboxed_command", {"command": "echo test | grep test"}
        )
        assert not res_pipe.success
        assert "Security violation" in str(res_pipe.output.get("stderr"))

        # 4. Command substitution
        res_sub = await env.execute_tool(
            "execute_sandboxed_command", {"command": "echo $(id)"}
        )
        assert not res_sub.success
        assert "Security violation" in str(res_sub.output.get("stderr"))

        # 5. Path traversal with ../
        res_trav = await env.execute_tool(
            "execute_sandboxed_command", {"command": "cat ../../secret.txt"}
        )
        assert not res_trav.success
        assert "Security violation" in str(res_trav.output.get("stderr"))

        # 6. Escaping cwd via cd
        res_cd = await env.execute_tool(
            "execute_sandboxed_command", {"command": "cd .."}
        )
        assert not res_cd.success
        assert "Security violation" in str(res_cd.output.get("stderr"))
    finally:
        env.cleanup()

