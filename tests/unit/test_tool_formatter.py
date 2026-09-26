"""Unit tests for tool formatting and tool call extraction for local/open-weights models."""

from agent_bench.models.tool_formatter import (
    extract_tool_calls_from_text,
    format_tools_system_prompt,
    inject_tools_into_messages,
)

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "Checks the balance for an account",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_pix_transfer",
            "description": "Transfers funds via PIX",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_account": {"type": "string"},
                    "pix_key": {"type": "string"},
                    "amount_cents": {"type": "integer"},
                },
                "required": ["source_account", "pix_key", "amount_cents"],
            },
        },
    },
]


def test_format_tools_system_prompt() -> None:
    """Verify system prompt contains tool descriptions and XML syntax instructions."""
    prompt = format_tools_system_prompt(SAMPLE_TOOLS)
    assert "Available Tools" in prompt
    assert "check_balance" in prompt
    assert "execute_pix_transfer" in prompt
    assert "<tool_call>" in prompt
    assert "</tool_call>" in prompt


def test_inject_tools_into_messages_without_existing_system() -> None:
    """Verify system message is inserted when conversation has no system turn."""
    messages = [{"role": "user", "content": "How much money do I have?"}]
    injected = inject_tools_into_messages(messages, SAMPLE_TOOLS)

    assert len(injected) == 2
    assert injected[0]["role"] == "system"
    assert "check_balance" in injected[0]["content"]
    assert injected[1]["content"] == "How much money do I have?"


def test_inject_tools_into_messages_with_existing_system() -> None:
    """Verify tool instructions are appended to an existing system turn."""
    messages = [
        {"role": "system", "content": "You are Athena Banking Assistant."},
        {"role": "user", "content": "Check balance."},
    ]
    injected = inject_tools_into_messages(messages, SAMPLE_TOOLS)

    assert len(injected) == 2
    assert injected[0]["role"] == "system"
    assert "You are Athena Banking Assistant." in injected[0]["content"]
    assert "check_balance" in injected[0]["content"]


def test_inject_tools_into_messages_none_tools() -> None:
    """Verify messages remain identical when tools is None or empty."""
    messages = [{"role": "user", "content": "Hello"}]
    assert inject_tools_into_messages(messages, None) == messages
    assert inject_tools_into_messages(messages, []) == messages


def test_extract_tool_calls_xml_format() -> None:
    """Verify extraction of standard XML <tool_call> tags."""
    response = (
        "I need to check your account balance first.\n"
        "<tool_call>\n"
        '{"name": "check_balance", "arguments": {"account_id": "ACC_9988"}}\n'
        "</tool_call>"
    )
    clean_text, tool_calls = extract_tool_calls_from_text(response)

    assert "I need to check your account balance first." in clean_text
    assert "<tool_call>" not in clean_text
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "check_balance"
    assert tool_calls[0]["arguments"] == {"account_id": "ACC_9988"}
    assert tool_calls[0]["id"].startswith("call_")


def test_extract_tool_calls_multiple_xml() -> None:
    """Verify extraction of multiple tool calls in a single response."""
    response = (
        "<tool_call>\n"
        '{"name": "validate_pix_key", "arguments": {"pix_key": "123.456.789-00"}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "check_balance", "arguments": {"account_id": "ACC_1"}}\n'
        "</tool_call>"
    )
    clean_text, tool_calls = extract_tool_calls_from_text(response)

    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "validate_pix_key"
    assert tool_calls[0]["arguments"] == {"pix_key": "123.456.789-00"}
    assert tool_calls[1]["name"] == "check_balance"
    assert tool_calls[1]["arguments"] == {"account_id": "ACC_1"}


def test_extract_tool_calls_react_format() -> None:
    """Verify extraction of ReAct Action and Action Input syntax."""
    response = (
        "Thought: I will check firewall rules to investigate suspicious traffic.\n"
        "Action: check_firewall_rules\n"
        'Action Input: {"server": "web-prod-01"}'
    )
    clean_text, tool_calls = extract_tool_calls_from_text(response)

    assert "Thought: I will check firewall rules" in clean_text
    assert "Action:" not in clean_text
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "check_firewall_rules"
    assert tool_calls[0]["arguments"] == {"server": "web-prod-01"}


def test_extract_tool_calls_markdown_json() -> None:
    """Verify extraction of markdown ```json blocks."""
    response = (
        "Invoking query tool:\n"
        "```json\n"
        '{"name": "query_siem_logs", "arguments": {"timeframe": "24h"}}\n'
        "```"
    )
    clean_text, tool_calls = extract_tool_calls_from_text(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "query_siem_logs"
    assert tool_calls[0]["arguments"] == {"timeframe": "24h"}


def test_extract_tool_calls_raw_json() -> None:
    """Verify extraction when the entire response is raw JSON."""
    response = '{"name": "escalate_to_soc", "arguments": {"incident_id": "INC-99"}}'
    clean_text, tool_calls = extract_tool_calls_from_text(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "escalate_to_soc"
    assert tool_calls[0]["arguments"] == {"incident_id": "INC-99"}


def test_extract_tool_calls_no_tools() -> None:
    """Verify that pure conversational text returns no tool calls."""
    response = "The current balance for account ACC_1234 is 5,420.50 BRL."
    clean_text, tool_calls = extract_tool_calls_from_text(response)

    assert clean_text == response
    assert tool_calls == []
