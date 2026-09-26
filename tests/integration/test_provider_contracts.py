"""Contract tests for OpenAI and Anthropic model adapters using recorded HTTP cassettes."""

import json

import httpx
import pytest

from agent_bench.models.anthropic_adapter import AnthropicModelAdapter
from agent_bench.models.openai_adapter import OpenAIModelAdapter

# Recorded OpenAI Chat Completion Cassettes
OPENAI_TEXT_RESPONSE_CASSETTE = {
    "id": "chatcmpl-A1B2C3D4E5F6",
    "object": "chat.completion",
    "created": 1726912800,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I am an enterprise banking assistant ready to help with PIX transfers.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 35,
        "completion_tokens": 17,
        "total_tokens": 52,
    },
}

OPENAI_TOOL_CALL_RESPONSE_CASSETTE = {
    "id": "chatcmpl-ToolCall12345",
    "object": "chat.completion",
    "created": 1726912801,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "check_balance",
                            "arguments": '{"account_id": "ACC_9988"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 85,
        "completion_tokens": 24,
        "total_tokens": 109,
    },
}

# Recorded Anthropic Messages API Cassettes
ANTHROPIC_TEXT_RESPONSE_CASSETTE = {
    "id": "msg_01A2B3C4D5E6",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {
            "type": "text",
            "text": "Fixed income investments such as CDBs guaranteed by the FGC provide lower risk.",
        }
    ],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 42,
        "output_tokens": 19,
    },
}

ANTHROPIC_TOOL_USE_RESPONSE_CASSETTE = {
    "id": "msg_01ToolUse888999",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {
            "type": "text",
            "text": "Checking client investment suitability before allocation.",
        },
        {
            "type": "tool_use",
            "id": "toolu_01X9Y8Z7",
            "name": "check_suitability",
            "input": {"client_id": "CLI_1234", "risk_profile": "conservative"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 96,
        "output_tokens": 38,
    },
}


@pytest.mark.asyncio
async def test_openai_adapter_text_completion_contract() -> None:
    """Validate OpenAIModelAdapter request format and response contract for text generation."""
    captured_requests: list[httpx.Request] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        # Verify request contract
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-openai-key"
        assert request.headers["content-type"] == "application/json"

        body = json.loads(request.read())
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "Hello banking assistant"}]
        assert body["temperature"] == 0.0
        assert body["seed"] == 42

        return httpx.Response(200, json=OPENAI_TEXT_RESPONSE_CASSETTE)

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = OpenAIModelAdapter(model_id="gpt-4o", api_key="test-openai-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    try:
        res = await adapter.generate(
            [{"role": "user", "content": "Hello banking assistant"}],
            temperature=0.0,
            seed=42,
        )

        assert len(captured_requests) == 1
        assert res.content == "Hello! I am an enterprise banking assistant ready to help with PIX transfers."
        assert res.tokens_in == 35
        assert res.tokens_out == 17
        assert res.latency_ms > 0
        assert res.tool_calls == []
        assert res.raw["id"] == "chatcmpl-A1B2C3D4E5F6"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_tool_calling_contract() -> None:
    """Validate OpenAIModelAdapter schema translation and parsing for function tool calls."""
    captured_requests: list[httpx.Request] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.read())
        # Verify tool schema was translated to OpenAI format {"type": "function", "function": ...}
        assert "tools" in body
        assert len(body["tools"]) == 1
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "check_balance"

        return httpx.Response(200, json=OPENAI_TOOL_CALL_RESPONSE_CASSETTE)

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = OpenAIModelAdapter(model_id="gpt-4o", api_key="test-openai-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    tools_schema = [
        {
            "name": "check_balance",
            "description": "Check user balance",
            "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}},
        }
    ]

    try:
        res = await adapter.generate(
            [{"role": "user", "content": "Check my balance"}],
            tools=tools_schema,
        )

        assert len(res.tool_calls) == 1
        tc = res.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "check_balance"
        assert tc["arguments"] == {"account_id": "ACC_9988"}
        assert res.tokens_in == 85
        assert res.tokens_out == 24
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_malformed_tool_call_arguments_fallback() -> None:
    """Validate OpenAIModelAdapter handles malformed tool arguments without crashing."""
    malformed_cassette = {
        "id": "chatcmpl-Malformed123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad123",
                            "type": "function",
                            "function": {
                                "name": "check_balance",
                                "arguments": "{malformed json",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=malformed_cassette)

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = OpenAIModelAdapter(model_id="gpt-4o", api_key="test-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    try:
        res = await adapter.generate([{"role": "user", "content": "test"}])
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["arguments"] == {"_raw": "{malformed json"}
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_error_contract() -> None:
    """Validate OpenAIModelAdapter error contract on API error (401 Unauthorized)."""
    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid API key provided", "type": "invalid_request_error"}},
        )

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = OpenAIModelAdapter(model_id="gpt-4o", api_key="invalid-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.generate([{"role": "user", "content": "ping"}])
        assert exc_info.value.response.status_code == 401
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_text_completion_contract() -> None:
    """Validate AnthropicModelAdapter request format and system prompt extraction."""
    captured_requests: list[httpx.Request] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        # Verify request contract
        assert request.method == "POST"
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-anthropic-key"
        assert request.headers["anthropic-version"] == "2023-06-01"

        body = json.loads(request.read())
        assert body["model"] == "claude-sonnet-4-20250514"
        # System message separated from messages list
        assert body["system"] == "You are a professional financial advisor."
        assert body["messages"] == [{"role": "user", "content": "What is a CDB?"}]

        return httpx.Response(200, json=ANTHROPIC_TEXT_RESPONSE_CASSETTE)

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = AnthropicModelAdapter(model_id="claude-sonnet-4-20250514", api_key="test-anthropic-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    try:
        messages = [
            {"role": "system", "content": "You are a professional financial advisor."},
            {"role": "user", "content": "What is a CDB?"},
        ]
        res = await adapter.generate(messages)

        assert len(captured_requests) == 1
        assert "guaranteed by the FGC" in res.content
        assert res.tokens_in == 42
        assert res.tokens_out == 19
        assert res.tool_calls == []
        assert res.raw["id"] == "msg_01A2B3C4D5E6"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_tool_use_contract() -> None:
    """Validate AnthropicModelAdapter tool_use schema translation and content block parsing."""
    captured_requests: list[httpx.Request] = []

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        body = json.loads(request.read())
        # Anthropic tool schema format: {"name": ..., "input_schema": ...}
        assert "tools" in body
        assert len(body["tools"]) == 1
        assert body["tools"][0]["name"] == "check_suitability"
        assert "input_schema" in body["tools"][0]

        return httpx.Response(200, json=ANTHROPIC_TOOL_USE_RESPONSE_CASSETTE)

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = AnthropicModelAdapter(model_id="claude-sonnet-4-20250514", api_key="test-anthropic-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    tools_schema = [
        {
            "name": "check_suitability",
            "description": "Checks suitability profile",
            "parameters": {"type": "object", "properties": {"client_id": {"type": "string"}}},
        }
    ]

    try:
        res = await adapter.generate(
            [{"role": "user", "content": "Check suitability for CLI_1234"}],
            tools=tools_schema,
        )

        assert "Checking client investment suitability" in res.content
        assert len(res.tool_calls) == 1
        tc = res.tool_calls[0]
        assert tc["id"] == "toolu_01X9Y8Z7"
        assert tc["name"] == "check_suitability"
        assert tc["arguments"] == {"client_id": "CLI_1234", "risk_profile": "conservative"}
        assert res.tokens_in == 96
        assert res.tokens_out == 38
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_error_contract() -> None:
    """Validate AnthropicModelAdapter error contract on 400 Bad Request."""
    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens is too large"}},
        )

    transport = httpx.MockTransport(mock_transport_handler)
    adapter = AnthropicModelAdapter(model_id="claude-sonnet-4-20250514", api_key="test-key")
    adapter._client = httpx.AsyncClient(transport=transport)

    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await adapter.generate([{"role": "user", "content": "test"}], max_tokens=100000)
        assert exc_info.value.response.status_code == 400
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_model_adapter_lifecycle_and_context_manager() -> None:
    """Verify that model adapters properly close HTTP clients and support async context managers."""
    # 1. Direct close on OpenAIModelAdapter
    openai_adapter = OpenAIModelAdapter(api_key="test-key")
    assert not openai_adapter._client.is_closed
    await openai_adapter.close()
    assert openai_adapter._client.is_closed
    # Calling close again must be safe and idempotent
    await openai_adapter.close()

    # 2. Async context manager on AnthropicModelAdapter
    async with AnthropicModelAdapter(api_key="test-key") as anthropic_adapter:
        assert not anthropic_adapter._client.is_closed
    assert anthropic_adapter._client.is_closed


@pytest.mark.asyncio
async def test_default_agent_runner_lifecycle() -> None:
    """Verify that DefaultAgentRunner closes its underlying model adapter."""
    from agent_bench.runners.case_runner import DefaultAgentRunner

    adapter = OpenAIModelAdapter(api_key="test-key")
    async with DefaultAgentRunner(system_id="test_sys", model=adapter) as runner:
        assert runner.system_id == "test_sys"
        assert not adapter._client.is_closed
    assert adapter._client.is_closed

