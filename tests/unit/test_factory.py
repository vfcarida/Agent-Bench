"""Unit tests for system_id to adapter/runner factory (T-WIRE-1)."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_bench.core.config import BenchConfig, ModelConfig, SystemConfig
from agent_bench.core.scenarios import Task
from agent_bench.models.anthropic_adapter import AnthropicModelAdapter
from agent_bench.models.factory import ConfigError, build_agent_runner, build_model_adapter
from agent_bench.models.openai_adapter import OpenAIModelAdapter
from agent_bench.models.stub import StubModelAdapter
from agent_bench.runners.case_runner import (
    DefaultAgentRunner,
    DefaultTaskEnvironment,
    run_single_case,
)
from agent_bench.runners.scripted import ScriptedAgentRunner


@pytest.fixture
def base_config() -> BenchConfig:
    return BenchConfig(
        models=[
            ModelConfig(
                model_id="stub-model",
                provider="stub",
                parameters={"responses": ["[STUB] Response 1", "[STUB] Response 2"]},
            ),
            ModelConfig(
                model_id="gpt-4o",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                parameters={"timeout": 30.0},
            ),
            ModelConfig(
                model_id="claude-sonnet",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                parameters={"timeout": 45.0},
            ),
            ModelConfig(
                model_id="internal-slm",
                provider="internal",
                endpoint="http://localhost:8000/v1",
                api_key_env="INTERNAL_KEY",
            ),
        ],
        systems=[
            SystemConfig(
                system_id="stub_system",
                architecture="prompt_only",
                model="stub-model",
            ),
            SystemConfig(
                system_id="openai_system",
                architecture="tool_calling_reactive",
                model="gpt-4o",
            ),
            SystemConfig(
                system_id="anthropic_system",
                architecture="planner_executor",
                model="claude-sonnet",
            ),
            SystemConfig(
                system_id="internal_system",
                architecture="prompt_only",
                model="internal-slm",
            ),
            SystemConfig(
                system_id="scripted_system",
                architecture="scripted_policy",
                model="none",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_t_wire_1_stub_system_drives_model_execute(base_config: BenchConfig):
    """T-WIRE-1: build_agent_runner('stub_system', cfg) returns runner invoking _model_execute."""
    runner = build_agent_runner("stub_system", base_config)

    assert isinstance(runner, DefaultAgentRunner)
    assert runner.system_id == "stub_system"
    assert runner.architecture == "model_adapter"

    dummy_task = Task(
        task_id="DUMMY_001",
        domain="pix_assist",
        name="Dummy Task",
        description="Dummy Description",
        input_messages=[{"role": "user", "content": "Hello"}],
    )
    env = DefaultTaskEnvironment()

    with patch(
        "agent_bench.runners.case_runner._model_execute",
        new_callable=AsyncMock,
    ) as mock_model_exec, patch(
        "agent_bench.runners.case_runner._stub_execute"
    ) as mock_stub_exec:
        mock_model_exec.return_value = {
            "response": "Model response",
            "final_state": {},
            "tools_called": [],
            "tokens_in": 10,
            "tokens_out": 5,
            "latency_ms": 25.0,
        }

        result, _traces = await runner.run_task(dummy_task, env)

        assert mock_model_exec.called
        assert not mock_stub_exec.called
        assert result["response"] == "Model response"


def test_t_wire_1_unknown_system_id_raises_config_error(base_config: BenchConfig):
    """Unknown system_id must raise ConfigError."""
    with pytest.raises(ConfigError) as exc_info:
        build_agent_runner("non_existent_system_123", base_config)

    assert "non_existent_system_123" in str(exc_info.value)
    assert "Available systems" in str(exc_info.value)


def test_unknown_model_id_raises_config_error():
    """System referencing a model not in config.models must raise ConfigError."""
    bad_config = BenchConfig(
        models=[],
        systems=[
            SystemConfig(
                system_id="orphan_system",
                architecture="prompt_only",
                model="missing_model",
            )
        ],
    )
    with pytest.raises(ConfigError) as exc_info:
        build_agent_runner("orphan_system", bad_config)

    assert "missing_model" in str(exc_info.value)


def test_unsupported_provider_raises_config_error():
    """ModelConfig with unsupported provider raises ConfigError."""
    unsupported_config = BenchConfig(
        models=[
            ModelConfig(
                model_id="custom_model",
                provider="unsupported_custom_provider",
            )
        ],
        systems=[
            SystemConfig(
                system_id="custom_system",
                architecture="prompt_only",
                model="custom_model",
            )
        ],
    )
    with pytest.raises(ConfigError) as exc_info:
        build_agent_runner("custom_system", unsupported_config)

    assert "Unsupported provider" in str(exc_info.value)


def test_scripted_policy_architecture_resolution(base_config: BenchConfig):
    """SystemConfig with architecture='scripted_policy' returns ScriptedAgentRunner."""
    runner = build_agent_runner("scripted_system", base_config)
    assert isinstance(runner, ScriptedAgentRunner)
    assert runner.architecture == "scripted_policy"


def test_build_model_adapters_configuration(base_config: BenchConfig):
    """Verifies each supported provider instantiates the right adapter with config."""
    # Stub
    stub_adapter = build_model_adapter(base_config.models[0])
    assert isinstance(stub_adapter, StubModelAdapter)
    assert stub_adapter.provider == "stub"

    # OpenAI
    openai_adapter = build_model_adapter(base_config.models[1])
    assert isinstance(openai_adapter, OpenAIModelAdapter)
    assert openai_adapter.provider == "openai"
    assert openai_adapter.model_id == "gpt-4o"

    # Anthropic
    anthropic_adapter = build_model_adapter(base_config.models[2])
    assert isinstance(anthropic_adapter, AnthropicModelAdapter)
    assert anthropic_adapter.provider == "anthropic"
    assert anthropic_adapter.model_id == "claude-sonnet"

    # Internal SLM
    internal_adapter = build_model_adapter(base_config.models[3])
    assert isinstance(internal_adapter, OpenAIModelAdapter)
    assert internal_adapter._base_url == "http://localhost:8000/v1"


@pytest.mark.asyncio
async def test_run_single_case_wires_factory(base_config: BenchConfig, tmp_path):
    """run_single_case must resolve runner via factory and invoke execution."""
    with patch(
        "agent_bench.runners.case_runner._model_execute",
        new_callable=AsyncMock,
    ) as mock_exec:
        mock_exec.return_value = {
            "response": "Executed single case",
            "final_state": {},
            "tools_called": [],
            "tokens_in": 15,
            "tokens_out": 10,
            "latency_ms": 30.0,
        }

        # PIX_001 exists in data/fixtures/pix_assist.yaml
        result = await run_single_case(
            task_id="PIX_001",
            system_id="stub_system",
            domain="pix_assist",
            config=base_config,
            output_dir=tmp_path,
        )

        assert result is not None
        assert mock_exec.called
