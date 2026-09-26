"""Unit tests for UserSimulator implementations and dynamic multi-turn execution."""

from unittest.mock import AsyncMock

import pytest

from agent_bench.core.adapters import ModelAdapter, ModelResponse
from agent_bench.core.artifacts import TraceEventType
from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig
from agent_bench.core.scenarios import Task
from agent_bench.core.user_simulator import (
    ModelUserSimulator,
    RuleBasedUserSimulator,
    ScriptedUserSimulator,
    UserSimulator,
)
from agent_bench.runners.case_runner import (
    execute_task,
)


@pytest.mark.asyncio
async def test_scripted_user_simulator_sequential() -> None:
    """Verify ScriptedUserSimulator steps through sequential replies and terminates."""
    sim = ScriptedUserSimulator(
        scripted_responses=["Resposta 1", "Resposta 2"],
        max_turns=3,
        default_finish_message="Obrigado, acabou.",
    )
    assert isinstance(sim, UserSimulator)

    task = Task(
        task_id="TASK_SIM_01",
        name="Task Sim 01",
        description="Sim test task",
        domain="pix_assist",
        input_messages=[{"role": "user", "content": "Olá"}],
    )

    turn1 = await sim.step("Qual seu nome?", [], task)
    assert turn1.content == "Resposta 1"
    assert not turn1.finished

    turn2 = await sim.step("Qual seu CPF?", [], task)
    assert turn2.content == "Resposta 2"
    assert not turn2.finished

    turn3 = await sim.step("Algo mais?", [], task)
    assert turn3.content == "Obrigado, acabou."
    assert turn3.finished

    # Test reset
    sim.reset(task)
    turn_reset = await sim.step("Qual seu nome?", [], task)
    assert turn_reset.content == "Resposta 1"


@pytest.mark.asyncio
async def test_scripted_user_simulator_patterns() -> None:
    """Verify ScriptedUserSimulator pattern matching overrides sequential responses."""
    sim = ScriptedUserSimulator(
        pattern_responses={
            r"cpf": "123.456.789-00",
            r"chave|pix": "user@test.com",
        },
        max_turns=3,
    )
    task = Task(
        task_id="TASK_SIM_02",
        name="Task Sim 02",
        description="Pattern test task",
        domain="pix_assist",
        input_messages=[],
    )

    turn1 = await sim.step("Por favor me informe seu CPF para continuar.", [], task)
    assert turn1.content == "123.456.789-00"
    assert not turn1.finished

    turn2 = await sim.step("Qual a sua chave pix de destino?", [], task)
    assert turn2.content == "user@test.com"
    assert not turn2.finished


@pytest.mark.asyncio
async def test_rule_based_user_simulator() -> None:
    """Verify RuleBasedUserSimulator dynamic slot-filling from task metadata."""
    sim = RuleBasedUserSimulator(max_turns=5)
    task = Task(
        task_id="TASK_SLOT_01",
        name="Task Slot 01",
        description="Slot test task",
        domain="pix_assist",
        input_messages=[],
        metadata={
            "user_slots": {
                "chave": "11999998888",
                "valor": "R$ 50,00",
            }
        },
    )
    sim.reset(task)

    turn1 = await sim.step("Qual é a chave do recebedor?", [], task)
    assert "11999998888" in turn1.content
    assert not turn1.finished

    turn2 = await sim.step("Qual o valor que deseja transferir?", [], task)
    assert "R$ 50,00" in turn2.content
    assert not turn2.finished

    turn3 = await sim.step("Confirma a transferência?", [], task)
    assert "confirmo" in turn3.content.lower()

    turn4 = await sim.step("Transferência realizada com sucesso!", [], task)
    assert turn4.finished


@pytest.mark.asyncio
async def test_model_user_simulator() -> None:
    """Verify ModelUserSimulator properly invokes model adapter and handles [FINISHED]."""
    mock_model = AsyncMock(spec=ModelAdapter)
    mock_model.generate.return_value = ModelResponse(
        content="[FINISHED] Sim, está tudo resolvido.",
        tokens_in=20,
        tokens_out=10,
        latency_ms=100.0,
    )

    sim = ModelUserSimulator(model=mock_model, max_turns=3)
    task = Task(
        task_id="TASK_MODEL_SIM",
        name="Task Model Sim",
        domain="pix_assist",
        description="Transferir 10 reais para Maria",
        input_messages=[{"role": "user", "content": "Quero transferir"}],
    )

    turn = await sim.step(
        "Transferência efetuada com sucesso!",
        history=[{"role": "user", "content": "Quero transferir"}],
        task=task,
    )
    assert turn.finished
    assert "Sim, está tudo resolvido." in turn.content
    assert mock_model.generate.called


@pytest.mark.asyncio
async def test_multi_turn_execution_with_user_simulator() -> None:
    """Verify DefaultAgentRunner executes multi-turn conversations with UserSimulator and logs traces."""
    task = Task(
        task_id="TASK_MULTI_TURN",
        domain="pix_assist",
        name="Multi Turn PIX",
        description="Transferencia com pergunta interativa",
        input_messages=[{"role": "user", "content": "Quero fazer um PIX."}],
        expected_final_state={"balance": 900.0},
        initial_state={"balance": 1000.0},
    )

    # Mock agent model that first asks a question, then upon receiving user answer executes transfer tool
    mock_agent_model = AsyncMock(spec=ModelAdapter)

    # Turn 1: Model asks for PIX key
    resp_step1 = ModelResponse(
        content="Qual é a chave PIX do destinatário?",
        tool_calls=[],
        tokens_in=15,
        tokens_out=10,
        latency_ms=80.0,
    )
    # Turn 2: Model calls transfer_pix tool
    resp_step2 = ModelResponse(
        content="Enviando PIX...",
        tool_calls=[
            {
                "id": "call_1",
                "name": "transfer_pix",
                "arguments": {"amount": 100.0, "pix_key": "dest@pix.com"},
            }
        ],
        tokens_in=30,
        tokens_out=15,
        latency_ms=90.0,
    )
    # Turn 3: Model confirms
    resp_step3 = ModelResponse(
        content="Transferência de R$ 100 realizada com sucesso!",
        tool_calls=[],
        tokens_in=45,
        tokens_out=12,
        latency_ms=70.0,
    )

    mock_agent_model.generate.side_effect = [resp_step1, resp_step2, resp_step3]

    simulator = ScriptedUserSimulator(
        scripted_responses=["A chave é dest@pix.com"],
        max_turns=3,
        default_finish_message="Obrigado!",
    )

    bench_cfg = BenchConfig(
        benchmark_version="1.0.0",
        suites=[
            SuiteConfig(
                suite_id="test_suite",
                name="Test",
                version="1.0.0",
                domains=["pix_assist"],
                systems=["test_system"],
            )
        ],
        systems=[SystemConfig(system_id="test_system", model="mock_model", architecture="tool_calling_reactive")],
    )

    case_result = await execute_task(
        task,
        system_id="test_system",
        config=bench_cfg,
        model=mock_agent_model,
        user_simulator=simulator,
    )

    # Verify task passed
    assert case_result.passed

    # Verify traces include USER_MESSAGE
    user_msg_traces = [
        tr for tr in case_result.traces if tr.event_type == TraceEventType.USER_MESSAGE
    ]
    assert len(user_msg_traces) >= 1
    assert user_msg_traces[0].data["content"] == "A chave é dest@pix.com"
