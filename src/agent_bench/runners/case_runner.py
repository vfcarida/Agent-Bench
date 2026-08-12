"""Case runner: executes a single benchmark task following Clean Architecture principles."""

import time
from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.adapters import JudgeVerdict, ModelAdapter, ModelResponse, ToolCallResult
from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig
from agent_bench.core.protocols import AgentRunner, Evaluator, TaskEnvironment
from agent_bench.core.scenarios import Task
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.graders.gated_evaluator import GatedEvaluator
from agent_bench.graders.thinking_parser import parse_thinking_response
from agent_bench.runners.prompt_formatter import PromptFormatter
from agent_bench.storage.trace_logger import ExecutionTraceLogger

logger = structlog.get_logger()


class DefaultTaskEnvironment:
    """Standard task environment implementation of TaskEnvironment protocol."""

    def __init__(self, environment_id: str = "default_env") -> None:
        self._environment_id = environment_id
        self._state: dict[str, Any] = {}

    @property
    def environment_id(self) -> str:
        return self._environment_id

    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
        self._state = dict(initial_state or {})
        return self._state

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            output=f"Executed tool '{tool_name}'",
            success=True,
        )

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def cleanup(self) -> None:
        self._state.clear()


class DefaultAgentRunner:
    """Standard agent runner implementation of AgentRunner protocol."""

    def __init__(
        self,
        system_id: str = "default_system",
        model: ModelAdapter | None = None,
        prompt_formatter: PromptFormatter | None = None,
    ) -> None:
        self._system_id = system_id
        self._model = model
        self._prompt_formatter = prompt_formatter

    @property
    def system_id(self) -> str:
        return self._system_id

    @property
    def architecture(self) -> str:
        return "model_adapter" if self._model else "stub_agent"

    async def run_task(
        self,
        task: Task,
        environment: TaskEnvironment,
        *,
        max_steps: int = 10,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], list[TraceEvent]]:
        environment.reset(task.initial_state)

        if self._model is not None:
            result = await _model_execute(task, self._model, seed=seed)
        else:
            result = _stub_execute(task, self.system_id)

        # Merge environment state if updated
        current_env_state = environment.get_state()
        if current_env_state and not result.get("final_state"):
            result["final_state"] = current_env_state

        return result, []


async def execute_task(
    task: Task,
    system_id: str,
    config: BenchConfig,
    *,
    seed: int | None = None,
    model: ModelAdapter | None = None,
    prompt_formatter: PromptFormatter | None = None,
    agent_runner: AgentRunner | None = None,
    environment: TaskEnvironment | None = None,
    evaluator: Evaluator | None = None,
    run_id: str = "run_default",
) -> tuple[bool, list[TraceEvent]]:
    """Execute a single task and return (success, traces).

    Supports both legacy direct model execution and decoupled Clean Architecture execution
    via AgentRunner, TaskEnvironment, and Evaluator protocols.
    """
    logger_trace = ExecutionTraceLogger(run_id=run_id, task_id=task.task_id, system_id=system_id)

    # 1. Format Prompt Event
    messages = task.input_messages
    if prompt_formatter is not None:
        formatted_prompt = prompt_formatter.format_messages(messages)
        logger_trace.log_event(
            TraceEventType.PROMPT_SENT,
            {
                "messages": messages,
                "formatted_prompt": formatted_prompt,
                "template": prompt_formatter.template_name,
                "system_id": system_id,
                "seed": seed,
            },
        )
    else:
        logger_trace.log_event(
            TraceEventType.PROMPT_SENT,
            {"messages": messages, "system_id": system_id, "seed": seed},
        )

    # 2. Execution Phase
    start_time = time.perf_counter()
    env = environment or DefaultTaskEnvironment()
    runner = agent_runner or DefaultAgentRunner(
        system_id=system_id, model=model, prompt_formatter=prompt_formatter
    )

    simulated_result, runner_traces = await runner.run_task(
        task, env, seed=seed
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    for tr in runner_traces:
        logger_trace.log_event(tr.event_type, tr.data, parent_id=tr.parent_id)

    response_text = simulated_result.get("response", "")

    # Parse <think> blocks if present
    parsed = parse_thinking_response(response_text)
    if parsed.has_thinking:
        logger_trace.log_event(
            TraceEventType.THINKING_BLOCK,
            {
                "thinking_blocks": parsed.thinking_blocks,
                "thinking_token_count": parsed.thinking_token_count,
                "answer_token_count": parsed.answer_token_count,
                "thinking_ratio": parsed.thinking_ratio,
            },
        )
        simulated_result["response"] = parsed.clean_response

    tokens_in = simulated_result.get("tokens_in", 100)
    tokens_out = simulated_result.get("tokens_out", 50)
    req_latency = simulated_result.get("latency_ms", elapsed_ms)

    logger_trace.log_model_call(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=req_latency,
        model_id=system_id,
        thinking_content=simulated_result.get("thinking_content"),
    )

    for tool_name in simulated_result.get("tools_called", []):
        logger_trace.log_event(
            TraceEventType.TOOL_CALL,
            {"tool_name": tool_name, "arguments": {}},
        )

    if simulated_result.get("retrieved_documents"):
        logger_trace.log_event(
            TraceEventType.RETRIEVAL_RESULT,
            {"documents": simulated_result["retrieved_documents"]},
        )

    # 3. Evaluation Phase (Gated short-circuit logic)
    eval_engine = evaluator or GatedEvaluator()
    verdict = await eval_engine.evaluate(task, simulated_result, logger_trace.traces)

    logger_trace.log_event(
        TraceEventType.JUDGE_DECISION,
        {
            "verdict": verdict.passed,
            "score": verdict.score,
            "reasoning": verdict.reasoning,
            "criteria": verdict.criteria,
            "judge_id": verdict.judge_id,
            "metadata": verdict.metadata,
        },
    )

    return verdict.passed, logger_trace.traces


async def _model_execute(
    task: Task,
    model: ModelAdapter,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Execute a task using a real ModelAdapter."""
    try:
        response: ModelResponse = await model.generate(
            task.input_messages,
            temperature=0.0,
            max_tokens=4096,
            seed=seed,
        )
        return {
            "response": response.content,
            "final_state": {},
            "tools_called": [tc.get("name", "") for tc in response.tool_calls],
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "latency_ms": response.latency_ms,
            "thinking_content": response.thinking_content,
        }
    except Exception as e:
        logger.error("model_execution_error", task_id=task.task_id, error=str(e))
        return {
            "response": f"[ERROR] Model execution failed: {e}",
            "final_state": {},
            "tools_called": [],
        }


def _stub_execute(task: Task, system_id: str) -> dict[str, Any]:
    """Stub execution — simulates system behavior based on task metadata."""
    if "happy_path" in task.tags:
        result: dict[str, Any] = {
            "response": _generate_stub_response(task),
            "final_state": task.expected_final_state,
            "tools_called": task.allowed_tools,
        }
        if task.domain == "investment_advisor" and "retrieval" in task.required_capabilities:
            result["retrieved_documents"] = _stub_retrieval_docs(task)
        return result
    elif "refusal" in task.tags:
        return {
            "response": _generate_refusal_response(task),
            "final_state": {},
            "refusal": True,
            "tools_called": [],
        }
    elif "edge_case" in task.tags:
        result = {
            "response": f"[STUB] Task {task.task_id} edge case handled.",
            "final_state": task.expected_final_state,
            "tools_called": task.allowed_tools[:2],
        }
        if "retrieval" in task.required_capabilities:
            result["retrieved_documents"] = _stub_retrieval_docs(task)
        return result
    else:
        return {
            "response": f"[STUB] Task {task.task_id} partial execution.",
            "final_state": {},
            "tools_called": [],
        }


def _generate_stub_response(task: Task) -> str:
    """Generate a plausible stub response for happy path tasks."""
    if task.domain == "investment_advisor":
        numeric_values = task.metadata.get("expected_numeric_values", {})
        parts = [f"[STUB] Task {task.task_id} completed."]
        for key, val in numeric_values.items():
            parts.append(f"{key}: {val}")
        parts.append("[Fonte: manual_produtos_rf_v3]")
        return " ".join(parts)
    return f"[STUB] Task {task.task_id} executed successfully."


def _generate_refusal_response(task: Task) -> str:
    """Generate a refusal response stub."""
    if "guarantee_prohibition" in task.tags:
        return "Não é possível garantir retornos em investimentos. Todo investimento envolve riscos."
    if "concentration_violation" in task.tags:
        return "Não posso recomendar concentração de 100% em um único ativo. A política limita a 50%."
    if "insider_trading" in task.tags:
        return "Não tenho e não posso fornecer informações privilegiadas. Isso viola regulamentações."
    if "suitability_mismatch" in task.tags:
        return "Criptoativos não são adequados para seu perfil conservador. Considere Tesouro Selic."
    return "[STUB] I cannot perform this action due to policy restrictions."


def _stub_retrieval_docs(task: Task) -> list[dict[str, Any]]:
    """Return stub retrieved documents for investment domain."""
    return [
        {
            "doc_id": "inv_001",
            "title": "CDB Pós-fixado - Características",
            "content": (
                "O CDB pós-fixado rende com base no CDI. Rentabilidade típica: 100-120% CDI. "
                "Garantido pelo FGC até R$250.000 por CPF/instituição."
            ),
            "source": "manual_produtos_rf_v3",
            "relevance_score": 0.9,
        },
        {
            "doc_id": "inv_002",
            "title": "Suitability - Classificação de Perfil",
            "content": (
                "Perfil Conservador: até 20% em renda variável. "
                "Perfil Moderado: até 40% em renda variável."
            ),
            "source": "politica_suitability_v2",
            "relevance_score": 0.95,
        },
    ]


async def run_single_case(
    task_id: str, system_id: str, domain: str, config: BenchConfig, output_dir: Path
) -> bool:
    """Run a single case by task_id lookup."""
    tasks = load_domain_tasks(domain)
    task = next((t for t in tasks if t.task_id == task_id), None)
    if not task:
        logger.error("task_not_found", task_id=task_id, domain=domain)
        return False
    success, _ = await execute_task(task, system_id, config)
    return success
