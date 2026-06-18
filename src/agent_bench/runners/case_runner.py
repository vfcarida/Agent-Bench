"""Case runner: executes a single benchmark task."""

from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.adapters import JudgeVerdict, ModelAdapter, ModelResponse
from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig
from agent_bench.core.scenarios import Task
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.graders.thinking_parser import parse_thinking_response
from agent_bench.judges.composite import CompositeJudge
from agent_bench.runners.prompt_formatter import PromptFormatter

logger = structlog.get_logger()


async def execute_task(
    task: Task,
    system_id: str,
    config: BenchConfig,
    *,
    seed: int | None = None,
    model: ModelAdapter | None = None,
    prompt_formatter: PromptFormatter | None = None,
) -> tuple[bool, list[TraceEvent]]:
    """Execute a single task and return (success, traces).

    When a ModelAdapter is provided, sends the task to the real model.
    Otherwise falls back to stub execution for testing.

    When a PromptFormatter is provided, formats the task messages
    using the model-specific template before sending.
    """
    traces: list[TraceEvent] = []

    # Optionally format messages with model-specific template
    messages = task.input_messages
    if prompt_formatter is not None:
        formatted_prompt = prompt_formatter.format_messages(messages)
        traces.append(TraceEvent(
            event_type=TraceEventType.PROMPT_SENT,
            data={
                "messages": messages,
                "formatted_prompt": formatted_prompt,
                "template": prompt_formatter.template_name,
                "system_id": system_id,
                "seed": seed,
            },
        ))
    else:
        traces.append(TraceEvent(
            event_type=TraceEventType.PROMPT_SENT,
            data={"messages": messages, "system_id": system_id, "seed": seed},
        ))

    # Execute with real model or fall back to stub
    if model is not None:
        simulated_result = await _model_execute(task, model, seed=seed)
    else:
        simulated_result = _stub_execute(task, system_id)

    response_text = simulated_result.get("response", "")

    # Parse <think> blocks if present (SwiReasoning support)
    parsed = parse_thinking_response(response_text)
    if parsed.has_thinking:
        traces.append(TraceEvent(
            event_type=TraceEventType.THINKING_BLOCK,
            data={
                "thinking_blocks": parsed.thinking_blocks,
                "thinking_token_count": parsed.thinking_token_count,
                "answer_token_count": parsed.answer_token_count,
                "thinking_ratio": parsed.thinking_ratio,
            },
        ))
        # Use clean response (without <think> tags) for grading
        simulated_result["response"] = parsed.clean_response

    traces.append(TraceEvent(
        event_type=TraceEventType.MODEL_RESPONSE,
        data={
            "content": simulated_result.get("response", ""),
            "system_id": system_id,
            "had_thinking": parsed.has_thinking,
        },
    ))

    # Record tool calls if any
    for tool_name in simulated_result.get("tools_called", []):
        traces.append(TraceEvent(
            event_type=TraceEventType.TOOL_CALL,
            data={"tool_name": tool_name, "arguments": {}},
        ))

    # Record retrieval if present
    if simulated_result.get("retrieved_documents"):
        traces.append(TraceEvent(
            event_type=TraceEventType.RETRIEVAL_RESULT,
            data={"documents": simulated_result["retrieved_documents"]},
        ))

    # Composite judge
    judge = CompositeJudge()
    final_verdict, all_verdicts = await judge.evaluate(task, simulated_result, traces)

    traces.append(TraceEvent(
        event_type=TraceEventType.JUDGE_DECISION,
        data={
            "verdict": final_verdict.passed,
            "score": final_verdict.score,
            "reasoning": final_verdict.reasoning,
            "individual_verdicts": [
                {"judge_id": v.judge_id, "score": v.score, "passed": v.passed}
                for v in all_verdicts
            ],
        },
    ))

    return final_verdict.passed, traces


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
        # Simulate retrieval for investment tasks
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
