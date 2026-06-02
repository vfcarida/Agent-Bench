"""Case runner: executes a single benchmark task."""

from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.adapters import JudgeVerdict
from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig
from agent_bench.core.scenarios import Task
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.judges.composite import CompositeJudge

logger = structlog.get_logger()


async def execute_task(
    task: Task,
    system_id: str,
    config: BenchConfig,
    *,
    seed: int | None = None,
) -> tuple[bool, list[TraceEvent]]:
    """Execute a single task and return (success, traces).

    Uses stub execution + composite judge (deterministic + numeric + grounding).
    """
    traces: list[TraceEvent] = []

    # Record the prompt
    traces.append(TraceEvent(
        event_type=TraceEventType.PROMPT_SENT,
        data={"messages": task.input_messages, "system_id": system_id, "seed": seed},
    ))

    # Execute (stub or real)
    simulated_result = _stub_execute(task, system_id)

    traces.append(TraceEvent(
        event_type=TraceEventType.MODEL_RESPONSE,
        data={"content": simulated_result.get("response", ""), "system_id": system_id},
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
