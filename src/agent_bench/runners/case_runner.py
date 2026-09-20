"""Case runner: executes a single benchmark task following Clean Architecture principles."""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.adapters import ModelAdapter, ModelResponse, SafetyVerdict, ToolCallResult
from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig
from agent_bench.core.protocols import AgentRunner, Evaluator, TaskEnvironment
from agent_bench.core.scenarios import Task
from agent_bench.core.settings import settings
from agent_bench.datasets.loader import load_domain_tasks
from agent_bench.graders.gated_evaluator import GatedEvaluator
from agent_bench.graders.thinking_parser import parse_thinking_response
from agent_bench.runners.prompt_formatter import PromptFormatter
from agent_bench.storage.trace_logger import ExecutionTraceLogger

logger = structlog.get_logger()


@dataclass
class CaseResult:
    """Result of executing a single task with real measured metrics."""

    passed: bool
    traces: list[TraceEvent]
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    safety_violated: bool = False
    safety_verdict: SafetyVerdict | None = None

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking as (passed, traces) for backwards compatibility."""
        return iter((self.passed, self.traces))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.passed, self.traces)[index]


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
) -> CaseResult:
    """Execute a single task and return CaseResult with real measured metrics.

    Supports both legacy direct model execution and decoupled Clean Architecture execution
    via AgentRunner, TaskEnvironment, and Evaluator protocols.
    """
    # 0. Resolve model pricing from config (falling back to settings defaults with warning)
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None

    sys_cfg = next((s for s in config.systems if s.system_id == system_id), None)
    model_id = sys_cfg.model if sys_cfg else None

    if model_id:
        model_cfg = next((m for m in config.models if m.model_id == model_id), None)
        if model_cfg is not None:
            cost_per_1k_input = model_cfg.price_per_1k_input
            cost_per_1k_output = model_cfg.price_per_1k_output

    if cost_per_1k_input is None or cost_per_1k_output is None:
        logger.warning(
            "pricing_fallback_to_defaults",
            system_id=system_id,
            model_id=model_id,
            cost_per_1k_input=settings.cost_per_1k_input_tokens,
            cost_per_1k_output=settings.cost_per_1k_output_tokens,
        )
        if cost_per_1k_input is None:
            cost_per_1k_input = settings.cost_per_1k_input_tokens
        if cost_per_1k_output is None:
            cost_per_1k_output = settings.cost_per_1k_output_tokens

    logger_trace = ExecutionTraceLogger(
        run_id=run_id,
        task_id=task.task_id,
        system_id=system_id,
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
    )

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

    tokens_in = simulated_result.get("tokens_in")
    tokens_out = simulated_result.get("tokens_out")
    req_latency = simulated_result.get("latency_ms")

    if tokens_in is None:
        tokens_in = sum(
            len(str(m.get("content", ""))) // 4
            for m in messages
            if isinstance(m, dict)
        )
    if tokens_out is None:
        tokens_out = len(response_text) // 4
    if req_latency is None:
        req_latency = elapsed_ms

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

    is_safety_violated = bool(verdict.metadata.get("safety_violation", False))
    return CaseResult(
        passed=verdict.passed and not is_safety_violated,
        traces=logger_trace.traces,
        latency_ms=logger_trace.total_latency_ms,
        tokens_in=logger_trace.tokens_in,
        tokens_out=logger_trace.tokens_out,
        cost_usd=logger_trace.total_cost_usd,
        safety_violated=is_safety_violated,
        safety_verdict=verdict.metadata.get("safety_verdict"),
    )


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
    """Honest domain-agnostic stub execution (smoke agent).

    This stub performs minimal smoke execution for pipeline connectivity and harness verification.
    It strictly does NOT read task.expected_final_state, task.allowed_tools,
    task.expected_refusal_mode, or expected_numeric_values from task metadata.
    It returns a fixed benign response, empty final state, and no tool calls.
    It peeks at no answer labels or gold references; refusal and state-check tasks
    fail honestly under this baseline smoke policy.
    """
    return {
        "response": "I need more information to proceed safely.",
        "final_state": {},
        "tools_called": [],
        "refusal": False,
    }


async def run_single_case(
    task_id: str,
    system_id: str,
    domain: str,
    config: BenchConfig,
    output_dir: Path,
    split: str = "dev",
) -> bool:
    """Run a single case by task_id lookup."""
    tasks = load_domain_tasks(domain, split=split)
    task = next((t for t in tasks if t.task_id == task_id), None)
    if not task:
        logger.error("task_not_found", task_id=task_id, domain=domain)
        return False
    from agent_bench.models.factory import build_agent_runner

    runner = build_agent_runner(system_id, config)
    case_res = await execute_task(task, system_id, config, agent_runner=runner)
    return case_res.passed
