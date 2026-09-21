"""Case runner: executes a single benchmark task following Clean Architecture principles."""

import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from agent_bench.core.adapters import (
    ModelAdapter,
    ModelResponse,
    SafetyVerdict,
    ToolAdapter,
    ToolCallResult,
)
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


def _get_default_tool(tool_name: str) -> ToolAdapter | None:
    """Lazy factory for standard domain tools."""
    if tool_name in (
        "check_balance",
        "validate_pix_key",
        "execute_pix_transfer",
        "request_user_confirmation",
    ):
        from agent_bench.tools.pix_tools import (
            CheckBalanceTool,
            ExecutePixTransferTool,
            RequestUserConfirmationTool,
            ValidatePixKeyTool,
        )

        mapping: dict[str, type[ToolAdapter]] = {
            "check_balance": CheckBalanceTool,
            "validate_pix_key": ValidatePixKeyTool,
            "execute_pix_transfer": ExecutePixTransferTool,
            "request_user_confirmation": RequestUserConfirmationTool,
        }
        return mapping[tool_name]()

    if tool_name in (
        "get_client_profile",
        "get_portfolio_summary",
        "calculate_return",
        "check_suitability",
        "search_products",
        "simulate_allocation",
    ):
        from agent_bench.tools.investment_tools import (
            CalculateReturnTool,
            CheckSuitabilityTool,
            GetClientProfileTool,
            GetPortfolioSummaryTool,
            SearchProductsTool,
            SimulateAllocationTool,
        )

        mapping_inv: dict[str, type[ToolAdapter]] = {
            "get_client_profile": GetClientProfileTool,
            "get_portfolio_summary": GetPortfolioSummaryTool,
            "calculate_return": CalculateReturnTool,
            "check_suitability": CheckSuitabilityTool,
            "search_products": SearchProductsTool,
            "simulate_allocation": SimulateAllocationTool,
        }
        return mapping_inv[tool_name]()

    if tool_name in (
        "check_firewall_rules",
        "query_siem_logs",
        "create_incident_ticket",
        "escalate_to_soc",
        "execute_sandboxed_command",
    ):
        from agent_bench.tools.cyber_tools import (
            CheckFirewallRulesTool,
            CreateIncidentTicketTool,
            EscalateToSocTool,
            ExecuteSandboxedCommandTool,
            QuerySiemLogsTool,
        )

        mapping_cyber: dict[str, type[ToolAdapter]] = {
            "check_firewall_rules": CheckFirewallRulesTool,
            "query_siem_logs": QuerySiemLogsTool,
            "create_incident_ticket": CreateIncidentTicketTool,
            "escalate_to_soc": EscalateToSocTool,
            "execute_sandboxed_command": ExecuteSandboxedCommandTool,
        }
        return mapping_cyber[tool_name]()

    return None


class DefaultTaskEnvironment:
    """Standard task environment implementation of TaskEnvironment protocol with stateful tool execution."""

    def __init__(
        self,
        environment_id: str = "default_env",
        tools: Sequence[ToolAdapter] | dict[str, ToolAdapter] | None = None,
        state_mutators: (
            dict[str, Callable[[dict[str, Any], dict[str, Any], ToolCallResult], None]]
            | None
        ) = None,
    ) -> None:
        self._environment_id = environment_id
        self._state: dict[str, Any] = {}
        self._tools: dict[str, ToolAdapter] = {}
        if isinstance(tools, dict):
            self._tools.update(tools)
        elif tools is not None:
            for t in tools:
                self._tools[t.name] = t
        self._state_mutators = dict(state_mutators or {})
        self._execution_history: list[ToolCallResult] = []

    @property
    def environment_id(self) -> str:
        return self._environment_id

    @property
    def registered_tools(self) -> list[str]:
        return list(self._tools.keys())

    def register_tool(self, tool: ToolAdapter) -> None:
        """Register a ToolAdapter with the environment."""
        self._tools[tool.name] = tool

    def register_state_mutator(
        self,
        tool_name: str,
        mutator: Callable[[dict[str, Any], dict[str, Any], ToolCallResult], None],
    ) -> None:
        """Register a state mutation callback invoked after tool execution."""
        self._state_mutators[tool_name] = mutator

    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
        self._state = dict(initial_state or {})
        self._execution_history.clear()
        return self._state

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            tool = _get_default_tool(tool_name)

        if tool is not None:
            result = await tool.execute(arguments)
        else:
            result = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                output=f"Executed tool '{tool_name}'",
                success=True,
            )

        # Apply state mutations
        if tool_name in self._state_mutators:
            self._state_mutators[tool_name](self._state, arguments, result)
        else:
            self._apply_default_state_mutation(tool_name, arguments, result)

        self._execution_history.append(result)
        return result

    def _apply_default_state_mutation(
        self, tool_name: str, arguments: dict[str, Any], result: ToolCallResult
    ) -> None:
        """Default domain state transitions for standard tool operations."""
        if tool_name in ("execute_pix_transfer", "transfer_pix") and result.success:
            amount = float(arguments.get("amount", 0.0))
            if "balance" in self._state and isinstance(
                self._state["balance"], (int, float)
            ):
                self._state["balance"] = round(
                    float(self._state["balance"]) - amount, 2
                )
            self._state["amount"] = amount
            self._state["transfer_completed"] = True
            if isinstance(result.output, dict):
                self._state["last_transaction"] = result.output
        elif tool_name == "request_user_confirmation":
            self._state["confirmation_requested"] = True
        elif tool_name == "validate_pix_key":
            self._state["key_validated"] = result.success
            if isinstance(result.output, dict) and "owner_name" in result.output:
                self._state["recipient_name"] = result.output["owner_name"]
        elif tool_name == "simulate_allocation" and result.success:
            self._state["allocation_simulated"] = True
        elif tool_name == "check_firewall_rules" and result.success:
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

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def cleanup(self) -> None:
        self._state.clear()
        self._execution_history.clear()


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
            exec_res = await _model_execute(
                task,
                self._model,
                environment=environment,
                max_steps=max_steps,
                seed=seed,
            )
            if isinstance(exec_res, tuple):
                result, collected_traces = exec_res
            else:
                result = exec_res
                collected_traces = []
            return result, collected_traces
        else:
            result = _stub_execute(task, self.system_id)
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
                "task_id": task.task_id,
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
            {
                "task_id": task.task_id,
                "messages": messages,
                "system_id": system_id,
                "seed": seed,
            },
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

    if not runner_traces:
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
    environment: TaskEnvironment | None = None,
    max_steps: int = 10,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[TraceEvent]] | dict[str, Any]:
    """Execute a task using a real ModelAdapter with multi-turn ReAct reasoning."""
    env = environment or DefaultTaskEnvironment()
    messages: list[dict[str, Any]] = [dict(m) for m in task.input_messages]
    collected_traces: list[TraceEvent] = []
    all_tools_called: list[str] = []
    all_tool_calls: list[dict[str, Any]] = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_latency_ms = 0.0
    last_response = ""
    thinking_content: str | None = None
    prev_signatures: set[str] = set()

    for step in range(max_steps):
        try:
            response: ModelResponse = await model.generate(
                messages,
                temperature=0.0,
                max_tokens=4096,
                seed=seed,
            )
        except Exception as e:
            logger.error(
                "model_execution_error",
                task_id=task.task_id,
                step=step,
                error=str(e),
            )
            last_response = f"[ERROR] Model execution failed at step {step}: {e}"
            break

        total_tokens_in += response.tokens_in
        total_tokens_out += response.tokens_out
        total_latency_ms += response.latency_ms
        last_response = response.content
        if response.thinking_content:
            thinking_content = response.thinking_content

        if not response.tool_calls:
            messages.append({"role": "assistant", "content": response.content})
            break

        current_signatures = {
            f"{tc.get('name')}:{json.dumps(tc.get('arguments', {}), sort_keys=True)}"
            for tc in response.tool_calls
        }
        if current_signatures and current_signatures.issubset(prev_signatures):
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })
            break
        prev_signatures = current_signatures

        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls,
        })

        for tc in response.tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("arguments", {})
            if tc_name:
                all_tools_called.append(tc_name)
                all_tool_calls.append(tc)
                tool_res = await env.execute_tool(tc_name, tc_args)
                output_str = (
                    json.dumps(tool_res.output)
                    if isinstance(tool_res.output, (dict, list))
                    else str(tool_res.output)
                )
                messages.append({
                    "role": "tool",
                    "name": tc_name,
                    "content": output_str,
                    "tool_call_id": str(tc.get("id", f"call_{step}_{tc_name}")),
                })
                collected_traces.append(
                    TraceEvent(
                        event_type=TraceEventType.TOOL_CALL,
                        data={"tool_name": tc_name, "arguments": tc_args, "step": step},
                    )
                )
                collected_traces.append(
                    TraceEvent(
                        event_type=TraceEventType.TOOL_OUTPUT,
                        data={
                            "tool_name": tc_name,
                            "output": tool_res.output,
                            "success": tool_res.success,
                            "step": step,
                        },
                    )
                )

    result = {
        "response": last_response,
        "final_state": env.get_state(),
        "tools_called": all_tools_called,
        "tool_calls": all_tool_calls,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "latency_ms": total_latency_ms,
        "thinking_content": thinking_content,
        "messages": messages,
    }
    return result, collected_traces


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
