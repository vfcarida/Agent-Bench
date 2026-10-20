"""Causal Counterfactual Trajectory Replay (`CounterfactualSandbox`).

Forks recorded `TraceEvent` trajectories at a specified decision step `t` and applies
controlled counterfactual interventions (e.g., overriding a tool's output to simulate
a fraud alert, mutating environment state, or injecting an adversarial user message)
without re-executing turns `< t`.

Grounded in:
- Counterfactual Off-Policy Evaluation (OPE) for Sequential Decision Processes
- Causal Guardrail Attribution in Stateful Agent Benchmarks
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.protocols import AgentRunner, Evaluator, TaskEnvironment
from agent_bench.core.scenarios import Task
from agent_bench.graders.gated_evaluator import GatedEvaluator
from agent_bench.runners.case_runner import DefaultTaskEnvironment

InterventionType = Literal[
    "override_tool_output",
    "mutate_state",
    "inject_user_turn",
    "override_response",
]


@dataclass
class CounterfactualIntervention:
    """Specification of a counterfactual intervention applied at `fork_step`."""

    fork_step: int = 1
    intervention_type: InterventionType = "mutate_state"
    target_tool: str | None = None
    replacement_output: Any = None
    state_mutations: dict[str, Any] = field(default_factory=dict)
    injected_user_message: str | None = None
    replacement_response: str | None = None


@dataclass
class CounterfactualReplayResult:
    """Comparison between original factual trajectory and counterfactual fork."""

    task_id: str
    fork_step: int
    intervention_type: str
    original_passed: bool
    counterfactual_passed: bool
    original_score: float
    counterfactual_score: float
    original_safety_violated: bool
    counterfactual_safety_violated: bool
    causal_effect_score: float
    outcome_flipped: bool
    safety_flipped: bool
    replayed_state: dict[str, Any] = field(default_factory=dict)
    counterfactual_traces: list[TraceEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["counterfactual_traces"] = [
            {"event_type": t.event_type.value, "data": t.data}
            for t in self.counterfactual_traces
        ]
        return data


class CounterfactualSandbox:
    """Deterministic trajectory replayer and counterfactual intervention sandbox."""

    def __init__(
        self,
        environment: TaskEnvironment | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self._environment = environment or DefaultTaskEnvironment(
            environment_id="counterfactual_sandbox"
        )
        self._evaluator = evaluator or GatedEvaluator()

    @staticmethod
    def _normalize_traces(
        raw_traces: Sequence[TraceEvent | dict[str, Any]],
    ) -> list[TraceEvent]:
        normalized: list[TraceEvent] = []
        for item in raw_traces:
            if isinstance(item, TraceEvent):
                normalized.append(item)
            elif isinstance(item, dict):
                raw_type = str(item.get("event_type", TraceEventType.AGENT_STEP.value))
                try:
                    ev_type = TraceEventType(raw_type)
                except ValueError:
                    ev_type = TraceEventType.AGENT_STEP
                normalized.append(
                    TraceEvent(
                        event_type=ev_type,
                        data=dict(item.get("data", item)),
                    )
                )
        return normalized

    def reconstruct_prefix(
        self,
        task: Task,
        traces: Sequence[TraceEvent | dict[str, Any]],
        fork_step: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[TraceEvent]]:
        """Reconstruct environment state, messages, and tool calls up to `fork_step`."""
        norm_traces = self._normalize_traces(traces)
        self._environment.reset(task.initial_state)

        messages: list[dict[str, Any]] = [dict(m) for m in task.input_messages]
        tools_called: list[str] = []
        prefix_traces: list[TraceEvent] = []

        step_counter = 0
        for tr in norm_traces:
            ev_step = tr.data.get("step")
            if isinstance(ev_step, int):
                current_step = ev_step
            else:
                if tr.event_type in (
                    TraceEventType.TOOL_CALL,
                    TraceEventType.MODEL_CALL,
                    TraceEventType.USER_ACTION,
                ):
                    step_counter += 1
                current_step = step_counter

            if current_step > fork_step:
                break

            prefix_traces.append(tr)

            if tr.event_type == TraceEventType.TOOL_CALL:
                t_name = str(tr.data.get("tool_name", tr.data.get("tool", "")))
                t_args = tr.data.get("arguments", {})
                if not isinstance(t_args, dict):
                    t_args = {}
                if t_name:
                    tools_called.append(t_name)
                    if hasattr(self._environment, "_apply_default_state_mutation"):
                        from agent_bench.core.adapters import ToolCallResult

                        dummy_res = ToolCallResult(
                            tool_name=t_name,
                            arguments=t_args,
                            output=tr.data.get("output", "replayed"),
                            success=True,
                        )
                        self._environment._apply_default_state_mutation(
                            t_name, t_args, dummy_res
                        )
            elif tr.event_type == TraceEventType.USER_ACTION:
                mutations = tr.data.get("state_mutations", {})
                if isinstance(mutations, dict) and mutations:
                    if hasattr(self._environment, "update_state"):
                        self._environment.update_state(mutations)
            elif tr.event_type == TraceEventType.USER_MESSAGE:
                content = str(tr.data.get("content", ""))
                if content:
                    messages.append({"role": "user", "content": content})
            elif tr.event_type == TraceEventType.MODEL_RESPONSE:
                content = str(tr.data.get("content", ""))
                if content:
                    messages.append({"role": "assistant", "content": content})

        return self._environment.get_state(), messages, tools_called, prefix_traces

    async def replay_with_intervention(
        self,
        task: Task,
        traces: Sequence[TraceEvent | dict[str, Any]],
        intervention: CounterfactualIntervention,
        *,
        original_execution_result: dict[str, Any] | None = None,
        continuation_runner: AgentRunner | None = None,
    ) -> CounterfactualReplayResult:
        """Fork trajectory at `intervention.fork_step`, apply intervention, and evaluate causal delta."""
        norm_traces = self._normalize_traces(traces)

        # 1. Evaluate factual (original) trajectory
        if original_execution_result is None:
            orig_tools = [
                str(t.data.get("tool_name", t.data.get("tool", "")))
                for t in norm_traces
                if t.event_type == TraceEventType.TOOL_CALL
                and t.data.get("tool_name", t.data.get("tool", ""))
            ]
            orig_resp = ""
            for t in reversed(norm_traces):
                if t.event_type == TraceEventType.MODEL_RESPONSE:
                    orig_resp = str(t.data.get("content", ""))
                    break
            orig_state, _, _, _ = self.reconstruct_prefix(task, norm_traces, fork_step=999999)
            original_execution_result = {
                "response": orig_resp or "Completed task execution.",
                "final_state": orig_state,
                "tools_called": orig_tools,
                "refusal": False,
            }

        orig_verdict = await self._evaluator.evaluate(
            task, original_execution_result, norm_traces
        )
        orig_safety_violated = bool(orig_verdict.metadata.get("safety_violation", False))
        orig_passed = orig_verdict.passed and not orig_safety_violated

        # 2. Reconstruct state & context at fork_step
        prefix_state, messages, prefix_tools, cf_traces = self.reconstruct_prefix(
            task, norm_traces, intervention.fork_step
        )
        cf_state = dict(prefix_state)
        cf_tools = list(prefix_tools)
        cf_response = str(original_execution_result.get("response", ""))

        # 3. Apply the counterfactual intervention at fork_step
        if intervention.intervention_type == "mutate_state":
            cf_state.update(intervention.state_mutations)
            if hasattr(self._environment, "update_state"):
                self._environment.update_state(intervention.state_mutations)
            cf_traces.append(
                TraceEvent(
                    event_type=TraceEventType.USER_ACTION,
                    data={
                        "actor": "counterfactual_sandbox",
                        "intervention_type": "mutate_state",
                        "fork_step": intervention.fork_step,
                        "state_mutations": dict(intervention.state_mutations),
                    },
                )
            )
        elif intervention.intervention_type == "override_tool_output":
            t_name = intervention.target_tool or (cf_tools[-1] if cf_tools else "overridden_tool")
            if t_name not in cf_tools:
                cf_tools.append(t_name)
            if isinstance(intervention.replacement_output, dict):
                cf_state.update(intervention.replacement_output)
                if hasattr(self._environment, "update_state"):
                    self._environment.update_state(intervention.replacement_output)
            cf_traces.append(
                TraceEvent(
                    event_type=TraceEventType.TOOL_OUTPUT,
                    data={
                        "tool_name": t_name,
                        "output": intervention.replacement_output,
                        "counterfactual_override": True,
                        "fork_step": intervention.fork_step,
                    },
                )
            )
        elif intervention.intervention_type == "inject_user_turn":
            msg_text = intervention.injected_user_message or ""
            messages.append({"role": "user", "content": msg_text})
            if intervention.state_mutations:
                cf_state.update(intervention.state_mutations)
                if hasattr(self._environment, "update_state"):
                    self._environment.update_state(intervention.state_mutations)
            cf_traces.append(
                TraceEvent(
                    event_type=TraceEventType.USER_MESSAGE,
                    data={
                        "role": "user",
                        "content": msg_text,
                        "counterfactual_override": True,
                        "fork_step": intervention.fork_step,
                    },
                )
            )
        elif intervention.intervention_type == "override_response":
            if intervention.replacement_response is not None:
                cf_response = intervention.replacement_response
            if intervention.state_mutations:
                cf_state.update(intervention.state_mutations)
                if hasattr(self._environment, "update_state"):
                    self._environment.update_state(intervention.state_mutations)
            cf_traces.append(
                TraceEvent(
                    event_type=TraceEventType.MODEL_RESPONSE,
                    data={
                        "content": cf_response,
                        "counterfactual_override": True,
                        "fork_step": intervention.fork_step,
                    },
                )
            )

        # 4. Optionally invoke continuation_runner from the forked state
        if continuation_runner is not None:
            forked_task = Task(
                task_id=task.task_id,
                domain=task.domain,
                name=task.name,
                description=task.description,
                input_messages=messages,
                initial_state=cf_state,
                expected_final_state=task.expected_final_state,
                allowed_tools=task.allowed_tools,
                expected_refusal_mode=task.expected_refusal_mode,
                severity=task.severity,
                metadata=task.metadata,
            )
            cont_result, cont_traces = await continuation_runner.run_task(
                forked_task, self._environment
            )
            cf_traces.extend(cont_traces)
            cf_state = dict(cont_result.get("final_state", self._environment.get_state()))
            for t_called in cont_result.get("tools_called", []):
                if t_called not in cf_tools:
                    cf_tools.append(t_called)
            if cont_result.get("response"):
                cf_response = str(cont_result["response"])
            cf_exec_result = {
                **cont_result,
                "response": cf_response,
                "final_state": cf_state,
                "tools_called": cf_tools,
            }
        else:
            cf_exec_result = {
                **original_execution_result,
                "response": cf_response,
                "final_state": cf_state,
                "tools_called": cf_tools,
            }

        # 5. Evaluate counterfactual outcome via GatedEvaluator (Phase 0 Safety + Phase 1/2)
        cf_verdict = await self._evaluator.evaluate(task, cf_exec_result, cf_traces)
        cf_safety_violated = bool(cf_verdict.metadata.get("safety_violation", False))
        cf_passed = cf_verdict.passed and not cf_safety_violated

        orig_effective_score = 0.0 if orig_safety_violated else float(orig_verdict.score)
        cf_effective_score = 0.0 if cf_safety_violated else float(cf_verdict.score)
        causal_delta = round(cf_effective_score - orig_effective_score, 6)

        return CounterfactualReplayResult(
            task_id=task.task_id,
            fork_step=intervention.fork_step,
            intervention_type=intervention.intervention_type,
            original_passed=orig_passed,
            counterfactual_passed=cf_passed,
            original_score=round(orig_effective_score, 6),
            counterfactual_score=round(cf_effective_score, 6),
            original_safety_violated=orig_safety_violated,
            counterfactual_safety_violated=cf_safety_violated,
            causal_effect_score=causal_delta,
            outcome_flipped=(orig_passed != cf_passed),
            safety_flipped=(orig_safety_violated != cf_safety_violated),
            replayed_state=cf_state,
            counterfactual_traces=cf_traces,
        )
