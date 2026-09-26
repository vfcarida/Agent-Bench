"""Callable agent runner bridging external frameworks (LangChain, CrewAI, custom functions).

Enables evaluating external autonomous agents by wrapping any synchronous or
asynchronous Python callable conforming to common agent invocation patterns.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Coroutine
from typing import Any

from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.protocols import AgentRunner, TaskEnvironment
from agent_bench.core.scenarios import Task


class EnvironmentProxy:
    """Proxy around TaskEnvironment that records tool call telemetry."""

    def __init__(self, target_env: TaskEnvironment) -> None:
        self._target_env = target_env
        self.recorded_tool_calls: list[str] = []
        self.recorded_events: list[TraceEvent] = []

    @property
    def environment_id(self) -> str:
        return getattr(self._target_env, "environment_id", "proxy_env")

    def get_state(self) -> dict[str, Any]:
        return self._target_env.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        if hasattr(self._target_env, "set_state"):
            self._target_env.set_state(state)
        elif hasattr(self._target_env, "_state"):
            self._target_env._state.update(state)  # type: ignore[union-attr]

    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._target_env.reset(initial_state)

    def cleanup(self) -> None:
        self._target_env.cleanup()

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.recorded_tool_calls.append(tool_name)
        call_event = TraceEvent(
            event_type=TraceEventType.TOOL_CALL,
            data={"tool_name": tool_name, "arguments": arguments},
        )
        self.recorded_events.append(call_event)

        try:
            result = await self._target_env.execute_tool(tool_name, arguments)
            out_event = TraceEvent(
                event_type=TraceEventType.TOOL_OUTPUT,
                parent_id=call_event.event_id,
                data={"tool_name": tool_name, "result": result},
            )
            self.recorded_events.append(out_event)
            return result
        except Exception as exc:
            err_event = TraceEvent(
                event_type=TraceEventType.ERROR,
                parent_id=call_event.event_id,
                data={"tool_name": tool_name, "error": str(exc)},
            )
            self.recorded_events.append(err_event)
            raise


class CallableAgentRunner(AgentRunner):
    """Bridge runner that adapts an arbitrary Python callable into an AgentRunner.

    Supported callable signatures:
    - ``fn(task, env)``: receives Task and wrapped TaskEnvironment
    - ``fn(prompt, tools, env)``: receives initial prompt string, allowed tools, and env
    - ``fn(messages, tools, env)``: receives input message dicts, allowed tools, and env
    - ``fn(task)``: receives Task object
    - ``fn(prompt)``: receives initial prompt string

    The callable may return either:
    - A ``str``: representing final agent text response
    - A ``dict[str, Any]``: containing fields such as 'response', 'final_state',
      'tokens_in', 'tokens_out', 'cost_usd', or 'safety_violation'.
    """

    def __init__(
        self,
        agent_fn: Callable[..., Any | Coroutine[Any, Any, Any]],
        name: str = "callable_agent",
        pass_environment: bool = True,
    ) -> None:
        self.agent_fn = agent_fn
        self.name = name
        self.pass_environment = pass_environment
        self._param_names = list(inspect.signature(agent_fn).parameters.keys())

    async def run(
        self,
        task: Task,
        environment: TaskEnvironment,
        max_steps: int = 10,
    ) -> tuple[dict[str, Any], list[TraceEvent]]:
        """Run the wrapped external agent callable on the task."""
        traces: list[TraceEvent] = []
        env_proxy = EnvironmentProxy(environment)

        prompt_str = task.input_messages[0]["content"] if task.input_messages else ""
        traces.append(
            TraceEvent(
                event_type=TraceEventType.USER_MESSAGE,
                data={"content": prompt_str, "task_id": task.task_id},
            )
        )

        start_time = time.perf_counter()
        raw_output: Any = None
        error_msg: str | None = None

        try:
            raw_output = await self._dispatch_call(task, prompt_str, env_proxy)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            traces.append(
                TraceEvent(
                    event_type=TraceEventType.ERROR,
                    data={"error": error_msg, "task_id": task.task_id},
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Incorporate tool events captured by proxy
        traces.extend(env_proxy.recorded_events)

        # Standardize return dictionary
        execution_result = self._build_execution_result(
            raw_output=raw_output,
            error_msg=error_msg,
            env_proxy=env_proxy,
            duration_ms=duration_ms,
        )

        traces.append(
            TraceEvent(
                event_type=TraceEventType.MODEL_RESPONSE,
                data={
                    "response": execution_result.get("response", ""),
                    "tools_called": execution_result.get("tools_called", []),
                },
            )
        )

        return execution_result, traces

    async def run_task(
        self,
        task: Task,
        environment: TaskEnvironment,
        *,
        max_steps: int = 10,
        seed: int | None = None,
        user_simulator: Any = None,
    ) -> tuple[dict[str, Any], list[TraceEvent]]:
        """Alias for execute_task runner compatibility."""
        return await self.run(task, environment, max_steps=max_steps)

    async def close(self) -> None:
        """Close runner resources if any."""
        pass

    async def __aenter__(self) -> CallableAgentRunner:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _dispatch_call(
        self,
        task: Task,
        prompt_str: str,
        env_proxy: EnvironmentProxy,
    ) -> Any:
        """Inspect signature and dispatch invocation with proper positional parameters."""
        num_params = len(self._param_names)

        # Dispatch based on parameter count and names
        if num_params == 1:
            first_param = self._param_names[0]
            if first_param in ("prompt", "input", "query", "user_goal"):
                res = self.agent_fn(prompt_str)
            else:
                res = self.agent_fn(task)
        elif num_params == 2:
            res = self.agent_fn(task, env_proxy)
        elif num_params >= 3:
            first_param = self._param_names[0]
            if first_param in ("messages", "history", "input_messages"):
                res = self.agent_fn(task.input_messages, task.allowed_tools, env_proxy)
            else:
                res = self.agent_fn(prompt_str, task.allowed_tools, env_proxy)
        else:
            res = self.agent_fn()

        if inspect.isawaitable(res):
            return await res
        return res

    def _build_execution_result(
        self,
        raw_output: Any,
        error_msg: str | None,
        env_proxy: EnvironmentProxy,
        duration_ms: float,
    ) -> dict[str, Any]:
        """Normalize raw output into a standard execution result dictionary."""
        final_state = env_proxy.get_state()

        if error_msg:
            return {
                "response": f"Execution failed: {error_msg}",
                "final_state": final_state,
                "tools_called": env_proxy.recorded_tool_calls,
                "latency_ms": duration_ms,
                "steps_taken": len(env_proxy.recorded_tool_calls) + 1,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "safety_violation": False,
                "error": error_msg,
            }

        if isinstance(raw_output, dict):
            resp_str = str(raw_output.get("response", raw_output.get("output", "")))
            tools_called = list(raw_output.get("tools_called", env_proxy.recorded_tool_calls))
            merged_state = {**final_state, **raw_output.get("final_state", {})}
            return {
                "response": resp_str,
                "final_state": merged_state,
                "tools_called": tools_called,
                "latency_ms": raw_output.get("latency_ms", duration_ms),
                "steps_taken": raw_output.get("steps_taken", len(tools_called) + 1),
                "tokens_in": int(raw_output.get("tokens_in", 0)),
                "tokens_out": int(raw_output.get("tokens_out", 0)),
                "cost_usd": float(raw_output.get("cost_usd", 0.0)),
                "safety_violation": bool(raw_output.get("safety_violation", False)),
                "safety_violation_reason": raw_output.get("safety_violation_reason"),
                "refusal": bool(raw_output.get("refusal", False)),
            }

        # Raw string output
        return {
            "response": str(raw_output or ""),
            "final_state": final_state,
            "tools_called": env_proxy.recorded_tool_calls,
            "latency_ms": duration_ms,
            "steps_taken": len(env_proxy.recorded_tool_calls) + 1,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "safety_violation": False,
        }
