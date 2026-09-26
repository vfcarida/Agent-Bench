# Python API Reference

Agent-Bench provides a modular, Clean Architecture-driven Python API structured around `@runtime_checkable` Python Protocols.

---

## 1. Core Protocols (`agent_bench.core.protocols`)

### `TaskEnvironment`
Manages state snapshotting, tool execution, and cleanup for an agent's task lifecycle.

```python
from agent_bench.core.protocols import TaskEnvironment

class CustomEnvironment(TaskEnvironment):
    async def reset(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Initialize or reset the environment state."""
        ...

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Dispatch a tool call against the environment state machine."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Return the current environment state snapshot."""
        ...

    async def close(self) -> None:
        """Release underlying system resources (Docker, subshells, sockets)."""
        ...
```

### `AgentRunner`
Drives an agent system against a `TaskEnvironment` up to `max_steps`.

```python
from agent_bench.core.protocols import AgentRunner

class CustomAgentRunner(AgentRunner):
    async def run_task(
        self,
        task: Task,
        env: TaskEnvironment,
        tools: list[ToolAdapter] | None = None,
        max_steps: int = 10,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], list[TraceEvent]]:
        """Execute multi-turn ReAct reasoning loop."""
        ...
```

### `Evaluator`
Performs multi-phase grading over task outputs and environment state diffs.

```python
from agent_bench.core.protocols import Evaluator

class CustomEvaluator(Evaluator):
    async def evaluate(
        self,
        task: Task,
        result: dict[str, Any],
        actual_state: dict[str, Any],
        traces: list[TraceEvent],
    ) -> EvalResult:
        """Evaluate agent run against gold specifications."""
        ...
```

---

## 2. Model Adapters (`agent_bench.models`)

All model adapters subclass [`ModelAdapter`](file:///c:/Users/vinicius/Documents/GeminiCodes/Agent-Bench/src/agent_bench/core/adapters.py) and provide consistent `generate()` signatures with token and latency accounting.

### Available Adapters:
- `OpenAIModelAdapter`: Direct asynchronous OpenAI API completions (`gpt-4o`, `gpt-4o-mini`, etc.).
- `AnthropicModelAdapter`: Direct Anthropic Messages API completions (`claude-3-5-sonnet-20241022`, etc.).
- `HuggingFacePipelineAdapter`: Local causal language model inference with tool prompting and extraction.
- `VLLMModelAdapter`: High-throughput local GPU inference via vLLM engine.
- `StubModelAdapter`: Zero-cost deterministic mock responses for CI and dry runs.

---

## 3. Runners (`agent_bench.runners`)

- `CaseRunner`: Executes a single evaluation case or task.
- `SuiteRunner`: Executes full benchmark suites with bounded async concurrency (`--concurrency`), bootstrap confidence intervals, and scorecard persistence.
- `CallableAgentRunner`: Universal bridge enabling LangChain, CrewAI, LangGraph, or custom Python agent functions to plug into Agent-Bench.
- `ScriptedAgentRunner`: Baseline deterministic rule engine for sanity checks.

---

## 4. Graders & Judges (`agent_bench.graders`, `agent_bench.judges`)

- `SafetyGate`: Zero-tolerance non-compensable refusal and forbidden action checker (Phase 0).
- `StateGrader`: Evaluates expected state transitions, balances, and numeric values with tolerance.
- `RubricGrader`: Evaluates weighted qualitative rubrics with critical gating.
- `ToolCallGrader`: Validates exact or subset matching of invoked tool signatures.
- `SemanticJudge`: LLM-as-a-judge qualitative evaluator with prompt injection defenses.
- `GroundingJudge`: Verifies factual claims against explicit evidence strings (`hallucination_rate`).

---

## 5. Metrics (`agent_bench.metrics`)

- `compute_pass_hat_k(results, k)`: Unbiased per-task $\text{Pass}@k$ combinatorial estimator.
- `compute_pass_k(results, k)`: High-risk $\text{Pass}^k$ metric measuring consistent success across all $k$ trials.
- `compute_bootstrap_ci(values, n_resamples=1000)`: Non-parametric 95% bootstrap confidence intervals.
- `compute_scorecard(...)`: Multi-dimensional reliability report across functional, safety, latency, and cost axes.
