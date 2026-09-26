# Architecture Overview

**Agent-Bench** is architected around **Clean Architecture** principles using Python `@runtime_checkable` `Protocol` interfaces. This decouples the agent's reasoning loop, the environment state machine, and the evaluation judges.

---

## 1. High-Level System Architecture

```text
                     +---------------------------------------+
                     |           bench CLI / CI / API        |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |              SuiteRunner              |
                     |  (Async Concurrency Pool / Semaphore) |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |              CaseRunner               |
                     +---------------------------------------+
                                         |
        +--------------------------------+-------------------------------+
        |                                |                               |
        v                                v                               v
+----------------+              +-----------------+             +-----------------+
|  AgentRunner   | <---tools--- | TaskEnvironment |             |    Evaluator    |
| (Multi-turn    | ---action--> | (State Machine  |             | (Phase 0, 1, 2) |
|  ReAct loop)   |              |  & Sandboxing)  |             +-----------------+
+----------------+              +-----------------+                      |
        |                                                                v
        v                                                       +-----------------+
+----------------+                                              | Hard Safety Gate|
|  ModelAdapter  |                                              | State Diffs     |
| (OpenAI / Anth |                                              | Semantic Judge  |
|  vLLM / Local) |                                              +-----------------+
+----------------+
```

---

## 2. Core Protocol Interfaces

All primary contracts reside in [`agent_bench.core.protocols`](file:///src/agent_bench/core/protocols.py):

### 2.1 `TaskEnvironment`
Manages task execution state, tool definitions, and environment lifecycle:
```python
@runtime_checkable
class TaskEnvironment(Protocol):
    @property
    def environment_id(self) -> str: ...
    def reset(self, initial_state: dict[str, Any] | None = None) -> dict[str, Any]: ...
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult: ...
    def get_state(self) -> dict[str, Any]: ...
    def cleanup(self) -> None: ...
```

Implementations:
- `DefaultTaskEnvironment`: In-memory state machine with domain tool dispatch.
- `SandboxedTaskEnvironment`: Isolated execution environment using Docker containers or hardened sub-process fallbacks.

### 2.2 `AgentRunner`
Orchestrates an agent's execution against a task and environment up to `max_steps`:
```python
@runtime_checkable
class AgentRunner(Protocol):
    async def run(
        self,
        task: Task,
        environment: TaskEnvironment,
        max_steps: int = 10,
    ) -> tuple[dict[str, Any], list[TraceEvent]]: ...
```

Implementations:
- `DefaultAgentRunner`: Autonomous multi-turn ReAct model driver with stall detection.
- `ScriptedAgentRunner`: Deterministic rule-based baseline agent.
- `CallableAgentRunner`: Universal bridge wrapping external agents (LangChain, CrewAI, custom functions).

### 2.3 `Evaluator`
Assesses execution traces and environment state mutations:
```python
@runtime_checkable
class Evaluator(Protocol):
    async def evaluate(
        self,
        task: Task,
        execution_result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict: ...
```

Primary implementation:
- `GatedEvaluator`: Hierarchical short-circuiting evaluator implementing the 3-phase gating protocol.

---

## 3. The Three-Phase Gated Evaluation Pipeline

To guarantee rigor and minimize evaluation API costs, Agent-Bench uses a non-compensable 3-phase pipeline:

```text
Execution Output & Traces
           |
           v
+--------------------------------------------------+
| Phase 0: Hard Safety Gate (evaluate_safety)       |
| - Refusal compliance (required refusals)         |
| - Forbidden state actions (e.g. illegal transfer)|
| - Disallowed tool calls in restricted contexts   |
+--------------------------------------------------+
     |                                    |
  Violation                            Cleared
     v                                    v
+------------------------+   +---------------------------------------+
| Score: 0.0, Failed     |   | Phase 1: Deterministic Assertion Gate |
| Short-circuit bypass   |   | - State equality & balance checks     |
+------------------------+   | - Regex & tool call sequence checks   |
                             +---------------------------------------+
                                  |                         |
                                Failed                   Passed
                                  v                         v
                             +--------------------+   +---------------------+
                             | Score: 0.0, Failed |   | Phase 2: LLM Judge  |
                             | Short-circuit      |   | (Qualitative Rubric)|
                             +--------------------+   +---------------------+
```

1. **Phase 0 (Hard Safety Gate)**: Non-compensable. If any safety violation occurs, the task immediately fails with score `0.0`. Quality judges are never invoked, preventing high fluency from masking critical security or compliance violations.
2. **Phase 1 (Deterministic Check)**: Verifies exact state mutations and structured deliverables. If deterministic checks fail, execution terminates early.
3. **Phase 2 (Subjective Semantic Judge)**: If enabled, an LLM judge (`SemanticJudge`) assesses fluency, qualitative clarity, and reasoning against domain rubrics.

---

## 4. Observability & Persistence

Every evaluation emits structured artifacts into `data/runs/<run_id>/`:
- `traces.jsonl`: Step-by-step event log (prompts, tool calls, tool outputs, thinking blocks, errors).
- `metrics.parquet`: High-throughput tabular metrics (latency percentiles, tokens, financial cost).
- `run_manifest.json`: Configuration snapshot and cryptographic hash for reproducible audit trails.
