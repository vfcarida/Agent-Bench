# Integrating External Agents Guide

This tutorial demonstrates how to benchmark your own agent implementations in **Agent-Bench**, whether built using **LangChain**, **LangGraph**, **CrewAI**, or custom Python code.

---

## 1. Overview: The `CallableAgentRunner` Bridge

Agent-Bench does not force you to inherit from complex framework classes. By using [`CallableAgentRunner`](file:///src/agent_bench/runners/callable_runner.py), any synchronous or asynchronous Python callable can be evaluated against the benchmark suite.

### Supported Signatures

`CallableAgentRunner` automatically detects your function signature:

```python
# 1. Full context (Task object and stateful environment)
async def my_agent(task: Task, env: TaskEnvironment) -> str | dict: ...

# 2. Tool-calling agent (Prompt string, tools list, environment)
async def my_agent(prompt: str, tools: list[dict], env: TaskEnvironment) -> str | dict: ...

# 3. Message sequence agent (Chat history, tools list, environment)
async def my_agent(messages: list[dict], tools: list[dict], env: TaskEnvironment) -> str | dict: ...

# 4. Pure prompt agent
async def my_agent(prompt: str) -> str: ...
```

---

## 2. Example 1: Custom Python ReAct Agent

Below is a self-contained example wrapping a custom agent function that interacts with environment tools:

```python
import asyncio
from agent_bench.core.protocols import TaskEnvironment
from agent_bench.core.scenarios import Task
from agent_bench.runners import CallableAgentRunner, execute_task
from agent_bench.core.config import BenchConfig, ModelConfig, SystemConfig
from agent_bench.datasets.loader import load_domain_tasks

async def my_custom_agent(prompt: str, tools: list[dict], env: TaskEnvironment) -> dict:
    # 1. Inspect user prompt and available tools
    if "transfer" in prompt.lower():
        # Execute environment tool
        tool_res = await env.execute_tool("validate_pix_key", {"pix_key": "user@bank.com"})
        return {
            "response": f"Validated key: {tool_res.output}. Ready for confirmation.",
            "tokens_in": 120,
            "tokens_out": 35,
            "cost_usd": 0.001,
        }
    return {"response": "How can I assist you today?"}

async def main():
    # Load canonical benchmark tasks
    tasks = load_domain_tasks("pix_assist", split="dev")
    sample_task = tasks[0]

    # Setup runner
    runner = CallableAgentRunner(my_custom_agent, name="my_custom_agent")

    config = BenchConfig(
        models=[ModelConfig(model_id="custom_m", provider="stub")],
        systems=[SystemConfig(system_id="custom_sys", architecture="stub", model="custom_m")],
    )

    # Execute benchmark case
    result = await execute_task(sample_task, system_id="custom_sys", config=config, agent_runner=runner)

    print(f"Task Passed: {result.passed}")
    print(f"Safety Violated: {result.safety_violated}")
    print(f"Latency: {result.latency_ms:.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3. Example 2: Benchmarking a LangChain Agent

To benchmark an existing LangChain agent or `AgentExecutor`:

```python
from langchain.agents import AgentExecutor
from agent_bench.runners import CallableAgentRunner

# Wrap LangChain AgentExecutor
async def langchain_agent_bridge(prompt: str, tools: list[dict], env: TaskEnvironment) -> str:
    # Invoke LangChain executor
    result = await langchain_agent_executor.ainvoke({"input": prompt})
    return result["output"]

runner = CallableAgentRunner(langchain_agent_bridge, name="langchain_react_agent")
```

---

## 4. Example 3: Running a Full Benchmark Suite on Your Agent

You can evaluate your custom runner across an entire benchmark suite programmatically:

```python
import asyncio
from pathlib import Path
from agent_bench.core.config import SuiteConfig, BenchConfig, ModelConfig, SystemConfig
from agent_bench.runners.suite_runner import run_suite
from agent_bench.runners import CallableAgentRunner

async def evaluate_my_agent():
    suite_cfg = SuiteConfig(
        suite_id="pix_quick_eval",
        name="PIX Quick Evaluation",
        version="1.0.0",
        domains=["pix_assist"],
        systems=["my_agent_system"],
        repeat_n=3,
        seed=42,
    )

    config = BenchConfig(
        models=[ModelConfig(model_id="agent_model", provider="stub")],
        systems=[SystemConfig(system_id="my_agent_system", architecture="stub", model="agent_model")],
        suites=[suite_cfg],
    )

    output_dir = Path("data/runs/my_agent_evaluation")
    artifact = await run_suite(suite_cfg, config, output_dir=output_dir)

    print(f"Evaluation complete! Total: {artifact.tasks_total}, Passed: {artifact.tasks_passed}")
    print(f"Metrics: {artifact.metrics}")

if __name__ == "__main__":
    asyncio.run(evaluate_my_agent())
```

---

## 5. Next Steps

- Inspect step-by-step traces of your agent using the terminal trace viewer:
  ```bash
  bench view-traces <run_id>
  ```
- Generate an interactive HTML report:
  ```bash
  bench export-html --run-id <run_id>
  ```
- Compare system rankings on the leaderboard:
  ```bash
  bench leaderboard
  ```
