# Agent-Bench Documentation

**Agent-Bench** is an evaluation and reliability benchmark framework for Large Language Model (LLM) agents operating as autonomous systems in compliance-critical, stateful enterprise domains.

---

## Key Differentiators

1. **Hard Safety Gating**: Phase 0 zero-tolerance refusal and forbidden action checks. Safety violations immediately fail the task and cannot be masked by functional success.
2. **Statistical Rigor**: Per-task $\text{Pass}@k$, high-risk $\text{Pass}^k$, and 95% non-parametric bootstrap confidence intervals.
3. **Real Cost & Latency Accounting**: Atomic step-by-step token and financial cost accounting in USD.
4. **Universal Agent Interoperability**: Protocol-driven architecture (`AgentRunner`, `TaskEnvironment`, `Evaluator`) supporting OpenAI, Anthropic, HuggingFace, vLLM, LangChain, CrewAI, LangGraph, and Inspect AI.

---

## Navigation Guide

### [Getting Started](getting-started/quickstart.md)
- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Architecture Overview](getting-started/architecture-overview.md)

### [User Guides](guides/adding-agent.md)
- [Plugging External Agents (LangChain, CrewAI)](guides/adding-agent.md)
- [Adding New Evaluation Domains](guides/adding-domain.md)
- [Running Local Models (vLLM, HuggingFace)](guides/running-local-models.md)
- [Inspect AI Interoperability](guides/inspect-ai-interoperability.md)

### [Methodology](methodology/safety-gating.md)
- [Hard Safety Gating](methodology/safety-gating.md)
- [Statistical Reliability Metrics](methodology/statistical-metrics.md)
- [Rubric & Tool Grading](methodology/rubric-and-tool-grading.md)
- [Inter-Annotator Agreement & Calibration](methodology/inter-annotator.md)

### [Reference](reference/cli.md)
- [CLI Reference](reference/cli.md)
- [Python API Reference](reference/api.md)
