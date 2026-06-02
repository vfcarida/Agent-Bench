# Architecture Overview

## Core Design

agent-bench is built around the concept of **comparing systems, not just models**. A system combines a model with tools, retrieval, memory, and orchestration strategy.

## Layered Architecture

```
┌─────────────────────────────────────┐
│           CLI Layer                   │
├─────────────────────────────────────┤
│         Runner Layer                  │
│  (SuiteRunner, CaseRunner)           │
├─────────────────────────────────────┤
│        Judge Layer                    │
│  (Deterministic, Semantic, Policy)   │
├─────────────────────────────────────┤
│       Adapter Layer                   │
│  (Model, Agent, Tool, Retrieval)     │
├─────────────────────────────────────┤
│       Core Layer                      │
│  (Task, Scenario, Metrics, Artifact) │
├─────────────────────────────────────┤
│      Storage & Governance            │
│  (JSONL, Parquet, Provenance)        │
└─────────────────────────────────────┘
```

## Execution Flow

1. CLI parses command and loads config
2. SuiteRunner iterates domains × systems × tasks × repetitions
3. CaseRunner executes a single task against a system adapter
4. System adapter orchestrates model + tools + retrieval
5. Judges evaluate the result (deterministic first)
6. Metrics are computed (pass@k, latency, cost, etc.)
7. Traces and artifacts are persisted
8. Report is generated

## Key Decisions

- **Pydantic for config, dataclasses for internal models**: Config needs validation; internal types need speed.
- **Async throughout**: All adapters are async for I/O-bound model calls.
- **Stub-first development**: Every adapter has a stub implementation for testing without APIs.
- **YAML configs, not code**: Non-engineers can modify benchmark parameters.
- **Deterministic judges prioritized**: Only use LLM judges when state-matching is insufficient.
- **pass@k via unbiased estimator**: Not naive sampling, uses combinatorial formula.

## Data Flow

```
configs/*.yaml → BenchConfig → SuiteRunner → [CaseRunner × N]
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              ModelAdapter    ToolAdapter    RetrievalAdapter
                                    │               │               │
                                    └───────────────┼───────────────┘
                                                    ▼
                                              JudgeAdapter
                                                    │
                                                    ▼
                                        RunArtifact + Traces
                                                    │
                                                    ▼
                                          Report + Scorecard
```
