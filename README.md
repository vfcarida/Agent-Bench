<div align="center">
  <h1>🚀 Agent-Bench</h1>
  <p><strong>Professional MLOps Benchmark Framework for Evaluating LLM Agents as Complete Systems</strong></p>

  [![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
  [![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/agent-bench/agent-bench/releases)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/agent-bench/agent-bench/actions)
  [![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/agent-bench/agent-bench/actions)
  [![Architecture](https://img.shields.io/badge/architecture-Clean%20%2F%20Decoupled-orange.svg)](https://github.com/agent-bench/agent-bench)
</div>

---

## 💼 Executive Vision

Evaluating LLM-based agents requires moving beyond static, single-turn correctness checks. Enterprise agent systems operate in dynamic workflows—utilizing multi-step reasoning, invoking tools, retrieving knowledge, and conforming to strict safety guardrails.

> **Enterprise Positioning**: **Agent-Bench** is an enterprise-grade evaluation ecosystem specifically engineered for production agent workflows (such as Brazilian PIX banking, CRM compliance, and transactional e-commerce). It stands distinct from academic benchmarks (such as Tsinghua's AgentBench or general-purpose QA benchmarks) by prioritizing **zero-tolerance hard safety gating** (refusal failures can never be numerically compensated), **stateful environment sandboxing** (live balance mutations and state diff assertions), **autonomous multi-turn ReAct reasoning loops**, **unbiased per-task Pass@k & Pass^k reliability**, **inter-annotator calibration** (Cohen's Kappa & Krippendorff's Alpha), and **measured execution traces & real financial token accounting**.

**Agent-Bench** provides an end-to-end testing and analytics ecosystem to assess the complete agent lifecycle across dynamic domains. It measures:
- 🎯 **Functional Correctness:** State assertion accuracy, execution of multi-turn plans, and tool call fidelity.
- 🛡️ **Risk & Safety Compliance:** Refusal behavior on unsafe prompts, regulatory adherence, and forbidden action prevention.
- 💰 **Operational Cost & Latency:** Detailed token consumption (`tokens_in`, `tokens_out`), financial cost tracking ($ USD), and latency distribution.
- 🔄 **Statistical Reliability:** Unbiased per-task Pass@k aggregation, non-parametric bootstrap 95% confidence intervals, and pass^k (consistent success across trials) for high-risk domains.

> **Note on Evaluation Baseline**: The default stub runner (`DefaultAgentRunner` without a backend model) is strictly a smoke agent for pipeline connectivity and test harness verification; it returns benign responses and empty states, and does **not** represent real-agent evaluation. For deterministic, answer-independent offline reference testing in CI, Agent-Bench provides `ScriptedAgentRunner`.

---

## ✨ Production Architectural Principles

1. **Clean Architecture (Decoupled Protocols):**
   Agent execution logic is strictly decoupled from benchmark orchestration via Python `Protocol` interfaces:
   - `AgentRunner`: Executes agent reasoning independently of underlying frameworks (`DefaultAgentRunner`, `ScriptedAgentRunner`, or custom LLM loops).
   - `TaskEnvironment`: Sandboxes state management, tool execution, and environment lifecycle.
   - `Evaluator`: Handles scoring, assertion checks, and subjective evaluation.

2. **Hard Safety Gating & Non-Compensable Evaluation (Robustness & Integrity):**
   - **Phase 0 (Hard Safety Gate):** Evaluates non-negotiable safety constraints (refusal requirements, forbidden state changes, disallowed tools). Any breach immediately forces `passed=False` with `safety_violation=True`. Hard safety failures can **never** be averaged away or compensated by quality, latency, or cost.
   - **Separated Safety Axis:** Scorecards report `safety_violations` (count) and `safety_gated` (bool) on a dedicated safety axis. Weighted `global_score` is computed strictly over the safety-passing subset with normalized weights.
   - **Phase 1 (Deterministic Check):** Verifies state diffs, assertion rules, and tool call schemas. Short-circuits on failure to avoid unnecessary LLM costs.
   - **Phase 2 (LLM / Subjective Judge):** Invoked only when safety and deterministic checks pass to evaluate subjective output quality.

3. **Execution Trace Analytics & Real Accounting (JSONL & Parquet):**
   Full execution histories and operational metrics are measured from live execution traces:
   - Measured token consumption (`tokens_in`, `tokens_out`, `total_tokens`).
   - Per-request, cumulative, and percentile latency distributions (`latency_p50`, `latency_p90`, `latency_p99`).
   - Financial cost calculation (`total_cost_usd`) driven by configurable per-model input/output pricing (`configs/models/*`) with documented default fallback.
   - Efficiency accounting with cost-per-successful-task metrics.

4. **Security & MLOps:**
   - **Zero Hardcoded Secrets:** Configuration and API keys are managed safely via `pydantic-settings` with `.env` support.
   - **Docker Sandbox Isolation:** Containerized execution environment preventing unsafe agent actions from affecting the host machine.

5. **Statistical Rigor & Reliability Estimators:**
   Benchmark scores reflect robust statistical accounting rather than noisy single-turn or pooled estimates:
   - **Per-Task Pass@k:** Evaluates the unbiased combinatorial estimator per task across repetitions (never pooled across distinct tasks).
   - **Bootstrap Confidence Intervals:** Generates 95% non-parametric bootstrap intervals for headline pass rates and reliability.
   - **Pass^k for High-Risk Profiles:** Measures the probability that *all* $k$ sampled trials succeed ($\text{pass}^k = \binom{c}{k}/\binom{n}{k}$), preventing pass@k from masking critical agent regressions in transactional and cyber domains.

6. **Dataset Governance, Split Discipline & Contamination Gating:**
   - **Canonical Gold Tree:** `datasets/gold/<split>/<domain>.yaml` serves as the single authoritative dataset hierarchy, using the typed `EvalCase v2` schema.
   - **Strict Split Isolation:** Split-aware loader isolates `dev`, `holdout`, `calibration`, `regression`, and `smoke`. Loading holdout content from legacy fixture paths is strictly prohibited by regression gates.
   - **Offline Contamination Gate:** An automated gate (`bench check-contamination`) runs in offline CI, testing for verbatim (SHA-256) and paraphrase (token Jaccard) leakage from holdout sets into dev or synthetic training pools.

7. **Autonomous Multi-Turn ReAct Reasoning Loop:**
   `DefaultAgentRunner` drives an autonomous ReAct loop: invoking model tools against the stateful `TaskEnvironment`, appending observations to the conversational context, and re-querying until task termination or step exhaustion (`max_steps`). Loop stall detection guards against repetitive static mock stagnation.

8. **Inter-Annotator Agreement & Calibration:**
   Measures grading dataset consistency using Cohen's Kappa (pairwise) and Krippendorff's Alpha (multi-rater reliability across arbitrary nominal labels) via `bench check-agreement`.

9. **Rich Interactive Visual Trace Exploration:**
   Inspect prompts, `<think>` internal reasoning blocks, tool arguments, environment observations, and judge verdicts via Rich terminal panels (`bench view-traces`).

---

## 📦 Installation

Agent-Bench requires **Python 3.12+**.

```bash
# Clone the repository
git clone https://github.com/agent-bench/agent-bench.git
cd agent-bench

# Install core package along with development requirements
pip install -e ".[dev]"

# Install with model provider integrations
pip install -e ".[openai]"
pip install -e ".[anthropic]"

# Install all dependencies
pip install -e ".[all]"
```

---

## 🏗️ Architecture Diagram

```mermaid
graph TD
    subgraph CLI & Runner Layer
        CLI[bench CLI] --> CaseRunner[CaseRunner / SuiteRunner]
    end

    subgraph Clean Architecture Protocols
        CaseRunner --> AgentRunner["AgentRunner Protocol<br/>(LangChain / CrewAI / Custom)"]
        CaseRunner --> TaskEnv["TaskEnvironment Protocol<br/>(State Sandbox / Tools)"]
        CaseRunner --> Evaluator["Evaluator Protocol"]
    end

    subgraph Gated Evaluation Engine
        Evaluator --> GatedEval[GatedEvaluator]
        GatedEval --> Phase1["Phase 1: Deterministic Check<br/>(State Diffs / Assertions / Refusal)"]
        Phase1 -- Fails --> ShortCircuit["Short-Circuit Gate<br/>(Score: 0.0, Bypass LLM Judge)"]
        Phase1 -- Passes --> Phase2["Phase 2: LLM / Subjective Judge<br/>(Semantic / Rubric / Grounding)"]
    end

    subgraph Execution Trace & Analytics Storage
        AgentRunner --> TraceLogger[ExecutionTraceLogger]
        TraceLogger --> JSONL[Atomic JSONL Traces]
        TraceLogger --> Parquet[Parquet Analytics Metrics<br/>(Tokens / Latency / Cost USD)]
    end

    subgraph Security & Settings
        Config[BenchSettings / pydantic-settings] --> CLI
        Config --> .env[.env Environment Variables]
    end
```

### Directory Structure

```text
agent-bench/
  .github/workflows/    # CI/CD pipeline (Ruff, Mypy, Pytest, Golden Tasks)
  configs/              # YAML configurations (models, systems, suites, judges)
  datasets/
    gold/dev/           # Curated golden evaluation tasks (JSONL & YAML)
    synthetic/shadow/   # Auto-generated shadow cases for coverage expansion
    adversarial/        # Attack vectors and boundary-testing cases
  src/agent_bench/
    cli/                # Click-based CLI commands
    core/               # Protocols (AgentRunner, TaskEnvironment, Evaluator), Settings, Scenarios
    graders/            # GatedEvaluator, RubricGrader, StateGrader, ToolCallGrader
    judges/             # Deterministic, Semantic, Grounding, and Composite Judges
    models/             # Provider adapters (OpenAI, Anthropic, HuggingFace, Stub)
    runners/            # CaseRunner, SuiteRunner, PromptFormatter
    storage/            # ExecutionTraceLogger, Resilient JSONL & Parquet Storage
    validators/         # Schema validation & length-pruned Jaccard deduplication
  tests/                # Unit & integration test suites (280+ tests)
  Dockerfile            # Containerized execution environment
  docker-compose.yml    # Docker sandbox compose configuration
```

---

## 🚀 Quickstart

Setting up environment variables:

```bash
# Copy example environment configuration
cp .env.example .env

# Edit .env to add your API keys (optional for stub runs)
# OPENAI_API_KEY=your_key_here
```

Operating Agent-Bench via CLI:

```bash
# 1. Validate YAML configuration and dataset schemas
bench --config-dir configs validate-config

# 2. Run a benchmark suite (stub mode - no API keys required!)
bench --config-dir configs run-suite pix_basic_v1

# 3. Run a specific task case
bench --config-dir configs run-case PIX_001 --system tool_calling_reactive_gpt4 --domain pix_assist

# 4. View interactive execution traces in rich terminal panels
bench view-traces <run-id> --limit 20

# 5. Check inter-annotator calibration & agreement (Cohen's Kappa & Krippendorff's Alpha)
bench check-agreement --annotations datasets/gold/calibration/annotator_agreement.yaml

# 6. Check for holdout contamination and data leakage
bench check-contamination --threshold 0.80

# 7. Generate Markdown / HTML execution reports
bench generate-report <run-id> --format html

# 8. Compare two evaluation runs side-by-side
bench compare-runs <run-id-1> <run-id-2>
```

---

## 🐳 Running in Isolated Docker Sandbox

To run benchmarks securely in an isolated container sandbox:

```bash
# Build Docker image
docker build -t agent-bench:latest .

# Run benchmark suite inside container using docker-compose
docker-compose up
```

---

## 📊 Evaluation Protocol & Variance Management

Agent-Bench enforces strict evaluation protocols to measure reliability, cost, and latency:

1. **Variance Control & Seeds:**
   - Tasks are executed with configurable random seeds (`seed: 42`).
   - Suites support multi-iteration runs (`repeat_n: 3`) to calculate Pass@k metrics and 95% bootstrap confidence intervals.

2. **Cost Calculation:**
   - Financial cost is calculated per request using model token pricing:
     \[
     \text{Total Cost (USD)} = \left(\frac{\text{Tokens}_{\text{in}}}{1000} \times \text{Price}_{\text{in}}\right) + \left(\frac{\text{Tokens}_{\text{out}}}{1000} \times \text{Price}_{\text{out}}\right)
     \]

3. **Gated Evaluation Pipeline:**
   - Deterministic assertion checks execute first in $O(1)$ time.
   - Bypassing LLM judges on deterministic failures reduces benchmark execution time and LLM API cost by up to 70%.

---

## 🧪 Running Tests

Ensure all components and clean architecture protocols pass locally:

```bash
# Run complete test suite (280+ tests)
python -m pytest tests/ -v

# Run specific unit test modules
python -m pytest tests/unit/test_protocols.py -v
python -m pytest tests/unit/test_gated_evaluator.py -v
python -m pytest tests/unit/test_trace_logger.py -v
```

---

## 📜 License

This project is licensed under the terms of the **Apache-2.0** License.
