<div align="center">
  <h1>🚀 Agent-Bench</h1>
  <p><strong>Multi-domain benchmark framework for evaluating LLM agents as complete systems</strong></p>

  [![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/agent-bench/agent-bench/actions)
  [![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/agent-bench/agent-bench/actions)
</div>

---

## 🌟 Why Agent-Bench?

Existing benchmarks often evaluate models in isolation. However, real-world agents combine LLMs with tool-calling capabilities, RAG pipelines, policy enforcement engines, and multi-step reasoning capabilities. **Agent-Bench** steps up to evaluate the *entire system* on realistic scenarios across multiple distinct domains, providing a holistic view of your agent's performance.

With Agent-Bench, you gain insights into:
- 🎯 **Functional correctness** — Did the agent execute the right sequence of tools and reach the correct final state?
- 🛡️ **Risk compliance** — Did the agent effectively refuse unsafe requests? Did it actively avoid forbidden or destructive actions?
- 💰 **Cost efficiency** — What is the token usage and API invocation cost per successful task?
- ⚡ **Latency** — What is the end-to-end response time of the agent system?
- 🔄 **Reliability** — What is the pass@k metric across repeated runs?

---

## ✨ Key Features

- **🔌 Provider-Agnostic Engine:** Seamlessly compare OpenAI, Anthropic, or any custom model on identical tasks.
- **🧩 4 Evaluation Families:**
  - `transactional_tools`: Multi-step tool orchestration with deterministic state changes.
  - `knowledge_rag_reasoning`: Information retrieval, synthesis, and factual accuracy.
  - `business_long_horizon`: Multi-turn planning with strict constraints.
  - `security_guardrails`: Adversarial robustness and strict policy compliance.
- **⚖️ Deterministic & Semantic Grading:** Supports state-based, tool-call-based, rubric-based, and composite grading strategies.
- **🧪 Synthetic Case Generation:** Robust data generation with configurable perturbations (noise, urgency, ambiguity, distractors) to stress-test your agents.
- **✅ Automated Schema Validation:** Built-in validation pipeline featuring strict consistency and deduplication checks.
- **🔐 Enterprise Governance:** Integrated provenance tracking, versioning, and PII redaction capabilities.
- **📊 Observability:** OpenTelemetry-compatible tracing (spans, events, structured export).
- **🛠️ Extensible Plugin Architecture:** Easily inject custom models, semantic judges, and retrieval adapters.
- **💻 CI/CD Native:** CLI-driven execution, clean exit codes, and automated threshold gates for your deployment pipelines.

---

## 📦 Installation

Agent-Bench requires **Python 3.12+**.

```bash
# Clone the repository
git clone https://github.com/agent-bench/agent-bench.git
cd agent-bench

# Install core package
pip install -e ".[dev]"

# Install with specific model providers
pip install -e ".[openai]"
pip install -e ".[anthropic]"

# Install everything
pip install -e ".[all]"
```

---

## 🚀 Quickstart

Agent-Bench is operated via a powerful and intuitive CLI tool: `bench`.

```bash
# Validate your configuration YAMLs
bench --config-dir configs validate-config

# Run a suite (stub mode - no real API keys needed for testing!)
bench --config-dir configs run-suite pix_basic_v1

# Run a specific single case
bench --config-dir configs run-case PIX_001 --system tool_calling_reactive_gpt4 --domain pix_assist

# Generate a detailed Markdown/HTML report for a run
bench generate-report <run-id>

# Compare two evaluation runs side-by-side
bench compare-runs <run-id-1> <run-id-2>
```

---

## 🏗️ Project Architecture

```text
agent-bench/
  configs/              # YAML configurations (models, systems, suites, judges)
  datasets/
    gold/dev/           # Human-curated evaluation golden cases
    synthetic/shadow/   # Auto-generated shadow cases for coverage expansion
    adversarial/        # Attack vectors and boundary-testing cases
  src/agent_bench/
    cli/                # Click-based CLI commands
    core/               # Schema definitions, configuration logic, base scenarios
    validators/         # Validation, consistency, and deduplication logic
    generators/         # Synthetic case generation pipelines
    graders/            # State, tool-call, rubric, and composite grading engines
    metrics/            # C.I., precision/recall computations, compliance scoring
    models/             # Provider implementations (stub, openai, anthropic)
    tools/              # Tool execution adapters
    judges/             # Deterministic and LLM-as-a-judge evaluators
    runners/            # Suite, single-case, and online evaluation orchestrators
    reports/            # Markdown & HTML leaderboard/report generators
    storage/            # Artifact persistence (JSONL, Parquet, local DB)
    governance/         # Case versioning, data provenance, and PII redaction
    utils/              # Observability tracing, plugins, shared tools
  tests/                # Comprehensive unit, integration, and golden test suites
```

---

## ⚖️ Grading & Weighting Profiles

Agent-Bench evaluates systems against multiple, customizable weighting profiles suited for your specific use-case:

| Profile | Functional | Risk | Cost | Latency | Reliability |
|---------|-----------|------|------|---------|-------------|
| **high_volume_low_risk** | 0.30 | 0.10 | 0.30 | 0.20 | 0.10 |
| **transactional_high_risk** | 0.25 | 0.40 | 0.10 | 0.10 | 0.15 |
| **advisory_regulated** | 0.30 | 0.30 | 0.10 | 0.05 | 0.25 |
| **ops_long_horizon** | 0.35 | 0.20 | 0.15 | 0.05 | 0.25 |
| **cyber_restricted** | 0.20 | 0.45 | 0.05 | 0.05 | 0.25 |

---

## 🛠️ Extending Agent-Bench

### Adding a New Domain
1. Create your cases in `datasets/gold/dev/<domain>.yaml`
2. Add the system configurations in `configs/domains/<domain>.yaml`
3. Include the domain in an evaluation suite under `configs/suites/`
4. *(Optional)* Add domain-specific logic in `src/agent_bench/tools/`
5. *(Optional)* Provide custom grading rules and state tracking

### Adding a New Model Provider
1. Implement the `ModelAdapter` interface in `src/agent_bench/models/<provider>.py`
2. Register the implementation in the plugin system or under `configs/models/`
3. Reference the new `model_id` in your system configs

---

## 🧪 Running Tests

Ensure all components are working locally:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
```

---

## 📜 License

This project is licensed under the terms of the **Apache-2.0** License.
