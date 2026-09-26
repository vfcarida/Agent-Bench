# Quickstart Guide

This guide will walk you through running your first agent evaluation with **Agent-Bench** in under 5 minutes.

---

## 1. Installation

Agent-Bench requires **Python 3.12+**. Install the core package with development tools:

```bash
# Clone the repository
git clone https://github.com/agent-bench/agent-bench.git
cd agent-bench

# Install editable package with development tools
pip install -e ".[dev]"
```

For specific model providers, install optional extras:
```bash
pip install -e ".[openai]"       # OpenAI GPT-4o / O1 models
pip install -e ".[anthropic]"    # Anthropic Claude 3.5 Sonnet
pip install -e ".[vllm]"         # Local vLLM high-throughput engine
pip install -e ".[all]"          # All providers and dependencies
```

---

## 2. Environment Configuration

Copy the example environment file and configure your API credentials (optional for stub and scripted runs):

```bash
cp .env.example .env
```

If evaluating proprietary models, specify your keys in `.env`:
```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 3. Verify Configuration & Datasets

Before running evaluation suites, validate your YAML configurations and dataset integrity:

```bash
# Validate benchmark configuration manifests
bench --config-dir configs validate-config

# Validate canonical gold evaluation datasets
python scripts/check_gold_integrity.py

# Verify zero contamination between holdout and dev splits
python scripts/check_contamination.py
```

---

## 4. Run an Evaluation Suite

### 4.1 Zero-Cost Smoke Run (Stub Mode)
To verify your evaluation pipeline without making external API calls:

```bash
bench --config-dir configs run-suite pix_basic_v1 --runner stub
```

### 4.2 Deterministic Baseline Run (Scripted Policy)
To run an honest, reproducible reference policy that executes domain actions:

```bash
bench --config-dir configs run-suite pix_basic_v1 --runner scripted
```

### 4.3 High-Throughput Concurrent Execution
Use the `--concurrency` flag to evaluate tasks in parallel using the asynchronous concurrency pool:

```bash
bench --config-dir configs run-suite pix_basic_v1 --runner scripted --concurrency 4
```

---

## 5. Inspect Execution Traces

Inspect prompts, internal thinking blocks, model tool calls, environment state mutations, and judge verdicts:

```bash
# Interactive terminal viewer
bench view-traces <run_id> --limit 20
```

---

## 6. Generate Evaluation Reports & Web Dashboards

Generate rich Markdown and interactive HTML dashboards:

```bash
# Generate interactive HTML dashboard with search filtering
bench generate-report <run_id> --format html --output-dir data/reports
```

Open `data/reports/<run_id>_report.html` in your browser to inspect scorecards, pass@k reliability curves, and cost/latency breakdowns.
