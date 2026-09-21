# How to Run Evaluations

This guide provides complete instructions for executing benchmark evaluations, running CI quality gates, and inspecting telemetry traces using the `bench` CLI.

---

## 1. Prerequisites & Environment Setup

Ensure your environment is configured with development dependencies:

```bash
# Install core package with development tools
pip install -e ".[dev]"

# (Optional) Copy and configure model provider API keys
cp .env.example .env
# Edit .env with OPENAI_API_KEY, ANTHROPIC_API_KEY if evaluating live models
```

Validate your configuration manifests before running:

```bash
bench --config-dir configs validate-config
```

---

## 2. Local Suite Execution

### 2.1 Running a Benchmark Suite

To execute a complete benchmark suite, pass the positional `suite_id` (defined in `configs/suites/<suite_id>.yaml`):

```bash
# Run the PIX Basic suite using the scripted reference runner (offline)
bench --config-dir configs run-suite pix_basic_v1 --runner scripted

# Run with custom repetition count and random seed for Pass@k estimation
bench --config-dir configs run-suite pix_basic_v1 --runner scripted --repeat 3 --seed 42

# Run against a specific dataset split (dev, holdout, calibration, regression)
bench --config-dir configs run-suite pix_basic_v1 --runner scripted --split dev
```

#### Available `--runner` options:
- `auto`: Dynamically resolves the runner and model adapter from `configs/systems/` (default).
- `scripted`: Uses deterministic, offline scripted agent policies (`ScriptedAgentRunner`).
- `stub`: Uses domain-agnostic smoke agent (`DefaultAgentRunner` with `StubModelAdapter`).

### 2.2 Enabling the LLM-as-a-Judge (Phase 2)

By default, qualitative evaluations run deterministic and rubric checks. To enable `SemanticJudge` in Phase 2:

```bash
bench --config-dir configs run-suite pix_basic_v1 \
  --runner scripted \
  --enable-llm-judge \
  --llm-judge-system prompt_only_gpt4
```

> **Note**: `--enable-llm-judge` requires passing the calibration gate experiment first. See [`docs/llm_judge_calibration_report.md`](llm_judge_calibration_report.md).

---

## 3. Running a Single Case

For targeted debugging and development, execute a single task using its unique `task_id`:

```bash
# Execute PIX_001 against tool_calling_reactive_gpt4 in domain pix_assist
bench --config-dir configs run-case PIX_001 \
  --system tool_calling_reactive_gpt4 \
  --domain pix_assist

# Execute a case from the holdout split
bench --config-dir configs run-case PIX_HOLDOUT_001 \
  --system tool_calling_reactive_gpt4 \
  --domain pix_assist \
  --split holdout
```

---

## 4. Contamination & Data Leakage Gating

Before running benchmarks or training synthetic cases, check for data leakage between holdout sets and dev or synthetic candidate pools:

```bash
# Check dataset leakage using default 0.80 Jaccard threshold
bench check-contamination

# Check with custom similarity threshold and directories
bench check-contamination \
  --threshold 0.85 \
  --holdout-dir datasets/gold/holdout \
  --dev-dir datasets/gold/dev
```

---

## 5. Continuous Integration (CI) Quality Gates

To evaluate whether a completed run satisfies deployment criteria:

```bash
# Check quality gate against minimum global, functional, and risk scores
bench gate <run_id> --min-global 0.60 --min-functional 0.50 --min-risk 0.70

# Check gate with a maximum allowed failure ceiling
bench gate <run_id> --max-failures 5
```

Exit Codes:
- `0`: All quality and safety thresholds satisfied.
- `1`: One or more thresholds breached (logs details to console).

---

## 6. Telemetry Traces & Analytics

### 6.1 Inspecting Execution Traces

Inspect events, prompt messages, and tool calls emitted during execution:

```bash
# View all traces for a run
bench view-traces <run_id>

# Filter traces for a specific task
bench view-traces <run_id> --task PIX_001 --limit 20
```

### 6.2 Running Columnar Analytics

Query aggregated operational metrics across persisted Parquet runs:

```bash
# Aggregate metric results across domains
bench analytics --by domain

# Aggregate metric results across systems
bench analytics --by system

# Inspect historical score trends for a specific system
bench analytics --system tool_calling_reactive_gpt4
```

---

## 7. Generating Reports & Visualizations

```bash
# Generate Markdown summary report
bench generate-report <run_id> --format markdown

# Generate interactive HTML dashboard
bench generate-report <run_id> --format html

# Compare two evaluation runs side-by-side
bench compare-runs <run_id_1> <run_id_2> --output data/reports/comparison.md
```

Generated reports include:
- Pass@1, Pass@3, Pass@5 and Pass^k reliability scores.
- Non-parametric 95% bootstrap confidence intervals.
- Operational latency percentiles ($p_{50}, p_{90}, p_{99}$).
- Financial cost per successful task (USD).
- Hard safety gate violation audit.
