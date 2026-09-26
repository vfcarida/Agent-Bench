# Command-Line Interface (CLI) Reference

The `bench` CLI provides commands for running benchmark suites, debugging execution traces, ranking systems, and auditing data integrity.

---

## Command Overview

| Command | Description |
| :--- | :--- |
| [`run-suite`](#1-bench-run-suite) | Run an evaluation benchmark suite |
| [`run-case`](#2-bench-run-case) | Execute a single benchmark task by ID |
| [`view-traces`](#3-bench-view-traces) | Interactive terminal visualizer for step-by-step traces |
| [`leaderboard`](#4-bench-leaderboard) | Display system ranking leaderboard |
| [`export-html`](#5-bench-export-html) | Generate interactive self-contained HTML dashboard |
| [`compare-runs`](#6-bench-compare-runs) | Compare metrics across two historical runs |
| [`analytics`](#7-bench-analytics) | Run cross-system and trend queries across runs |
| [`validate-config`](#8-bench-validate-config) | Validate YAML benchmark configurations |
| [`check-agreement`](#9-bench-check-agreement) | Calculate inter-annotator agreement (Kappa & Alpha) |
| [`list-suites`](#10-bench-list-suites) | List available evaluation suites |
| [`list-models`](#11-bench-list-models) | List configured model endpoints |
| [`list-systems`](#12-bench-list-systems) | List configured agent systems |
| [`list-plugins`](#13-bench-list-plugins) | List registered domain, tool, and model plugins |
| [`export-inspect`](#14-bench-export-inspect) | Export datasets to Inspect AI dataset format |
| [`export-inspect-log`](#15-bench-export-inspect-log) | Export run traces to Inspect AI EvalLog format |

---

## 1. `bench run-suite`

Executes a benchmark suite across defined systems and tasks.

```bash
bench run-suite <suite_id> [OPTIONS]
```

### Options
- `--config-dir PATH`: Directory containing configuration manifests. *(Default: `configs`)*
- `--output-dir PATH`: Directory to write run artifacts. *(Default: `data/runs`)*
- `--repeat-n INTEGER`: Number of repetitions per task for statistical estimation. *(Default: from suite)*
- `--concurrency / -c INTEGER`: Maximum concurrent tasks to execute in parallel. *(Default: `1`)*
- `--system TEXT`: Override systems to evaluate.
- `--split TEXT`: Dataset split to evaluate (`dev`, `holdout`). *(Default: `dev`)*
- `--seed INTEGER`: Random seed for reproducibility.
- `--enable-llm-judge`: Enable Phase 2 subjective LLM judge evaluation.
- `--judge-model TEXT`: Model ID for LLM judge.

### Example
```bash
bench run-suite pix_basic_v1 --repeat-n 3 --concurrency 5 --split dev
```

---

## 2. `bench run-case`

Executes a single benchmark task in isolation for debugging.

```bash
bench run-case <task_id> [OPTIONS]
```

### Options
- `--system-id TEXT`: System identifier to evaluate. *(Required)*
- `--domain TEXT`: Domain of the task (`pix_assist`, `cyber_sandbox`, etc.). *(Required)*
- `--split TEXT`: Dataset split. *(Default: `dev`)*

### Example
```bash
bench run-case PIX_DEV_001 --domain pix_assist --system-id scripted_system
```

---

## 3. `bench view-traces`

Launches the rich interactive terminal trace visualizer. Displays step-by-step thinking blocks, tool calls, tool outputs, and judge decisions.

```bash
bench view-traces <run_id> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing runs. *(Default: `data/runs`)*
- `--task-id TEXT`: Filter display to a single task ID.

### Example
```bash
bench view-traces run_20260925_143000
```

---

## 4. `bench leaderboard`

Displays system rankings across benchmark runs.

```bash
bench leaderboard [OPTIONS]
```

### Options
- `--domain TEXT`: Filter by specific domain.
- `--top INTEGER`: Number of top entries to show. *(Default: `20`)*
- `--format [table|markdown|html|json]`: Output presentation format. *(Default: `table`)*
- `--leaderboard-path PATH`: Path to leaderboard JSON file.

### Example
```bash
bench leaderboard --domain pix_assist --format table
bench leaderboard --format markdown > LEADERBOARD.md
```

---

## 5. `bench export-html`

Generates an interactive, standalone HTML report with real-time search, latency graphs, and scorecard breakdowns.

```bash
bench export-html [OPTIONS]
```

### Options
- `--run-id TEXT`: Run identifier to export.
- `--runs-dir PATH`: Directory containing runs. *(Default: `data/runs`)*
- `--output PATH`: Target HTML file path. *(Default: `data/reports/<run_id>.html`)*

### Example
```bash
bench export-html --run-id run_20260925_143000
```

---

## 6. `bench compare-runs`

Compares two benchmark runs side-by-side, computing score deltas and regressions.

```bash
bench compare-runs <baseline_run_id> <candidate_run_id> [OPTIONS]
```

---

## 7. `bench analytics`

Runs analytics queries across historical benchmark runs.

```bash
bench analytics [OPTIONS]
```

### Options
- `--by [system|domain]`: Aggregation axis.
- `--system TEXT`: Filter by specific system ID.
- `--domain TEXT`: Filter by domain for cross-system comparisons.

---

## 8. `bench validate-config`

Validates all YAML configuration files in `configs/` against Pydantic schemas.

```bash
bench validate-config [OPTIONS]
```

---

## 9. `bench check-agreement`

Calculates multi-annotator agreement metrics (**Cohen's Kappa** and **Krippendorff's Alpha**) to calibrate judge and human ratings.

```bash
bench check-agreement [OPTIONS]
```

### Options
- `--annotations PATH`: Path to multi-annotator YAML/JSON ratings file.

---

## 10. `bench list-suites`

Lists all benchmark suites configured in `configs/suites/`.

```bash
bench list-suites [OPTIONS]
```

---

## 11. `bench list-models`

Lists all configured LLM and local model endpoints in `configs/models/`.

```bash
bench list-models [OPTIONS]
```

---

## 12. `bench list-systems`

Lists all configured agent systems in `configs/systems/`.

```bash
bench list-systems [OPTIONS]
```

---

## 13. `bench list-plugins`

Lists all registered domain tools, mutators, and model adapters discovered in the runtime environment.

```bash
bench list-plugins [OPTIONS]
```

---

## 14. `bench export-inspect`

Exports Agent-Bench datasets into standard UK/US AI Safety Institute **Inspect AI** dataset format (`inspect_evals`).

```bash
bench export-inspect [OPTIONS]
```

### Options
- `--domain TEXT`: Evaluation domain to export (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`).
- `--split [dev|holdout]`: Dataset split to export *(Default: `dev`)*.
- `--output PATH`: Target output JSON/JSONL file path.

---

## 15. `bench export-inspect-log`

Converts completed Agent-Bench run traces into Inspect AI `EvalLog` compatible format.

```bash
bench export-inspect-log <run_id> [OPTIONS]
```

### Options
- `--output PATH`: Target output path for Inspect eval log JSON.
