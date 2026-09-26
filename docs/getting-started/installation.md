# Installation Guide

This guide walks you through setting up **Agent-Bench** for evaluating autonomous LLM agents in stateful environments.

---

## Prerequisites

- **Python**: `>= 3.12` (Python 3.12 or 3.13 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Docker** *(Optional, recommended for sandboxed execution)*: Docker Engine 24.0+ or Docker Desktop

Verify your Python installation:
```bash
python --version
# Expected: Python 3.12.x or higher
```

---

## Quick Installation

### 1. From PyPI

Install the core package:
```bash
pip install agent-bench
```

### 2. From Source (Development)

Clone the repository and install in editable mode:
```bash
git clone https://github.com/vfcarida/Agent-Bench.git
cd Agent-Bench
pip install -e ".[dev]"
```

Alternatively, use the automated setup script:
```bash
bash scripts/setup.sh
# or on Windows PowerShell:
make dev
```

---

## Provider Optional Extras

Depending on the models and providers you intend to benchmark, install the corresponding extras:

| Extra | Description | Command |
| :--- | :--- | :--- |
| `openai` | OpenAI API provider (`gpt-4o`, `o1`, etc.) | `pip install "agent-bench[openai]"` |
| `anthropic` | Anthropic Claude API (`claude-3-5-sonnet`, etc.) | `pip install "agent-bench[anthropic]"` |
| `local` | Hugging Face Transformers & vLLM for local weights | `pip install "agent-bench[local]"` |
| `all` | All model providers and analysis libraries | `pip install "agent-bench[all]"` |
| `dev` | Development tools (pytest, ruff, mypy, build) | `pip install "agent-bench[dev]"` |

Install multiple extras simultaneously:
```bash
pip install -e ".[openai,anthropic,dev]"
```

---

## Environment Variables Configuration

If you evaluate commercial API models, export your API keys:

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Verifying the Installation

Run the CLI self-test:
```bash
bench --help
```

Validate benchmark configuration files:
```bash
bench validate-config
```

Run the offline unit test suite:
```bash
pytest tests/ -p no:deepeval -q
```
Expected output: All unit tests pass without errors or warnings.
