#!/usr/bin/env bash
# Developer setup script for Agent-Bench
set -euo pipefail

echo "=========================================="
echo " Setting up Agent-Bench Development Env   "
echo "=========================================="

# Check Python version
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "-> Using Python: $PYTHON_BIN (v$PY_VERSION)"

# Upgrade pip and install package with dev extras
echo "-> Installing Agent-Bench with development dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e ".[dev]"

# Install pre-commit hooks if git repository
if [ -d ".git" ]; then
    echo "-> Configuring git pre-commit hooks..."
    if command -v pre-commit &> /dev/null; then
        pre-commit install
    fi
fi

# Run quick verification checks
echo "-> Validating configurations and dataset integrity..."
"$PYTHON_BIN" -m agent_bench.cli.main --config-dir configs validate-config
"$PYTHON_BIN" scripts/check_gold_integrity.py
"$PYTHON_BIN" scripts/check_contamination.py

echo "=========================================="
echo " Environment setup completed successfully!"
echo " Run 'pytest tests/ -p no:deepeval' to execute tests."
echo "=========================================="
