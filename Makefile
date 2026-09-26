# Makefile for Agent-Bench
# Usage:
#   make dev             # Install dependencies and pre-commit hooks
#   make test            # Run unit and integration test suite
#   make lint            # Check formatting and linting with ruff
#   make typecheck       # Strict type check with mypy
#   make check-all       # Run all local CI gates (lint, typecheck, tests, integrity)
#   make build           # Build distribution packages (wheel + sdist)
#   make clean           # Remove build caches and artifacts

.PHONY: help dev install test lint format typecheck check-gold check-contamination check-all build clean

help:
	@echo "Agent-Bench Development Commands:"
	@echo "  make dev                 Install editable package with dev tools and pre-commit hooks"
	@echo "  make test                Run pytest test suite"
	@echo "  make lint                Run ruff linting"
	@echo "  make format              Format code with ruff"
	@echo "  make typecheck           Run strict type check with mypy"
	@echo "  make check-gold          Verify canonical gold dataset integrity"
	@echo "  make check-contamination Check for dataset leakage between holdout and dev"
	@echo "  make check-all           Run all quality gates and tests"
	@echo "  make build               Build source distribution and wheel"
	@echo "  make clean               Clean build artifacts and caches"

dev: install
	pre-commit install

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest tests/ -v -p no:deepeval

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/agent_bench

check-gold:
	python scripts/check_gold_integrity.py

check-contamination:
	python scripts/check_contamination.py

check-all: lint typecheck check-gold check-contamination test

build:
	python -m pip install --upgrade build twine
	python -m build
	twine check dist/*

clean:
	python -c "import shutil, os, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True) + ['.pytest_cache', '.mypy_cache', '.ruff_cache', 'dist', 'build', '*.egg-info']]"
