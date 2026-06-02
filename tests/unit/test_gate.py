"""Unit tests for CI gate."""

import json
import os
import pytest
from pathlib import Path

from agent_bench.cli.gate import check_gate


@pytest.fixture
def run_artifact(tmp_path):
    runs_dir = tmp_path / "data" / "runs"
    runs_dir.mkdir(parents=True)
    data = {
        "run_id": "gate-test-run-001",
        "tasks_total": 10,
        "tasks_passed": 8,
        "tasks_failed": 2,
        "scorecards": [
            {
                "system_id": "sys_a",
                "domain": "pix",
                "global_score": 0.75,
                "functional_score": 0.8,
                "risk_score": 0.9,
            },
        ],
    }
    (runs_dir / "gate-test-run-001.json").write_text(json.dumps(data))
    return tmp_path


def test_gate_passes(run_artifact):
    old_cwd = os.getcwd()
    os.chdir(run_artifact)
    try:
        result = check_gate("gate-test-run-001", min_global=0.6, min_functional=0.5, min_risk=0.7)
        assert result is True
    finally:
        os.chdir(old_cwd)


def test_gate_fails_global_threshold(run_artifact):
    old_cwd = os.getcwd()
    os.chdir(run_artifact)
    try:
        result = check_gate("gate-test-run-001", min_global=0.9)
        assert result is False
    finally:
        os.chdir(old_cwd)


def test_gate_fails_max_failures(run_artifact):
    old_cwd = os.getcwd()
    os.chdir(run_artifact)
    try:
        result = check_gate("gate-test-run-001", max_failures=1)
        assert result is False
    finally:
        os.chdir(old_cwd)


def test_gate_not_found(tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    (tmp_path / "data" / "runs").mkdir(parents=True)
    try:
        result = check_gate("nonexistent-run")
        assert result is False
    finally:
        os.chdir(old_cwd)
