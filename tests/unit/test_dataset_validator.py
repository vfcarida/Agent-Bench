"""Unit tests for dataset validator."""

import pytest
from pathlib import Path

import yaml

from agent_bench.datasets.validator import validate_dataset, validate_all_datasets


@pytest.fixture
def valid_dataset(tmp_path):
    data = {
        "domain": "test_domain",
        "version": "1.0.0",
        "tasks": [
            {
                "task_id": "T_001",
                "name": "Test task",
                "description": "A test",
                "input_messages": [{"role": "user", "content": "hello"}],
                "severity": "medium",
                "business_criticality": "operational",
                "tags": ["happy_path"],
                "expected_final_state": {"done": True},
                "allowed_tools": ["tool_a"],
            }
        ],
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture
def invalid_dataset(tmp_path):
    data = {
        "domain": "broken",
        "tasks": [
            {"name": "no id"},  # missing task_id and input_messages
            {"task_id": "DUP", "name": "first", "input_messages": [{"role": "user", "content": "x"}]},
            {"task_id": "DUP", "name": "duplicate", "input_messages": [{"role": "user", "content": "y"}]},
        ],
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.dump(data))
    return path


def test_valid_dataset(valid_dataset):
    result = validate_dataset(valid_dataset)
    assert result.valid is True
    assert result.task_count == 1
    assert len(result.errors) == 0


def test_invalid_dataset_missing_fields(invalid_dataset):
    result = validate_dataset(invalid_dataset)
    assert result.valid is False
    assert len(result.errors) >= 2  # missing task_id + duplicate


def test_nonexistent_file(tmp_path):
    result = validate_dataset(tmp_path / "nope.yaml")
    assert result.valid is False


def test_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("{{invalid yaml: [")
    result = validate_dataset(path)
    assert result.valid is False


def test_warnings_for_missing_optional(tmp_path):
    data = {
        "domain": "test",
        "tasks": [
            {
                "task_id": "T_001",
                "name": "Minimal",
                "input_messages": [{"role": "user", "content": "hi"}],
                # no tags, no description, no allowed_tools
            }
        ],
    }
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.dump(data))
    result = validate_dataset(path)
    assert result.valid is True
    assert len(result.warnings) >= 2  # tags + description


def test_validate_all_datasets(tmp_path):
    for name in ["a.yaml", "b.yaml"]:
        data = {"domain": name[0], "version": "1.0", "tasks": [
            {"task_id": f"{name[0]}_001", "name": "t", "input_messages": [{"role": "user", "content": "x"}]}
        ]}
        (tmp_path / name).write_text(yaml.dump(data))
    results = validate_all_datasets(tmp_path)
    assert len(results) == 2
    assert all(r.valid for r in results)


def test_validate_real_fixtures():
    """Validate all real fixture files in the project."""
    fixtures_dir = Path(__file__).parents[2] / "data" / "fixtures"
    if not fixtures_dir.exists():
        pytest.skip("Fixtures dir not found")
    results = validate_all_datasets(fixtures_dir)
    for r in results:
        assert r.valid, f"{r.path.name} invalid: {[i.message for i in r.errors]}"
