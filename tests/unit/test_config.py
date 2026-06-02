"""Unit tests for configuration loading."""

from pathlib import Path

import pytest

from agent_bench.core.config import BenchConfig, ModelConfig, SuiteConfig, SystemConfig, load_config


@pytest.fixture
def config_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "test.yaml").write_text("""
models:
  - model_id: test-model
    provider: stub
    parameters: {}
""")
    suites_dir = tmp_path / "suites"
    suites_dir.mkdir()
    (suites_dir / "test.yaml").write_text("""
suite_id: test_suite
name: Test Suite
version: "1.0.0"
domains:
  - pix_assist
systems:
  - stub_system
""")
    domains_dir = tmp_path / "domains"
    domains_dir.mkdir()
    (domains_dir / "test.yaml").write_text("""
systems:
  - system_id: stub_system
    architecture: prompt_only
    model: test-model
""")
    return tmp_path


def test_load_config(config_dir):
    config = load_config(config_dir)
    assert len(config.models) == 1
    assert config.models[0].model_id == "test-model"
    assert len(config.suites) == 1
    assert config.suites[0].suite_id == "test_suite"
    assert len(config.systems) == 1


def test_config_hash_deterministic():
    c1 = BenchConfig(models=[ModelConfig(model_id="a", provider="b")])
    c2 = BenchConfig(models=[ModelConfig(model_id="a", provider="b")])
    assert c1.config_hash() == c2.config_hash()


def test_invalid_architecture():
    with pytest.raises(Exception):
        SystemConfig(system_id="x", architecture="invalid_arch", model="m")
