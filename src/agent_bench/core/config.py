"""Configuration loading and validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class ModelConfig(BaseModel):
    model_id: str
    provider: str
    endpoint: str | None = None
    api_key_env: str | None = None
    parameters: dict[str, Any] = {}


class SystemConfig(BaseModel):
    system_id: str
    architecture: str
    model: str
    tools: list[str] = []
    retrieval: str | None = None
    memory: bool = False
    max_steps: int = 10

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        valid = {
            "prompt_only",
            "rag_basic",
            "rag_hybrid",
            "tool_calling_reactive",
            "planner_executor",
            "planner_executor_with_memory",
        }
        if v not in valid:
            raise ValueError(f"Architecture must be one of {valid}")
        return v


class SuiteConfig(BaseModel):
    suite_id: str
    name: str
    version: str
    description: str = ""
    domains: list[str]
    systems: list[str]
    repeat_n: int = 1
    seed: int | None = 42
    weighting_profile: str = "transactional_high_risk"
    timeout_ms: int = 30000


class BenchConfig(BaseModel):
    models: list[ModelConfig] = []
    systems: list[SystemConfig] = []
    suites: list[SuiteConfig] = []

    def config_hash(self) -> str:
        content = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Path) -> BenchConfig:
    """Load all YAML configs from the config directory."""
    models = []
    systems = []
    suites = []

    models_dir = config_dir / "models"
    if models_dir.exists():
        for f in models_dir.glob("*.yaml"):
            data = load_yaml(f)
            if "models" in data:
                models.extend([ModelConfig(**m) for m in data["models"]])
            else:
                models.append(ModelConfig(**data))

    suites_dir = config_dir / "suites"
    if suites_dir.exists():
        for f in suites_dir.glob("*.yaml"):
            data = load_yaml(f)
            if "suites" in data:
                suites.extend([SuiteConfig(**s) for s in data["suites"]])
            else:
                suites.append(SuiteConfig(**data))

    domains_dir = config_dir / "domains"
    if domains_dir.exists():
        for f in domains_dir.glob("*.yaml"):
            data = load_yaml(f)
            if "systems" in data:
                systems.extend([SystemConfig(**s) for s in data["systems"]])

    return BenchConfig(models=models, systems=systems, suites=suites)
