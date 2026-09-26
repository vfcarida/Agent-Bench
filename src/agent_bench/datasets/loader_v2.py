"""V2 dataset loader for EvalCase instances."""

from pathlib import Path
from typing import Any

import yaml

from agent_bench.core.schema_migration import migrate_fixture_file
from agent_bench.core.schema_v2 import EvalCase


def _load_yaml_cases(file_path: Path) -> list[dict[str, Any]]:
    with open(file_path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]  # type: ignore[no-any-return]
    return [data]


def _dict_to_eval_case(d: dict[str, Any]) -> EvalCase:
    return EvalCase.from_dict(d)



_SOURCE_TO_DIR = {
    "human_gold": "gold",
    "synthetic_shadow": "synthetic/shadow",
    "synthetic_candidate": "synthetic/candidates",
    "adversarial": "adversarial",
    "calibration": "gold/calibration",
}


def load_eval_cases(
    base_dir: Path,
    family: str | None = None,
    split: str | None = None,
    source_type: str | None = None,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    yaml_files: list[Path] = []

    if source_type and split:
        sub = _SOURCE_TO_DIR.get(source_type, "gold")
        search_dir = base_dir / sub / split if "/" not in sub else base_dir / sub
        if search_dir.exists():
            yaml_files.extend(search_dir.glob("*.yaml"))
            yaml_files.extend(search_dir.glob("*.yml"))
    elif split:
        for sub_dir in base_dir.rglob(split):
            if sub_dir.is_dir():
                yaml_files.extend(sub_dir.glob("*.yaml"))
                yaml_files.extend(sub_dir.glob("*.yml"))
    else:
        yaml_files.extend(base_dir.rglob("*.yaml"))
        yaml_files.extend(base_dir.rglob("*.yml"))

    for yf in yaml_files:
        for raw in _load_yaml_cases(yf):
            case = _dict_to_eval_case(raw)
            if family and case.family != family:
                continue
            if source_type and case.source_type != source_type:
                continue
            cases.append(case)

    return cases


def load_from_legacy(domain: str) -> list[EvalCase]:
    project_root = Path(__file__).resolve().parents[3]
    fixtures_dir = project_root / "data" / "fixtures"
    fixture_file = fixtures_dir / f"{domain}.yaml"

    if not fixture_file.exists():
        return []

    migrated = migrate_fixture_file(fixture_file)
    return [_dict_to_eval_case(d) for d in migrated]
