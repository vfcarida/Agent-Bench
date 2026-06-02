"""Parquet export for analytics-friendly metric storage."""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def save_metrics_parquet(
    records: list[dict[str, Any]], output_path: Path
) -> Path:
    """Save metric records as a Parquet file.

    Each record should have:
    - run_id, system_id, task_id, domain
    - metric_name, metric_value, metric_category
    - Additional context fields
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        # Create empty file with schema
        schema = pa.schema([
            ("run_id", pa.string()),
            ("system_id", pa.string()),
            ("task_id", pa.string()),
            ("domain", pa.string()),
            ("metric_name", pa.string()),
            ("metric_value", pa.float64()),
            ("metric_category", pa.string()),
            ("passed", pa.bool_()),
            ("timestamp", pa.string()),
        ])
        table = pa.table({f.name: [] for f in schema}, schema=schema)
        pq.write_table(table, output_path)  # type: ignore[no-untyped-call]
        return output_path

    # Normalize records to flat structure
    columns: dict[str, list[Any]] = {
        "run_id": [],
        "system_id": [],
        "task_id": [],
        "domain": [],
        "metric_name": [],
        "metric_value": [],
        "metric_category": [],
        "passed": [],
        "timestamp": [],
    }

    for r in records:
        columns["run_id"].append(r.get("run_id", ""))
        columns["system_id"].append(r.get("system_id", ""))
        columns["task_id"].append(r.get("task_id", ""))
        columns["domain"].append(r.get("domain", ""))
        columns["metric_name"].append(r.get("metric_name", ""))
        columns["metric_value"].append(float(r.get("metric_value", 0.0)))
        columns["metric_category"].append(r.get("metric_category", ""))
        columns["passed"].append(bool(r.get("passed", False)))
        columns["timestamp"].append(str(r.get("timestamp", "")))

    table = pa.table(columns)
    pq.write_table(table, output_path)  # type: ignore[no-untyped-call]
    return output_path


def load_metrics_parquet(path: Path) -> list[dict[str, Any]]:
    """Load metrics from Parquet file."""
    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    return table.to_pylist()  # type: ignore[no-any-return]


def save_comparison_parquet(
    systems: list[str],
    scorecards: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Save system comparison as Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[Any]] = {
        "system_id": [],
        "domain": [],
        "functional_score": [],
        "risk_score": [],
        "cost_score": [],
        "latency_score": [],
        "reliability_score": [],
        "global_score": [],
        "weighting_profile": [],
    }

    for sc in scorecards:
        columns["system_id"].append(sc.get("system_id", ""))
        columns["domain"].append(sc.get("domain", ""))
        columns["functional_score"].append(float(sc.get("functional_score", 0)))
        columns["risk_score"].append(float(sc.get("risk_score", 0)))
        columns["cost_score"].append(float(sc.get("cost_score", 0)))
        columns["latency_score"].append(float(sc.get("latency_score", 0)))
        columns["reliability_score"].append(float(sc.get("reliability_score", 0)))
        columns["global_score"].append(float(sc.get("global_score", 0)))
        columns["weighting_profile"].append(sc.get("weighting_profile", ""))

    table = pa.table(columns)
    pq.write_table(table, output_path)  # type: ignore[no-untyped-call]
    return output_path
