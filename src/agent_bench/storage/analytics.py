"""Parquet analytics: aggregation queries across runs."""

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa


def load_run_metrics(run_dir: Path) -> pa.Table | None:
    """Load metrics parquet from a run directory."""
    parquet_path = run_dir / "metrics.parquet"
    if not parquet_path.exists():
        # Try inside run_id subdirectory
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            p = subdir / "metrics.parquet"
            if p.exists():
                parquet_path = p
                break
        else:
            return None
    return pq.read_table(parquet_path)  # type: ignore[no-untyped-call]


def aggregate_by_system(runs_dir: Path) -> list[dict[str, Any]]:
    """Aggregate metrics across all runs, grouped by system."""
    all_tables = []
    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            table = load_run_metrics(run_path)
            if table is not None:
                all_tables.append(table)

    if not all_tables:
        return []

    combined = pa.concat_tables(all_tables)

    # Group by system_id and compute aggregates
    results = []
    system_ids = pc.unique(combined.column("system_id")).to_pylist()  # type: ignore[attr-defined]

    for sys_id in system_ids:
        mask = pc.equal(combined.column("system_id"), sys_id)  # type: ignore[attr-defined]
        subset = combined.filter(mask)

        values = subset.column("metric_value").to_pylist()
        passed = subset.column("passed").to_pylist()

        results.append({
            "system_id": sys_id,
            "total_evaluations": len(values),
            "mean_score": sum(values) / len(values) if values else 0,
            "pass_rate": sum(passed) / len(passed) if passed else 0,
            "min_score": min(values) if values else 0,
            "max_score": max(values) if values else 0,
        })

    return sorted(results, key=lambda r: r["mean_score"], reverse=True)


def aggregate_by_domain(runs_dir: Path) -> list[dict[str, Any]]:
    """Aggregate metrics across all runs, grouped by domain."""
    all_tables = []
    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            table = load_run_metrics(run_path)
            if table is not None:
                all_tables.append(table)

    if not all_tables:
        return []

    combined = pa.concat_tables(all_tables)
    results = []
    domains = pc.unique(combined.column("domain")).to_pylist()  # type: ignore[attr-defined]

    for domain in domains:
        mask = pc.equal(combined.column("domain"), domain)  # type: ignore[attr-defined]
        subset = combined.filter(mask)
        values = subset.column("metric_value").to_pylist()
        passed = subset.column("passed").to_pylist()

        results.append({
            "domain": domain,
            "total_evaluations": len(values),
            "mean_score": sum(values) / len(values) if values else 0,
            "pass_rate": sum(passed) / len(passed) if passed else 0,
        })

    return results


def trend_over_runs(runs_dir: Path, system_id: str | None = None) -> list[dict[str, Any]]:
    """Show score trend over time for a system (or all systems)."""
    run_results = []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        table = load_run_metrics(run_path)
        if table is None:
            continue

        if system_id:
            mask = pc.equal(table.column("system_id"), system_id)  # type: ignore[attr-defined]
            table = table.filter(mask)

        if table.num_rows == 0:
            continue

        values = table.column("metric_value").to_pylist()
        timestamps = table.column("timestamp").to_pylist()
        passed = table.column("passed").to_pylist()

        run_results.append({
            "run_dir": run_path.name[:8],
            "timestamp": timestamps[0] if timestamps else "",
            "evaluations": len(values),
            "mean_score": sum(values) / len(values) if values else 0,
            "pass_rate": sum(passed) / len(passed) if passed else 0,
        })

    return run_results


def cross_system_comparison(runs_dir: Path, domain: str) -> list[dict[str, Any]]:
    """Compare all systems on a specific domain across runs."""
    all_tables = []
    for run_path in runs_dir.iterdir():
        if run_path.is_dir():
            table = load_run_metrics(run_path)
            if table is not None:
                all_tables.append(table)

    if not all_tables:
        return []

    combined = pa.concat_tables(all_tables)
    # Filter by domain
    mask = pc.equal(combined.column("domain"), domain)  # type: ignore[attr-defined]
    domain_table = combined.filter(mask)

    if domain_table.num_rows == 0:
        return []

    results = []
    system_ids = pc.unique(domain_table.column("system_id")).to_pylist()  # type: ignore[attr-defined]

    for sys_id in system_ids:
        sys_mask = pc.equal(domain_table.column("system_id"), sys_id)  # type: ignore[attr-defined]
        subset = domain_table.filter(sys_mask)
        values = subset.column("metric_value").to_pylist()
        passed = subset.column("passed").to_pylist()

        results.append({
            "system_id": sys_id,
            "domain": domain,
            "evaluations": len(values),
            "mean_score": sum(values) / len(values) if values else 0,
            "pass_rate": sum(passed) / len(passed) if passed else 0,
        })

    return sorted(results, key=lambda r: r["mean_score"], reverse=True)
