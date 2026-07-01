"""JSONL trace and artifact storage."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_bench.core.artifacts import RunArtifact, TraceEvent


def save_traces_jsonl(traces: list[TraceEvent], output_path: Path) -> Path:
    """Save traces as JSONL file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for trace in traces:
                record = {
                    "event_id": trace.event_id,
                    "event_type": trace.event_type.value,
                    "timestamp": trace.timestamp.isoformat(),
                    "data": trace.data,
                    "parent_id": trace.parent_id,
                }
                f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise
    return output_path


def load_traces_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load traces from a JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def save_run_manifest(artifact: RunArtifact, output_dir: Path) -> Path:
    """Save run manifest as JSON with all metadata atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": artifact.run_id,
        "suite_id": artifact.suite_id,
        "system_id": artifact.system_id,
        "model_id": artifact.model_id,
        "started_at": artifact.started_at.isoformat(),
        "finished_at": artifact.finished_at.isoformat() if artifact.finished_at else None,
        "config_hash": artifact.config_hash,
        "benchmark_version": artifact.benchmark_version,
        "seed": artifact.seed,
        "tasks_total": artifact.tasks_total,
        "tasks_passed": artifact.tasks_passed,
        "tasks_failed": artifact.tasks_failed,
        "duration_ms": artifact.duration_ms,
        "metrics": artifact.metrics,
        "metadata": artifact.metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / f"{artifact.run_id}_manifest.json"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2, default=str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise
    return path


def save_metrics_jsonl(
    metrics: list[dict[str, Any]], output_path: Path
) -> Path:
    """Save per-task metrics as JSONL for analysis atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for m in metrics:
                f.write(json.dumps(m, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise
    return output_path
