"""Unit tests for storage (JSONL and Parquet)."""

import json
from datetime import datetime, timezone

import pytest

from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType
from agent_bench.storage.jsonl import (
    load_traces_jsonl,
    save_metrics_jsonl,
    save_run_manifest,
    save_traces_jsonl,
)
from agent_bench.storage.parquet import load_metrics_parquet, save_metrics_parquet


class TestJSONL:
    def test_save_and_load_traces(self, tmp_path):
        traces = [
            TraceEvent(event_type=TraceEventType.PROMPT_SENT, data={"msg": "hello"}),
            TraceEvent(event_type=TraceEventType.MODEL_RESPONSE, data={"content": "world"}),
        ]
        path = tmp_path / "traces.jsonl"
        save_traces_jsonl(traces, path)

        loaded = load_traces_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["event_type"] == "prompt_sent"
        assert loaded[1]["data"]["content"] == "world"

    def test_save_metrics(self, tmp_path):
        metrics = [
            {"run_id": "r1", "task_id": "t1", "metric_name": "task_success", "metric_value": 1.0},
            {"run_id": "r1", "task_id": "t2", "metric_name": "task_success", "metric_value": 0.0},
        ]
        path = tmp_path / "metrics.jsonl"
        save_metrics_jsonl(metrics, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_save_manifest(self, tmp_path):
        artifact = RunArtifact(suite_id="s1", system_id="sys1")
        artifact.tasks_total = 5
        artifact.tasks_passed = 3
        artifact.finalize()
        path = save_run_manifest(artifact, tmp_path)
        data = json.loads(path.read_text())
        assert data["suite_id"] == "s1"
        assert data["tasks_passed"] == 3


class TestParquet:
    def test_save_and_load_metrics(self, tmp_path):
        records = [
            {
                "run_id": "r1", "system_id": "sys_a", "task_id": "t1",
                "domain": "pix", "metric_name": "task_success",
                "metric_value": 1.0, "metric_category": "functional",
                "passed": True, "timestamp": "2026-01-01T00:00:00",
            },
            {
                "run_id": "r1", "system_id": "sys_a", "task_id": "t2",
                "domain": "pix", "metric_name": "latency_ms",
                "metric_value": 500.0, "metric_category": "latency",
                "passed": True, "timestamp": "2026-01-01T00:00:01",
            },
        ]
        path = tmp_path / "metrics.parquet"
        save_metrics_parquet(records, path)
        assert path.exists()

        loaded = load_metrics_parquet(path)
        assert len(loaded) == 2
        assert loaded[0]["metric_name"] == "task_success"
        assert loaded[1]["metric_value"] == 500.0

    def test_empty_parquet(self, tmp_path):
        path = tmp_path / "empty.parquet"
        save_metrics_parquet([], path)
        assert path.exists()
        loaded = load_metrics_parquet(path)
        assert len(loaded) == 0
