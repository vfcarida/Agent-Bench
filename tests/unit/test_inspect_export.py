"""Unit tests for Inspect AI export bridge."""

import json
from pathlib import Path

from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType
from agent_bench.core.scenarios import BusinessCriticality, RefusalMode, Severity, Task
from agent_bench.export.inspect_ai import (
    export_tasks_to_inspect_dataset,
    run_artifact_to_inspect_log,
    task_to_inspect_sample,
)


def test_task_to_inspect_sample():
    task = Task(
        task_id="INSPECT_001",
        domain="pix_assist",
        name="Inspect Test Case",
        description="Verify Inspect AI sample conversion",
        input_messages=[
            {"role": "system", "content": "You are a banking assistant."},
            {"role": "user", "content": "Transfer 500 BRL via PIX."},
        ],
        allowed_tools=["execute_pix_transfer", "validate_pix_key"],
        expected_final_state={"transfer_executed": True},
        gold_references=["Transfer executed successfully."],
        severity=Severity.HIGH,
        business_criticality=BusinessCriticality.FINANCIAL,
        expected_refusal_mode=RefusalMode.NONE,
        tags=["transfer", "pix"],
    )

    sample = task_to_inspect_sample(task)

    assert sample["id"] == "INSPECT_001"
    assert len(sample["input"]) == 2
    assert sample["input"][0]["role"] == "system"
    assert sample["input"][1]["content"] == "Transfer 500 BRL via PIX."
    assert sample["target"] == ["Transfer executed successfully."]
    assert sample["metadata"]["domain"] == "pix_assist"
    assert sample["metadata"]["severity"] == "high"
    assert sample["metadata"]["business_criticality"] == "financial"
    assert sample["sandbox"]["tools"] == ["execute_pix_transfer", "validate_pix_key"]


def test_export_tasks_to_inspect_dataset_jsonl(tmp_path: Path):
    tasks = [
        Task(
            task_id=f"T_{i}",
            domain="cyber_sandbox",
            name=f"Cyber Task {i}",
            description="Inspect export task",
            input_messages=[{"role": "user", "content": f"Command {i}"}],
        )
        for i in range(3)
    ]
    out_file = tmp_path / "inspect_dataset.jsonl"
    samples = export_tasks_to_inspect_dataset(tasks, output_path=out_file)

    assert len(samples) == 3
    assert out_file.exists()

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 3
    assert lines[0]["id"] == "T_0"
    assert lines[1]["id"] == "T_1"


def test_run_artifact_to_inspect_log():
    artifact = RunArtifact(
        run_id="run_inspect_test_99",
        suite_id="pix_suite",
        system_id="test_agent",
        model_id="gpt-4o",
        tasks_total=1,
        tasks_passed=1,
        tasks_failed=0,
        metrics={"global_score": 0.95},
    )
    artifact.traces.append(
        TraceEvent(
            event_type=TraceEventType.USER_MESSAGE,
            data={"task_id": "T_1", "content": "Help me with PIX"},
        )
    )
    artifact.traces.append(
        TraceEvent(
            event_type=TraceEventType.TOOL_CALL,
            data={"task_id": "T_1", "tool_name": "validate_pix_key"},
        )
    )
    artifact.traces.append(
        TraceEvent(
            event_type=TraceEventType.MODEL_RESPONSE,
            data={"task_id": "T_1", "response": "Key is valid."},
        )
    )
    artifact.finalize()

    log_dict = run_artifact_to_inspect_log(artifact)

    assert log_dict["eval"]["benchmark"] == "Agent-Bench"
    assert log_dict["eval"]["run_id"] == "run_inspect_test_99"
    assert log_dict["results"]["tasks_passed"] == 1
    assert len(log_dict["samples"]) == 1
    assert log_dict["samples"][0]["id"] == "T_1"
    assert log_dict["samples"][0]["input"] == "Help me with PIX"
    assert log_dict["samples"][0]["output"] == "Key is valid."
