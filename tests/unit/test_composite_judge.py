"""Unit tests for composite judge."""

import pytest

from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.scenarios import RefusalMode, Task
from agent_bench.judges.composite import CompositeJudge


@pytest.fixture
def judge():
    return CompositeJudge()


def _make_task(**kwargs) -> Task:
    defaults = {
        "task_id": "TEST_C01",
        "domain": "test",
        "name": "Composite test",
        "description": "",
        "input_messages": [{"role": "user", "content": "test"}],
        "metadata": {},
    }
    defaults.update(kwargs)
    return Task(**defaults)


@pytest.mark.asyncio
async def test_deterministic_only(judge):
    task = _make_task(expected_final_state={"done": True})
    result = {"final_state": {"done": True}, "response": "Done."}
    final, all_v = await judge.evaluate(task, result, [])
    assert final.passed is True
    assert len(all_v) == 1  # only deterministic


@pytest.mark.asyncio
async def test_with_numeric(judge):
    task = _make_task(
        expected_final_state={"done": True},
        metadata={"expected_numeric_values": {"rate": 10.5}},
    )
    result = {"final_state": {"done": True}, "response": "A taxa é 10,5%."}
    final, all_v = await judge.evaluate(task, result, [])
    assert len(all_v) == 2  # deterministic + numeric
    assert final.passed is True


@pytest.mark.asyncio
async def test_with_retrieval(judge):
    task = _make_task(expected_final_state={"done": True})
    result = {
        "final_state": {"done": True},
        "response": "CDB rende 110% CDI.",
        "retrieved_documents": [
            {"content": "CDB rende 110% CDI.", "source": "manual"}
        ],
    }
    final, all_v = await judge.evaluate(task, result, [])
    assert len(all_v) == 2  # deterministic + grounding
    assert final.passed is True


@pytest.mark.asyncio
async def test_all_judges(judge):
    task = _make_task(
        expected_final_state={"done": True},
        metadata={"expected_numeric_values": {"val": 100.0}},
        gold_references=["Value is 100"],
    )
    result = {
        "final_state": {"done": True},
        "response": "The value is 100. [manual_rf]",
        "retrieved_documents": [
            {"content": "Value is 100.", "source": "manual_rf"}
        ],
    }
    final, all_v = await judge.evaluate(task, result, [])
    assert len(all_v) == 3  # deterministic + numeric + grounding
    assert final.judge_id == "composite"


@pytest.mark.asyncio
async def test_refusal_with_composite(judge):
    task = _make_task(expected_refusal_mode=RefusalMode.POLITE_DECLINE)
    result = {"response": "Cannot do that.", "refusal": True}
    final, all_v = await judge.evaluate(task, result, [])
    assert final.passed is True
