"""Unit tests for deterministic judge."""

import pytest

from agent_bench.core.scenarios import RefusalMode, Task
from agent_bench.judges.deterministic import DeterministicJudge


@pytest.fixture
def judge():
    return DeterministicJudge()


def _make_task(**kwargs) -> Task:
    defaults = {
        "task_id": "TEST_001",
        "domain": "test",
        "name": "Test task",
        "description": "",
        "input_messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return Task(**defaults)


@pytest.mark.asyncio
async def test_state_match_full(judge):
    task = _make_task(expected_final_state={"balance": 100, "status": "done"})
    result = {"final_state": {"balance": 100, "status": "done"}}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is True
    assert verdict.score == 1.0


@pytest.mark.asyncio
async def test_state_match_partial(judge):
    task = _make_task(expected_final_state={"a": 1, "b": 2, "c": 3})
    result = {"final_state": {"a": 1, "b": 99, "c": 3}}
    verdict = await judge.evaluate(task, result, [])
    assert abs(verdict.score - 0.67) < 0.1


@pytest.mark.asyncio
async def test_state_match_none(judge):
    task = _make_task(expected_final_state={"x": 1})
    result = {"final_state": {"y": 2}}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is False
    assert verdict.score == 0.0


@pytest.mark.asyncio
async def test_refusal_expected_and_detected(judge):
    task = _make_task(expected_refusal_mode=RefusalMode.POLITE_DECLINE)
    result = {"refusal": True, "response": "Cannot do that."}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_refusal_expected_but_not_detected(judge):
    task = _make_task(expected_refusal_mode=RefusalMode.POLITE_DECLINE)
    result = {"response": "Sure, done!", "refusal": False}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_no_expected_state_with_response(judge):
    task = _make_task()
    result = {"response": "Here is your answer"}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is True
