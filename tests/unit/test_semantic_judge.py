"""Unit tests for semantic judge."""

import pytest

from agent_bench.core.scenarios import Task
from agent_bench.judges.semantic import SemanticJudge, _parse_judge_response
from agent_bench.models.stub import StubModelAdapter


class TestParseJudgeResponse:
    def test_valid_json(self):
        content = '{"correctness": 0.9, "completeness": 0.8, "safety": 1.0, "helpfulness": 0.7, "tone": 0.9, "aggregate": 0.85, "reasoning": "Good response"}'
        result = _parse_judge_response(content)
        assert result["aggregate"] == 0.85

    def test_json_in_code_block(self):
        content = '```json\n{"aggregate": 0.7, "reasoning": "ok"}\n```'
        result = _parse_judge_response(content)
        assert result["aggregate"] == 0.7

    def test_malformed_fallback(self):
        content = "This is not JSON at all"
        result = _parse_judge_response(content)
        assert result["aggregate"] == 0.5  # fallback

    def test_partial_json(self):
        content = 'Some text {"aggregate": 0.6, "reasoning": "partial"} more text'
        result = _parse_judge_response(content)
        assert result["aggregate"] == 0.6


@pytest.mark.asyncio
async def test_semantic_judge_with_stub():
    # Stub model returns a JSON-like response
    stub = StubModelAdapter(responses=['{"correctness": 0.8, "completeness": 0.7, "safety": 1.0, "helpfulness": 0.8, "tone": 0.9, "aggregate": 0.82, "reasoning": "Good quality response"}'])
    judge = SemanticJudge(model=stub)

    task = Task(
        task_id="T_001",
        domain="test",
        name="Test",
        description="Test task",
        input_messages=[{"role": "user", "content": "hello"}],
    )
    result = {"response": "Hello! How can I help?"}
    verdict = await judge.evaluate(task, result, [])

    assert verdict.score == 0.82
    assert verdict.passed is True
    assert verdict.judge_id == "semantic_stub-model"


@pytest.mark.asyncio
async def test_semantic_judge_error_handling():
    # Stub that returns garbage
    stub = StubModelAdapter(responses=["not json at all"])
    judge = SemanticJudge(model=stub)

    task = Task(
        task_id="T_001",
        domain="test",
        name="Test",
        description="Test",
        input_messages=[{"role": "user", "content": "x"}],
    )
    result = {"response": "y"}
    verdict = await judge.evaluate(task, result, [])

    # Should not crash, returns fallback
    assert verdict.score == 0.5
    assert verdict.passed is False
