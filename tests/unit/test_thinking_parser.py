"""Tests for the SwiReasoning thinking parser and reasoning metrics."""

from __future__ import annotations

import pytest

from agent_bench.graders.state_grader import GradeResult
from agent_bench.graders.thinking_parser import (
    ParsedResponse,
    ThinkingAwareGraderMiddleware,
    parse_thinking_response,
)
from agent_bench.metrics.reasoning_metrics import (
    aggregate_reasoning_metrics,
    compute_entropy_efficiency,
    compute_overthinking_index,
    compute_reasoning_quality,
    compute_thinking_depth,
)


# ──────────────────────────────────────────────────────────────────
# parse_thinking_response Tests
# ──────────────────────────────────────────────────────────────────


class TestParseThinkingResponse:
    """Test the <think> block parser."""

    def test_no_thinking_blocks(self):
        """Response without <think> tags should pass through unchanged."""
        response = "The answer is 42."
        parsed = parse_thinking_response(response)

        assert parsed.raw_response == response
        assert parsed.clean_response == response
        assert parsed.thinking_blocks == []
        assert parsed.has_thinking is False
        assert parsed.thinking_ratio == 0.0

    def test_single_thinking_block(self):
        """Single <think> block should be extracted."""
        response = "<think>Let me work through this step by step.</think>The answer is 42."
        parsed = parse_thinking_response(response)

        assert parsed.has_thinking is True
        assert len(parsed.thinking_blocks) == 1
        assert "step by step" in parsed.thinking_blocks[0]
        assert "The answer is 42." in parsed.clean_response
        assert "<think>" not in parsed.clean_response
        assert "</think>" not in parsed.clean_response

    def test_multiple_thinking_blocks(self):
        """Multiple <think> blocks should all be extracted."""
        response = (
            "<think>First thought.</think>"
            "Intermediate text. "
            "<think>Second thought.</think>"
            "Final answer."
        )
        parsed = parse_thinking_response(response)

        assert parsed.has_thinking is True
        assert len(parsed.thinking_blocks) == 2
        assert "First thought." in parsed.thinking_blocks[0]
        assert "Second thought." in parsed.thinking_blocks[1]
        assert "Intermediate text." in parsed.clean_response
        assert "Final answer." in parsed.clean_response

    def test_multiline_thinking_block(self):
        """<think> blocks can contain newlines."""
        response = (
            "<think>\nLine 1.\nLine 2.\nLine 3.\n</think>\n"
            "The answer is clear."
        )
        parsed = parse_thinking_response(response)

        assert parsed.has_thinking is True
        assert "Line 1." in parsed.thinking_blocks[0]
        assert "Line 2." in parsed.thinking_blocks[0]
        assert "The answer is clear." in parsed.clean_response

    def test_empty_response(self):
        """Empty response should return empty parsed result."""
        parsed = parse_thinking_response("")

        assert parsed.raw_response == ""
        assert parsed.clean_response == ""
        assert parsed.thinking_blocks == []
        assert parsed.has_thinking is False

    def test_thinking_only_response(self):
        """Response with only <think> blocks should have empty clean response."""
        response = "<think>Just thinking, no answer.</think>"
        parsed = parse_thinking_response(response)

        assert parsed.has_thinking is True
        assert parsed.clean_response == ""
        assert parsed.thinking_ratio > 0.0

    def test_token_counts(self):
        """Token counts should be reasonable estimates."""
        response = "<think>one two three four five</think>answer here now"
        parsed = parse_thinking_response(response)

        assert parsed.thinking_token_count > 0
        assert parsed.answer_token_count > 0
        assert parsed.total_token_count == parsed.thinking_token_count + parsed.answer_token_count

    def test_thinking_ratio(self):
        """Thinking ratio should be between 0 and 1."""
        response = "<think>thinking content here</think>answer"
        parsed = parse_thinking_response(response)

        assert 0.0 < parsed.thinking_ratio < 1.0


# ──────────────────────────────────────────────────────────────────
# ThinkingAwareGraderMiddleware Tests
# ──────────────────────────────────────────────────────────────────


class TestThinkingAwareGraderMiddleware:
    """Test the grader middleware that strips <think> blocks."""

    def _make_mock_grader(self, return_score: float = 0.9):
        """Create a mock grader that records what it received."""

        class MockGrader:
            def __init__(self):
                self.last_response = None

            def grade(self, case, actual_state=None, actual_tool_calls=None,
                      actual_response="", judge_fn=None):
                self.last_response = actual_response
                return GradeResult(
                    score=return_score,
                    passed=return_score >= 0.6,
                    strategy="mock",
                    details={},
                )

        return MockGrader()

    def test_strips_thinking_before_grading(self):
        """Middleware should pass clean response to inner grader."""
        mock = self._make_mock_grader()
        middleware = ThinkingAwareGraderMiddleware(mock)

        result = middleware.grade(
            case={},
            actual_response="<think>reasoning here</think>The actual answer.",
        )

        assert "<think>" not in mock.last_response
        assert "The actual answer." in mock.last_response

    def test_thinking_analysis_in_details(self):
        """Grade result should include thinking analysis metadata."""
        mock = self._make_mock_grader()
        middleware = ThinkingAwareGraderMiddleware(mock)

        result = middleware.grade(
            case={},
            actual_response="<think>I think therefore I am.</think>42",
        )

        assert "thinking_analysis" in result.details
        analysis = result.details["thinking_analysis"]
        assert analysis["had_thinking"] is True
        assert analysis["thinking_blocks_count"] == 1
        assert analysis["thinking_token_count"] > 0

    def test_no_thinking_passthrough(self):
        """Without <think> blocks, response passes through unchanged."""
        mock = self._make_mock_grader()
        middleware = ThinkingAwareGraderMiddleware(mock)

        result = middleware.grade(
            case={},
            actual_response="Just a normal answer.",
        )

        assert mock.last_response == "Just a normal answer."
        assert result.details["thinking_analysis"]["had_thinking"] is False

    def test_preserves_grade_result(self):
        """Middleware should preserve the inner grader's score."""
        mock = self._make_mock_grader(return_score=0.85)
        middleware = ThinkingAwareGraderMiddleware(mock)

        result = middleware.grade(
            case={},
            actual_response="<think>thinking</think>answer",
        )

        assert result.score == 0.85
        assert result.passed is True


# ──────────────────────────────────────────────────────────────────
# Reasoning Metrics Tests
# ──────────────────────────────────────────────────────────────────


class TestOverthinkingIndex:
    """Test the overthinking index metric."""

    def test_no_thinking(self):
        """No thinking should give index of 0."""
        parsed = parse_thinking_response("just an answer")
        assert compute_overthinking_index(parsed) == 0.0

    def test_with_thinking(self):
        """Thinking response should give positive index."""
        parsed = parse_thinking_response("<think>lots of reasoning</think>short answer")
        index = compute_overthinking_index(parsed)
        assert 0.0 < index < 1.0

    def test_all_thinking(self):
        """All-thinking response should give index near 1.0."""
        parsed = parse_thinking_response("<think>everything is thinking here</think>")
        index = compute_overthinking_index(parsed)
        assert index == 1.0  # No answer tokens


class TestEntropyEfficiency:
    """Test the entropy efficiency metric."""

    def test_high_score_low_thinking(self):
        """High score with low thinking = high efficiency."""
        parsed = parse_thinking_response("<think>brief</think>correct answer")
        grade = GradeResult(score=1.0, passed=True, strategy="test")

        efficiency = compute_entropy_efficiency(parsed, grade)
        assert efficiency > 0

    def test_zero_score(self):
        """Zero score should give zero efficiency."""
        parsed = parse_thinking_response("<think>long thinking</think>wrong answer")
        grade = GradeResult(score=0.0, passed=False, strategy="test")

        efficiency = compute_entropy_efficiency(parsed, grade)
        assert efficiency == 0.0


class TestThinkingDepth:
    """Test thinking depth analysis."""

    def test_no_blocks(self):
        """No thinking blocks should return zero metrics."""
        parsed = parse_thinking_response("no thinking")
        depth = compute_thinking_depth(parsed)

        assert depth["block_count"] == 0
        assert depth["avg_block_tokens"] == 0

    def test_multiple_blocks(self):
        """Multiple blocks should be counted correctly."""
        parsed = parse_thinking_response(
            "<think>first block here</think>"
            "middle"
            "<think>second block here too</think>"
            "end"
        )
        depth = compute_thinking_depth(parsed)

        assert depth["block_count"] == 2
        assert depth["avg_block_tokens"] > 0
        assert depth["max_block_tokens"] > 0


class TestAggregateReasoningMetrics:
    """Test suite-level metric aggregation."""

    def test_empty_inputs(self):
        """Empty inputs should return zero metrics."""
        result = aggregate_reasoning_metrics([], [])

        assert result["mean_overthinking_index"] == 0.0
        assert result["thinking_usage_rate"] == 0.0

    def test_mixed_responses(self):
        """Mix of thinking and non-thinking responses."""
        parsed_list = [
            parse_thinking_response("<think>thinking</think>answer1"),
            parse_thinking_response("answer2 without thinking"),
            parse_thinking_response("<think>more thinking</think>answer3"),
        ]
        grades = [
            GradeResult(score=0.9, passed=True, strategy="test"),
            GradeResult(score=0.8, passed=True, strategy="test"),
            GradeResult(score=0.7, passed=True, strategy="test"),
        ]

        result = aggregate_reasoning_metrics(parsed_list, grades)

        assert 0.0 < result["thinking_usage_rate"] <= 1.0
        assert result["total_thinking_tokens"] > 0
        assert result["total_answer_tokens"] > 0


class TestReasoningQuality:
    """Test composite reasoning quality assessment."""

    def test_perfect_no_thinking(self):
        """Perfect score without thinking should have high ROI."""
        parsed = parse_thinking_response("perfect answer")
        grade = GradeResult(score=1.0, passed=True, strategy="test")

        quality = compute_reasoning_quality(parsed, grade)
        assert quality["correctness"] == 1.0
        assert quality["reasoning_roi"] == 1.0  # No thinking penalty
        assert quality["had_thinking"] == 0.0

    def test_failed_with_heavy_thinking(self):
        """Failed response with heavy thinking should have low ROI."""
        parsed = parse_thinking_response(
            "<think>very very long thinking chain that goes on and on</think>wrong"
        )
        grade = GradeResult(score=0.2, passed=False, strategy="test")

        quality = compute_reasoning_quality(parsed, grade)
        assert quality["correctness"] == 0.2
        assert quality["reasoning_roi"] < quality["correctness"]  # Penalized
