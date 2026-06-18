"""SwiReasoning-specific metric computations.

Captures efficiency metrics for Athena's <think> block reasoning,
including Overthinking Index and Entropy Efficiency.
"""

from __future__ import annotations

from typing import Any

from agent_bench.graders.state_grader import GradeResult
from agent_bench.graders.thinking_parser import ParsedResponse


def compute_overthinking_index(parsed: ParsedResponse) -> float:
    """Ratio of thinking tokens to total tokens. Lower = more efficient.

    An overthinking index of 0.0 means no thinking was done.
    An overthinking index of 1.0 means the entire response was thinking
    with no actual answer.

    A healthy range for complex reasoning tasks is 0.3–0.6.

    Args:
        parsed: A ParsedResponse from the thinking parser.

    Returns:
        Float between 0.0 and 1.0 representing the thinking ratio.
    """
    return parsed.thinking_ratio


def compute_entropy_efficiency(
    parsed: ParsedResponse,
    grade_result: GradeResult,
) -> float:
    """Score per thinking token — how much value per unit of reasoning.

    Higher values mean the model is extracting more value from its
    reasoning process. Defined as: grade_score / max(thinking_tokens, 1)

    This helps identify:
    - Models that reason efficiently (high score, low thinking)
    - Models that overthink (low score, high thinking)
    - Models that don't reason enough (low score, low thinking)

    Args:
        parsed: A ParsedResponse from the thinking parser.
        grade_result: The grading result for the same response.

    Returns:
        Entropy efficiency score (unbounded positive float).
    """
    thinking_tokens = max(parsed.thinking_token_count, 1)
    return grade_result.score / thinking_tokens


def compute_thinking_depth(parsed: ParsedResponse) -> dict[str, Any]:
    """Analyze the structure and depth of thinking blocks.

    Args:
        parsed: A ParsedResponse from the thinking parser.

    Returns:
        Dict with block count, average length, max length, and total tokens.
    """
    if not parsed.thinking_blocks:
        return {
            "block_count": 0,
            "avg_block_tokens": 0,
            "max_block_tokens": 0,
            "total_thinking_tokens": 0,
        }

    block_lengths = [max(1, len(block.split())) for block in parsed.thinking_blocks]

    return {
        "block_count": len(parsed.thinking_blocks),
        "avg_block_tokens": sum(block_lengths) / len(block_lengths),
        "max_block_tokens": max(block_lengths),
        "total_thinking_tokens": parsed.thinking_token_count,
    }


def compute_reasoning_quality(
    parsed: ParsedResponse,
    grade_result: GradeResult,
) -> dict[str, float]:
    """Composite reasoning quality assessment.

    Combines overthinking index, entropy efficiency, and correctness
    into a single quality profile.

    Args:
        parsed: A ParsedResponse from the thinking parser.
        grade_result: The grading result for the same response.

    Returns:
        Dict with quality metrics.
    """
    overthinking = compute_overthinking_index(parsed)
    efficiency = compute_entropy_efficiency(parsed, grade_result)

    # Reasoning ROI: did thinking actually help?
    # If model got it right with minimal thinking, that's ideal.
    # If model thought a lot but still failed, that's worst case.
    if parsed.has_thinking:
        # Penalize high thinking with low scores
        roi = grade_result.score * (1 - overthinking * 0.5)
    else:
        roi = grade_result.score

    return {
        "overthinking_index": overthinking,
        "entropy_efficiency": efficiency,
        "reasoning_roi": roi,
        "correctness": grade_result.score,
        "had_thinking": 1.0 if parsed.has_thinking else 0.0,
    }


def aggregate_reasoning_metrics(
    parsed_responses: list[ParsedResponse],
    grade_results: list[GradeResult],
) -> dict[str, float]:
    """Suite-level reasoning efficiency summary.

    Aggregates individual response metrics into a suite-level
    summary for reporting and leaderboard display.

    Args:
        parsed_responses: List of ParsedResponse objects for all cases.
        grade_results: Corresponding list of GradeResult objects.

    Returns:
        Dict with aggregated reasoning metrics.
    """
    if not parsed_responses or not grade_results:
        return {
            "mean_overthinking_index": 0.0,
            "mean_entropy_efficiency": 0.0,
            "mean_reasoning_roi": 0.0,
            "thinking_usage_rate": 0.0,
            "total_thinking_tokens": 0,
            "total_answer_tokens": 0,
        }

    n = min(len(parsed_responses), len(grade_results))

    overthinking_values: list[float] = []
    efficiency_values: list[float] = []
    roi_values: list[float] = []
    thinking_count = 0
    total_thinking_tokens = 0
    total_answer_tokens = 0

    for i in range(n):
        parsed = parsed_responses[i]
        grade = grade_results[i]

        quality = compute_reasoning_quality(parsed, grade)
        overthinking_values.append(quality["overthinking_index"])
        efficiency_values.append(quality["entropy_efficiency"])
        roi_values.append(quality["reasoning_roi"])

        if parsed.has_thinking:
            thinking_count += 1

        total_thinking_tokens += parsed.thinking_token_count
        total_answer_tokens += parsed.answer_token_count

    return {
        "mean_overthinking_index": sum(overthinking_values) / n,
        "mean_entropy_efficiency": sum(efficiency_values) / n,
        "mean_reasoning_roi": sum(roi_values) / n,
        "thinking_usage_rate": thinking_count / n,
        "total_thinking_tokens": total_thinking_tokens,
        "total_answer_tokens": total_answer_tokens,
    }
