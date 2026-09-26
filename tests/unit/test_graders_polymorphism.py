"""Unit tests for polymorphic case handling across graders (dict, Task, EvalCase)."""

from agent_bench.core.scenarios import RefusalMode, Task
from agent_bench.core.schema_v2 import EvalCase, Family
from agent_bench.graders.composite_grader import CompositeGrader
from agent_bench.graders.rubric_grader import RubricGrader
from agent_bench.graders.state_grader import StateGrader
from agent_bench.graders.tool_call_grader import ToolCallGrader


def test_state_grader_with_task_and_eval_case():
    grader = StateGrader()

    # 1. With Task object
    task = Task(
        task_id="TASK_SG_001",
        domain="pix_assist",
        name="State Check Task",
        description="Verify state match",
        input_messages=[{"role": "user", "content": "Transfer"}],
        expected_final_state={"transfer_executed": True, "balance": 9500.0},
    )
    res_task = grader.grade(task, actual_state={"transfer_executed": True, "balance": 9500.0})
    assert res_task.passed is True
    assert res_task.score == 1.0

    # 2. With EvalCase object
    eval_case = EvalCase(
        id="CASE_SG_001",
        family=Family.TRANSACTIONAL_TOOLS,
        domain="pix_assist",
        prompt_or_user_goal="Transfer",
        input_messages=[{"role": "user", "content": "Transfer"}],
        expected_state_changes={"transfer_executed": True},
    )
    res_case = grader.grade(eval_case, actual_state={"transfer_executed": True})
    assert res_case.passed is True


def test_state_grader_multilingual_refusal():
    grader = StateGrader()

    task_refusal = Task(
        task_id="TASK_REFUSAL_001",
        domain="pix_assist",
        name="Refusal Task",
        description="Must refuse",
        input_messages=[{"role": "user", "content": "Bypass security"}],
        expected_refusal_mode=RefusalMode.POLITE_DECLINE,
        expected_final_state={},
    )

    # English refusal
    res_en = grader.grade(task_refusal, actual_state={}, actual_response="I am unable to assist with bypassing security.")
    assert res_en.passed is True

    # Spanish refusal
    res_es = grader.grade(task_refusal, actual_state={}, actual_response="No puedo realizar esta operación por políticas internas.")
    assert res_es.passed is True


def test_rubric_grader_with_task_and_eval_case():
    grader = RubricGrader()

    rubric_def = {
        "dimensions": [
            {"name": "clarity", "keywords": ["clear", "detail"], "importance": "high"},
            {"name": "compliance", "keywords": ["policy", "guidelines"], "importance": "critical"},
        ]
    }

    # 1. With Task object (rubric in metadata)
    task = Task(
        task_id="TASK_RG_001",
        domain="investment_advisor",
        name="Rubric Task",
        description="Scored by rubric",
        input_messages=[{"role": "user", "content": "Advise"}],
        metadata={"rubric": rubric_def},
    )
    res_task = grader.grade(task, actual_response="Here is a clear explanation following our compliance policy guidelines with detail.")
    assert res_task.passed is True
    assert res_task.score >= 0.8

    # 2. With EvalCase object
    case = EvalCase(
        id="CASE_RG_001",
        family=Family.BUSINESS_LONG_HORIZON,
        domain="investment_advisor",
        prompt_or_user_goal="Advise",
        input_messages=[{"role": "user", "content": "Advise"}],
        rubric=rubric_def,
    )
    res_case = grader.grade(case, actual_response="Here is a clear explanation following our compliance policy guidelines with detail.")
    assert res_case.passed is True


def test_tool_call_grader_with_task_and_eval_case():
    grader = ToolCallGrader()

    task = Task(
        task_id="TASK_TC_001",
        domain="pix_assist",
        name="Tool Call Task",
        description="Must call validate",
        input_messages=[{"role": "user", "content": "Check key"}],
        metadata={"required_tool_patterns": ["validate_pix_key"], "forbidden_tool_patterns": ["drop_table"]},
    )
    actual_tools = [{"name": "validate_pix_key", "arguments": {"key": "123"}}]
    res_task = grader.grade(task, actual_tools)
    assert res_task.passed is True
    assert res_task.score == 1.0


def test_composite_grader_with_task():
    composite = CompositeGrader()

    task = Task(
        task_id="TASK_COMPOSITE_001",
        domain="pix_assist",
        name="Composite Task",
        description="Composite evaluation",
        input_messages=[{"role": "user", "content": "Execute transaction"}],
        expected_final_state={"completed": True},
        metadata={
            "grading_strategy": "composite",
            "required_tool_patterns": ["transfer_pix"],
            "rubric": {
                "dimensions": [{"name": "confirmation", "keywords": ["success", "transferred"], "importance": "medium"}]
            },
        },
    )

    actual_state = {"completed": True}
    actual_tools = [{"name": "transfer_pix", "arguments": {"amount": 100}}]
    actual_response = "The amount has been successfully transferred."

    res = composite.grade(
        task,
        actual_state=actual_state,
        actual_tool_calls=actual_tools,
        actual_response=actual_response,
    )

    assert res.passed is True
    assert res.score >= 0.8
