"""Unit tests for Task and EvalCase schema unification, aliases, and round-trips."""

from agent_bench.core.scenarios import (
    BusinessCriticality,
    RefusalMode,
    Severity,
    Task,
)
from agent_bench.core.schema_v2 import EvalCase


def test_task_aliases() -> None:
    """Verify Task property aliases match EvalCase attribute names."""
    task = Task(
        task_id="TASK_001",
        domain="pix_assist",
        name="PIX Test",
        description="Test description",
        input_messages=[{"role": "user", "content": "Transfer 100 BRL to bob@email.com"}],
        initial_state={"balance": 500},
        expected_final_state={"balance": 400},
        severity=Severity.HIGH,
        business_criticality=BusinessCriticality.FINANCIAL,
        task_version="2.1.0",
    )

    # Aliases
    assert task.id == "TASK_001"
    assert task.prompt == "Transfer 100 BRL to bob@email.com"
    assert task.prompt_or_user_goal == "Transfer 100 BRL to bob@email.com"
    assert task.expected_state == {"balance": 400}
    assert task.expected_state_changes == {"balance": 400}
    assert task.version == "2.1.0"
    assert task.risk_level == "high"

    # Setter aliases
    task.id = "TASK_002"
    assert task.task_id == "TASK_002"
    task.version = "2.2.0"
    assert task.task_version == "2.2.0"


def test_eval_case_aliases() -> None:
    """Verify EvalCase property aliases match Task attribute names."""
    case = EvalCase(
        id="CASE_001",
        family="transactional_tools",
        domain="pix_assist",
        prompt_or_user_goal="Check balance",
        input_messages=[{"role": "user", "content": "Check balance"}],
        expected_state_changes={"checked": True},
        version="1.5.0",
    )

    assert case.task_id == "CASE_001"
    assert case.prompt == "Check balance"
    assert case.expected_final_state == {"checked": True}
    assert case.task_version == "1.5.0"

    # Setter aliases
    case.task_id = "CASE_002"
    assert case.id == "CASE_002"
    case.task_version = "1.6.0"
    assert case.version == "1.6.0"


def test_round_trip_task_to_eval_case_and_back() -> None:
    """Verify seamless bidirectional conversion between Task and EvalCase."""
    original_task = Task(
        task_id="TASK_PIX_99",
        domain="pix_assist",
        name="Transfer test",
        description="Perform PIX transfer",
        input_messages=[{"role": "user", "content": "Send 50 BRL via PIX"}],
        initial_state={"account": "ACC_1", "balance": 200},
        expected_final_state={"account": "ACC_1", "balance": 150},
        allowed_tools=["validate_pix_key", "execute_pix_transfer"],
        required_capabilities=["validate_recipient"],
        expected_refusal_mode=RefusalMode.NONE,
        gold_references=["DOC_PIX_RULES"],
        evidence_strings=[{"claim": "Limit is 1000", "evidence": "Sec 4.1"}],
        answer_format="tool_use",
        expected_deliverables=["transaction_receipt"],
        severity=Severity.HIGH,
        business_criticality=BusinessCriticality.FINANCIAL,
        tags=["pix", "transfer"],
        task_version="1.0.0",
        metadata={"author": "eval_team"},
    )

    # Task -> EvalCase
    eval_case = original_task.to_eval_case()
    assert isinstance(eval_case, EvalCase)
    assert eval_case.id == original_task.task_id
    assert eval_case.domain == original_task.domain
    assert eval_case.expected_state_changes == original_task.expected_final_state
    assert eval_case.allowed_tools == original_task.allowed_tools
    assert eval_case.evidence_strings == original_task.evidence_strings
    assert eval_case.metadata["author"] == "eval_team"

    # EvalCase -> Task
    reconstructed_task = eval_case.to_task()
    assert isinstance(reconstructed_task, Task)
    assert reconstructed_task.task_id == original_task.task_id
    assert reconstructed_task.domain == original_task.domain
    assert reconstructed_task.input_messages == original_task.input_messages
    assert reconstructed_task.expected_final_state == original_task.expected_final_state
    assert reconstructed_task.allowed_tools == original_task.allowed_tools
    assert reconstructed_task.gold_references == original_task.gold_references
    assert reconstructed_task.evidence_strings == original_task.evidence_strings
    assert reconstructed_task.severity == original_task.severity


def test_task_from_dict_eval_case_schema() -> None:
    """Verify Task.from_dict accepts EvalCase-formatted dictionary."""
    eval_case_dict = {
        "id": "CASE_YAML_01",
        "domain": "cyber_sandbox",
        "family": "security_guardrails",
        "prompt_or_user_goal": "Check open firewall ports",
        "input_messages": [{"role": "user", "content": "Check ports"}],
        "expected_state_changes": {"ports_checked": True},
        "risk_level": "critical",
        "business_criticality": "regulatory",
        "version": "3.0.0",
    }
    task = Task.from_dict(eval_case_dict)
    assert task.task_id == "CASE_YAML_01"
    assert task.domain == "cyber_sandbox"
    assert task.expected_final_state == {"ports_checked": True}
    assert task.severity == Severity.CRITICAL
    assert task.business_criticality == BusinessCriticality.REGULATORY
    assert task.task_version == "3.0.0"


def test_eval_case_from_dict_task_schema() -> None:
    """Verify EvalCase.from_dict accepts Task-formatted dictionary."""
    task_dict = {
        "task_id": "TASK_LEGACY_01",
        "domain": "pix_assist",
        "name": "Legacy Task",
        "description": "Legacy description",
        "prompt": "Legacy prompt",
        "expected_final_state": {"legacy_success": True},
        "severity": "medium",
        "task_version": "1.2.0",
    }
    case = EvalCase.from_dict(task_dict)
    assert case.id == "TASK_LEGACY_01"
    assert case.domain == "pix_assist"
    assert case.expected_state_changes == {"legacy_success": True}
    assert case.expected_final_state == {"legacy_success": True}
    assert case.version == "1.2.0"
