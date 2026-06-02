"""Unit tests for grounding judge."""

import pytest

from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.scenarios import Task
from agent_bench.judges.grounding import GroundingJudge


@pytest.fixture
def judge():
    return GroundingJudge()


def _make_task(**kwargs) -> Task:
    defaults = {
        "task_id": "TEST_G01",
        "domain": "investment_advisor",
        "name": "Grounding test",
        "description": "",
        "input_messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return Task(**defaults)


@pytest.mark.asyncio
async def test_grounded_response(judge):
    task = _make_task()
    result = {
        "response": "O CDB rende 110% do CDI, garantido pelo FGC até R$250.000.",
        "retrieved_documents": [
            {
                "doc_id": "inv_001",
                "title": "CDB Pós-fixado",
                "content": "O CDB pós-fixado rende com base no CDI. Rentabilidade típica: 100-120% CDI. Garantido pelo FGC até R$250.000.",
                "source": "manual_rf",
            }
        ],
    }
    verdict = await judge.evaluate(task, result, [])
    assert verdict.score > 0.5
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_ungrounded_response(judge):
    task = _make_task()
    result = {
        "response": "Este fundo rende 500% ao ano com risco zero e garantia total.",
        "retrieved_documents": [
            {
                "doc_id": "inv_003",
                "title": "Fundos Multimercado",
                "content": "Fundos multimercado podem investir em renda fixa e variável. Taxa: 1,5-2,5%.",
                "source": "catalogo_fundos",
            }
        ],
    }
    verdict = await judge.evaluate(task, result, [])
    assert verdict.score < 0.7


@pytest.mark.asyncio
async def test_no_retrieval_docs(judge):
    task = _make_task()
    result = {"response": "Some response without docs.", "retrieved_documents": []}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is False
    assert "No retrieved documents" in verdict.reasoning


@pytest.mark.asyncio
async def test_retrieval_from_traces(judge):
    task = _make_task()
    result = {"response": "CDB rende 100% CDI com FGC."}
    traces = [
        TraceEvent(
            event_type=TraceEventType.RETRIEVAL_RESULT,
            data={
                "documents": [
                    {"content": "CDB rende 100% CDI. FGC garante.", "source": "src1"}
                ]
            },
        )
    ]
    verdict = await judge.evaluate(task, result, traces)
    assert verdict.score > 0.5


@pytest.mark.asyncio
async def test_citation_verification(judge):
    task = _make_task()
    result = {
        "response": "Conforme [manual_rf], o CDB rende 110% CDI.",
        "retrieved_documents": [
            {"doc_id": "d1", "title": "Manual RF", "content": "CDB rende 110% CDI.", "source": "manual_rf"}
        ],
    }
    verdict = await judge.evaluate(task, result, [])
    assert verdict.metadata["citation_score"] > 0.5
