"""Unit tests for numeric correctness judge."""

import pytest

from agent_bench.core.scenarios import Task
from agent_bench.judges.numeric import NumericCorrectnessJudge, _extract_numbers, _parse_br_number


@pytest.fixture
def judge():
    return NumericCorrectnessJudge(default_tolerance=0.02)


def _make_task(**kwargs) -> Task:
    defaults = {
        "task_id": "TEST_N01",
        "domain": "test",
        "name": "Numeric test",
        "description": "",
        "input_messages": [{"role": "user", "content": "calc"}],
        "metadata": {},
    }
    defaults.update(kwargs)
    return Task(**defaults)


class TestNumberExtraction:
    def test_br_format(self):
        assert _parse_br_number("1.000,50") == 1000.50

    def test_comma_decimal(self):
        assert _parse_br_number("13,365") == 13.365

    def test_plain_int(self):
        assert _parse_br_number("250000") == 250000.0

    def test_extract_from_text(self):
        text = "Rendimento bruto: R$ 1.000,50. Taxa: 13,365%. Total: 55180 reais."
        numbers = _extract_numbers(text)
        assert 1000.50 in numbers or any(abs(n - 1000.5) < 1 for n in numbers)
        assert 13.365 in numbers or any(abs(n - 13.365) < 0.1 for n in numbers)

    def test_percentage(self):
        numbers = _extract_numbers("Taxa de 22,5%")
        assert any(abs(n - 22.5) < 0.1 for n in numbers)


@pytest.mark.asyncio
async def test_correct_numeric_values(judge):
    task = _make_task(
        metadata={"expected_numeric_values": {"return_pct": 13.365, "net_value": 55180.0}}
    )
    result = {"response": "Retorno bruto de 13,365%. Valor líquido: R$55.180,00."}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is True
    assert verdict.score >= 0.8


@pytest.mark.asyncio
async def test_incorrect_numeric_values(judge):
    task = _make_task(
        metadata={"expected_numeric_values": {"return_pct": 13.365, "net_value": 55180.0}}
    )
    result = {"response": "Retorno de 20%. Valor: R$60.000."}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is False


@pytest.mark.asyncio
async def test_no_expectations(judge):
    task = _make_task(metadata={})
    result = {"response": "Sem números esperados."}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is True


@pytest.mark.asyncio
async def test_tolerance_override(judge):
    task = _make_task(
        metadata={
            "expected_numeric_values": {"approx_val": 100.0},
            "numeric_tolerances": {"approx_val": 0.10},
        }
    )
    result = {"response": "O valor é aproximadamente 108."}
    verdict = await judge.evaluate(task, result, [])
    assert verdict.passed is True
