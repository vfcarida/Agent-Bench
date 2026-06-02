"""Unit tests for model and tool adapters."""

import pytest

from agent_bench.models.stub import StubModelAdapter
from agent_bench.tools.pix_tools import CheckBalanceTool, ValidatePixKeyTool


@pytest.mark.asyncio
async def test_stub_model():
    model = StubModelAdapter(responses=["Hello", "World"])
    r1 = await model.generate([{"role": "user", "content": "hi"}])
    assert r1.content == "Hello"
    r2 = await model.generate([{"role": "user", "content": "hi"}])
    assert r2.content == "World"
    r3 = await model.generate([{"role": "user", "content": "hi"}])
    assert r3.content == "Hello"  # wraps around


@pytest.mark.asyncio
async def test_check_balance_tool():
    tool = CheckBalanceTool()
    assert tool.name == "check_balance"
    result = await tool.execute({})
    assert result.output["balance"] == 5000.00


@pytest.mark.asyncio
async def test_validate_pix_key_valid():
    tool = ValidatePixKeyTool()
    result = await tool.execute({"pix_key": "123.456.789-00"})
    assert result.output["valid"] is True


@pytest.mark.asyncio
async def test_validate_pix_key_invalid():
    tool = ValidatePixKeyTool()
    result = await tool.execute({"pix_key": "xyz-invalida-key"})
    assert result.output["valid"] is False
    assert result.success is False
