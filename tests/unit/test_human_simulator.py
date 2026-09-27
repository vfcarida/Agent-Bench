"""Unit tests for HumanUserSimulator and interactive CLI integration."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from agent_bench.cli.main import generate_security_suite
from agent_bench.core.scenarios import Task
from agent_bench.core.user_simulator import HumanUserSimulator


def _make_dummy_task() -> Task:
    return Task(
        task_id="DUMMY_SIM_001",
        name="Simulator Test",
        description="Test human simulator integration",
        domain="pix_assist",
        input_messages=[{"role": "user", "content": "I want to make a transfer"}],
    )


class TestHumanUserSimulator:
    """Test suite verifying interactive human user simulator capabilities."""

    @pytest.mark.asyncio
    async def test_human_simulator_step_interactive(self) -> None:
        inputs = ["Yes, transfer to Alice", "/exit"]
        outputs: list[str] = []

        def mock_input(prompt: str) -> str:
            return inputs.pop(0)

        def mock_output(msg: str) -> None:
            outputs.append(msg)

        sim = HumanUserSimulator(
            input_fn=mock_input,
            output_fn=mock_output,
        )
        assert sim.simulator_id == "human"

        task = _make_dummy_task()

        # Step 1: Normal reply
        turn1 = await sim.step("Who is the recipient?", [], task)
        assert turn1.content == "Yes, transfer to Alice"
        assert not turn1.finished
        assert len(outputs) == 1
        assert "Who is the recipient?" in outputs[0]

        # Step 2: Exit command
        turn2 = await sim.step("Anything else?", [], task)
        assert turn2.finished

    @pytest.mark.asyncio
    async def test_human_simulator_handles_interrupt(self) -> None:
        def mock_interrupt(prompt: str) -> str:
            raise KeyboardInterrupt()

        sim = HumanUserSimulator(input_fn=mock_interrupt, output_fn=lambda _: None)
        task = _make_dummy_task()

        turn = await sim.step("What is your account?", [], task)
        assert turn.finished
        assert turn.content == ""

    @pytest.mark.asyncio
    async def test_human_simulator_handles_eof(self) -> None:
        def mock_eof(prompt: str) -> str:
            raise EOFError()

        sim = HumanUserSimulator(input_fn=mock_eof, output_fn=lambda _: None)
        task = _make_dummy_task()

        turn = await sim.step("What is your account?", [], task)
        assert turn.finished
        assert turn.content == ""


class TestGenerateSecuritySuiteCLI:
    """Test suite verifying generate-security-suite CLI command."""

    def test_generate_security_suite_command(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out_yaml = tmp_path / "test_sec_suite.yaml"

        result = runner.invoke(
            generate_security_suite,
            [
                "--count",
                "6",
                "--domain",
                "cyber_sandbox",
                "--lang",
                "en",
                "--output",
                str(out_yaml),
                "--seed",
                "100",
            ],
        )

        assert result.exit_code == 0
        assert out_yaml.exists()

        with open(out_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "cases" in data
        assert len(data["cases"]) == 6
        for case in data["cases"]:
            assert case["domain"] == "cyber_sandbox"
            assert "lang_en" in case["tags"]
            assert case["metadata"]["language"] == "en"
