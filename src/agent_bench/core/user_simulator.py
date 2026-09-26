"""User simulator protocol and implementations for dynamic multi-turn interactions.

Following Clean Architecture principles, this module defines the UserSimulator protocol
to simulate interactive user behavior, clarifying questions, and slot-filling in conversational
agent benchmarks.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_bench.core.adapters import ModelAdapter
from agent_bench.core.scenarios import Task


@dataclass
class UserTurn:
    """A simulated user response turn."""

    content: str
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class UserSimulator(Protocol):
    """Protocol for dynamic multi-turn user simulation in conversational agent benchmarks."""

    @property
    def simulator_id(self) -> str:
        """Unique identifier of the user simulator implementation."""
        ...

    async def step(
        self,
        agent_message: str,
        history: Sequence[dict[str, Any]],
        task: Task,
    ) -> UserTurn:
        """Generates the next simulated user utterance in response to the agent.

        Args:
            agent_message: The latest response or question from the agent.
            history: Full message history of the dialogue so far.
            task: The active benchmark task containing user profile or scenario instructions.

        Returns:
            UserTurn containing the user's reply and a completion indicator.
        """
        ...

    def reset(self, task: Task | None = None) -> None:
        """Resets the simulator internal state for a new conversation session."""
        ...


class ScriptedUserSimulator:
    """Deterministic, scripted user simulator for multi-turn conversational benchmark cases.

    Supports sequential responses and regex pattern matching to simulate realistic user
    dialogue flows without external model dependencies.
    """

    def __init__(
        self,
        scripted_responses: Sequence[str] | None = None,
        pattern_responses: dict[str, str] | None = None,
        max_turns: int = 5,
        default_finish_message: str = "Thank you, that resolves my issue.",
        simulator_id: str = "scripted_user_simulator",
    ) -> None:
        self._scripted_responses: list[str] = list(scripted_responses or [])
        self._pattern_responses: dict[str, str] = dict(pattern_responses or {})
        self._max_turns: int = max_turns
        self._default_finish_message: str = default_finish_message
        self._simulator_id: str = simulator_id
        self._turn_index: int = 0

    @property
    def simulator_id(self) -> str:
        return self._simulator_id

    def reset(self, task: Task | None = None) -> None:
        self._turn_index = 0

    async def step(
        self,
        agent_message: str,
        history: Sequence[dict[str, Any]],
        task: Task,
    ) -> UserTurn:
        if self._turn_index >= self._max_turns:
            return UserTurn(content=self._default_finish_message, finished=True)

        # 1. Check pattern-matching triggers
        for pattern, response in self._pattern_responses.items():
            if re.search(pattern, agent_message, re.IGNORECASE):
                self._turn_index += 1
                is_last = self._turn_index >= self._max_turns
                return UserTurn(
                    content=response,
                    finished=is_last,
                    metadata={"matched_pattern": pattern, "turn": self._turn_index},
                )

        # 2. Sequential response fallback
        if self._turn_index < len(self._scripted_responses):
            resp = self._scripted_responses[self._turn_index]
            self._turn_index += 1
            is_last = self._turn_index >= self._max_turns
            return UserTurn(
                content=resp,
                finished=is_last,
                metadata={"turn": self._turn_index},
            )

        return UserTurn(
            content=self._default_finish_message,
            finished=True,
            metadata={"exhausted": True},
        )


class RuleBasedUserSimulator:
    """Slot-filling user simulator that supplies requested information from task metadata.

    Inspects slot mappings from task metadata (e.g., 'user_slots' or 'profile') and provides
    the value whenever the agent inquires about a known slot keyword.
    """

    def __init__(
        self,
        slots: dict[str, str] | None = None,
        max_turns: int = 5,
        default_finish_message: str = "Perfeito, obrigado!",
        simulator_id: str = "rule_based_user_simulator",
    ) -> None:
        self._default_slots: dict[str, str] = dict(slots or {})
        self._max_turns: int = max_turns
        self._default_finish_message: str = default_finish_message
        self._simulator_id: str = simulator_id
        self._turn_index: int = 0
        self._active_slots: dict[str, str] = {}

    @property
    def simulator_id(self) -> str:
        return self._simulator_id

    def reset(self, task: Task | None = None) -> None:
        self._turn_index = 0
        self._active_slots = dict(self._default_slots)
        if task and hasattr(task, "metadata") and isinstance(task.metadata, dict):
            task_slots = task.metadata.get("user_slots") or task.metadata.get("slots")
            if isinstance(task_slots, dict):
                for k, v in task_slots.items():
                    self._active_slots[str(k).lower()] = str(v)

    async def step(
        self,
        agent_message: str,
        history: Sequence[dict[str, Any]],
        task: Task,
    ) -> UserTurn:
        self._turn_index += 1
        if self._turn_index > self._max_turns:
            return UserTurn(content=self._default_finish_message, finished=True)

        lowered_agent_msg = agent_message.lower()

        # Match slots from active dictionary
        for slot_key, slot_val in self._active_slots.items():
            if slot_key in lowered_agent_msg:
                is_last = self._turn_index >= self._max_turns
                return UserTurn(
                    content=f"Meu {slot_key} é {slot_val}.",
                    finished=is_last,
                    metadata={"slot": slot_key, "value": slot_val},
                )

        # Check confirmation intent
        if any(w in lowered_agent_msg for w in ["confirma", "confirmar", "posso prosseguir", "deseja"]):
            return UserTurn(content="Sim, confirmo a operação.", finished=False)

        # If agent indicates completion
        if any(w in lowered_agent_msg for w in ["realizada", "concluída", "sucesso", "finalizado", "pronto"]):
            return UserTurn(content=self._default_finish_message, finished=True)

        return UserTurn(content="Pode prosseguir com a solicitação.", finished=False)


class ModelUserSimulator:
    """LLM-backed user simulator for open-ended multi-turn dialogues."""

    def __init__(
        self,
        model: ModelAdapter,
        persona_prompt: str = (
            "You are a customer interacting with an automated service assistant. "
            "Stay in character, respond concisely, and provide missing parameters when requested. "
            "When the assistant completes your request, say '[FINISHED] Thank you!'."
        ),
        max_turns: int = 5,
        simulator_id: str = "model_user_simulator",
    ) -> None:
        self._model = model
        self._persona_prompt = persona_prompt
        self._max_turns = max_turns
        self._simulator_id = simulator_id
        self._turn_count = 0

    @property
    def simulator_id(self) -> str:
        return self._simulator_id

    def reset(self, task: Task | None = None) -> None:
        self._turn_count = 0

    async def step(
        self,
        agent_message: str,
        history: Sequence[dict[str, Any]],
        task: Task,
    ) -> UserTurn:
        self._turn_count += 1
        if self._turn_count > self._max_turns:
            return UserTurn(content="Thank you, all done.", finished=True)

        user_goal = task.description or (
            task.input_messages[0]["content"] if task.input_messages else ""
        )
        prompt_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    f"{self._persona_prompt}\n"
                    f"Your overall task/goal: {user_goal}\n"
                    "If the assistant fulfilled your request, output [FINISHED] followed by your final message."
                ),
            },
        ]
        for msg in history:
            prompt_messages.append({
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", "")),
            })

        resp = await self._model.generate(
            prompt_messages,
            temperature=0.3,
            max_tokens=256,
        )
        text = resp.content.strip()
        finished = False
        if "[FINISHED]" in text or self._turn_count >= self._max_turns:
            finished = True
            text = text.replace("[FINISHED]", "").strip()
            if not text:
                text = "Thank you, that's all."

        return UserTurn(content=text, finished=finished)
