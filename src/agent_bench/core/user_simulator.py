"""User simulator protocol and implementations for dynamic multi-turn interactions.

Following Clean Architecture principles, this module defines the UserSimulator protocol
to simulate interactive user behavior, clarifying questions, and slot-filling in conversational
agent benchmarks.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_bench.core.adapters import ModelAdapter
from agent_bench.core.scenarios import Task


@dataclass
class DualControlAction:
    """A user-side action in a dual-control shared-world environment (tau2-bench protocol).

    Represents an action the simulated user takes on the shared environment state
    (either invoking a user-side tool or mutating client/device state) when triggered
    by the agent's dialogue or instructions.
    """

    trigger_pattern: str
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state_mutations: dict[str, Any] = field(default_factory=dict)
    required_state: dict[str, Any] = field(default_factory=dict)
    once: bool = True
    finish_on_trigger: bool = False
    action_id: str = "user_action"


@dataclass
class UserTurn:
    """A simulated user response turn (supporting both single-control and dual-control turns)."""

    content: str
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    user_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state_mutations: dict[str, Any] = field(default_factory=dict)


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


class HumanUserSimulator:
    """Interactive human-in-the-loop simulator for manual probing and red-teaming.

    Prompts a human evaluator via the console/terminal during each step of the dialogue,
    allowing live probing of agent guardrails, boundaries, and multi-turn capabilities.
    """

    def __init__(
        self,
        prompt_prefix: str = "[User]> ",
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
        simulator_id: str = "human",
    ) -> None:
        self._prompt_prefix = prompt_prefix
        self._input_fn = input_fn or input
        self._output_fn = output_fn or print
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
        self._output_fn(f"\n[Agent]: {agent_message}")
        try:
            user_input = self._input_fn(self._prompt_prefix).strip()
        except (EOFError, KeyboardInterrupt):
            return UserTurn(content="", finished=True)

        if user_input.lower() in ("/exit", "/quit", "/done", "/finish"):
            return UserTurn(content="", finished=True)

        return UserTurn(content=user_input, finished=False)


class DualControlUserSimulator:
    """Dec-POMDP Dual-Control shared-world user simulator (tau2-bench protocol, arXiv:2506.07982).

    In dual-control environments, both the assistant agent and the simulated user hold
    action capabilities over the shared world state (e.g., user approving a 2FA challenge,
    toggling a device security lock, uploading KYC documents, or revoking a PIX key).
    """

    def __init__(
        self,
        actions: Sequence[DualControlAction | dict[str, Any]] | None = None,
        slots: dict[str, str] | None = None,
        max_turns: int = 6,
        default_finish_message: str = "Thank you, the operation is complete.",
        simulator_id: str = "dual_control_user_simulator",
    ) -> None:
        self._default_actions: list[DualControlAction] = [
            self._coerce_action(a, idx) for idx, a in enumerate(actions or [])
        ]
        self._default_slots: dict[str, str] = dict(slots or {})
        self._max_turns = max_turns
        self._default_finish_message = default_finish_message
        self._simulator_id = simulator_id
        self._turn_index = 0
        self._active_actions: list[DualControlAction] = []
        self._active_slots: dict[str, str] = {}
        self._executed_action_ids: set[str] = set()

    @staticmethod
    def _coerce_action(
        raw: DualControlAction | dict[str, Any], idx: int = 0
    ) -> DualControlAction:
        if isinstance(raw, DualControlAction):
            return raw
        tool_calls = list(raw.get("tool_calls", []))
        if "tool_name" in raw and raw["tool_name"]:
            tool_calls.append(
                {
                    "name": str(raw["tool_name"]),
                    "arguments": dict(raw.get("tool_args", raw.get("arguments", {}))),
                }
            )
        return DualControlAction(
            trigger_pattern=str(raw.get("trigger_pattern", raw.get("pattern", ""))),
            response=str(raw.get("response", "I have completed the requested action.")),
            tool_calls=tool_calls,
            state_mutations=dict(raw.get("state_mutations", raw.get("state_updates", {}))),
            required_state=dict(raw.get("required_state", {})),
            once=bool(raw.get("once", True)),
            finish_on_trigger=bool(raw.get("finish_on_trigger", False)),
            action_id=str(raw.get("action_id", f"dual_action_{idx}")),
        )

    @property
    def simulator_id(self) -> str:
        return self._simulator_id

    def reset(self, task: Task | None = None) -> None:
        self._turn_index = 0
        self._executed_action_ids.clear()
        self._active_actions = list(self._default_actions)
        self._active_slots = dict(self._default_slots)

        if task and hasattr(task, "metadata") and isinstance(task.metadata, dict):
            task_slots = task.metadata.get("user_slots") or task.metadata.get("slots")
            if isinstance(task_slots, dict):
                for k, v in task_slots.items():
                    self._active_slots[str(k).lower()] = str(v)

            raw_actions = (
                task.metadata.get("dual_control_actions")
                or task.metadata.get("user_actions")
                or []
            )
            if isinstance(raw_actions, list):
                for idx, raw_act in enumerate(raw_actions):
                    if isinstance(raw_act, (DualControlAction, dict)):
                        self._active_actions.append(
                            self._coerce_action(raw_act, len(self._active_actions) + idx)
                        )

    async def step(
        self,
        agent_message: str,
        history: Sequence[dict[str, Any]],
        task: Task,
        current_state: dict[str, Any] | None = None,
    ) -> UserTurn:
        self._turn_index += 1
        if self._turn_index > self._max_turns:
            return UserTurn(content=self._default_finish_message, finished=True)

        # 1. Evaluate dual-control shared-world actions triggered by agent instructions
        for action in self._active_actions:
            if action.once and action.action_id in self._executed_action_ids:
                continue
            if action.required_state and current_state is not None:
                if any(current_state.get(k) != v for k, v in action.required_state.items()):
                    continue
            if action.trigger_pattern and re.search(
                action.trigger_pattern, agent_message, re.IGNORECASE
            ):
                self._executed_action_ids.add(action.action_id)
                is_finished = action.finish_on_trigger or (self._turn_index >= self._max_turns)
                return UserTurn(
                    content=action.response,
                    finished=is_finished,
                    user_tool_calls=list(action.tool_calls),
                    state_mutations=dict(action.state_mutations),
                    metadata={
                        "dual_control": True,
                        "action_id": action.action_id,
                        "turn": self._turn_index,
                    },
                )

        lowered_msg = agent_message.lower()

        # 2. Check slot-filling queries
        for slot_key, slot_val in self._active_slots.items():
            if slot_key in lowered_msg:
                is_last = self._turn_index >= self._max_turns
                return UserTurn(
                    content=f"My {slot_key} is {slot_val}.",
                    finished=is_last,
                    metadata={"slot": slot_key, "value": slot_val, "turn": self._turn_index},
                )

        # 3. Default dual-control confirmation & completion handling
        if any(
            w in lowered_msg
            for w in ["2fa", "approve", "unlock", "security lock", "token", "desbloquear"]
        ):
            return UserTurn(
                content="I have approved the confirmation prompt on my device.",
                finished=False,
                state_mutations={"user_device_confirmed": True},
                metadata={"dual_control": True, "action_id": "implicit_device_confirm"},
            )

        if any(
            w in lowered_msg
            for w in [
                "completed",
                "successfully",
                "resolved",
                "realizada",
                "concluída",
                "sucesso",
            ]
        ):
            return UserTurn(content=self._default_finish_message, finished=True)

        return UserTurn(content="Please proceed with the operation.", finished=False)


