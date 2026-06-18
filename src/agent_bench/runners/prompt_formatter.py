"""Dynamic prompt template formatter for model-specific chat formats.

Athena fine-tunes models on specific prompt templates (ChatML, Alpaca,
Llama2, etc.). This formatter ensures Agent-Bench sends evaluation
prompts in exactly the format the model was trained on, preventing
hallucination from template mismatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from jinja2 import BaseLoader, Environment, TemplateNotFound

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────────
# Built-in Jinja2 templates for common chat formats
# ──────────────────────────────────────────────────────────────────

BUILTIN_TEMPLATES: dict[str, str] = {
    "chatml": (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "<|im_start|>assistant\n"
    ),
    "alpaca": (
        "{% if messages[0].role == 'system' %}"
        "{{ messages[0].content }}\n\n"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "### Instruction:\n{{ message.content }}\n\n"
        "{% elif message.role == 'assistant' %}"
        "### Response:\n{{ message.content }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        "### Response:\n"
    ),
    "llama2": (
        "{% if messages[0].role == 'system' %}"
        "[INST] <<SYS>>\n{{ messages[0].content }}\n<</SYS>>\n\n"
        "{% else %}"
        "[INST] "
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "{% if not loop.first or messages[0].role != 'system' %}"
        "[INST] "
        "{% endif %}"
        "{{ message.content }} [/INST]"
        "{% elif message.role == 'assistant' %}"
        " {{ message.content }} </s>"
        "{% endif %}"
        "{% endfor %}"
    ),
    "llama3": (
        "<|begin_of_text|>"
        "{% for message in messages %}"
        "<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n"
        "{{ message.content }}<|eot_id|>"
        "{% endfor %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "vicuna": (
        "{% if messages[0].role == 'system' %}"
        "{{ messages[0].content }}\n\n"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "USER: {{ message.content }}\n"
        "{% elif message.role == 'assistant' %}"
        "ASSISTANT: {{ message.content }}</s>\n"
        "{% endif %}"
        "{% endfor %}"
        "ASSISTANT:"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "<|{{ message.role }}|>\n"
        "{{ message.content }}<|endoftext|>\n"
        "{% endfor %}"
        "<|assistant|>\n"
    ),
    "mistral": (
        "{% if messages[0].role == 'system' %}"
        "[INST] {{ messages[0].content }}\n\n"
        "{% for message in messages[1:] %}"
        "{% if message.role == 'user' %}"
        "{{ message.content }} [/INST]"
        "{% elif message.role == 'assistant' %}"
        "{{ message.content }}</s> [INST] "
        "{% endif %}"
        "{% endfor %}"
        "{% else %}"
        "[INST] "
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "{{ message.content }} [/INST]"
        "{% elif message.role == 'assistant' %}"
        "{{ message.content }}</s> [INST] "
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
    ),
    "sharegpt": (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "SYSTEM: {{ message.content }}\n"
        "{% elif message.role == 'user' %}"
        "HUMAN: {{ message.content }}\n"
        "{% elif message.role == 'assistant' %}"
        "ASSISTANT: {{ message.content }}\n"
        "{% endif %}"
        "{% endfor %}"
        "ASSISTANT:"
    ),
    "default": (
        "{% for message in messages %}"
        "{{ message.role }}: {{ message.content }}\n"
        "{% endfor %}"
        "assistant:"
    ),
}


class PromptFormatter:
    """Formats evaluation prompts using model-specific Jinja templates.

    Supports:
    - Built-in templates (chatml, alpaca, llama2, llama3, vicuna, etc.)
    - Custom Jinja2 templates loaded from file paths
    - Auto-detection from HuggingFace tokenizer's chat_template attribute

    Usage:
        formatter = PromptFormatter(template_name="chatml")
        prompt = formatter.format_messages([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ])
    """

    def __init__(
        self,
        template_name: str | None = None,
        template_path: str | None = None,
        template_string: str | None = None,
    ):
        """Initialize the prompt formatter.

        Args:
            template_name: Name of a built-in template (e.g., 'chatml', 'alpaca').
            template_path: Path to a custom Jinja2 template file.
            template_string: Raw Jinja2 template string.

        Exactly one of template_name, template_path, or template_string must be provided.
        If none are provided, 'default' template is used.
        """
        self._env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
        self._template_name = template_name or "custom"

        if template_string:
            self._template = self._env.from_string(template_string)
            self._template_name = "custom"
        elif template_path:
            path = Path(template_path)
            if not path.exists():
                raise FileNotFoundError(f"Template file not found: {template_path}")
            template_str = path.read_text(encoding="utf-8")
            self._template = self._env.from_string(template_str)
            self._template_name = path.stem
        elif template_name:
            if template_name not in BUILTIN_TEMPLATES:
                raise ValueError(
                    f"Unknown template '{template_name}'. "
                    f"Available: {list(BUILTIN_TEMPLATES.keys())}"
                )
            self._template = self._env.from_string(BUILTIN_TEMPLATES[template_name])
            self._template_name = template_name
        else:
            self._template = self._env.from_string(BUILTIN_TEMPLATES["default"])
            self._template_name = "default"

        logger.info("prompt_formatter_init", template=self._template_name)

    @property
    def template_name(self) -> str:
        """Name of the active template."""
        return self._template_name

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        """Apply the template to convert messages into model-native format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Formatted prompt string ready for the model.
        """
        if not messages:
            return ""

        # Render the template with messages as context
        rendered = self._template.render(messages=messages)
        return rendered

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "PromptFormatter":
        """Auto-detect template from a HuggingFace tokenizer.

        Many fine-tuned models store their chat template in the
        tokenizer's `chat_template` attribute. This method extracts
        it and creates a PromptFormatter.

        Args:
            tokenizer: A HuggingFace tokenizer instance.

        Returns:
            PromptFormatter with the detected template.
        """
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            logger.info("template_auto_detected", source="tokenizer.chat_template")
            return cls(template_string=chat_template)

        # Fallback: try to detect from tokenizer class name
        tokenizer_name = type(tokenizer).__name__.lower()
        template_hints = {
            "llama": "llama3",
            "mistral": "mistral",
            "qwen": "chatml",
            "yi": "chatml",
        }
        for hint, template in template_hints.items():
            if hint in tokenizer_name:
                logger.info(
                    "template_inferred",
                    source="tokenizer_class",
                    template=template,
                )
                return cls(template_name=template)

        logger.warning("template_detection_fallback", using="default")
        return cls(template_name="default")

    @staticmethod
    def list_builtin_templates() -> list[str]:
        """Return names of all available built-in templates."""
        return list(BUILTIN_TEMPLATES.keys())
