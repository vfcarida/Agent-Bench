"""Tests for the prompt template formatter."""

from __future__ import annotations

import pytest

from agent_bench.runners.prompt_formatter import (
    BUILTIN_TEMPLATES,
    PromptFormatter,
)


# ──────────────────────────────────────────────────────────────────
# Built-in Template Tests
# ──────────────────────────────────────────────────────────────────


class TestBuiltinTemplates:
    """Test each built-in prompt template format."""

    SAMPLE_MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    MULTI_TURN_MESSAGES = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
    ]

    def test_chatml_format(self):
        """ChatML should use <|im_start|> and <|im_end|> markers."""
        formatter = PromptFormatter(template_name="chatml")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "<|im_start|>system" in result
        assert "You are a helpful assistant." in result
        assert "<|im_end|>" in result
        assert "<|im_start|>user" in result
        assert "What is 2+2?" in result
        assert "<|im_start|>assistant" in result

    def test_alpaca_format(self):
        """Alpaca should use ### Instruction: and ### Response: markers."""
        formatter = PromptFormatter(template_name="alpaca")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "### Instruction:" in result
        assert "What is 2+2?" in result
        assert "### Response:" in result

    def test_llama2_format(self):
        """Llama2 should use [INST] and <<SYS>> markers."""
        formatter = PromptFormatter(template_name="llama2")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "[INST]" in result
        assert "<<SYS>>" in result
        assert "You are a helpful assistant." in result
        assert "[/INST]" in result

    def test_llama3_format(self):
        """Llama3 should use header_id markers."""
        formatter = PromptFormatter(template_name="llama3")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_vicuna_format(self):
        """Vicuna should use USER: and ASSISTANT: markers."""
        formatter = PromptFormatter(template_name="vicuna")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "USER:" in result
        assert "What is 2+2?" in result
        assert "ASSISTANT:" in result

    def test_zephyr_format(self):
        """Zephyr should use <|role|> markers."""
        formatter = PromptFormatter(template_name="zephyr")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert "<|endoftext|>" in result

    def test_mistral_format(self):
        """Mistral should use [INST] markers."""
        formatter = PromptFormatter(template_name="mistral")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "[INST]" in result
        assert "What is 2+2?" in result
        assert "[/INST]" in result

    def test_sharegpt_format(self):
        """ShareGPT should use HUMAN: and ASSISTANT: markers."""
        formatter = PromptFormatter(template_name="sharegpt")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "SYSTEM:" in result
        assert "HUMAN:" in result
        assert "ASSISTANT:" in result

    def test_default_format(self):
        """Default should use simple role: content format."""
        formatter = PromptFormatter(template_name="default")
        result = formatter.format_messages(self.SAMPLE_MESSAGES)

        assert "system:" in result
        assert "user:" in result
        assert "assistant:" in result

    def test_multi_turn_conversation(self):
        """Templates should handle multi-turn conversations."""
        for template_name in BUILTIN_TEMPLATES:
            formatter = PromptFormatter(template_name=template_name)
            result = formatter.format_messages(self.MULTI_TURN_MESSAGES)

            # All templates should include both user messages
            assert "Hello" in result
            assert "How are you?" in result


# ──────────────────────────────────────────────────────────────────
# Edge Case Tests
# ──────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases for prompt formatting."""

    def test_empty_messages(self):
        """Empty message list should return empty string."""
        formatter = PromptFormatter(template_name="chatml")
        result = formatter.format_messages([])
        assert result == ""

    def test_user_only_messages(self):
        """Messages without system message should work."""
        formatter = PromptFormatter(template_name="chatml")
        messages = [{"role": "user", "content": "Hello!"}]
        result = formatter.format_messages(messages)

        assert "Hello!" in result
        assert "<|im_start|>user" in result

    def test_system_only_message(self):
        """System-only message should work."""
        formatter = PromptFormatter(template_name="chatml")
        messages = [{"role": "system", "content": "You are helpful."}]
        result = formatter.format_messages(messages)

        assert "You are helpful." in result


# ──────────────────────────────────────────────────────────────────
# Custom Template Tests
# ──────────────────────────────────────────────────────────────────


class TestCustomTemplates:
    """Test custom Jinja template support."""

    def test_custom_template_string(self):
        """Custom template string should work."""
        template = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}"
        formatter = PromptFormatter(template_string=template)

        messages = [
            {"role": "user", "content": "Hello!"},
        ]
        result = formatter.format_messages(messages)

        assert "[user] Hello!" in result
        assert formatter.template_name == "custom"

    def test_custom_template_from_file(self, tmp_path):
        """Custom template from file should work."""
        template_file = tmp_path / "my_template.jinja2"
        template_file.write_text(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        )

        formatter = PromptFormatter(template_path=str(template_file))

        messages = [{"role": "user", "content": "test"}]
        result = formatter.format_messages(messages)

        assert "user: test" in result
        assert formatter.template_name == "my_template"

    def test_nonexistent_template_file_raises(self):
        """Loading from non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PromptFormatter(template_path="/nonexistent/template.jinja2")


# ──────────────────────────────────────────────────────────────────
# Initialization Tests
# ──────────────────────────────────────────────────────────────────


class TestPromptFormatterInit:
    """Test PromptFormatter initialization."""

    def test_unknown_template_raises(self):
        """Unknown built-in template should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            PromptFormatter(template_name="nonexistent_template")

    def test_default_template_when_none(self):
        """No arguments should use default template."""
        formatter = PromptFormatter()
        assert formatter.template_name == "default"

    def test_list_builtin_templates(self):
        """list_builtin_templates should return all template names."""
        templates = PromptFormatter.list_builtin_templates()
        assert "chatml" in templates
        assert "alpaca" in templates
        assert "llama2" in templates
        assert "llama3" in templates
        assert len(templates) >= 8


# ──────────────────────────────────────────────────────────────────
# Auto-Detection Tests
# ──────────────────────────────────────────────────────────────────


class TestTemplateAutoDetection:
    """Test auto-detection from tokenizer (mocked)."""

    def test_detect_from_chat_template(self):
        """Should use tokenizer's chat_template attribute."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = (
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        )

        formatter = PromptFormatter.from_tokenizer(mock_tokenizer)
        result = formatter.format_messages([{"role": "user", "content": "test"}])
        assert "user: test" in result

    def test_detect_from_tokenizer_class_name(self):
        """Should infer template from tokenizer class name."""
        from unittest.mock import MagicMock

        class LlamaTokenizer:
            pass

        mock_tokenizer = LlamaTokenizer()

        formatter = PromptFormatter.from_tokenizer(mock_tokenizer)
        assert formatter.template_name == "llama3"

    def test_fallback_to_default(self):
        """Unknown tokenizer should fall back to default."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock(spec=[])
        mock_tokenizer.__class__ = type("UnknownTokenizer", (), {})

        formatter = PromptFormatter.from_tokenizer(mock_tokenizer)
        assert formatter.template_name == "default"
