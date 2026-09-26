"""Unit tests for international enterprise redactions (SSN, IBAN, AWS keys, GitHub tokens, API keys)."""

from agent_bench.governance.redaction import RedactionEngine, default_denylist


class TestInternationalRedaction:
    def setup_method(self) -> None:
        self.engine = RedactionEngine(default_denylist())

    def test_redact_us_ssn(self) -> None:
        text = "Customer SSN is 123-45-6789 for tax identification."
        redacted = self.engine.redact_text(text)
        assert "123-45-6789" not in redacted
        assert "[SSN_REDACTED]" in redacted

    def test_redact_iban(self) -> None:
        text = "Wire transfer funds to IBAN DE89370400440532013000."
        redacted = self.engine.redact_text(text)
        assert "DE89370400440532013000" not in redacted
        assert "[IBAN_REDACTED]" in redacted

    def test_redact_aws_access_key(self) -> None:
        text = "AWS credentials leaked: AKIAIOSFODNN7EXAMPLE in debug trace."
        redacted = self.engine.redact_text(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[AWS_KEY_REDACTED]" in redacted

    def test_redact_github_pat(self) -> None:
        text = "Token: ghp_1234567890abcdefghijklmnopqrstuvwxyzAB"
        redacted = self.engine.redact_text(text)
        assert "ghp_1234567890abcdefghijklmnopqrstuvwxyzAB" not in redacted
        assert "[GITHUB_PAT_REDACTED]" in redacted

    def test_redact_modern_openai_project_key(self) -> None:
        text = "OpenAI key: sk-proj-abc123xyz456longsecretkeyhere998877"
        redacted = self.engine.redact_text(text)
        assert "sk-proj-abc123xyz456longsecretkeyhere998877" not in redacted
        assert "[API_KEY_REDACTED]" in redacted

    def test_redact_anthropic_api_key(self) -> None:
        text = "Anthropic key: sk-ant-api03-abcdef1234567890longtokencontenthere"
        redacted = self.engine.redact_text(text)
        assert "sk-ant-api03-abcdef1234567890longtokencontenthere" not in redacted
        assert "[API_KEY_REDACTED]" in redacted

    def test_redact_nested_dict_mixed_pii(self) -> None:
        data = {
            "user": "Alice",
            "ssn": "987-65-4321",
            "credentials": {
                "token": "ghp_securetoken1234567890abcdefghijklmnopqr",
                "iban": "GB29XABC10123456789012",
            },
        }
        redacted = self.engine.redact_data(data)
        assert redacted["ssn"] == "[SSN_REDACTED]"
        assert redacted["credentials"]["token"] == "[GITHUB_PAT_REDACTED]"
        assert redacted["credentials"]["iban"] == "[IBAN_REDACTED]"
