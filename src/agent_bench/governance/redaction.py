"""Redaction engine: strip sensitive patterns before storage."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RedactionRule:
    name: str
    pattern: str
    replacement: str = "[REDACTED]"
    compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.compiled = re.compile(self.pattern)


def default_denylist() -> list[RedactionRule]:
    """Default patterns for Brazilian banking PII."""
    return [
        RedactionRule(
            name="cpf",
            pattern=r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}',
            replacement="[CPF_REDACTED]",
        ),
        RedactionRule(
            name="cnpj",
            pattern=r'\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}',
            replacement="[CNPJ_REDACTED]",
        ),
        RedactionRule(
            name="card_number",
            pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            replacement="[CARD_REDACTED]",
        ),
        RedactionRule(
            name="email",
            pattern=r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            replacement="[EMAIL_REDACTED]",
        ),
        RedactionRule(
            name="phone_br",
            pattern=r'\(?\d{2}\)?\s?\d{4,5}-?\d{4}',
            replacement="[PHONE_REDACTED]",
        ),
        RedactionRule(
            name="account_number",
            pattern=r'\b\d{4,6}-[\dXx]\b',
            replacement="[ACCOUNT_REDACTED]",
        ),
        RedactionRule(
            name="api_key",
            pattern=r'(sk-|pk-|key-)[a-zA-Z0-9]{20,}',
            replacement="[API_KEY_REDACTED]",
        ),
    ]


class RedactionEngine:
    """Applies redaction rules to text and structured data."""

    def __init__(self, rules: list[RedactionRule] | None = None):
        self._rules = rules or default_denylist()

    @property
    def rules(self) -> list[RedactionRule]:
        return self._rules

    def add_rule(self, rule: RedactionRule) -> None:
        self._rules.append(rule)

    def redact_text(self, text: str) -> str:
        """Apply all redaction rules to a string."""
        result = text
        for rule in self._rules:
            result = rule.compiled.sub(rule.replacement, result)
        return result

    def redact_dict(self, data: dict[str, Any], *, depth: int = 10) -> dict[str, Any]:
        """Recursively redact all string values in a dict."""
        if depth <= 0:
            return data
        result = {}
        for key, value in data.items():
            result[key] = self._redact_value(value, depth - 1)
        return result

    def redact_list(self, items: list[Any], *, depth: int = 10) -> list[Any]:
        """Recursively redact all string values in a list."""
        return [self._redact_value(item, depth) for item in items]

    def _redact_value(self, value: Any, depth: int) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        elif isinstance(value, dict):
            return self.redact_dict(value, depth=depth)
        elif isinstance(value, list):
            return [self._redact_value(v, depth) for v in value]
        return value

    def scan(self, text: str) -> list[dict[str, Any]]:
        """Scan text for sensitive patterns without redacting. Returns matches."""
        findings = []
        for rule in self._rules:
            matches = rule.compiled.findall(text)
            if matches:
                findings.append({
                    "rule": rule.name,
                    "count": len(matches),
                    "samples": [m[:4] + "..." for m in matches[:3]],
                })
        return findings
