# Security Policy

## Supported Versions

We actively provide security patches and updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

The Agent-Bench team takes the security of our benchmarking infrastructure, agent runtime sandboxes, and dataset governance seriously.

If you discover a security vulnerability (such as prompt injection escape, unauthorized command execution in sandboxes, credential leakage, or dataset tampering):

1. **Do NOT open a public GitHub issue.**
2. Send an email to **security@agent-bench.org** with:
   - A detailed description of the vulnerability.
   - Exact steps to reproduce or proof-of-concept (PoC) code.
   - Affected modules, runners, or tool adapters.
   - Potential impact on benchmark integrity or host systems.
3. We will acknowledge receipt of your report within 48 hours.
4. We will coordinate a remediation patch and issue a CVE/advisory in accordance with responsible disclosure timelines (typically 30-90 days).

---

## Sandbox & Tool Execution Security

When developing or executing benchmarks against external models:

- **Isolated Sandboxes**: Tool adapters should only interact with synthetic, in-memory mock environments or isolated container sandboxes.
- **Credential Safety**: Never commit API keys or authentication tokens to test fixtures, traces, or artifacts. Always inject credentials through environment variables.
- **Data Governance**: Real customer personally identifiable information (PII) is strictly forbidden in dataset repositories. All data must be synthetic or irreversibly anonymized.
