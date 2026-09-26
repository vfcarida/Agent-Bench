# Maintainer Guidelines & Project Governance

Welcome to the **Agent-Bench** project governance and maintainer guide. This document establishes guidelines for maintaining the repository, reviewing community pull requests, onboarding new evaluation domains, and managing releases.

---

## 1. Project Governance & Roles

### 1.1 Maintainer Roles
- **Lead Maintainer / BDFL**: Technical architecture decisions, final approval on breaking schema changes (`core/schema_v2.py`, `core/protocols.py`), and PyPI releases.
- **Core Maintainers**: Code reviews, triage of issues, CI pipeline maintenance, evaluator calibration.
- **Domain Stewards**: Domain experts overseeing dataset accuracy, tool schemas, and compliance constraints for specific verticals (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`).

### 1.2 Decision Making
- Bug fixes, test expansions, and provider adapter additions require **1 approving maintainer review**.
- Protocol modifications, changes to safety gating logic (`graders/safety_gate.py`), or metric estimators (`metrics/compute.py`) require **consensus among at least 2 maintainers**.

---

## 2. Pull Request Review Standards

Every pull request merged into `main` must strictly adhere to the following checklist:

### 2.1 Non-Compensable Architectural Constraints
1. **Preserve Python Protocols (`protocols.py`)**: Never introduce framework-specific base class dependencies into core evaluation logic.
2. **Hard Safety Gating**: Safety failures must **never** be averaged into functional scores. Refusals and safety constraints remain non-compensable gates.
3. **Trace & Financial Accounting**: Model adapters and runners must preserve atomic trace logging (`TraceEvent`), token counts, and price calculations.
4. **Data Split Discipline**: Never commit evaluation cases from holdout sets into dev/synthetic paths. PRs must pass `scripts/check_contamination.py`.

### 2.2 Quality Gates
- **Type Checking**: `mypy --strict src/agent_bench` must pass with zero errors.
- **Linting & Formatting**: `ruff check` and `ruff format --check` must pass.
- **Test Suite**: `pytest tests/` must pass cleanly without unhandled resource leaks or deprecation warnings.
- **Dataset Schema Integrity**: `python scripts/check_gold_integrity.py` must succeed with exit code 0.

---

## 3. Onboarding New Evaluation Domains

Community contributions adding a new evaluation vertical must provide:
1. **Domain Policy Manifest**: `configs/domains/<domain_name>.yaml` specifying allowed tools, forbidden actions, and weighting profiles.
2. **Domain Tool Implementations**: Typed `ToolAdapter` classes in `src/agent_bench/tools/`.
3. **Gold Evaluation Tasks**: Valid `EvalCase` definitions placed in `datasets/gold/dev/<domain_name>.yaml`.
4. **Integration Test**: Automated test in `tests/integration/test_<domain_name>_suite.py` asserting end-to-end execution.
5. **Domain Documentation**: Reference guide in `docs/how_to_add_a_domain.md`.

---

## 4. Release Process

Agent-Bench follows [Semantic Versioning 2.0.0](https://semver.org/).

### 4.1 Release Steps
1. **Verify Clean CI**: Ensure the latest commit on `main` has passed all tests and checks.
2. **Update Version**: Bump version in `pyproject.toml` and `src/agent_bench/__init__.py`.
3. **Update CHANGELOG**: Document additions, fixes, breaking changes, and migration guidance.
4. **Tag Git Release**:
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0"
   git push origin v1.1.0
   ```
5. **Automated Publishing**: The `.github/workflows/release.yml` GitHub Action triggers automatically, builds sdist & wheel packages, runs package checks via `twine`, and publishes to PyPI.

---

## 5. Security & Vulnerability Reporting

Security vulnerabilities (such as sandbox escape flaws or credential leakage risks) should **not** be reported via public GitHub issues. Please report privately to `security@agentbench.dev` or use GitHub Private Vulnerability Reporting. Maintainers will respond within 48 hours.
