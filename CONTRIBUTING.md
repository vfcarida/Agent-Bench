# Contributing to Agent-Bench

Thank you for your interest in contributing to **Agent-Bench**! We welcome contributions from engineers, AI researchers, domain experts, and community contributors.

This guide outlines our development setup, coding standards, testing workflows, and PR lifecycle.

---

## 1. Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## 2. Development Setup

### Prerequisites
- **Python 3.12+**
- Git
- Virtual environment tool (`venv`, `uv`, or `conda`)

### Local Installation
1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Agent-Bench.git
   cd Agent-Bench
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. Install editable package with development dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

---

## 3. Architecture & Coding Standards

Agent-Bench is architected following Clean Architecture and Protocol interfaces:

- **Protocols First**: Components interact via protocols defined in `agent_bench.core.protocols` (`TaskEnvironment`, `AgentRunner`, `Evaluator`) and `agent_bench.core.adapters` (`ModelAdapter`, `ToolAdapter`).
- **Deterministic Prioritization**: Favor deterministic code-based graders (`ExactMatch`, `ToolMatch`, `StateCheck`) over subjective LLM judges whenever feasible.
- **Two-Phase Safety Gating**: Phase 0 enforces non-compensable hard safety constraints. Phase 1 computes functional scores only if Phase 0 passes.
- **Typing & Linting**: Strict static typing is enforced via `mypy` and code formatting/linting via `ruff`.
  - All new functions and public methods must include complete PEP 484 type annotations.
  - Zero tolerance for untyped `Any` returns or unvalidated casts.
- **Language Policy**: All code, docstrings, comments, error messages, and documentation must be written in technically precise, professional English.

---

## 4. Quality Verification Commands

Before opening a pull request, run the complete local quality gate:

### Linting & Formatting
```bash
ruff check src/ tests/
ruff format --check src/ tests/
```
Auto-fix formatting and trivial lint issues with:
```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Static Type Checking
```bash
mypy src/agent_bench
```

### Test Suite Execution
```bash
pytest tests/unit/ tests/integration/ -v
```

### Contamination & Data Leakage Gate
```bash
bench check-contamination
```

### Configuration Validation
```bash
bench --config-dir configs validate-config
```

---

## 5. Adding New Benchmark Domains

To contribute a new domain:

1. **Author Evaluation Cases**:
   - Create `datasets/gold/dev/<domain_id>.yaml` following the `EvalCase v2` schema.
   - Include realistic scenarios (`happy_path`, `edge_case`, `refusal`, `policy_conflict`).
   - Consult [docs/annotation_guide.md](docs/annotation_guide.md) for field specifications.
2. **Define Domain System & Tools**:
   - Create `configs/domains/<domain_id>.yaml` declaring tools, parameters, and system configs.
   - If specialized mock tools are needed, implement `ToolAdapter` classes in `src/agent_bench/tools/<domain_id>_tools.py`.
3. **Verify Locally**:
   ```bash
   bench validate-datasets --data-dir datasets/gold/dev
   bench run-case <task_id> --domain <domain_id> --system mock
   ```

---

## 6. Git Commit & Pull Request Guidelines

### Commit Conventions
We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature or capability
- `fix:` Bug fix or defect resolution
- `docs:` Documentation improvements
- `refactor:` Code refactoring without behavior modification
- `test:` Adding or updating tests
- `perf:` Performance improvements with benchmark evidence
- `chore:` Dependency or build updates

### Pull Request Lifecycle
1. Branch from `main`: `git checkout -b feat/my-improvement`.
2. Ensure all quality checks (`ruff`, `mypy`, `pytest`, `check-contamination`) pass locally.
3. Open a Pull Request referencing related issues.
4. CI will execute the offline validation suite automatically.
5. Address maintainer feedback promptly.
