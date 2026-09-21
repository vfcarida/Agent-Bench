## Description
Briefly describe the changes introduced by this pull request, including motivation and context.

Fixes #(issue)

## Type of Change
- [ ] Bug fix (`fix: ...`)
- [ ] New feature (`feat: ...`)
- [ ] Refactoring (`refactor: ...`)
- [ ] Performance improvement (`perf: ...`)
- [ ] Documentation update (`docs: ...`)
- [ ] Dataset / Test addition (`test: ...`)
- [ ] Infrastructure / Chore (`chore: ...`)

## Validation & Verification
Please confirm that your local environment satisfies all quality gates:
- [ ] `ruff check src/ tests/` passes with 0 errors
- [ ] `mypy src/agent_bench` passes strict type checking
- [ ] `pytest tests/` passes with 100% success rate
- [ ] `bench check-contamination` passes with zero detected leakage
- [ ] Documentation updated to reflect changes (in English)

## Breaking Changes
- [ ] No breaking changes
- [ ] Yes, this PR introduces breaking changes (detailed below)

## Checklist
- [ ] Code follows project Clean Architecture principles and Protocol conventions
- [ ] All new functions and public methods include complete type annotations
- [ ] No real customer PII or sensitive keys included in datasets or traces
