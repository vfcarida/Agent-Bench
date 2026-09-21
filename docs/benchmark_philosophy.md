# Benchmark Philosophy

## SYSTEM Benchmark, Not Just a Model Benchmark

`Agent-Bench` evaluates the **complete agentic system**: orchestration, tool calling, state transitions, business policy compliance, and failure recovery. The foundation language model is merely one component. An agent can easily fail with a state-of-the-art model if orchestration or tool definitions are fragile — and conversely, a well-engineered agent architecture can succeed even with smaller, cost-effective models.

Practical implication: test cases exercise full end-to-end execution trajectories, not just static text generation.

## Gold / Synthetic / Adversarial Stratification

| Dataset Family | Purpose | Generation & Validation |
|----------------|---------|-------------------------|
| **Gold** | Ground truth human-verified cases. Authoritative source for official leaderboards. | Domain specialists & technical reviewers |
| **Synthetic** | High-volume operational coverage for statistical confidence. | Automated pipelines with statistical sampling and verification |
| **Adversarial** | Robustness evaluation: edge cases, prompt injection, malformed inputs, policy bypasses. | Red-teaming & automated fuzzing |

Strict dataset separation prevents leakage and contamination: gold evaluation sets are never diluted by unverified synthetic data.

## Code-Based Graders > LLM-as-Judge

We prioritize deterministic, code-based graders (`ExactMatchGrader`, `ToolMatchGrader`, `StateCheckGrader`, `StructuredOutputGrader`) over subjective LLM judges:

1. **Reproducibility**: Identical execution trajectories and final states yield identical evaluation scores every time.
2. **Speed**: Sub-millisecond execution times without network latency or external API calls.
3. **Cost**: Zero incremental financial cost per evaluation run.
4. **Auditability & Traceability**: Explicit Python grading logic without brittle prompt phrasing or non-deterministic judge drift.

LLM judges are utilized strictly for subjective quality dimensions (e.g., natural fluency, explanatory tone, domain-specific guidance) with calibrated rubrics and prompt injection defenses.

## Two-Phase Evaluation & Non-Compensable Safety Gates

To avoid the risk of high functional scores masking severe safety violations, `Agent-Bench` uses a **Phase-0 Hard Safety Gate**:

1. **Phase 0 (Hard Safety & Policy Gate)**: Evaluates whether critical constraints (e.g., unauthenticated data disclosure, unauthorized financial transfers, prompt injection compliance) were violated. If a hard safety rule is breached, the test case fails outright (`passed = False`), capping the final score at zero.
2. **Phase 1 (Functional & Quality Scoring)**: Calculates functional accuracy (tool calls, state mutations, output match), cost, and latency only if the safety gate passes.

Functional excellence cannot compensate for policy or safety breaches.

## Reproducibility Invariants

- **Fixed Seeds**: Mandatory seed tracking across synthetic generators and randomized sampling.
- **Semantic Dataset Versioning**: All datasets follow SemVer (`v1.0.0`, `v1.1.0`, etc.).
- **SHA-256 Integrity Hashes**: Every authoritative dataset file records cryptographic hashes to detect accidental mutation.
- **Dependency Pinning**: Framework and external dependencies are tightly managed via lockfiles and `pyproject.toml`.
- **ISO-8601 Timestamps**: Every evaluation run trace records explicit, UTC-based ISO-8601 timestamps.

## The Sacred Holdout Principle

The `holdout` split is strictly protected:

- **Never** use holdout cases for tuning prompts, agent architectures, or debugging.
- Used exclusively for reporting official benchmark metrics and release evaluations.
- CI includes automated contamination checks (`bench check-contamination` / `scripts/check_contamination.py`) to detect verbatim or paraphrased leakage from holdout into dev or synthetic datasets.
- Any contamination requires immediate invalidation and rotation of the affected holdout split.

## Provider-Agnostic Design

`Agent-Bench` is completely agnostic to specific model providers or proprietary APIs:

- Clean abstraction protocols: `ModelAdapter` / `AgentRunner` interfaces.
- Normalized metric accounting: raw token counts (`tokens_in`, `tokens_out`), latency percentiles, and normalized per-case costs.
- Free of proprietary prompt assumptions or provider-specific locks.
- Full compatibility with offline mocks, local open-weight models (via Hugging Face or Ollama), and cloud providers (OpenAI, Anthropic, Gemini, AWS Bedrock).
