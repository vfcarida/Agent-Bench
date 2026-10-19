# Agent-Bench Architecture & Evolution Roadmap

This document defines the architectural north star, engineering maturity governance, verified capability matrix, and next-horizon research tracks for **Agent-Bench**.

---

## 1. Vision & Architectural North Star

**Agent-Bench** is a multi-domain, safety-gated evaluation framework for tool-augmented and multi-agent AI systems across regulated enterprise verticals (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`).

Its architecture is anchored on four scientific and executive pillars:
1. **Non-Compensable Safety Gating** (*AgentHarm* `arXiv:2410.09024`; *Agent-SafetyBench* `arXiv:2412.14470`; *Gartner AI TRiSM*): Safety or policy violations are never averaged into functional quality scores. Phase 0 safety gates short-circuit execution before deterministic (Phase 1) or calibrated semantic judge (Phase 2) evaluation.
2. **Multi-Trial Reliability (`pass^k` vs. `pass@k`)** (*$\tau$-bench* `arXiv:2406.12045`; *$\tau^2$-bench* `arXiv:2506.07982`): High-risk transactional and cybersecurity profiles evaluate worst-case consistency across $k$ trials ($\text{pass}\hat{\ }k = \binom{c}{k}/\binom{n}{k}$) alongside optimistic capability ($\text{pass}@k$) and 95% BCa/percentile bootstrap confidence intervals.
3. **Joint Cost-Quality Pareto Optimization & Graduated Autonomy** (*AI Agents That Matter* `arXiv:2407.01502`; *BCG Graduated Autonomy*): Systems are evaluated on a safety-constrained 2D Pareto frontier (`functional_score` vs. `cost_per_successful_task` with hypervolume indicator) and mapped into deployment readiness tiers (`Tier 1 — Shadow` $\rightarrow$ `Tier 4 — Full Bounded Autonomy`, or `Blocked — Safety Gate Triggered`).
4. **Psychometric & Structural Diagnostics** (*FINESSE-Bench* `arXiv:2605.15482`; *tinyBenchmarks* `arXiv:2402.14992`): Automated run diagnostics compute adjacent-difficulty degradation ($\Delta_{\text{easy}\to\text{expert}}$), cross-domain balanced profiles, cross-family transfer gaps, and item-level discriminative power (mid-band $10\%\text{--}90\%$ task rate, Gini coefficient, and point-biserial discrimination alerts).

---

## 2. Maturity Tiers & Verification Governance

Every subsystem in Agent-Bench is governed by an explicit, test-verified maturity taxonomy. A component may only advance to a higher tier when all objective quality gates for that tier are satisfied in CI:

| Maturity Tier | Engineering & Verification Requirements |
| :--- | :--- |
| **`Stable`** | Complete implementation; strict static typing (`mypy src` with 0 errors); automated unit + integration test coverage ($\ge 80\%$ module coverage); deterministic execution with seeded RNG; documented in `docs/` and CLI/SDK reference. |
| **`Beta`** | Complete core implementation and passing unit tests; API surface stabilized; edge-case or multi-provider stress testing in progress. |
| **`Alpha`** | Functional prototype with unit tests in mock/stub mode; requires live-credential calibration or multi-environment validation before production reliance. |
| **`Experimental`** | Behind an explicit opt-in flag (e.g., `--enable-llm-judge`); requires calibration governance (`bench calibrate-judge` `GO` verdict) before headline reporting. |
| **`Planned`** | Architectural specification and research anchor defined; scheduled for upcoming release milestone. |

---

## 3. Capability & Milestone Matrix

| Track / Subsystem | Architectural Scope | Research & Industry Anchor | Maturity Status | Verification Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Core Protocols & Schema v2** | Structural `Protocol` interfaces (`core/protocols.py`), 12-dimension `EvalCase` schema (`core/schema_v2.py`), PEP 561 typed SDK (`py.typed`) | Hexagonal Architecture; PEP 544 / PEP 561 | **`Stable`** | `tests/unit/test_protocols.py`, `tests/unit/test_schema_v2_validators_and_generators.py` |
| **Two-Phase Gated Evaluator** | Phase 0 Hard Safety Gate $\rightarrow$ Phase 1 Deterministic State/Tool Graders $\rightarrow$ Phase 2 Calibrated Semantic Judge | *AgentHarm* (`arXiv:2410.09024`); *Agent-SafetyBench* (`arXiv:2412.14470`) | **`Stable`** | `tests/unit/test_safety_gate.py`, `tests/unit/test_gated_evaluator.py` |
| **Reliability & Bootstrap Statistics** | Per-task `pass@k`, `pass^k`, and seeded 95% bootstrap confidence intervals | *$\tau$-bench* (`arXiv:2406.12045`) | **`Stable`** | `tests/unit/test_statistics.py`, `tests/unit/test_metrics.py` |
| **Safety-Constrained Pareto Frontier & Autonomy Tiers** | Non-dominated frontier, 2D hypervolume indicator, and BCG Graduated Autonomy Tier classification (`metrics/pareto.py`) | *AI Agents That Matter* (`arXiv:2407.01502`); *BCG AI at Scale* | **`Stable`** | `tests/unit/test_pareto_and_psychometrics.py` |
| **Psychometric & Difficulty Diagnostics** | Difficulty degradation ($\Delta$), discriminative power (IRT mid-band, Gini), balanced profiles, and transfer gaps (`bench diagnose-run`) | *FINESSE-Bench* (`arXiv:2605.15482`); *tinyBenchmarks* (`arXiv:2402.14992`) | **`Stable`** | `tests/unit/test_pareto_and_psychometrics.py`, `tests/unit/test_reports_and_loader_v2.py` |
| **Multi-Domain Sandboxes & Tools** | Stateful environments and tools for `pix_assist`, `investment_advisor`, `sme_business_advisor`, and `cyber_sandbox` | *BankerToolBench* (`arXiv:2604.11304`); *ToolEmu* (`arXiv:2309.15817`) | **`Stable`** | `tests/integration/test_all_domains_suite.py`, `tests/unit/test_tools.py` |
| **Dataset Governance & Contamination Detection** | SHA-256 provenance registry, PII redaction (`CPF`, `CNPJ`, `PIX`), and n-gram Jaccard contamination auditing | *FinanceBench* (`arXiv:2311.11944`); *Gartner AI TRiSM* | **`Stable`** | `tests/unit/test_contamination.py`, `tests/unit/test_governance_observability_and_online_eval.py` |
| **Markdown, HTML & Parquet Reporting** | Automated run reports, multi-system comparison tables, interactive Chart.js HTML dashboards, and PyArrow analytics | *HELM* (`stanford-crfm/helm`); *Inspect AI* (`UKGovernmentBEIS/inspect_ai`) | **`Stable`** | `tests/unit/test_reports_and_loader_v2.py` |
| **Calibrated Semantic Judge (`SemanticJudge`)** | Rubric-based LLM-as-a-Judge with prompt-injection defense and Cohen's $\kappa$ calibration gate (`judges/calibration.py`) | *MT-Bench* (`arXiv:2306.05685`) | **`Experimental`** | `tests/unit/test_judge_calibration.py`, `tests/unit/test_semantic_judge.py` |

---

## 4. Next-Horizon Frontier Tracks

### Horizon Track 1 — Dual-Control Shared-World User Simulation ($\tau^2$-bench Protocol)
- **Scope:** Extend `src/agent_bench/simulation/` from single-control user persona simulation to a **Dec-POMDP dual-control environment** where both the simulated user and the support agent can invoke world-state mutation tools (e.g., user toggling app security locks, revoking PIX keys, or uploading documents mid-conversation).
- **Research Anchor:** Barres et al. (2025), *$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment* (`arXiv:2506.07982`).
- **Target Status:** `Planned` for v1.1.

### Horizon Track 2 — 2PL Item Response Theory (IRT) Adaptive Subset Selection
- **Scope:** Build on `src/agent_bench/metrics/discriminative_power.py` to fit a 2-Parameter Logistic (2PL) Item Response Theory model ($\theta_j$ agent ability, $a_i$ task discrimination, $b_i$ task difficulty) across historical run matrices, enabling a `bench run-suite --fast-anchor` mode that estimates full-suite scores using a 20% anchor subset with bounded estimation error.
- **Research Anchor:** Polo et al. (2024), *tinyBenchmarks: evaluating LLMs with fewer examples* (ICML 2024, `arXiv:2402.14992`).
- **Target Status:** `Planned` for v1.1.

### Horizon Track 3 — OpenTelemetry GenAI Semantic Conventions & Model Context Protocol (MCP) Bridge
- **Scope:** Export `SpanCollector` traces in native OpenTelemetry GenAI semantic format (`gen_ai.system`, `gen_ai.usage.input_tokens`, `gen_ai.agent.autonomy_tier`) and provide an MCP server/client adapter so external MCP-compliant agents and tools can be benchmarked with zero custom glue code.
- **Research Anchor:** OpenTelemetry GenAI Semantic Conventions; Anthropic Model Context Protocol (MCP).
- **Target Status:** `Planned` for v1.2.
