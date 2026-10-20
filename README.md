<div align="center">
  <h1>🚀 Agent-Bench</h1>
  <p><strong>Professional MLOps Benchmark Framework for Evaluating LLM Agents as Complete Systems</strong></p>

  [![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
  [![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/vfcarida/Agent-Bench/releases)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Build Status](https://img.shields.io/badge/tests-566%20passed-brightgreen.svg)](https://github.com/vfcarida/Agent-Bench/actions)
  [![Coverage](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)](https://github.com/vfcarida/Agent-Bench/actions)
  [![Type Checked](https://img.shields.io/badge/mypy-0%20errors%20%7C%20PEP%20561-blue.svg)](https://github.com/vfcarida/Agent-Bench)
  [![Architecture](https://img.shields.io/badge/architecture-Clean%20%2F%20Decoupled-orange.svg)](https://github.com/vfcarida/Agent-Bench)
</div>

---

## 💼 Executive Vision

Evaluating LLM-based agents requires moving beyond static, single-turn correctness checks. Enterprise agent systems operate in dynamic workflows—utilizing multi-step reasoning, invoking tools, retrieving knowledge, and conforming to strict safety guardrails.

> **Enterprise Positioning**: **Agent-Bench** is an enterprise-grade evaluation ecosystem specifically engineered for production agent workflows (such as Brazilian PIX banking, CRM compliance, and transactional e-commerce). It stands distinct from academic benchmarks (such as Tsinghua's AgentBench or general-purpose QA benchmarks) by prioritizing **zero-tolerance hard safety gating** (refusal failures can never be numerically compensated), **stateful environment sandboxing** (live balance mutations and state diff assertions), **autonomous multi-turn ReAct reasoning loops**, **unbiased per-task Pass@k & Pass^k reliability**, **inter-annotator calibration** (Cohen's Kappa & Krippendorff's Alpha), and **measured execution traces & real financial token accounting**.

**Agent-Bench** provides an end-to-end testing and analytics ecosystem to assess the complete agent lifecycle across dynamic domains. It measures:
- 🎯 **Functional Correctness:** State assertion accuracy, execution of multi-turn plans, and tool call fidelity.
- 🛡️ **Risk & Safety Compliance:** Refusal behavior on unsafe prompts, regulatory adherence, and forbidden action prevention.
- 💰 **Operational Cost, Latency & Pareto Efficiency:** Detailed token consumption (`tokens_in`, `tokens_out`), financial cost tracking ($ USD), latency distributions, and **Safety-Constrained Cost-Quality Pareto Frontiers** with 2D hypervolume indicators.
- 🔄 **Statistical Reliability & Graduated Autonomy:** Unbiased per-task `Pass@k` aggregation, non-parametric bootstrap 95% confidence intervals, `Pass^k` (consistent success across all $k$ trials) for high-risk domains, and automated classification into **Graduated Autonomy Tiers** (`Tier 1 — Shadow` $\rightarrow$ `Tier 4 — Full Bounded Autonomy`).
- 📐 **Psychometric & Structural Diagnostics:** Adjacent-difficulty degradation ($\Delta_{\text{easy}\to\text{expert}}$), cross-domain balanced profiles, cross-family transfer gaps, and IRT-inspired benchmark discriminative power (`bench diagnose-run`).

> **Note on Evaluation Baseline**: The default stub runner (`DefaultAgentRunner` without a backend model) is strictly a smoke agent for pipeline connectivity and test harness verification; it returns benign responses and empty states, and does **not** represent real-agent evaluation. For deterministic, answer-independent offline reference testing in CI, Agent-Bench provides `ScriptedAgentRunner`.

---

## ✨ Production Architectural Principles

1. **Clean Architecture (Decoupled Protocols):**
   Agent execution logic is strictly decoupled from benchmark orchestration via Python `Protocol` interfaces:
   - `AgentRunner`: Executes agent reasoning independently of underlying frameworks (`DefaultAgentRunner`, `ScriptedAgentRunner`, or custom LLM loops).
   - `TaskEnvironment`: Sandboxes state management, tool execution, and environment lifecycle.
   - `Evaluator`: Handles scoring, assertion checks, and subjective evaluation.
   - `UserSimulator`: Dynamically simulates conversational user personas, interactive clarifications, slot-filling, and **$\tau^2$-Bench Dec-POMDP dual-control shared-world actions** (`ScriptedUserSimulator`, `RuleBasedUserSimulator`, `ModelUserSimulator`, `DualControlUserSimulator`).

2. **Hard Safety Gating & Non-Compensable Evaluation (Robustness & Integrity):**
   - **Phase 0 (Hard Safety Gate):** Evaluates non-negotiable safety constraints (refusal requirements, forbidden state changes, disallowed tools). Any breach immediately forces `passed=False` with `safety_violation=True`. Hard safety failures can **never** be averaged away or compensated by quality, latency, or cost.
   - **Separated Safety Axis:** Scorecards report `safety_violations` (count) and `safety_gated` (bool) on a dedicated safety axis. Weighted `global_score` is computed strictly over the safety-passing subset with normalized weights.
   - **Phase 1 (Deterministic Check):** Verifies state diffs, assertion rules, and tool call schemas. Short-circuits on failure to avoid unnecessary LLM costs.
   - **Phase 2 (LLM / Subjective Judge):** Invoked only when safety and deterministic checks pass to evaluate subjective output quality.

3. **Execution Trace Analytics & Real Accounting (JSONL & Parquet):**
   Full execution histories and operational metrics are measured from live execution traces:
   - Measured token consumption (`tokens_in`, `tokens_out`, `total_tokens`).
   - Per-request, cumulative, and percentile latency distributions (`latency_p50`, `latency_p90`, `latency_p99`).
   - Financial cost calculation (`total_cost_usd`) driven by configurable per-model input/output pricing (`configs/models/*`) with documented default fallback.
   - Efficiency accounting with cost-per-successful-task metrics.

4. **Security & MLOps:**
   - **Zero Hardcoded Secrets:** Configuration and API keys are managed safely via `pydantic-settings` with `.env` support.
   - **Docker Sandbox Isolation:** Containerized execution environment preventing unsafe agent actions from affecting the host machine.

5. **Statistical Rigor & Reliability Estimators:**
   Benchmark scores reflect robust statistical accounting rather than noisy single-turn or pooled estimates:
   - **Per-Task Pass@k:** Evaluates the unbiased combinatorial estimator per task across repetitions (never pooled across distinct tasks).
   - **Bootstrap Confidence Intervals:** Generates 95% non-parametric bootstrap intervals for headline pass rates and reliability.
   - **Pass^k for High-Risk Profiles:** Measures the probability that *all* $k$ sampled trials succeed ($\text{pass}^k = \binom{c}{k}/\binom{n}{k}$), preventing pass@k from masking critical agent regressions in transactional and cyber domains.

6. **Dataset Governance, Split Discipline & Contamination Gating:**
   - **Canonical Gold Tree:** `datasets/gold/<split>/<domain>.yaml` serves as the single authoritative dataset hierarchy, using the typed `EvalCase v2` schema.
   - **Strict Split Isolation:** Split-aware loader isolates `dev`, `holdout`, `calibration`, `regression`, and `smoke`. Loading holdout content from legacy fixture paths is strictly prohibited by regression gates.
   - **Offline Contamination Gate:** An automated gate (`bench check-contamination`) runs in offline CI, testing for verbatim (SHA-256) and paraphrase (token Jaccard) leakage from holdout sets into dev or synthetic training pools.

7. **Autonomous Multi-Turn ReAct Reasoning Loop:**
   `DefaultAgentRunner` drives an autonomous ReAct loop: invoking model tools against the stateful `TaskEnvironment`, appending observations to the conversational context, and re-querying until task termination or step exhaustion (`max_steps`). Loop stall detection guards against repetitive static mock stagnation.

8. **Inter-Annotator Agreement & Calibration:**
   Measures grading dataset consistency using Cohen's Kappa (pairwise) and Krippendorff's Alpha (multi-rater reliability across arbitrary nominal labels) via `bench check-agreement`.

9. **Rich Interactive Visual Trace Exploration:**
   Inspect prompts, `<think>` internal reasoning blocks, tool arguments, environment observations, and judge verdicts via Rich terminal panels (`bench view-traces`).

10. **Safety-Constrained Pareto Frontier, Graduated Autonomy & 2PL IRT Psychometric Diagnostics:**
    - **Safety-Constrained Pareto Frontier (`metrics/pareto.py`):** Evaluates systems across the joint space of functional quality and `cost_per_successful_task` (with 2D hypervolume indicators), automatically disqualifying any system that triggers the Phase 0 safety gate (*AI Agents That Matter*, `arXiv:2407.01502`).
    - **Graduated Autonomy Tiers:** Classifies each evaluated system into enterprise deployment readiness tiers (`Tier 1 — Shadow / Advisory`, `Tier 2 — Human-in-the-Loop`, `Tier 3 — Guided Autonomy`, `Tier 4 — Full Bounded Autonomy`, or `Blocked — Safety Gate Triggered`).
    - **2PL IRT Fast-Anchor & Run Diagnostics (`metrics/irt.py`, `bench diagnose-run`):** Fits a 2-Parameter Logistic Item Response Theory model ($\theta_j, a_i, b_i$), enables stratified Fisher Information fast-anchor evaluation (`bench run-suite --fast-anchor`), and computes FINESSE-Bench difficulty degradation ($\Delta$), cross-domain balanced profiles, and cross-family transfer gaps.
11. **OpenTelemetry GenAI Semantic Conventions & Model Context Protocol (MCP) Bridge:**
    - **OTLP GenAI Span Export (`bench export-otel`):** Exports benchmark runs and traces in native OpenTelemetry GenAI `resourceSpans` JSON (`gen_ai.system`, `gen_ai.usage.*`, `gen_ai.agent.autonomy_tier`, `gen_ai.safety.gated`).
    - **Bidirectional MCP Bridge (`tools/mcp_bridge.py`):** Exposes any `TaskEnvironment` and `ToolAdapter` registry as a JSON-RPC 2.0 MCP (`2024-11-05`) server (`MCPToolServer`) and consumes external MCP servers as native `ToolAdapter` instances (`MCPClientToolAdapter`, `discover_mcp_tools`).
12. **Multi-Agent Byzantine Resilience, Causal Counterfactual Replay & Online Drift Assurance:**
    - **Byzantine Fault Tolerance (`runners/byzantine_orchestrator.py`):** Stress-tests multi-agent handoffs against hallucinated state, poisoned tool outputs, indirect prompt injection, and multi-agent collusion (`ByzantineOrchestratorRunner`, `compute_byzantine_resilience_metrics`).
    - **Causal Counterfactual Replay (`runners/counterfactual.py`, `bench replay-counterfactual`):** Forks recorded trajectories at step $t$ to evaluate counterfactual state mutations, tool overrides, or user interventions without re-running earlier steps.
    - **Shadow-Traffic Online Drift Monitor (`runners/online_eval.py`, `bench monitor-drift`):** Monitors production JSONL or OTLP GenAI spans against a certified baseline using tabular **CUSUM** change-point detectors and **Jensen-Shannon Divergence** (`OnlineDriftMonitor`, `CUSUMDriftDetector`).

---

## 📦 Installation

Agent-Bench requires **Python 3.12+**.

```bash
# Clone the repository
git clone https://github.com/vfcarida/Agent-Bench.git
cd Agent-Bench

# Install core package along with development requirements
pip install -e ".[dev]"

# Install with model provider integrations
pip install -e ".[openai]"
pip install -e ".[anthropic]"

# Install all dependencies
pip install -e ".[all]"
```

---

## 🏗️ Architecture Diagram

```mermaid
graph TD
    subgraph CLI & Runner Layer
        CLI[bench CLI] --> CaseRunner[CaseRunner / SuiteRunner]
    end

    subgraph Clean Architecture Protocols
        CaseRunner --> AgentRunner["AgentRunner Protocol<br/>(LangChain / CrewAI / Custom)"]
        CaseRunner --> TaskEnv["TaskEnvironment Protocol<br/>(State Sandbox / Tools)"]
        CaseRunner --> Evaluator["Evaluator Protocol"]
    end

    subgraph Gated Evaluation Engine
        Evaluator --> GatedEval[GatedEvaluator]
        GatedEval --> Phase1["Phase 1: Deterministic Check<br/>(State Diffs / Assertions / Refusal)"]
        Phase1 -- Fails --> ShortCircuit["Short-Circuit Gate<br/>(Score: 0.0, Bypass LLM Judge)"]
        Phase1 -- Passes --> Phase2["Phase 2: LLM / Subjective Judge<br/>(Semantic / Rubric / Grounding)"]
    end

    subgraph Execution Trace & Analytics Storage
        AgentRunner --> TraceLogger[ExecutionTraceLogger]
        TraceLogger --> JSONL[Atomic JSONL Traces]
        TraceLogger --> Parquet[Parquet Analytics Metrics<br/>(Tokens / Latency / Cost USD)]
    end

    subgraph Security & Settings
        Config[BenchSettings / pydantic-settings] --> CLI
        Config --> .env[.env Environment Variables]
    end
```

### Directory Structure

```text
agent-bench/
  .github/workflows/    # CI/CD pipeline (Ruff, Mypy, Pytest, Golden Tasks)
  configs/              # YAML configurations (models, systems, suites, judges)
  datasets/
    gold/dev/           # Curated golden evaluation tasks (JSONL & YAML)
    synthetic/shadow/   # Auto-generated shadow cases for coverage expansion
    adversarial/        # Attack vectors and boundary-testing cases
  src/agent_bench/
    cli/                # Click-based CLI commands (run-suite, diagnose-run, view-traces, etc.)
    core/               # Protocols (AgentRunner, TaskEnvironment, Evaluator), Settings, Schema v2
    graders/            # GatedEvaluator, SafetyGate, RubricGrader, StateGrader, ToolCallGrader
    judges/             # Deterministic, Semantic, Grounding, CrossArtifact, and Calibration Harness
    metrics/            # Pass@k/Pass^k, Pareto Frontier, Autonomy Tiers, Difficulty & IRT Diagnostics
    models/             # Provider adapters (OpenAI, Anthropic, HuggingFace, vLLM, Stub)
    reports/            # Markdown, HTML (Chart.js), Run Comparator, and Leaderboard generators
    runners/            # CaseRunner, SuiteRunner, ScriptedAgentRunner, PromptFormatter
    storage/            # ExecutionTraceLogger, Resilient JSONL & Parquet Analytics Storage
    tools/              # DomainToolRegistry, PIX/Investment/SME/Cyber tools, and MCP JSON-RPC Bridge
    validators/         # Schema v2 validation, contamination gating & Jaccard deduplication
  tests/                # Unit & integration test suites (591 passing tests, 89% coverage)
  Dockerfile            # Containerized execution environment
  docker-compose.yml    # Docker sandbox compose configuration
```

---

## 🚀 Quickstart

Setting up environment variables:

```bash
# Copy example environment configuration
cp .env.example .env

# Edit .env to add your API keys (optional for stub runs)
# OPENAI_API_KEY=your_key_here
```

Operating Agent-Bench via CLI:

```bash
# 1. Validate YAML configuration and dataset schemas
bench --config-dir configs validate-config

# 2. Run a benchmark suite (stub mode - no API keys required!)
bench --config-dir configs run-suite pix_basic_v1 --concurrency 4

# 3. Run a specific task case
bench --config-dir configs run-case PIX_001 --system tool_calling_reactive_gpt4 --domain pix_assist

# 4. View interactive execution traces in rich terminal panels
bench view-traces <run-id> --limit 20

# 5. Check inter-annotator calibration & agreement (Cohen's Kappa & Krippendorff's Alpha)
bench check-agreement --annotations datasets/gold/calibration/annotator_agreement.yaml

# 6. Check for holdout contamination and data leakage
bench check-contamination --threshold 0.80

# 7. Diagnose Safety-Constrained Pareto Frontier, Autonomy Tiers & Psychometrics
bench diagnose-run <run-id> --format table

# 8. Generate Markdown / HTML execution reports
bench generate-report <run-id> --format html

# 9. Compare two evaluation runs side-by-side
bench compare-runs <run-id-1> <run-id-2>

# 10. Export OpenTelemetry GenAI Semantic Convention spans (OTLP/JSON)
bench export-otel <run-id> --output otel_genai_spans.json

# 11. Aggregate multi-institution benchmarks under (epsilon, delta)-Differential Privacy
bench aggregate-federated data/federated/submissions.json --epsilon 0.8 --min-sites 3

# 12. Generate & verify SHA-256 Merkle Tree Selective-Disclosure Attestations
bench attest-run <run-id> -o attestation.json
bench verify-attestation attestation.json

# 13. Run closed-loop 2PL IRT-guided evolutionary red-team fuzzing (MAP-Elites)
bench evolve-redteam --domain pix_assist --generations 3 -o evolved_redteam.json

# 14. Generate & verify Hardware-Bound TEE Enclave Attestation Quotes (PCR0/PCR1/PCR2/PCR8)
bench attest-tee <run-id> --tee-arch aws_nitro_enclave --nonce audit_2026 -o tee_quote.json
bench verify-tee-quote tee_quote.json --expected-nonce audit_2026

# 15. Fold multiple runs into a constant-size Nova Relaxed R1CS IVC proof & verify
bench fold-zk-proofs run_1.json run_2.json run_3.json -o ivc_bundle.json
bench verify-ivc ivc_bundle.json

# 16. Simulate mid-episode macroeconomic & regulatory regime shifts
bench simulate-regime-shift --domain pix_assist --steps 6 -o regime_report.json

# 17. Synthesize mechanistic SAE/CLT circuit clamping & evaluate Safety Recovery Rate
bench steer-circuit --input-dim 4 --top-k 3 -o steer_report.json

# 18. Prove & verify Zero-Knowledge Federated Leaderboard (Nova IVC + Differential Privacy)
bench prove-federated-leaderboard subs.json --epsilon 1.0 -o zk_fed_bundle.json
bench verify-federated-leaderboard zk_fed_bundle.json

# 19. Simulate multi-agent market microstructure & liquidity contagion
bench simulate-contagion --agents 4 --steps 8 --stabilizing -o contagion_report.json

# 20. Build & verify Cryptographic Proof-Carrying Agent Trajectories (PCAT)
bench verify-pcat --trajectory-id pcat_cli_001 -o pcat_report.json

# 21. Execute sequential Bayesian 2PL IRT Computerized Adaptive Testing (CAT)
bench simulate-cat --system-id cat_agent_v1 --true-theta 0.85 --target-se 0.42 -o cat_report.json

# 22. Audit cross-jurisdictional statutory compliance (BACEN/LGPD, EU AI Act/GDPR, US SEC/FINRA)
bench audit-jurisdictions --system-id global_bank_agent -o regulatory_report.json
```

---

## 🐳 Running in Isolated Docker Sandbox

To run benchmarks securely in an isolated container sandbox:

```bash
# Build Docker image
docker build -t agent-bench:latest .

# Run benchmark suite inside container using docker-compose
docker-compose up
```

---

## 📊 Evaluation Protocol & Variance Management

Agent-Bench enforces strict evaluation protocols to measure reliability, cost, and latency:

1. **Variance Control & Seeds:**
   - Tasks are executed with configurable random seeds (`seed: 42`).
   - Suites support multi-iteration runs (`repeat_n: 3`) to calculate Pass@k metrics and 95% bootstrap confidence intervals.

2. **Cost Calculation:**
   - Financial cost is calculated per request using model token pricing:
     \[
     \text{Total Cost (USD)} = \left(\frac{\text{Tokens}_{\text{in}}}{1000} \times \text{Price}_{\text{in}}\right) + \left(\frac{\text{Tokens}_{\text{out}}}{1000} \times \text{Price}_{\text{out}}\right)
     \]

3. **Gated Evaluation Pipeline:**
   - Deterministic assertion checks execute first in $O(1)$ time.
   - Bypassing LLM judges on deterministic failures reduces benchmark execution time and LLM API cost by up to 70%.

---

## 🧪 Running Tests

Ensure all components and clean architecture protocols pass locally:

```bash
# Run complete test suite (614 passed tests)
python -m pytest tests/ -v

# Run specific unit test modules
python -m pytest tests/unit/test_protocols.py -v
python -m pytest tests/unit/test_gated_evaluator.py -v
python -m pytest tests/unit/test_pareto_and_psychometrics.py -v
python -m pytest tests/unit/test_dual_control_irt_mcp_otel.py -v
python -m pytest tests/unit/test_byzantine_counterfactual_and_drift.py -v
python -m pytest tests/unit/test_formal_multimodal_federated_and_coverage.py -v
python -m pytest tests/unit/test_zk_attestation_fuzzer_gui_and_coverage.py -v
python -m pytest tests/unit/test_tee_activation_cdp_and_adapters_coverage.py -v
python -m pytest tests/unit/test_hybrid_retrieval_episodic_memory_and_collusion.py -v
python -m pytest tests/unit/test_sae_attribution_zk_r1cs_and_policy_synthesizer.py -v
python -m pytest tests/unit/test_circuit_tracer_nova_ivc_and_regime_shift.py -v
python -m pytest tests/unit/test_activation_steering_zk_federated_and_contagion.py -v
python -m pytest tests/unit/test_pcat_bayesian_cat_and_regulatory_harmonizer.py -v
```

---

## 📚 Documentation & Guides

| Category | Guide | Description |
| :--- | :--- | :--- |
| **Getting Started** | [Installation Guide](docs/getting-started/installation.md) | Setup requirements, provider extras, and environment variables |
| | [Quickstart Guide](docs/getting-started/quickstart.md) | 5-minute smoke evaluation and pipeline walk-through |
| | [Architecture Overview](docs/getting-started/architecture-overview.md) | Clean Architecture protocols, multi-turn loops, and gated pipelines |
| **Developer Guides** | [Integrating External Agents](docs/guides/adding-agent.md) | Benchmark LangChain, LangGraph, CrewAI, or custom Python agents |
| | [Adding a New Domain](docs/guides/adding-domain.md) | Create custom domain scenarios, state mutators, and tools |
| | [Running Local Models](docs/guides/running-local-models.md) | Benchmark open-weights models using vLLM or Hugging Face |
| | [Inspect AI Interoperability](docs/guides/inspect-ai-interoperability.md) | Evaluate Agent-Bench tasks with UK/US AISI Inspect AI |
| **Methodology** | [Hard Safety Gating](docs/methodology/safety-gating.md) | Non-compensable Phase 0 safety invariants and refusal checks |
| | [Statistical Reliability](docs/methodology/statistical-metrics.md) | Mathematical formulation of Pass@k, Pass^k, and bootstrap CIs |
| | [Pareto Frontier & Psychometrics](docs/methodology/pareto-and-psychometrics.md) | Cost-quality Pareto frontier, Graduated Autonomy Tiers, and IRT diagnostics |
| | [Rubric & Tool Grading](docs/methodology/rubric-and-tool-grading.md) | Hierarchical importance weights and critical gate capping |
| | [Inter-Annotator Agreement](docs/methodology/inter-annotator.md) | Multi-rater Cohen's Kappa and Krippendorff's Alpha calibration |
| **Governance & Roadmap** | [Architecture & Evolution Roadmap](ROADMAP.md) | Maturity governance tiers, verified capability matrix, and frontier tracks |
| | [Scientific & Executive References](docs/REFERENCES.md) | Primary academic papers, Gartner/BCG/McKinsey frameworks, and standards |
| **Reference** | [CLI Reference Manual](docs/reference/cli.md) | Complete reference for all `bench` CLI commands and options |
| | [Python API Reference](docs/reference/api.md) | Typed SDK protocols, runners, evaluators, and metric functions |

---

## 🗺️ Roadmap & Maturity Governance

See [`ROADMAP.md`](ROADMAP.md) for the complete capability matrix and maturity tier governance (`Stable`, `Beta`, `Alpha`, `Experimental`, `Planned`):
- **Shipped (`Stable`):** Clean Architecture Protocols & Schema v2 (`py.typed`), Two-Phase Gated Evaluator (`Phase 0` Safety Gate $\rightarrow$ `Phase 1` Deterministic $\rightarrow$ `Phase 2` Judge), Formal Symbolic Trajectory Verification (`SymbolicTrajectoryVerifier` catching transient TOCTOU & tool precondition breaches), Cryptographic Proof-Carrying Agent Trajectories (`PCATVerifier`, `ProofCarryingStepEnvelope`, `bench verify-pcat`), Mechanistic Activation Probing & CoT-Action Faithfulness Monitoring (`LinearConceptProbe`, `CoTFaithfulnessMonitor`, `ActivationSafetyMonitor`), Sparse Autoencoder (SAE) Monosemantic Feature Dictionary & Causal Circuit Attribution (`SparseAutoencoderDictionary`, `SAECircuitAttributor`), Cross-Layer Transcoder (CLT) & Multi-Step Attribution Graph Extractor (`CrossLayerTranscoder`, `CircuitGraphTracer`, `AttributionGraph`), Mechanistic Circuit-Guided Automated Steering & Activation Clamping (`MechanisticActivationSteerer`, `FeatureClampIntervention`, `bench steer-circuit`), Per-Task `Pass@k` & `Pass^k` with 95% Bootstrap CIs, $\tau^2$-Bench Dual-Control Shared-World User Simulation (`DualControlUserSimulator`), Safety-Constrained Pareto Frontier & BCG Graduated Autonomy Tiers (`metrics/pareto.py`), 2PL IRT Fast-Anchor Evaluation (`metrics/irt.py`, `--fast-anchor`), Sequential Bayesian Computerized Adaptive Testing (`BayesianCATEngine`, `bench simulate-cat`), & Psychometric Diagnostics (`bench diagnose-run`), OpenTelemetry GenAI Span Export (`bench export-otel`) & MCP JSON-RPC 2.0 Bridge (`tools/mcp_bridge.py`), Multi-Agent Byzantine Fault Tolerance (`ByzantineOrchestratorRunner`), Game-Theoretic Multi-Agent Collusion, Sybil & Shapley Attribution Auditor (`MultiAgentCollusionDetector`, `ShapleyContributionAuditor`, `bench audit-collusion`), Causal Counterfactual Trajectory Replay (`CounterfactualSandbox`, `bench replay-counterfactual`), Long-Horizon Episodic Memory Consolidation & Interference Stress-Tester (`EpisodicMemoryStressRunner`, `MemoryInterferenceInjector`, `bench stress-memory`), Continual World-Model Adaptation & Non-Stationary Regime-Shift Simulator (`NonStationaryWorldSimulator`, `RegimeShiftShock`, `bench simulate-regime-shift`), Multi-Agent Macroeconomic Market Microstructure & Contagion Sandbox (`SystemicContagionSandbox`, `bench simulate-contagion`), Continuous Shadow-Traffic Online Drift Monitoring (`OnlineDriftMonitor`, `bench monitor-drift`), Hybrid BM25 + Dense Vector RRF Retrieval & RAG Poisoning Defense (`HybridRRFCorpusRetriever`, `CorpusPoisoningSanitizer`), Multi-Modal Document & UI-Action Financial Sandbox (`MultiModalFinancialSandbox`), Multi-Screen GUI/DOM Navigation & Accessibility Tree Runtime (`GUINavigationRuntime`), CDP Browser Bridge & Network Egress Firewall (`CDPBrowserSandbox`, `NetworkEgressFirewall`), Privacy-Preserving Federated Benchmark Aggregation (`FederatedBenchmarkAggregator`, `bench aggregate-federated`), Cryptographic Merkle Execution Attestation (`MerkleExecutionAttestor`, `bench attest-run`, `bench verify-attestation`), Hardware-Enforced TEE Enclave Attestation (`ConfidentialEnclaveAttestor`, `bench attest-tee`, `bench verify-tee-quote`), Verifiable Zero-Knowledge R1CS Circuit Compiler & Witness Verifier (`ZKEvaluationCircuitCompiler`, `bench compile-zk-circuit`, `bench verify-zk-circuit`), Incrementally Verifiable Computation (IVC) & Nova Relaxed R1CS Folding (`NovaIVCFolder`, `IVCAccumulatorBundle`, `bench fold-zk-proofs`, `bench verify-ivc`), Zero-Knowledge Federated Leaderboard Proofs (`ZKFederatedLeaderboardAggregator`, `bench prove-federated-leaderboard`, `bench verify-federated-leaderboard`), Cross-Lingual & Multi-Jurisdictional Regulatory Compliance Harmonizer (`RegulatoryComplianceHarmonizer`, `bench audit-jurisdictions`), Autonomous Self-Healing Sandbox Policy Synthesizer (`SelfHealingPolicySynthesizer`, `bench synthesize-policy`), Closed-Loop IRT Evolutionary Red-Team Fuzzer (`EvolutionaryRedTeamFuzzer`, `bench evolve-redteam`), Multi-Domain Sandboxes (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`), and Dataset Provenance & Contamination Gating.
- **Shipped (`Experimental`):** Rubric-based `SemanticJudge` with prompt-injection defense and Cohen's $\kappa$ calibration gate (`bench calibrate-judge`).
- **External Hardware & Post-Quantum Horizon (`Planned`):** Physical Bare-Metal Confidential GPU Attestation Bus Integration and Post-Quantum LatticeFold Proofs.

---

## 📜 License & Authorship

This project is licensed under the terms of the **Apache-2.0** License. See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff) for citation metadata.

*Designed and architected by [Vinicius Caridá](https://github.com/vfcarida). Implementation and verification accelerated with AI-assisted engineering workflows.*
