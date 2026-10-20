# Command-Line Interface (CLI) Reference

The `bench` CLI provides commands for running benchmark suites, debugging execution traces, ranking systems, and auditing data integrity.

---

## Command Overview

| Command | Description |
| :--- | :--- |
| [`run-suite`](#1-bench-run-suite) | Run an evaluation benchmark suite |
| [`run-case`](#2-bench-run-case) | Execute a single benchmark task by ID |
| [`view-traces`](#3-bench-view-traces) | Interactive terminal visualizer for step-by-step traces |
| [`leaderboard`](#4-bench-leaderboard) | Display system ranking leaderboard |
| [`export-html`](#5-bench-export-html) | Generate interactive self-contained HTML dashboard |
| [`compare-runs`](#6-bench-compare-runs) | Compare metrics across two historical runs |
| [`analytics`](#7-bench-analytics) | Run cross-system and trend queries across runs |
| [`validate-config`](#8-bench-validate-config) | Validate YAML benchmark configurations |
| [`check-agreement`](#9-bench-check-agreement) | Calculate inter-annotator agreement (Kappa & Alpha) |
| [`list-suites`](#10-bench-list-suites) | List available evaluation suites |
| [`list-models`](#11-bench-list-models) | List configured model endpoints |
| [`list-systems`](#12-bench-list-systems) | List configured agent systems |
| [`list-plugins`](#13-bench-list-plugins) | List registered domain, tool, and model plugins |
| [`export-inspect`](#14-bench-export-inspect) | Export datasets to Inspect AI dataset format |
| [`export-inspect-log`](#15-bench-export-inspect-log) | Export run traces to Inspect AI EvalLog format |
| [`generate-security-suite`](#16-bench-generate-security-suite) | Generate OWASP Top 10 synthetic adversarial security suite |
| [`diagnose-run`](#17-bench-diagnose-run) | Display Safety-Constrained Pareto Frontier, Autonomy Tiers, and 2PL IRT diagnostics |
| [`export-otel`](#18-bench-export-otel) | Export benchmark run traces in OpenTelemetry GenAI Semantic Conventions (OTLP/JSON) |
| [`replay-counterfactual`](#19-bench-replay-counterfactual) | Fork a recorded trajectory at step $t$ and evaluate counterfactual state mutations |
| [`monitor-drift`](#20-bench-monitor-drift) | Monitor shadow traffic (JSONL or OTLP GenAI JSON) for CUSUM and JSD drift |
| [`aggregate-federated`](#21-bench-aggregate-federated) | Aggregate multi-institution benchmark submissions under $(\varepsilon, \delta)$-Differential Privacy |
| [`attest-run`](#22-bench-attest-run) | Generate a SHA-256 Merkle Tree & Selective-Disclosure attestation bundle for a run |
| [`verify-attestation`](#23-bench-verify-attestation) | Verify HMAC signature and all `MerkleInclusionProof` claims in an attestation bundle |
| [`evolve-redteam`](#24-bench-evolve-redteam) | Run closed-loop 2PL IRT-guided evolutionary red-team fuzzing (`MAPElitesArchive`) |
| [`attest-tee`](#25-bench-attest-tee) | Bind a run's Merkle root into a hardware-enforced TEE Enclave Attestation Quote |
| [`verify-tee-quote`](#26-bench-verify-tee-quote) | Verify a TEE Enclave Attestation Quote, PCR measurements, and bound Merkle proofs |
| [`stress-memory`](#27-bench-stress-memory) | Execute long-horizon episodic memory consolidation and interference stress testing |
| [`audit-collusion`](#28-bench-audit-collusion) | Audit multi-agent committee voting logs for Sybil coalitions and Shapley attribution |
| [`compile-zk-circuit`](#29-bench-compile-zk-circuit) | Compile a benchmark run into a Goldilocks $\mathbb{F}_p$ R1CS arithmetic circuit and proof |
| [`verify-zk-circuit`](#30-bench-verify-zk-circuit) | Verify a compiled R1CS arithmetic circuit and Fiat-Shamir witness proof |
| [`synthesize-policy`](#31-bench-synthesize-policy) | Synthesize self-healing LTL, egress firewall, and RAG policies from red-team breakthroughs |
| [`fold-zk-proofs`](#32-bench-fold-zk-proofs) | Fold multiple benchmark runs into a constant-size Nova Relaxed R1CS IVC accumulator proof |
| [`verify-ivc`](#33-bench-verify-ivc) | Verify a Nova-style Relaxed R1CS IVC accumulator bundle and all cross-term folding steps |
| [`simulate-regime-shift`](#34-bench-simulate-regime-shift) | Simulate mid-episode macroeconomic, regulatory, and cyber regime shifts in sandboxes |
| [`steer-circuit`](#35-bench-steer-circuit) | Synthesize mechanistic SAE/CLT clamping interventions and evaluate Safety Recovery Rate |
| [`prove-federated-leaderboard`](#36-bench-prove-federated-leaderboard) | Verify institutional R1CS proofs, fold via Nova IVC, and release a DP federated leaderboard |
| [`verify-federated-leaderboard`](#37-bench-verify-federated-leaderboard) | Verify a Zero-Knowledge Federated Leaderboard bundle (Nova IVC + DP commitment) |
| [`simulate-contagion`](#38-bench-simulate-contagion) | Simulate multi-agent market microstructure liquidity contagion and herding |

---

## 1. `bench run-suite`

Executes a benchmark suite across defined systems and tasks.

```bash
bench run-suite <suite_id> [OPTIONS]
```

### Options
- `--config-dir PATH`: Directory containing configuration manifests. *(Default: `configs`)*
- `--output-dir PATH`: Directory to write run artifacts. *(Default: `data/runs`)*
- `--repeat-n INTEGER`: Number of repetitions per task for statistical estimation. *(Default: from suite)*
- `--concurrency / -c INTEGER`: Maximum concurrent tasks to execute in parallel. *(Default: `1`)*
- `--system TEXT`: Override systems to evaluate.
- `--split TEXT`: Dataset split to evaluate (`dev`, `holdout`). *(Default: `dev`)*
- `--seed INTEGER`: Random seed for reproducibility.
- `--enable-llm-judge`: Enable Phase 2 subjective LLM judge evaluation.
- `--judge-model TEXT`: Model ID for LLM judge.
- `--sprt / --early-stopping`: Enable Wald's Sequential Probability Ratio Test for early stopping on conclusive task repetitions.
- `--user-simulator [none|scripted|rule_based|dual_control|human]`: Attach interactive user simulator for multi-turn or Dec-POMDP dual-control conversational evaluation.
- `--fast-anchor`: Evaluate only an IRT-stratified anchor subset of tasks and estimate full-suite scores via 2-Parameter Logistic (2PL) Item Response Theory (*tinyBenchmarks*, `arXiv:2402.14992`).
- `--anchor-fraction FLOAT`: Fraction of suite tasks to sample in `--fast-anchor` mode. *(Default: `0.25`)*

### Example
```bash
bench run-suite pix_basic_v1 --repeat-n 3 --concurrency 5 --split dev
bench run-suite pix_basic_v1 --fast-anchor --anchor-fraction 0.25 --user-simulator dual_control
```

---

## 2. `bench run-case`

Executes a single benchmark task in isolation for debugging.

```bash
bench run-case <task_id> [OPTIONS]
```

### Options
- `--system-id TEXT`: System identifier to evaluate. *(Required)*
- `--domain TEXT`: Domain of the task (`pix_assist`, `cyber_sandbox`, etc.). *(Required)*
- `--split TEXT`: Dataset split. *(Default: `dev`)*

### Example
```bash
bench run-case PIX_DEV_001 --domain pix_assist --system-id scripted_system
```

---

## 3. `bench view-traces`

Launches the rich interactive terminal trace visualizer. Displays step-by-step thinking blocks, tool calls, tool outputs, and judge decisions.

```bash
bench view-traces <run_id> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing runs. *(Default: `data/runs`)*
- `--task-id TEXT`: Filter display to a single task ID.

### Example
```bash
bench view-traces run_20260925_143000
```

---

## 4. `bench leaderboard`

Displays system rankings across benchmark runs.

```bash
bench leaderboard [OPTIONS]
```

### Options
- `--domain TEXT`: Filter by specific domain.
- `--top INTEGER`: Number of top entries to show. *(Default: `20`)*
- `--format [table|markdown|html|json]`: Output presentation format. *(Default: `table`)*
- `--leaderboard-path PATH`: Path to leaderboard JSON file.

### Example
```bash
bench leaderboard --domain pix_assist --format table
bench leaderboard --format markdown > LEADERBOARD.md
```

---

## 5. `bench export-html`

Generates an interactive, standalone HTML report with real-time search, latency graphs, and scorecard breakdowns.

```bash
bench export-html [OPTIONS]
```

### Options
- `--run-id TEXT`: Run identifier to export.
- `--runs-dir PATH`: Directory containing runs. *(Default: `data/runs`)*
- `--output PATH`: Target HTML file path. *(Default: `data/reports/<run_id>.html`)*

### Example
```bash
bench export-html --run-id run_20260925_143000
```

---

## 6. `bench compare-runs`

Compares two benchmark runs side-by-side, computing score deltas and regressions.

```bash
bench compare-runs <baseline_run_id> <candidate_run_id> [OPTIONS]
```

---

## 7. `bench analytics`

Runs analytics queries across historical benchmark runs.

```bash
bench analytics [OPTIONS]
```

### Options
- `--by [system|domain]`: Aggregation axis.
- `--system TEXT`: Filter by specific system ID.
- `--domain TEXT`: Filter by domain for cross-system comparisons.

---

## 8. `bench validate-config`

Validates all YAML configuration files in `configs/` against Pydantic schemas.

```bash
bench validate-config [OPTIONS]
```

---

## 9. `bench check-agreement`

Calculates multi-annotator agreement metrics (**Cohen's Kappa** and **Krippendorff's Alpha**) to calibrate judge and human ratings.

```bash
bench check-agreement [OPTIONS]
```

### Options
- `--annotations PATH`: Path to multi-annotator YAML/JSON ratings file.

---

## 10. `bench list-suites`

Lists all benchmark suites configured in `configs/suites/`.

```bash
bench list-suites [OPTIONS]
```

---

## 11. `bench list-models`

Lists all configured LLM and local model endpoints in `configs/models/`.

```bash
bench list-models [OPTIONS]
```

---

## 12. `bench list-systems`

Lists all configured agent systems in `configs/systems/`.

```bash
bench list-systems [OPTIONS]
```

---

## 13. `bench list-plugins`

Lists all registered domain tools, mutators, and model adapters discovered in the runtime environment.

```bash
bench list-plugins [OPTIONS]
```

---

## 14. `bench export-inspect`

Exports Agent-Bench datasets into standard UK/US AI Safety Institute **Inspect AI** dataset format (`inspect_evals`).

```bash
bench export-inspect [OPTIONS]
```

### Options
- `--domain TEXT`: Evaluation domain to export (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`).
- `--split [dev|holdout]`: Dataset split to export *(Default: `dev`)*.
- `--output PATH`: Target output JSON/JSONL file path.

---

## 15. `bench export-inspect-log`

Converts completed Agent-Bench run traces into Inspect AI `EvalLog` compatible format.

```bash
bench export-inspect-log <run_id> [OPTIONS]
```

### Options
- `--output PATH`: Target output path for Inspect eval log JSON.

---

## 16. `bench generate-security-suite`

Generates synthetic adversarial evaluation test cases aligned with the OWASP Top 10 for LLMs.

```bash
bench generate-security-suite [OPTIONS]
```

### Options
- `--count INTEGER`: Number of security cases to generate. *(Default: `20`)*
- `--domain TEXT`: Target domain (`cyber_sandbox`, `investment_advisor`, `pix_assist`).
- `--lang [auto|en|pt]`: Target prompt language. *(Default: `auto`)*
- `--output / -o PATH`: Output YAML dataset file path. *(Default: `datasets/synthetic/security_generated.yaml`)*
- `--seed INTEGER`: Random generation seed. *(Default: `42`)*

### Example
```bash
bench generate-security-suite --count 50 --domain cyber_sandbox --lang en --output datasets/synthetic/cyber_adversarial.yaml
```

---

## 17. `bench diagnose-run`

Displays the Safety-Constrained Cost-Quality Pareto Frontier, BCG Graduated Autonomy Tier classifications, 2D hypervolume indicator, difficulty degradation profiles, and IRT-inspired psychometric discriminative power for a completed benchmark run.

```bash
bench diagnose-run <run_id> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing run artifacts. *(Default: `data/runs`)*
- `--format [table|json]`: Output format (`table` renders Rich terminal tables; `json` outputs structured diagnostic JSON). *(Default: `table`)*

### Example
```bash
bench diagnose-run run_20261019_120000 --format table
bench diagnose-run run_20261019_120000 --format json
```

---

## 18. `bench export-otel`

Exports a completed benchmark run's manifest, scorecards, and execution traces in native **OpenTelemetry GenAI Semantic Conventions** (`OTLP/JSON` `resourceSpans` format with `gen_ai.*` attributes).

```bash
bench export-otel <run_id> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing run artifacts. *(Default: `data/runs`)*
- `--output / -o PATH`: Output OTLP GenAI JSON file path. If omitted, prints OTLP JSON to stdout.

### Example
```bash
bench export-otel run_20261019_120000 --output data/runs/run_20261019_120000/otel_genai.json
```

---

## 19. `bench replay-counterfactual`

Forks a recorded trajectory at `--fork-step` and applies a counterfactual state mutation (`--mutate-state`) without re-executing earlier steps, evaluating the causal delta on task success and Phase 0 safety gating.

```bash
bench replay-counterfactual <run_id> --task-id <task_id> [OPTIONS]
```

### Options
- `--task-id TEXT`: Task ID to fork and replay counterfactually. *(Required)*
- `--domain TEXT`: Domain of the task. *(Default: `pix_assist`)*
- `--fork-step INTEGER`: Trajectory step index to fork at. *(Default: `1`)*
- `--mutate-state TEXT`: JSON dictionary of counterfactual state mutations. *(Default: `"{}"`)*
- `--runs-dir PATH`: Directory containing run artifacts. *(Default: `data/runs`)*

### Example
```bash
bench replay-counterfactual run_20261019_120000 --task-id PIX_DEV_001 --fork-step 2 --mutate-state '{"unauthorized_transfer": true}'
```

---

## 20. `bench monitor-drift`

Evaluates shadow production traffic (JSONL traces or OTLP GenAI `resourceSpans` JSON) against a certified baseline run using two-sided tabular **CUSUM** change-point detection (latency and cost) and **Jensen-Shannon Divergence** (tool-call distribution shift).

```bash
bench monitor-drift <baseline_run_id> <shadow_path> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing the baseline run artifact. *(Default: `data/runs`)*

### Example
```bash
bench monitor-drift run_20261019_120000 data/shadow/production_traces.jsonl
```

---

## 21. `bench aggregate-federated`

Aggregates multi-institution benchmark scorecards and 2PL IRT item pass rates using calibrated $(\varepsilon, \delta)$-Differential Privacy (Laplace mechanism) and $k$-anonymity site thresholds without sharing raw execution traces.

```bash
bench aggregate-federated <submissions_path> [OPTIONS]
```

### Options
- `--epsilon FLOAT`: Differential privacy $\varepsilon$ budget for the aggregation query. *(Default: `1.0`)*
- `--delta FLOAT`: Differential privacy $\delta$ budget. *(Default: `0.0`)*
- `--min-sites INTEGER`: Minimum $k$-anonymity participating sites required to release aggregated statistics. *(Default: `2`)*
- `--seed INTEGER`: Deterministic RNG seed for calibrated Laplace noise. *(Default: `42`)*

### Example
```bash
bench aggregate-federated data/federated/site_submissions.json --epsilon 0.8 --min-sites 3
```

---

## 22. `bench attest-run`

Generates a cryptographic **SHA-256 Merkle Tree** commitment and **Selective-Disclosure Attestation Bundle** (`RunAttestationBundle`) signed with HMAC-SHA256 for external regulatory verification without exposing raw prompts or PII.

```bash
bench attest-run <run_id> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing run artifacts. *(Default: `data/runs`)*
- `--signing-key TEXT`: HMAC-SHA256 signing key. *(Default: `agent-bench-default-audit-key`)*
- `--output / -o PATH`: Optional output JSON file path.

### Example
```bash
bench attest-run run_20261019_120000 -o data/runs/run_20261019_120000_attestation.json
```

---

## 23. `bench verify-attestation`

Cryptographically verifies the HMAC-SHA256 signature and every logarithmic `MerkleInclusionProof` in a `RunAttestationBundle`.

```bash
bench verify-attestation <attestation_path> [OPTIONS]
```

### Options
- `--signing-key TEXT`: HMAC-SHA256 signing key. *(Default: `agent-bench-default-audit-key`)*

### Example
```bash
bench verify-attestation data/runs/run_20261019_120000_attestation.json
```

---

## 24. `bench evolve-redteam`

Executes closed-loop 2PL IRT Fisher-guided evolutionary red-teaming (`EvolutionaryRedTeamFuzzer`) over a 2D `MAPElitesArchive` (`OWASP Category x Complexity Tier`).

```bash
bench evolve-redteam [OPTIONS]
```

### Options
- `--domain TEXT`: Target domain (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`). *(Default: `pix_assist`)*
- `--generations INTEGER`: Number of evolutionary generations. *(Default: `3`)*
- `--offspring INTEGER`: Offspring evaluated per generation. *(Default: `8`)*
- `--seed INTEGER`: Deterministic RNG seed. *(Default: `42`)*
- `--output / -o PATH`: Optional output JSON file path.

### Example
```bash
bench evolve-redteam --domain pix_assist --generations 4 --offspring 8 -o datasets/adversarial/evolved_pix.json
```

---

## 25. `bench attest-tee`

Binds a benchmark run's `RunAttestationBundle` SHA-256 Merkle root into a hardware-enforced **TEE Enclave Attestation Quote** (`TEEAttestationQuote`) with `PCR0` (runtime image), `PCR1` (suite config), `PCR2` (model lineage), and `PCR8` (enclave signer) measurements.

```bash
bench attest-tee <run_id> [OPTIONS]
```

### Options
- `--runs-dir PATH`: Directory containing run artifacts. *(Default: `data/runs`)*
- `--tee-arch TEXT`: TEE architecture (`aws_nitro_enclave`, `intel_tdx`, `amd_sev_snp`). *(Default: `aws_nitro_enclave`)*
- `--nonce TEXT`: Auditor anti-replay nonce.
- `--signing-key TEXT`: Enclave hardware signing key.
- `--output / -o PATH`: Optional output JSON file path.

### Example
```bash
bench attest-tee run_20261019_120000 --tee-arch aws_nitro_enclave --nonce audit_2026 -o tee_quote.json
```

---

## 26. `bench verify-tee-quote`

Verifies a `TEEAttestationQuote` signature, anti-replay nonce, `user_data` Merkle root binding, Golden PCR reference policy, and all inner `MerkleInclusionProof` claims.

```bash
bench verify-tee-quote <quote_path> [OPTIONS]
```

### Options
- `--signing-key TEXT`: Enclave hardware signing key.
- `--expected-nonce TEXT`: Expected auditor anti-replay nonce.

### Example
```bash
bench verify-tee-quote tee_quote.json --expected-nonce audit_2026
```

---

## 27. `bench stress-memory`

Executes a long-horizon multi-session episodic memory consolidation and interference stress test (`EpisodicMemoryStressRunner`, `MemoryInterferenceInjector`), computing **Constraint Retention Rate (CRR)**, **Temporal State Resolution Accuracy (TSRA)**, and **Interference Susceptibility Index (ISI)**.

```bash
bench stress-memory [OPTIONS]
```

### Options
- `--distractors INTEGER`: Number of proactive distractor turns to inject. *(Default: `8`)*
- `--compaction-interval INTEGER`: Turns between context compaction checkpoints. *(Default: `4`)*
- `--system-id TEXT`: System identifier for the stress report. *(Default: `episodic_agent`)*
- `--output / -o PATH`: Optional output JSON file path.

### Example
```bash
bench stress-memory --distractors 8 --compaction-interval 4 -o memory_stress.json
```

---

## 28. `bench audit-collusion`

Audits a multi-agent committee voting log for Sybil voting coalitions, covert steganographic signaling, and exact Coalition **Shapley Value** attribution (`MultiAgentCollusionDetector`, `ShapleyContributionAuditor`).

```bash
bench audit-collusion <committee_log_path> [OPTIONS]
```

### Options
- `--output / -o PATH`: Optional output JSON file path.

### Example
```bash
bench audit-collusion data/committee_votes.json -o collusion_audit.json
```

---

## 29. `bench compile-zk-circuit`

Compiles a benchmark evaluation run into a Goldilocks prime field ($\mathbb{F}_p, p = 2^{64} - 2^{32} + 1$) **Rank-1 Constraint System (R1CS)** arithmetic circuit (`ZKArithmeticCircuit`) and generates a Fiat-Shamir non-interactive witness proof (`R1CSWitnessProof`) enforcing Phase 0 safety non-violation, 16-bit score range proof, 3-bit autonomy tier range proof, and Merkle root commitment binding.

```bash
bench compile-zk-circuit <run_path> [OPTIONS]
```

### Options
- `--threshold-score FLOAT`: Minimum required global score in $[0.0, 1.0]$. *(Default: `0.70`)*
- `--min-tier INTEGER`: Minimum required graduated autonomy tier. *(Default: `2`)*
- `--output / -o PATH`: Optional output JSON bundle path.

### Example
```bash
bench compile-zk-circuit data/runs/run_001.json --threshold-score 0.75 --min-tier 2 -o zk_bundle.json
```

---

## 30. `bench verify-zk-circuit`

Verifies a compiled R1CS arithmetic circuit and Fiat-Shamir witness proof bundle (`ZKEvaluationCircuitCompiler.verify_proof`).

```bash
bench verify-zk-circuit <bundle_path>
```

### Example
```bash
bench verify-zk-circuit zk_bundle.json
```

---

## 31. `bench synthesize-policy`

Synthesizes self-healing bounded LTL `SymbolicInvariant` rules, outbound egress firewall domain blocks, and RAG injection trigger signatures from discovered red-team breakthroughs (`SelfHealingPolicySynthesizer`), verifying 100% adversarial block rate and 0% benign false-positive regression.

```bash
bench synthesize-policy <breakthroughs_path> [OPTIONS]
```

### Options
- `--patch-id TEXT`: Identifier for the synthesized policy patch. *(Default: `self_healing_patch_v1`)*
- `--output / -o PATH`: Optional output JSON file path.

### Example
```bash
bench synthesize-policy data/redteam_breakthroughs.json --patch-id patch_v2 -o policy_patch.json
```

---

## 32. `bench fold-zk-proofs`

Folds multiple benchmark evaluation run manifests into a single constant-size **Nova-style Relaxed R1CS Incrementally Verifiable Computation (IVC)** accumulator bundle (`IVCAccumulatorBundle`) over the Goldilocks prime field ($\mathbb{F}_p, p = 2^{64} - 2^{32} + 1$) with exact cross-term $T$ verification (`NovaIVCFolder`).

```bash
bench fold-zk-proofs <run_path_1> <run_path_2> [...] [OPTIONS]
```

### Options
- `--threshold-score FLOAT`: Minimum required global score in $[0.0, 1.0]$. *(Default: `0.70`)*
- `--min-tier INTEGER`: Minimum required graduated autonomy tier. *(Default: `2`)*
- `--output / -o PATH`: Optional output JSON bundle path.

### Example
```bash
bench fold-zk-proofs data/runs/run_001.json data/runs/run_002.json data/runs/run_003.json -o ivc_bundle.json
```

---

## 33. `bench verify-ivc`

Cryptographically verifies a Nova-style Relaxed R1CS IVC accumulator bundle (`NovaIVCFolder.verify_ivc_bundle`), checking every step's cross-term $T$, Fiat-Shamir folding challenge $r_i$, transcript hash chain, and final folded equation $(A Z_n) \circ (B Z_n) \equiv u_n \cdot (C Z_n) + E_n \pmod p$.

```bash
bench verify-ivc <bundle_path>
```

### Example
```bash
bench verify-ivc ivc_bundle.json
```

---

## 34. `bench simulate-regime-shift`

Simulates mid-episode non-stationary macroeconomic, regulatory, and cybersecurity threat-level regime shifts (`NonStationaryWorldSimulator`, `RegimeShiftShock`) across domain sandboxes (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`), measuring **Adaptation Lag**, **Post-Shift Regret**, and **Policy Stale-Rule Violation Rate**.

```bash
bench simulate-regime-shift [OPTIONS]
```

### Options
- `--domain TEXT`: Target domain (`pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`). *(Default: `pix_assist`)*
- `--turns INTEGER`: Total episode horizon turns. *(Default: `8`)*
- `--adaptive / --rigid`: Toggle whether the simulated policy adapts to regime shift signals or remains locked to pre-shift assumptions. *(Default: `--adaptive`)*
- `--output / -o PATH`: Optional output JSON report path.

### Example
```bash
bench simulate-regime-shift --domain pix_assist --turns 8 --adaptive -o regime_adaptation.json
```

---

## 35. `bench steer-circuit`

Synthesizes mechanistic circuit clamping interventions (`FeatureClampIntervention`) from a traced `AttributionGraph` and evaluates **Safety Recovery Rate (SRR)** and **Capability Preservation Ratio (CPR)** (`MechanisticActivationSteerer`).

```bash
bench steer-circuit [OPTIONS]
```

### Options
- `--input-dim INTEGER`: Residual stream dimensionality. *(Default: `4`)*
- `--top-k INTEGER`: SAE Top-K sparsity. *(Default: `3`)*
- `--output / -o PATH`: Optional output JSON report path.

### Example
```bash
bench steer-circuit --input-dim 4 --top-k 3 -o steer_report.json
```

---

## 36. `bench prove-federated-leaderboard`

Verifies institutional R1CS witness proofs over the Goldilocks prime field $\mathbb{F}_p$, filters out Byzantine metric-poisoning submissions, folds all valid runs into a single constant-size Nova IVC accumulator (`NovaIVCFolder`), and releases an $(\varepsilon, \delta)$-Differentially Private federated leaderboard (`ZKFederatedLeaderboardAggregator`).

```bash
bench prove-federated-leaderboard <submissions_path> [OPTIONS]
```

### Options
- `--epsilon FLOAT`: Differential privacy $\varepsilon$ budget. *(Default: `1.0`)*
- `--min-sites INTEGER`: Minimum $k$-anonymity participating sites. *(Default: `2`)*
- `--threshold-score FLOAT`: Minimum required R1CS global score. *(Default: `0.70`)*
- `--output / -o PATH`: Optional output JSON bundle path.

### Example
```bash
bench prove-federated-leaderboard data/federated/zk_submissions.json --epsilon 1.0 -o zk_fed_bundle.json
```

---

## 37. `bench verify-federated-leaderboard`

Cryptographically verifies a `ZKFederatedLeaderboardBundle` (`ZKFederatedLeaderboardAggregator.verify_federated_bundle`), checking $k$-anonymity satisfaction, the folded Nova IVC accumulator equation, and the SHA-256 DP commitment hash.

```bash
bench verify-federated-leaderboard <bundle_path>
```

### Example
```bash
bench verify-federated-leaderboard zk_fed_bundle.json
```

---

## 38. `bench simulate-contagion`

Simulates multi-agent market microstructure liquidity contagion, endogenous limit order-book price impact, pro-cyclical fire-sale herding, and circuit-breaker compliance (`SystemicContagionSandbox`).

```bash
bench simulate-contagion [OPTIONS]
```

### Options
- `--agents INTEGER`: Number of interacting institutional agents. *(Default: `4`)*
- `--steps INTEGER`: Total order-book simulation steps. *(Default: `8`)*
- `--stabilizing / --herding`: Toggle stabilizing liquidity-providing policies vs. pro-cyclical herding policies. *(Default: `--stabilizing`)*
- `--output / -o PATH`: Optional output JSON report path.

### Example
```bash
bench simulate-contagion --agents 4 --steps 8 --stabilizing -o contagion_report.json
```

---

## 39. `bench verify-pcat`

Builds and cryptographically verifies a **Proof-Carrying Agent Trajectory (PCAT)** (`PCATVerifier`, `ProofCarryingStepEnvelope`), checking tool pre/post-condition contracts, numeric bounds, forbidden argument patterns, and SHA-256 parent-to-child envelope hash chaining.

```bash
bench verify-pcat [OPTIONS]
```

### Options
- `--trajectory-id TEXT`: Trajectory identifier. *(Default: `pcat_cli_001`)*
- `--output / -o PATH`: Optional output JSON report path.

### Example
```bash
bench verify-pcat --trajectory-id pcat_cli_001 -o pcat_report.json
```

---

## 40. `bench simulate-cat`

Executes a sequential **Bayesian 2PL IRT Computerized Adaptive Testing (CAT)** session (`BayesianCATEngine`), updating the Maximum A Posteriori ($\hat{\theta}_t$) ability estimate via damped Newton-Raphson optimization, selecting the next item via Maximum Fisher Information, and stopping dynamically once $\text{SE}(\hat{\theta}_t) \le \text{target\_se}$.

```bash
bench simulate-cat [OPTIONS]
```

### Options
- `--system-id TEXT`: System identifier. *(Default: `cat_agent_v1`)*
- `--true-theta FLOAT`: Simulated true latent ability $\theta$. *(Default: `0.85`)*
- `--target-se FLOAT`: Target posterior standard error stopping threshold. *(Default: `0.42`)*
- `--max-items INTEGER`: Maximum adaptive items to administer. *(Default: `8`)*
- `--output / -o PATH`: Optional output JSON report path.

### Example
```bash
bench simulate-cat --system-id cat_agent_v1 --true-theta 0.85 --target-se 0.42 -o cat_report.json
```

---

## 41. `bench audit-jurisdictions`

Audits multi-jurisdictional statutory compliance across **Brazil (`BR_BACEN_LGPD`)**, **European Union (`EU_AI_ACT_GDPR`)**, and **United States (`US_SEC_FINRA_NIST`)**, synthesizing **Strictest-Compatible Superset** conflict-of-laws resolutions (`RegulatoryComplianceHarmonizer`).

```bash
bench audit-jurisdictions [OPTIONS]
```

### Options
- `--system-id TEXT`: System identifier. *(Default: `global_bank_agent`)*
- `--output / -o PATH`: Optional output JSON report path.

### Example
```bash
bench audit-jurisdictions --system-id global_bank_agent -o regulatory_report.json
```

